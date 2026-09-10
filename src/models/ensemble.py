"""
Ensemble / Model Selector
==========================
Decides which model's prediction to use at runtime:
  - Online model (incremental, adapting)
  - Baseline model (static, validated safety net)

Selector logic:
  1. Shadow mode: online model predicts but output is not used → always use baseline
  2. Normal mode:
     a. Use online model if its rolling AUROC > baseline_auroc − rollback_gap
     b. Auto-rollback to baseline if online AUROC drops below threshold
     c. Rollback is logged and an alert is raised to the monitoring dashboard
  3. Output: dict with prediction, model_used, confidence, flags

Also handles prediction caching (don't re-score same patient within cooldown).
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import yaml

log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class EnsembleSelector:
    """
    Selects between online and baseline models based on real-time performance.

    Usage:
        selector = EnsembleSelector(online_model, baseline_model)
        result = selector.predict(patient_id, features)
    """

    def __init__(self, online_model, baseline_model, cfg_path: str = "config/settings.yaml"):
        self.cfg = _load_cfg(cfg_path)
        self.online = online_model
        self.baseline = baseline_model

        self.rollback_gap: float = self.cfg["baseline"]["rollback_auroc_gap"]
        self.alert_threshold: float = self.cfg["alerting"]["risk_threshold"]
        self.cooldown_minutes: int = self.cfg["alerting"]["alert_cooldown_minutes"]
        self.shadow_mode: bool = self.cfg["online_model"]["shadow_mode"]

        # Track which model is currently active
        self._active_model: str = "online"  # "online" | "baseline"
        self._rollback_count: int = 0
        self._rollback_events: list[dict] = []

        # Per-patient alert cooldown: patient_id → last_alert_time
        self._last_alert: dict[str, datetime] = {}

        # Rolling AUROC comparison
        self._baseline_auroc: float = baseline_model.auroc
        log.info(
            f"[EnsembleSelector] shadow_mode={self.shadow_mode}, "
            f"baseline_auroc={self._baseline_auroc:.3f}, rollback_gap={self.rollback_gap}"
        )

    # ------------------------------------------------------------------
    def predict(
        self, patient_id: str, features: dict[str, float], now: datetime | None = None
    ) -> dict[str, Any]:
        """
        Return a prediction result dict:
          {
            patient_id, timestamp, risk_score, model_used, alert,
            online_score, baseline_score, active_model
          }
        """
        now = now or datetime.utcnow()

        # Always compute both for monitoring
        online_score = self.online.predict(features)
        baseline_score = self.baseline.predict(features)

        # Determine which score to use
        self._maybe_rollback(now)

        if self.shadow_mode or self._active_model == "baseline":
            used_score = baseline_score
            model_used = "baseline"
        else:
            used_score = online_score
            model_used = "online"

        # Alert decision (with cooldown)
        alert = self._should_alert(patient_id, used_score, now)

        result = {
            "patient_id": patient_id,
            "timestamp": now.isoformat(),
            "risk_score": round(used_score, 4),
            "model_used": model_used,
            "active_model": self._active_model,
            "shadow_mode": self.shadow_mode,
            "alert": alert,
            "alert_threshold": self.alert_threshold,
            "online_score": round(online_score, 4),
            "baseline_score": round(baseline_score, 4),
            "online_rolling_auroc": round(self.online.rolling_auroc, 4),
            "baseline_auroc": round(self._baseline_auroc, 4),
        }
        return result

    # ------------------------------------------------------------------
    def _maybe_rollback(self, now: datetime) -> None:
        """Check if online model should be rolled back to baseline."""
        online_auroc = self.online.rolling_auroc
        if online_auroc != online_auroc:  # NaN check
            return

        threshold = self._baseline_auroc - self.rollback_gap

        if self._active_model == "online" and online_auroc < threshold:
            self._active_model = "baseline"
            self._rollback_count += 1
            event = {
                "event": "rollback_to_baseline",
                "timestamp": now.isoformat(),
                "online_auroc": online_auroc,
                "baseline_auroc": self._baseline_auroc,
                "gap": self._baseline_auroc - online_auroc,
                "rollback_number": self._rollback_count,
            }
            self._rollback_events.append(event)
            log.warning(
                f"[EnsembleSelector] ⚠️  ROLLBACK #{self._rollback_count}: "
                f"online AUROC {online_auroc:.3f} < baseline {self._baseline_auroc:.3f} "
                f"− {self.rollback_gap} = {threshold:.3f}"
            )

        elif self._active_model == "baseline" and online_auroc >= threshold + 0.02:
            # Hysteresis: 2% buffer before switching back
            self._active_model = "online"
            log.info(
                f"[EnsembleSelector] ✅  RESTORED online model: AUROC {online_auroc:.3f}"
            )

    # ------------------------------------------------------------------
    def _should_alert(
        self, patient_id: str, score: float, now: datetime
    ) -> bool:
        if score < self.alert_threshold:
            return False
        last = self._last_alert.get(patient_id)
        if last and (now - last) < timedelta(minutes=self.cooldown_minutes):
            return False
        self._last_alert[patient_id] = now
        return True

    # ------------------------------------------------------------------
    def force_rollback(self) -> None:
        """Manually force rollback to baseline (e.g., clinician override)."""
        self._active_model = "baseline"
        log.warning("[EnsembleSelector] Manual rollback to baseline triggered.")

    def restore_online(self) -> None:
        """Manually restore online model."""
        self._active_model = "online"
        log.info("[EnsembleSelector] Online model manually restored.")

    # ------------------------------------------------------------------
    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def rollback_count(self) -> int:
        return self._rollback_count

    @property
    def rollback_events(self) -> list[dict]:
        return self._rollback_events.copy()

    def status(self) -> dict:
        return {
            "active_model": self._active_model,
            "shadow_mode": self.shadow_mode,
            "online_auroc": self.online.rolling_auroc,
            "baseline_auroc": self._baseline_auroc,
            "rollback_count": self._rollback_count,
            "rollback_threshold": self._baseline_auroc - self.rollback_gap,
        }
