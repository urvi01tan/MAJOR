"""
Concept Drift Detector
======================
Monitors multiple signals for distributional shift in the patient data stream.

Three complementary detectors run in parallel:

1. ADWIN (Adaptive Windowing) — tracks feature distribution drift
   - Monitors the mean of prediction probabilities over an adaptive window
   - Detects when the distribution of incoming data changes significantly

2. DDM (Drift Detection Method) — tracks model error rate
   - Uses Gaussian control charts on binary prediction errors
   - Raises "warning" before raising "drift" alert

3. Page-Hinkley Test — detects abrupt mean shifts
   - Sensitive to sudden spikes in prediction scores
   - Good for catching sensor recalibrations or ICU population shifts

On drift detected:
  - Logs the drift event to audit trail
  - Raises a flag visible on the monitoring dashboard
  - Optionally triggers model snapshot before/after
"""

from __future__ import annotations

import logging
import math
from collections import deque
from datetime import datetime
from typing import Any

import yaml

log = logging.getLogger(__name__)

try:
    from river import drift as river_drift
    from river.drift.binary import DDM as RiverDDM
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    RiverDDM = None
    log.error("River drift module not available. pip install river")


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Drift event
# ---------------------------------------------------------------------------

class DriftEvent:
    __slots__ = ["detector", "timestamp", "metric_value", "severity", "details"]

    def __init__(
        self,
        detector: str,
        timestamp: datetime,
        metric_value: float,
        severity: str,
        details: dict,
    ):
        self.detector = detector
        self.timestamp = timestamp
        self.metric_value = metric_value
        self.severity = severity  # "warning" | "drift"
        self.details = details

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "timestamp": self.timestamp.isoformat(),
            "metric_value": self.metric_value,
            "severity": self.severity,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Page-Hinkley detector (pure Python, no River dependency)
# ---------------------------------------------------------------------------

class PageHinkley:
    """
    Page-Hinkley test for detecting upward shifts in mean.
    Fires when cumulative sum of deviations exceeds threshold.
    """

    def __init__(self, threshold: float = 50.0, alpha: float = 0.005):
        self.threshold = threshold
        self.alpha = alpha  # allowable increase per sample
        self._sum = 0.0
        self._n = 0
        self._mean = 0.0
        self.drift_detected = False

    def update(self, value: float) -> bool:
        """Update with new value. Returns True if drift detected."""
        self._n += 1
        self._mean += (value - self._mean) / self._n
        self._sum = max(0.0, self._sum + (value - self._mean - self.alpha))
        self.drift_detected = self._sum > self.threshold
        return self.drift_detected

    def reset(self) -> None:
        self._sum = 0.0
        self._n = 0
        self._mean = 0.0
        self.drift_detected = False


# ---------------------------------------------------------------------------
# Main DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """
    Runs ADWIN, DDM, and Page-Hinkley in parallel.
    Call update() with each new prediction error/score.
    Fires callbacks and maintains event log when drift is detected.
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        cfg = _load_cfg(cfg_path)
        dc = cfg["drift"]

        # ADWIN: monitors prediction score distribution
        if RIVER_AVAILABLE:
            self._adwin = river_drift.ADWIN(delta=dc["adwin_delta"])
            # DDM moved to river.drift.binary in River >= 0.21
            self._ddm = RiverDDM(
                warning_threshold=dc.get("ddm_warning_level", 2.0),
                drift_threshold=dc.get("ddm_drift_level", 3.0),
            )
        else:
            self._adwin = None
            self._ddm = None

        # Page-Hinkley: River's native implementation
        ph_threshold = dc["page_hinkley_threshold"]
        ph_delta = dc["page_hinkley_alpha"]  # River uses 'delta' for the allowed change
        if RIVER_AVAILABLE:
            self._ph = river_drift.PageHinkley(
                threshold=ph_threshold,
                delta=ph_delta,
            )
        else:
            # Pure-Python fallback
            self._ph = PageHinkley(threshold=ph_threshold, alpha=ph_delta)
        self._ph_threshold = ph_threshold
        self._ph_delta = ph_delta

        self.check_every = dc["check_every_n"]
        self._n_updates = 0
        self._drift_events: list[DriftEvent] = []
        self._in_warning = False

        # Rolling buffers for dashboard display
        self._adwin_scores: deque[float] = deque(maxlen=500)
        self._error_rates: deque[float] = deque(maxlen=500)
        self._ph_sums: deque[float] = deque(maxlen=500)

    def update(
        self,
        prediction_score: float,
        true_label: int | None = None,
        now: datetime | None = None,
    ) -> list[DriftEvent]:
        """
        Update all detectors with new prediction.

        Args:
            prediction_score: model's output probability [0, 1]
            true_label: ground truth label (if known); used by DDM
            now: timestamp for event logging

        Returns:
            List of new DriftEvent objects (empty if no drift)
        """
        self._n_updates += 1
        now = now or datetime.utcnow()
        new_events: list[DriftEvent] = []

        if math.isnan(prediction_score):
            return new_events

        self._adwin_scores.append(prediction_score)

        # -- ADWIN --
        if self._adwin is not None:
            self._adwin.update(prediction_score)
            if self._adwin.drift_detected:
                evt = DriftEvent(
                    detector="ADWIN",
                    timestamp=now,
                    metric_value=prediction_score,
                    severity="drift",
                    details={"n_samples": self._n_updates},
                )
                new_events.append(evt)
                self._drift_events.append(evt)
                self._adwin = self._adwin.__class__(delta=self._adwin.delta)  # reset
                log.warning(f"[DriftDetector] 🔴 ADWIN drift at n={self._n_updates}")

        # -- DDM (requires binary error label) --
        if self._ddm is not None and true_label is not None:
            pred_class = 1 if prediction_score >= 0.5 else 0
            error = int(pred_class != true_label)
            self._error_rates.append(error)
            self._ddm.update(error)

            if self._ddm.drift_detected:
                evt = DriftEvent(
                    detector="DDM",
                    timestamp=now,
                    metric_value=float(error),
                    severity="drift",
                    details={"n_samples": self._n_updates},
                )
                new_events.append(evt)
                self._drift_events.append(evt)
                self._ddm = RiverDDM()  # reset
                log.warning(f"[DriftDetector] 🔴 DDM drift at n={self._n_updates}")

            elif self._ddm.warning_detected:
                if not self._in_warning:
                    self._in_warning = True
                    evt = DriftEvent(
                        detector="DDM",
                        timestamp=now,
                        metric_value=float(error),
                        severity="warning",
                        details={"n_samples": self._n_updates},
                    )
                    new_events.append(evt)
                    self._drift_events.append(evt)
                    log.warning(f"[DriftDetector] 🟡 DDM warning at n={self._n_updates}")
            else:
                self._in_warning = False

        # -- Page-Hinkley --
        self._ph.update(prediction_score)
        ph_fired = self._ph.drift_detected if hasattr(self._ph, 'drift_detected') else False
        # Get internal sum for dashboard display
        ph_sum = getattr(self._ph, '_sum_increase', getattr(self._ph, '_sum', 0.0))
        self._ph_sums.append(ph_sum)
        if ph_fired:
            ph_sum_val = getattr(self._ph, '_sum_increase', getattr(self._ph, '_sum', 0.0))
            evt = DriftEvent(
                detector="PageHinkley",
                timestamp=now,
                metric_value=ph_sum_val,
                severity="drift",
                details={
                    "ph_sum": ph_sum_val,
                    "ph_threshold": self._ph_threshold,
                    "n_samples": self._n_updates,
                },
            )
            new_events.append(evt)
            self._drift_events.append(evt)
            # Re-instantiate to reset state
            if RIVER_AVAILABLE:
                self._ph = river_drift.PageHinkley(
                    threshold=self._ph_threshold,
                    delta=self._ph_delta,
                )
            else:
                self._ph = PageHinkley(threshold=self._ph_threshold, alpha=self._ph_delta)
            log.warning(f"[DriftDetector] 🔴 Page-Hinkley drift at n={self._n_updates}")

        return new_events

    # ------------------------------------------------------------------
    @property
    def drift_events(self) -> list[dict]:
        return [e.to_dict() for e in self._drift_events]

    @property
    def total_drift_count(self) -> int:
        return len(self._drift_events)

    @property
    def last_drift(self) -> dict | None:
        if not self._drift_events:
            return None
        return self._drift_events[-1].to_dict()

    def drift_score_series(self) -> dict[str, list[float]]:
        """Return recent detector time-series for dashboard charts."""
        return {
            "adwin_scores": list(self._adwin_scores),
            "error_rates": list(self._error_rates),
            "ph_sums": list(self._ph_sums),
        }

    def status(self) -> dict:
        ph_sum = getattr(self._ph, '_sum_increase', getattr(self._ph, '_sum', 0.0))
        return {
            "n_updates": self._n_updates,
            "total_drift_events": self.total_drift_count,
            "in_warning": self._in_warning,
            "last_drift": self.last_drift,
            "ph_current_sum": ph_sum,
            "ph_threshold": self._ph_threshold,
        }
