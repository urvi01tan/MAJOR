"""
Main Pipeline Orchestrator
===========================
Ties all components together into a single streaming pipeline.

Flow per event:
  Event → WindowAggregator → MissingDataHandler
       → EnsembleSelector.predict()
       → DriftDetector.update()
       → Explainer.explain()
       → AuditLogger.log_prediction()
       → ModelRegistry.maybe_snapshot()
       → Online model delayed label update (via enqueue_for_update)

Also handles:
  - Clinician feedback ingestion → online model labelled update
  - Periodic stats reporting
  - Graceful shutdown (flush + snapshot)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from src.audit.logger import AuditLogger
from src.drift.detector import DriftDetector
from src.explainability.explainer import Explainer
from src.features.missing_handler import MissingDataHandler
from src.features.window_aggregator import WindowAggregator
from src.models.baseline_model import BaselineModel
from src.models.ensemble import EnsembleSelector
from src.models.online_model import OnlineModel
from src.models.registry import ModelRegistry

log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class EWSPipeline:
    """
    Early Warning System pipeline.

    Usage:
        pipeline = EWSPipeline()
        pipeline.start()
        for event in stream:
            result = pipeline.process_event(event)
        pipeline.shutdown()
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        self.cfg = _load_cfg(cfg_path)
        self._cfg_path = cfg_path
        self._n_events = 0
        self._n_predictions = 0
        self._n_alerts = 0
        self._started_at: datetime | None = None
        self._lock = threading.Lock()

        # Recent results buffer for dashboard
        self._recent_results: deque[dict] = deque(maxlen=500)
        self._patient_latest: dict[str, dict] = {}

        # Callbacks (e.g., dashboard refresh trigger)
        self._alert_callbacks: list[Callable[[dict], None]] = []

        log.info("[EWSPipeline] Initialising components …")
        self._init_components()

    def _init_components(self) -> None:
        # Feature layer
        self.aggregator = WindowAggregator(self._cfg_path)
        self.imputer = MissingDataHandler(self._cfg_path)

        # Models
        self.online_model = OnlineModel(self._cfg_path)
        self.baseline_model = BaselineModel(self._cfg_path)
        try:
            self.baseline_model.load()
            log.info("[EWSPipeline] Baseline model loaded.")
        except FileNotFoundError:
            log.warning(
                "[EWSPipeline] Baseline model not found — run scripts/train_baseline.py. "
                "Using online model only (shadow mode disabled)."
            )

        self.selector = EnsembleSelector(self.online_model, self.baseline_model, self._cfg_path)
        self.registry = ModelRegistry(self.online_model, self._cfg_path)
        self.drift_detector = DriftDetector(self._cfg_path)
        self.explainer = Explainer(self.online_model, self.baseline_model)
        self.audit = AuditLogger(self._cfg_path)

        # Label tracking per admission (for delayed online updates)
        self._label_map: dict[str, int] = {}  # hadm_id → label

    # ------------------------------------------------------------------
    def start(self) -> "EWSPipeline":
        self._started_at = datetime.utcnow()
        log.info("[EWSPipeline] ▶  Pipeline started.")
        return self

    def shutdown(self) -> None:
        """Gracefully flush and snapshot before shutdown."""
        log.info("[EWSPipeline] Shutting down — taking final snapshot …")
        self.registry.snapshot(reason="shutdown")
        log.info(
            f"[EWSPipeline] Done. Events={self._n_events}, "
            f"Predictions={self._n_predictions}, Alerts={self._n_alerts}"
        )

    # ------------------------------------------------------------------
    def process_event(self, event: dict) -> dict | None:
        """
        Process a single streaming event.

        Args:
            event: dict from StreamSimulator (patient_id, feature_name, value, timestamp…)

        Returns:
            Prediction result dict, or None if prediction was not triggered.
        """
        with self._lock:
            self._n_events += 1

            patient_id = event["patient_id"]
            hadm_id = event.get("hadm_id")
            stay_id = event.get("stay_id")
            ts: datetime = event["timestamp"]

            # 1. Update feature windows
            features_raw = self.aggregator.update(event)
            if not features_raw:
                return None

            # 2. Impute missing values
            features = self.imputer.impute_and_update(features_raw)

            # 3. Predict
            result = self.selector.predict(patient_id, features, now=ts)
            result["hadm_id"] = hadm_id
            result["stay_id"] = stay_id
            self._n_predictions += 1

            # 4. Detect drift
            drift_events = self.drift_detector.update(
                prediction_score=result["risk_score"],
                now=ts,
            )
            for de in drift_events:
                self.audit.log_drift_event(
                    de.to_dict(), self.online_model.version, action_taken="logged_only"
                )
                # Take pre-drift snapshot for rollback safety
                self.registry.snapshot(reason=f"pre_drift_{de.detector}")

            # 5. Generate explanation (only for alerts or high-risk)
            explanation = None
            if result["risk_score"] >= self.cfg["alerting"]["risk_threshold"] * 0.8:
                explanation = self.explainer.explain(
                    features, result["risk_score"], model_used=result["model_used"]
                )

            # 6. Audit log
            prediction_id = self.audit.log_prediction(result, features, explanation)
            result["prediction_id"] = prediction_id
            result["explanation"] = explanation

            # 7. Enqueue for online model update (label arrives later)
            self.online_model.enqueue_for_update(features, ts)

            # 8. If label is known for this admission, provide it
            if hadm_id and hadm_id in self._label_map:
                self.online_model.provide_label(hadm_id, self._label_map[hadm_id], ts)

            # 9. Auto-snapshot
            self.registry.maybe_snapshot()

            # 10. Alert callbacks
            if result.get("alert"):
                self._n_alerts += 1
                for cb in self._alert_callbacks:
                    try:
                        cb(result)
                    except Exception:
                        pass

            # 11. Store for dashboard
            self._recent_results.append(result)
            self._patient_latest[patient_id] = result

            return result

    def process_batch(self, events: list[dict]) -> list[dict]:
        """Process a batch of events, returning results for each."""
        results = []
        for event in events:
            r = self.process_event(event)
            if r:
                results.append(r)
        return results

    # ------------------------------------------------------------------
    def ingest_feedback(
        self,
        prediction_id: str,
        patient_id: str,
        hadm_id: str,
        feedback: str,
        clinician_id: str = "anonymous",
        notes: str = "",
    ) -> None:
        """
        Ingest clinician feedback and use it to update the online model.
        feedback: "true_positive" | "false_positive"
        """
        label = 1 if feedback == "true_positive" else 0
        self._label_map[hadm_id] = label
        self.online_model.provide_label(hadm_id, label)

        # Log feedback
        self.audit.log_feedback(prediction_id, patient_id, feedback, clinician_id, notes)
        self.audit.log_model_update(self.online_model, trigger="clinician_feedback", n_samples=1)

        log.info(f"[EWSPipeline] Clinician feedback: {feedback} for patient {patient_id}")

    def load_labels(self, label_map: dict[str, int]) -> None:
        """Load batch of known labels {hadm_id → label} from historical data."""
        self._label_map.update(label_map)
        for hadm_id, label in label_map.items():
            self.online_model.provide_label(hadm_id, label)
        log.info(f"[EWSPipeline] Loaded {len(label_map)} labels from label map.")

    # ------------------------------------------------------------------
    def add_alert_callback(self, callback: Callable[[dict], None]) -> None:
        self._alert_callbacks.append(callback)

    # ------------------------------------------------------------------
    def recent_results(self, n: int = 50) -> list[dict]:
        results = list(self._recent_results)
        return results[-n:]

    def high_risk_patients(self) -> list[dict]:
        threshold = self.cfg["alerting"]["risk_threshold"]
        patients = list(self._patient_latest.values())
        return sorted(
            [p for p in patients if p["risk_score"] >= threshold],
            key=lambda x: x["risk_score"],
            reverse=True,
        )

    def status(self) -> dict:
        uptime = None
        if self._started_at:
            uptime = (datetime.utcnow() - self._started_at).total_seconds()
        return {
            "uptime_seconds": uptime,
            "n_events_processed": self._n_events,
            "n_predictions": self._n_predictions,
            "n_alerts": self._n_alerts,
            "n_patients_tracked": self.aggregator.patient_count(),
            "active_model": self.selector.active_model,
            "online_auroc": self.online_model.rolling_auroc,
            "baseline_auroc": self.baseline_model.auroc,
            "rollback_count": self.selector.rollback_count,
            "total_drift_events": self.drift_detector.total_drift_count,
            "model_snapshots": self.registry.version_count(),
        }
