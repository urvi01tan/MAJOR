from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime
from typing import Any, Callable

import yaml

from src.audit.logger import AuditLogger
from src.clinical.scores import clinical_bundle
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
    def __init__(self, cfg_path: str = "config/settings.yaml"):
        self.cfg = _load_cfg(cfg_path)
        self._cfg_path = cfg_path
        self._n_events = 0
        self._n_predictions = 0
        self._n_alerts = 0
        self._started_at: datetime | None = None
        self._lock = threading.Lock()

        self._recent_results: deque[dict] = deque(maxlen=800)
        self._patient_history: dict[str, deque] = {}
        self._patient_latest: dict[str, dict] = {}
        self._acknowledged: set[str] = set()
        self._watchlist: set[str] = set()
        self._alert_callbacks: list[Callable[[dict], None]] = []
        self._label_map: dict[str, int] = {}
        self._alert_timestamps: deque[datetime] = deque(maxlen=5000)

        self._init_components()

    def _init_components(self) -> None:
        self.aggregator = WindowAggregator(self._cfg_path)
        self.imputer = MissingDataHandler(self._cfg_path)

        self.online_model = OnlineModel(self._cfg_path)
        self.baseline_model = BaselineModel(self._cfg_path)
        try:
            self.baseline_model.load()
            log.info("[EWSPipeline] Baseline model loaded.")
        except FileNotFoundError:
            log.warning(
                "[EWSPipeline] Baseline model not found — run scripts/train_baseline.py"
            )

        self.selector = EnsembleSelector(self.online_model, self.baseline_model, self._cfg_path)
        self.registry = ModelRegistry(self.online_model, self._cfg_path)
        self.drift_detector = DriftDetector(self._cfg_path)
        self.explainer = Explainer(self.online_model, self.baseline_model)
        self.audit = AuditLogger(self._cfg_path)

    def start(self) -> "EWSPipeline":
        self._started_at = datetime.utcnow()
        log.info("[EWSPipeline] Pipeline started.")
        return self

    def shutdown(self) -> None:
        log.info("[EWSPipeline] Shutting down — saving snapshot.")
        self.registry.snapshot(reason="shutdown")

    def process_event(self, event: dict) -> dict | None:
        with self._lock:
            self._n_events += 1

            patient_id = event["patient_id"]
            hadm_id = event.get("hadm_id") or patient_id
            stay_id = event.get("stay_id")
            ts: datetime = event["timestamp"]
            sepsis_label = event.get("sepsis_label")

            features_raw = self.aggregator.update(event)
            if not features_raw:
                return None

            features = self.imputer.impute_and_update(features_raw)

            result = self.selector.predict(patient_id, features, now=ts)
            result["hadm_id"] = hadm_id
            result["stay_id"] = stay_id
            result["acknowledged"] = patient_id in self._acknowledged
            result["on_watchlist"] = patient_id in self._watchlist
            self._n_predictions += 1

            vitals = self.aggregator.latest_vitals(patient_id)
            clinical = clinical_bundle(vitals)
            result.update(clinical)

            drift_events = self.drift_detector.update(
                prediction_score=result["risk_score"],
                now=ts,
            )
            for de in drift_events:
                self.audit.log_drift_event(
                    de.to_dict(), self.online_model.version, action_taken="logged_only"
                )
                self.registry.snapshot(reason=f"pre_drift_{de.detector}")

            explanation = None
            if result["risk_score"] >= self.cfg["alerting"]["risk_threshold"] * 0.8:
                explanation = self.explainer.explain(
                    features, result["risk_score"], model_used=result["model_used"]
                )

            prediction_id = self.audit.log_prediction(result, features, explanation)
            result["prediction_id"] = prediction_id
            result["explanation"] = explanation

            eventual = None
            if sepsis_label is not None:
                eventual = int(sepsis_label)
            elif patient_id in self._label_map:
                eventual = self._label_map[patient_id]
            elif hadm_id in self._label_map:
                eventual = self._label_map[hadm_id]

            self.online_model.enqueue_for_update(
                features, ts, patient_id=patient_id, eventual_label=eventual
            )

            self.registry.maybe_snapshot()

            if result.get("alert"):
                self._n_alerts += 1
                self._acknowledged.discard(patient_id)
                for cb in self._alert_callbacks:
                    try:
                        cb(result)
                    except Exception:
                        pass

            self._recent_results.append(result)
            self._patient_latest[patient_id] = result

            # Track per-patient history (last 200 predictions per patient)
            if patient_id not in self._patient_history:
                self._patient_history[patient_id] = deque(maxlen=200)
            self._patient_history[patient_id].append({
                "timestamp": result.get("timestamp"),
                "risk_score": result.get("risk_score", 0),
                "online_score": result.get("online_score", 0),
                "baseline_score": result.get("baseline_score", 0),
                "alert": result.get("alert", False),
                "news2": result.get("news2"),
                "sofa": result.get("sofa"),
                "severity": result.get("severity"),
            })

            if result.get("alert"):
                self._alert_timestamps.append(ts)

            return result

    def process_batch(self, events: list[dict]) -> list[dict]:
        results = []
        for event in events:
            r = self.process_event(event)
            if r:
                results.append(r)
        return results

    def ingest_feedback(
        self,
        prediction_id: str,
        patient_id: str,
        hadm_id: str | None = None,
        feedback: str = "true_positive",
        clinician_id: str = "anonymous",
        notes: str = "",
    ) -> None:
        label = 1 if feedback == "true_positive" else 0
        key = hadm_id or patient_id
        self._label_map[key] = label
        self._label_map[patient_id] = label
        self.online_model.provide_label(patient_id, label)
        self.online_model.provide_label(key, label)

        self.audit.log_feedback(prediction_id, patient_id, feedback, clinician_id, notes)
        self.audit.log_model_update(self.online_model, trigger="clinician_feedback", n_samples=1)
        log.info(f"[EWSPipeline] Clinician feedback: {feedback} for patient {patient_id}")

    def load_labels(self, label_map: dict[str, int]) -> None:
        self._label_map.update(label_map)
        for key, label in label_map.items():
            self.online_model.provide_label(key, label)
        log.info(f"[EWSPipeline] Loaded {len(label_map)} labels from label map.")

    def acknowledge_alert(self, patient_id: str) -> None:
        self._acknowledged.add(patient_id)
        latest = self._patient_latest.get(patient_id)
        if latest:
            latest["acknowledged"] = True

    def toggle_watchlist(self, patient_id: str) -> bool:
        if patient_id in self._watchlist:
            self._watchlist.discard(patient_id)
            return False
        self._watchlist.add(patient_id)
        return True

    def set_alert_threshold(self, threshold: float) -> None:
        self.cfg["alerting"]["risk_threshold"] = threshold
        self.selector.set_alert_threshold(threshold)

    def add_alert_callback(self, callback: Callable[[dict], None]) -> None:
        self._alert_callbacks.append(callback)

    def recent_results(self, n: int = 50) -> list[dict]:
        results = list(self._recent_results)
        return results[-n:]

    def high_risk_patients(self, include_acknowledged: bool = True) -> list[dict]:
        threshold = self.cfg["alerting"]["risk_threshold"]
        patients = list(self._patient_latest.values())
        filtered = [p for p in patients if p.get("risk_score", 0) >= threshold]
        if not include_acknowledged:
            filtered = [p for p in filtered if not p.get("acknowledged")]
        return sorted(filtered, key=lambda x: x.get("risk_score", 0), reverse=True)

    def census(self) -> list[dict]:
        patients = list(self._patient_latest.values())
        return sorted(patients, key=lambda x: x.get("risk_score", 0), reverse=True)

    def patient_risk_history(self, patient_id: str) -> list[dict]:
        """Return time-series of risk predictions for a specific patient."""
        hist = self._patient_history.get(patient_id)
        return list(hist) if hist else []

    def all_patient_risk_history(self, last_n: int = 100) -> dict[str, list[dict]]:
        """Return risk history for all tracked patients (for heatmap)."""
        return {
            pid: list(hist)[-last_n:]
            for pid, hist in self._patient_history.items()
        }

    def alert_rate_per_hour(self, window_hours: float = 1.0) -> float:
        """Return number of alerts fired in the last `window_hours`."""
        from datetime import timedelta
        if not self._alert_timestamps:
            return 0.0
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [t for t in self._alert_timestamps if t >= cutoff]
        return len(recent) / window_hours

    def watchlist_patients(self) -> list[dict]:
        return [
            p for p in self._patient_latest.values()
            if p.get("patient_id") in self._watchlist
        ]

    def shift_summary(self) -> dict[str, Any]:
        census = self.census()
        risk_bands = {"critical": 0, "high": 0, "watch": 0, "stable": 0}
        for p in census:
            band = p.get("severity", "stable")
            if band not in risk_bands:
                band = "stable"
            risk_bands[band] += 1
        audit = self.audit.shift_stats()
        return {
            "n_patients": len(census),
            "risk_bands": risk_bands,
            "n_watchlist": len(self._watchlist),
            "n_acknowledged": len(self._acknowledged),
            **audit,
            **self.status(),
        }

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
            "alert_threshold": self.selector.alert_threshold,
            "n_online_updates": self.online_model.n_updates,
        }
