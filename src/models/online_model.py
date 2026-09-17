from __future__ import annotations

import json
import logging
import math
import pickle
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


log = logging.getLogger(__name__)

try:
    from river import (
        linear_model,
        optim,
        preprocessing,
        tree,
    )
    from river.metrics import ROCAUC
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    log.error("River not installed. Run: pip install river")


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Delayed-label queue entry
# ---------------------------------------------------------------------------

class _DelayedEntry:
    __slots__ = ("features", "enqueued_at", "label", "patient_id")

    def __init__(
        self,
        features: dict,
        enqueued_at: datetime,
        patient_id: str | None = None,
        label: int | None = None,
    ):
        self.features = features
        self.enqueued_at = enqueued_at
        self.patient_id = patient_id
        self.label = label


# ---------------------------------------------------------------------------
# Online Model
# ---------------------------------------------------------------------------

class OnlineModel:
    """
    Incrementally-updated patient deterioration predictor.

    Lifecycle:
      1. predict(features) → risk score in [0, 1]
      2. enqueue_for_update(features, event_time) → queues the sample
      3. When outcome is confirmed: provide_label(hadm_id, label)
      4. Queued entries with confirmed labels are consumed by _drain_queue()
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        if not RIVER_AVAILABLE:
            raise RuntimeError("River library required. pip install river")
        self.cfg = _load_cfg(cfg_path)
        self.model_type = self.cfg["online_model"]["type"]
        self.prediction_window_hours = self.cfg["outcome"]["prediction_window_hours"]

        self._model = self._build_model()
        self._rolling_auroc = ROCAUC()
        self._n_predictions = 0
        self._n_updates = 0
        self.version = 1
        self.created_at = datetime.utcnow().isoformat()

        # Delayed label queue: deque of _DelayedEntry
        self._label_queue: deque[_DelayedEntry] = deque(maxlen=10_000)
        # Confirmed labels: {hadm_id → (label, confirmed_at)}
        self._confirmed_labels: dict[str, tuple[int, datetime]] = {}

        # Rolling error buffer for external drift detectors
        self._recent_errors: deque[float] = deque(maxlen=200)

        log.info(f"[OnlineModel] Initialised: type={self.model_type}, version={self.version}")

    # ------------------------------------------------------------------
    def _build_model(self):
        """Build River pipeline: scaler → classifier."""
        if self.model_type == "hoeffding_adaptive_tree":
            classifier = tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=100,
                delta=1e-5,
                leaf_prediction="nba",  # Naive Bayes Adaptive
                nb_threshold=0,
                seed=42,
            )
        elif self.model_type == "logistic_regression":
            classifier = linear_model.LogisticRegression(
                optimizer=optim.SGD(lr=self.cfg["online_model"]["learning_rate"]),
                l2=1e-4,
            )
        else:
            raise ValueError(f"Unknown online model type: {self.model_type}")

        # Wrap with StandardScaler for numerical stability
        return preprocessing.StandardScaler() | classifier

    # ------------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> float:
        """Return risk probability in [0, 1]. Handles NaN-safe input."""
        clean = {k: (v if not math.isnan(v) else 0.0) for k, v in features.items()}
        proba = self._model.predict_proba_one(clean)
        score = proba.get(1, 0.0) if proba else 0.0
        self._n_predictions += 1
        return float(score)

    # ------------------------------------------------------------------
    def enqueue_for_update(
        self,
        features: dict[str, float],
        event_time: datetime,
        patient_id: str | None = None,
        eventual_label: int | None = None,
    ) -> None:
        """Queue a feature sample; trained after the prediction window elapses."""
        if eventual_label is None and patient_id and patient_id in self._confirmed_labels:
            eventual_label = self._confirmed_labels[patient_id][0]
        entry = _DelayedEntry(
            features=features,
            enqueued_at=event_time,
            patient_id=patient_id,
            label=eventual_label,
        )
        self._label_queue.append(entry)
        self._drain_queue(event_time)

    # ------------------------------------------------------------------
    def provide_label(
        self, hadm_id: str, label: int, confirmed_at: datetime | None = None
    ) -> None:
        """
        Provide a confirmed outcome label for an admission.
        label: 1 = deteriorated (sepsis), 0 = stable
        """
        self._confirmed_labels[hadm_id] = (label, confirmed_at or datetime.utcnow())

    # ------------------------------------------------------------------
    def _drain_queue(self, now: datetime) -> None:
        """
        Update model with queued entries whose label delay has passed
        (prediction_window_hours since enqueueing).
        """
        delay = timedelta(hours=self.prediction_window_hours)
        while self._label_queue:
            entry = self._label_queue[0]
            if now - entry.enqueued_at < delay:
                break  # rest of queue is even newer
            self._label_queue.popleft()

            if entry.label is None and entry.patient_id:
                stored = self._confirmed_labels.get(entry.patient_id)
                if stored:
                    entry.label = stored[0]
            if entry.label is None:
                continue

            clean = {
                k: (v if not math.isnan(v) else 0.0)
                for k, v in entry.features.items()
            }
            self._model.learn_one(clean, entry.label)
            self._n_updates += 1

            # Update rolling AUROC
            pred = self.predict(clean)
            self._rolling_auroc.update(entry.label, pred)
            self._recent_errors.append(abs(entry.label - round(pred)))

    # ------------------------------------------------------------------
    def update_from_batch(
        self,
        samples: list[tuple[dict, int]],
    ) -> int:
        """
        Directly update model from (features, label) pairs — used during
        baseline→online warm-start or when labels are immediately available.
        Returns number of samples processed.
        """
        processed = 0
        for features, label in samples:
            clean = {k: (v if not math.isnan(v) else 0.0) for k, v in features.items()}
            self._model.learn_one(clean, label)
            pred = self.predict(clean)
            self._rolling_auroc.update(label, pred)
            self._recent_errors.append(abs(label - round(pred)))
            processed += 1
        self._n_updates += processed
        return processed

    # ------------------------------------------------------------------
    @property
    def rolling_auroc(self) -> float:
        return float(self._rolling_auroc.get())

    @property
    def n_predictions(self) -> int:
        return self._n_predictions

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def recent_error_rate(self) -> float:
        if not self._recent_errors:
            return float("nan")
        return float(sum(self._recent_errors) / len(self._recent_errors))

    # ------------------------------------------------------------------
    def get_feature_weights(self) -> dict[str, float]:
        """
        Extract model weights for explainability.
        Works for LogisticRegression; returns tree depth info for HAT.
        """
        inner = self._model[-1]
        if hasattr(inner, "weights"):
            return dict(inner.weights)
        return {}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "version": self.version,
                "created_at": self.created_at,
                "n_predictions": self._n_predictions,
                "n_updates": self._n_updates,
                "model_type": self.model_type,
            }, f)
        log.info(f"[OnlineModel] State saved → {path}")

    def load_state(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._model = state["model"]
        self.version = state["version"]
        self.created_at = state["created_at"]
        self._n_predictions = state["n_predictions"]
        self._n_updates = state["n_updates"]
        self.model_type = state["model_type"]
        log.info(f"[OnlineModel] State loaded ← {path} (version {self.version})")

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self.model_type,
            "created_at": self.created_at,
            "n_predictions": self._n_predictions,
            "n_updates": self._n_updates,
            "rolling_auroc": self.rolling_auroc,
            "recent_error_rate": self.recent_error_rate,
        }
