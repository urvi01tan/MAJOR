from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class BaselineModel:
    def __init__(self, cfg_path: str = "config/settings.yaml"):
        self.cfg = _load_cfg(cfg_path)
        self.model_path = Path(self.cfg["paths"]["baseline_model"])
        self.meta_path = Path(self.cfg["paths"]["baseline_meta"])
        self._model = None
        self._feature_names: list[str] = []
        self._meta: dict = {}
        self.version = "baseline_v1"

    def load(self) -> "BaselineModel":
        """Load the pre-trained model from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {self.model_path}.\n"
                "Train it first: python scripts/train_baseline.py"
            )
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._feature_names = payload.get("feature_names", [])
        self.version = payload.get("version", "baseline_v1")
        log.info(f"[BaselineModel] Loaded version={self.version}, features={len(self._feature_names)}")

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self._meta = json.load(f)
        return self

    def predict(self, features: dict[str, float]) -> float:
        """
        Return risk probability in [0, 1].
        Missing features are imputed to 0.0 (conservative).
        """
        if self._model is None:
            log.warning("[BaselineModel] Not loaded — returning 0.0. Train with scripts/train_baseline.py")
            return 0.0

        x = np.array(
            [features.get(f, 0.0) if not math.isnan(features.get(f, float("nan"))) else 0.0
             for f in self._feature_names],
            dtype=float
        ).reshape(1, -1)

        try:
            proba = self._model.predict_proba(x)[0]
            # proba is [P(class=0), P(class=1)]
            return float(proba[1])
        except Exception as exc:
            log.error(f"[BaselineModel] predict error: {exc}")
            return 0.0

    def get_feature_importance(self) -> dict[str, float]:
        """
        Return feature importance / coefficients for explainability.
        """
        if self._model is None:
            return {}
        inner = self._model
        # Unwrap Pipeline if needed
        if hasattr(inner, "steps"):
            inner = inner.steps[-1][1]

        if hasattr(inner, "coef_"):
            # LogisticRegression
            coefs = inner.coef_[0]
            return {f: float(c) for f, c in zip(self._feature_names, coefs)}
        elif hasattr(inner, "feature_importances_"):
            # XGBoost / RandomForest
            imps = inner.feature_importances_
            return {f: float(imp) for f, imp in zip(self._feature_names, imps)}
        return {}

    @property
    def auroc(self) -> float:
        return float(self._meta.get("test_auroc", float("nan")))

    @property
    def sensitivity(self) -> float:
        return float(self._meta.get("test_sensitivity", float("nan")))

    @property
    def specificity(self) -> float:
        return float(self._meta.get("test_specificity", float("nan")))

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "model_type": self._meta.get("model_type", "unknown"),
            "n_train_samples": self._meta.get("n_train", 0),
            "test_auroc": self.auroc,
            "test_sensitivity": self.sensitivity,
            "test_specificity": self.specificity,
            "n_features": len(self._feature_names),
        }
