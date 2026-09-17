from __future__ import annotations

import math
from typing import Any

import numpy as np
import yaml


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


CLINICAL_DEFAULTS: dict[str, float] = {
    "HR":               80.0,
    "O2Sat":            97.0,
    "Temp":             37.0,
    "SBP":             120.0,
    "MAP":              80.0,
    "DBP":              75.0,
    "Resp":             16.0,
    "EtCO2":            35.0,
    # Labs
    "BaseExcess":        0.0,
    "HCO3":             24.0,
    "FiO2":             21.0,
    "pH":                7.4,
    "PaCO2":            40.0,
    "SaO2":             98.0,
    "AST":              30.0,
    "BUN":              14.0,
    "Alkalinephos":     80.0,
    "Calcium":           9.0,
    "Chloride":        102.0,
    "Creatinine":        0.9,
    "Bilirubin_direct":  0.2,
    "Glucose":         100.0,
    "Lactate":           1.5,
    "Magnesium":         2.0,
    "Phosphate":         3.5,
    "Potassium":         4.0,
    "Bilirubin_total":   0.7,
    "TroponinI":         0.02,
    "Hct":              40.0,
    "Hgb":              13.5,
    "PTT":              30.0,
    "WBC":               8.0,
    "Fibrinogen":      250.0,
    "Platelets":       230.0,
    # Demographics
    "Age":              60.0,
    "Gender":            0.0,
    "Unit1":             0.0,
    "Unit2":             0.0,
    "HospAdmTime":       0.0,
    "ICULOS":            1.0,
    # Derived features
    "shock_index_60":    0.7,
    "pulse_pressure_60": 45.0,
    "pf_ratio_proxy":  450.0,
    "lactate_rising":    0.0,
}

# For statistics on the same base feature, propagate the same default
_STAT_SUFFIXES = ["mean", "std", "min", "max", "slope", "count"]


def _base_feature(col_name: str) -> str | None:
    """Extract base feature name from a column like 'HR__w60_mean'."""
    parts = col_name.split("__")
    return parts[0] if len(parts) > 1 else None


def _get_default(col_name: str) -> float:
    """Return a sensible imputation default for a feature column."""
    # Exact match first
    if col_name in CLINICAL_DEFAULTS:
        return CLINICAL_DEFAULTS[col_name]
    # Try base feature
    base = _base_feature(col_name)
    if base in CLINICAL_DEFAULTS:
        # std/slope default to 0 (no variability assumed)
        if "__std" in col_name or "__slope" in col_name:
            return 0.0
        # count defaults to 0 (no observations)
        if "__count" in col_name:
            return 0.0
        return CLINICAL_DEFAULTS[base]
    return 0.0


# ---------------------------------------------------------------------------
# Online running median (reservoir approximation)
# ---------------------------------------------------------------------------

class OnlineMedian:
    """
    Approximate online median using a reservoir of recent values.
    Efficient for streaming settings.
    """

    def __init__(self, buffer_size: int = 1000):
        self._buffer: list[float] = []
        self._buffer_size = buffer_size
        self._count = 0

    def update(self, value: float) -> None:
        if math.isnan(value):
            return
        if len(self._buffer) < self._buffer_size:
            self._buffer.append(value)
        else:
            # Reservoir sampling: replace with decreasing probability
            idx = np.random.randint(0, self._count + 1)
            if idx < self._buffer_size:
                self._buffer[idx] = value
        self._count += 1

    @property
    def median(self) -> float | None:
        if not self._buffer:
            return None
        return float(np.median(self._buffer))

    @property
    def count(self) -> int:
        return self._count


# ---------------------------------------------------------------------------
# Main missing data handler
# ---------------------------------------------------------------------------

class MissingDataHandler:
    """
    Imputes NaN values in feature vectors using online population statistics.

    Usage:
        handler = MissingDataHandler()
        # During stream:
        filled = handler.impute(feature_vector)
        handler.update_population_stats(feature_vector)  # update running medians
    """

    def __init__(self, cfg_path: str = "config/settings.yaml", buffer_size: int = 1000):
        self._cfg = _load_cfg(cfg_path)
        self._medians: dict[str, OnlineMedian] = {}
        self._buffer_size = buffer_size
        self._n_imputed_total = 0

    def _get_median(self, col: str) -> OnlineMedian:
        if col not in self._medians:
            self._medians[col] = OnlineMedian(self._buffer_size)
        return self._medians[col]

    def update_population_stats(self, features: dict[str, float]) -> None:
        """Update running medians with observed (non-NaN) feature values."""
        for col, val in features.items():
            if not math.isnan(val) and not col.endswith("__missing"):
                self._get_median(col).update(val)

    def impute(self, features: dict[str, float]) -> dict[str, float]:
        """
        Return a copy of `features` with NaN values imputed.

        Strategy:
          1. Population running median (if enough observations: n >= 10)
          2. Clinical default prior
          3. 0.0 (last resort)

        Missing indicator features (__missing, __staleness) are preserved as-is.
        """
        filled: dict[str, float] = {}
        for col, val in features.items():
            if not math.isnan(val):
                filled[col] = val
                continue

            # Preserve missing indicators (NaN means feature column itself is new)
            if "__missing" in col or "__staleness" in col:
                filled[col] = 1.0  # treat unknown staleness as missing
                continue

            # Population median
            med_tracker = self._get_median(col)
            if med_tracker.count >= 10 and med_tracker.median is not None:
                filled[col] = med_tracker.median
            else:
                filled[col] = _get_default(col)

            self._n_imputed_total += 1

        return filled

    def impute_and_update(self, features: dict[str, float]) -> dict[str, float]:
        """Convenience: impute then update stats from original (non-imputed) vector."""
        self.update_population_stats(features)
        return self.impute(features)

    @property
    def imputation_count(self) -> int:
        return self._n_imputed_total

    def stats_summary(self) -> dict[str, int]:
        """Return count of observations per tracked feature."""
        return {col: med.count for col, med in self._medians.items()}
