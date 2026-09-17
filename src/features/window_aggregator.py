from __future__ import annotations

import math
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import yaml


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class _WindowBuffer:
    def __init__(self, max_window_minutes: int):
        self._max_window = timedelta(minutes=max_window_minutes)
        self._data: deque[tuple[datetime, float]] = deque()

    def add(self, ts: datetime, value: float) -> None:
        self._data.append((ts, value))
        self._evict(ts)

    def _evict(self, now: datetime) -> None:
        """Remove entries older than the maximum window."""
        cutoff = now - self._max_window
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def get_window(self, now: datetime, minutes: int) -> list[tuple[datetime, float]]:
        """Return entries within the last `minutes` of `now`."""
        cutoff = now - timedelta(minutes=minutes)
        return [(ts, v) for ts, v in self._data if ts >= cutoff]


# ---------------------------------------------------------------------------
# Statistical aggregators
# ---------------------------------------------------------------------------

def _compute_stats(values: list[float], timestamps: list[datetime]) -> dict[str, float]:
    """Compute mean/std/min/max/slope/count for a list of (ts, val) pairs."""
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "slope": float("nan"), "count": 0}

    arr = np.array(values, dtype=float)
    stats: dict[str, float] = {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)) if n > 1 else 0.0,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "count": float(n),
    }

    # Trend slope: linear regression of value over time (seconds)
    if n >= 3:
        t_seconds = np.array(
            [(ts - timestamps[0]).total_seconds() for ts in timestamps], dtype=float
        )
        t_mean = t_seconds.mean()
        v_mean = arr.mean()
        denom = ((t_seconds - t_mean) ** 2).sum()
        stats["slope"] = float(
            ((t_seconds - t_mean) * (arr - v_mean)).sum() / denom
        ) if denom > 1e-9 else 0.0
    else:
        stats["slope"] = float("nan")

    return stats


# ---------------------------------------------------------------------------
# Feature list — PhysioNet CinC 2019 columns
# ---------------------------------------------------------------------------

# Vitals (updated every hour)
VITAL_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

# Labs (sparse — may be missing for many hours)
LAB_FEATURES = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

# Demographics (static per patient — still included as features)
DEMOGRAPHIC_FEATURES = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

CLINICAL_FEATURES = VITAL_FEATURES + LAB_FEATURES + DEMOGRAPHIC_FEATURES


# ---------------------------------------------------------------------------
# Main aggregator
# ---------------------------------------------------------------------------

class WindowAggregator:
    """
    Maintains per-patient sliding windows and produces feature vectors.

    Usage:
        agg = WindowAggregator()
        for event in stream:
            features = agg.update(event)
            model.predict(features)
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        cfg = _load_cfg(cfg_path)
        self.window_sizes: list[int] = cfg["features"]["window_sizes_minutes"]
        self.staleness_threshold = cfg["features"]["staleness_threshold_minutes"]
        self.max_window = max(self.window_sizes)

        # buffers[patient_id][feature_name] → _WindowBuffer
        self._buffers: dict[str, dict[str, _WindowBuffer]] = defaultdict(
            lambda: {f: _WindowBuffer(self.max_window) for f in CLINICAL_FEATURES}
        )
        # Track last observation time per patient + feature
        self._last_obs: dict[str, dict[str, datetime]] = defaultdict(dict)

    def update(self, event: dict) -> dict[str, float]:
        """
        Ingest a new observation event and return the updated feature vector.

        Args:
            event: dict with keys patient_id, feature_name, value, timestamp

        Returns:
            Flat feature dict {feature_W_stat: value, ...}
        """
        pid = event["patient_id"]
        feature = event["feature_name"]
        ts: datetime = event["timestamp"]
        value: float = event["value"]

        if feature not in CLINICAL_FEATURES:
            return {}

        # Range-clip obviously erroneous values
        value = self._clip(feature, value)
        if math.isnan(value):
            return {}

        # Update buffer
        self._buffers[pid][feature].add(ts, value)
        self._last_obs[pid][feature] = ts

        return self.get_feature_vector(pid, ts)

    def get_feature_vector(self, patient_id: str, now: datetime) -> dict[str, float]:
        """
        Compute the full feature vector for a patient at time `now`.
        Returns NaN for windows where no data is present.
        """
        vector: dict[str, float] = {}

        for feature in CLINICAL_FEATURES:
            buf = self._buffers[patient_id].get(feature)
            last_ts = self._last_obs.get(patient_id, {}).get(feature)

            # Staleness indicator
            if last_ts is None:
                staleness = float("nan")
                missing = 1.0
            else:
                staleness = (now - last_ts).total_seconds() / 60.0  # minutes
                missing = 1.0 if staleness > self.staleness_threshold else 0.0

            vector[f"{feature}__missing"] = missing
            vector[f"{feature}__staleness_min"] = staleness if not math.isnan(staleness) else -1.0

            if buf is None:
                for w in self.window_sizes:
                    for stat in ["mean", "std", "min", "max", "slope", "count"]:
                        vector[f"{feature}__w{w}_{stat}"] = float("nan")
                continue

            for w in self.window_sizes:
                window_data = buf.get_window(now, w)
                if window_data:
                    timestamps, values = zip(*window_data)
                    stats = _compute_stats(list(values), list(timestamps))
                else:
                    stats = {"mean": float("nan"), "std": float("nan"),
                             "min": float("nan"), "max": float("nan"),
                             "slope": float("nan"), "count": 0.0}

                for stat, val in stats.items():
                    vector[f"{feature}__w{w}_{stat}"] = val

        # Derived features
        vector.update(self._derived_features(vector))
        return vector

    def _derived_features(self, v: dict[str, float]) -> dict[str, float]:
        """Compute clinically meaningful derived features."""
        derived: dict[str, float] = {}

        # Shock index = HR / SBP
        hr = v.get("HR__w60_mean", float("nan"))
        sbp = v.get("SBP__w60_mean", float("nan"))
        if not math.isnan(hr) and not math.isnan(sbp) and sbp > 0:
            derived["shock_index_60"] = hr / sbp
        else:
            derived["shock_index_60"] = float("nan")

        # Pulse pressure = SBP - DBP
        dbp = v.get("DBP__w60_mean", float("nan"))
        if not math.isnan(sbp) and not math.isnan(dbp):
            derived["pulse_pressure_60"] = sbp - dbp
        else:
            derived["pulse_pressure_60"] = float("nan")

        # P/F ratio proxy = O2Sat / FiO2 (scaled)
        spo2 = v.get("O2Sat__w60_mean", float("nan"))
        fio2 = v.get("FiO2__w60_mean", float("nan"))
        if not math.isnan(spo2) and not math.isnan(fio2) and fio2 > 0:
            derived["pf_ratio_proxy"] = spo2 / (fio2 / 100.0)
        else:
            derived["pf_ratio_proxy"] = float("nan")

        # Lactate trend (key sepsis indicator)
        lactate_slope = v.get("Lactate__w240_slope", float("nan"))
        derived["lactate_rising"] = float(
            0.0 if math.isnan(lactate_slope) else (1.0 if lactate_slope > 0 else 0.0)
        )

        return derived

    # ------------------------------------------------------------------
    @staticmethod
    def _clip(feature: str, value: float) -> float:
        """Clip physiologically impossible values (sensor/entry errors)."""
        RANGES: dict[str, tuple[float, float]] = {
            # Vitals
            "HR":          (0, 300),
            "O2Sat":       (50, 100),
            "Temp":        (25, 45),       # °C (dataset already in Celsius)
            "SBP":         (20, 300),
            "MAP":         (10, 250),
            "DBP":         (0, 200),
            "Resp":        (0, 80),
            "EtCO2":       (0, 100),
            # Labs
            "BaseExcess":  (-30, 30),
            "HCO3":        (0, 60),
            "FiO2":        (0, 100),
            "pH":          (6.5, 8.0),
            "PaCO2":       (0, 150),
            "SaO2":        (0, 100),
            "AST":         (0, 10000),
            "BUN":         (0, 300),
            "Alkalinephos":(0, 5000),
            "Calcium":     (0, 20),
            "Chloride":    (70, 160),
            "Creatinine":  (0, 30),
            "Bilirubin_direct": (0, 50),
            "Glucose":     (0, 2000),
            "Lactate":     (0, 30),
            "Magnesium":   (0, 10),
            "Phosphate":   (0, 20),
            "Potassium":   (1, 10),
            "Bilirubin_total": (0, 50),
            "TroponinI":   (0, 1000),
            "Hct":         (0, 100),
            "Hgb":         (0, 25),
            "PTT":         (0, 300),
            "WBC":         (0, 500),
            "Fibrinogen":  (0, 2000),
            "Platelets":   (0, 2000),
            # Demographics (no clipping needed — pass through)
            "Age":         (0, 120),
            "Gender":      (0, 1),
            "Unit1":       (0, 1),
            "Unit2":       (0, 1),
            "HospAdmTime": (-500, 500),
            "ICULOS":      (0, 10000),
        }
        lo, hi = RANGES.get(feature, (float("-inf"), float("inf")))
        if value < lo or value > hi:
            return float("nan")
        return value

    def feature_names(self) -> list[str]:
        """Return the full ordered list of feature names."""
        names = []
        for feature in CLINICAL_FEATURES:
            names.append(f"{feature}__missing")
            names.append(f"{feature}__staleness_min")
            for w in self.window_sizes:
                for stat in ["mean", "std", "min", "max", "slope", "count"]:
                    names.append(f"{feature}__w{w}_{stat}")
        names += ["shock_index_60", "pulse_pressure_60", "pf_ratio_proxy", "lactate_rising"]
        return names

    def patient_count(self) -> int:
        return len(self._buffers)

    def latest_vitals(self, patient_id: str) -> dict[str, float]:
        """Most recent raw observation per feature for a patient."""
        out: dict[str, float] = {}
        buffers = self._buffers.get(patient_id, {})
        for feature, buf in buffers.items():
            if buf._data:
                out[feature] = buf._data[-1][1]
        return out

    def tracked_patient_ids(self) -> list[str]:
        return list(self._buffers.keys())
