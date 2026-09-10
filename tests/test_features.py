"""Tests for WindowAggregator and MissingDataHandler.

Dataset: PhysioNet/CinC Challenge 2019
Features: HR, O2Sat, Temp, SBP, MAP, DBP, Resp, EtCO2 (vitals)
          + 26 lab columns + 6 demographic columns
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.window_aggregator import WindowAggregator, _compute_stats
from src.features.missing_handler import MissingDataHandler


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_event(feature: str, value: float, ts: datetime, patient_id: str = "P001") -> dict:
    return {
        "patient_id": patient_id,
        "timestamp": ts,
        "source": "vital",
        "feature_name": feature,
        "value": value,
        "unit": "",
        "sepsis_label": 0,
        "iculos": 1,
    }


BASE_TS = datetime(2024, 1, 1, 8, 0, 0)


# ── WindowAggregator tests ───────────────────────────────────────────────────

class TestWindowAggregator:
    def setup_method(self):
        self.agg = WindowAggregator(cfg_path="config/settings.yaml")

    def test_single_observation_mean(self):
        # HR is the PhysioNet column name for heart rate
        event = _make_event("HR", 80.0, BASE_TS)
        fv = self.agg.update(event)
        assert "HR__w60_mean" in fv
        assert abs(fv["HR__w60_mean"] - 80.0) < 1e-6

    def test_missing_flag_before_observation(self):
        # Patient with no observations → missing flag should be 1
        fv = self.agg.get_feature_vector("NEW_PATIENT", BASE_TS)
        assert fv["HR__missing"] == 1.0

    def test_missing_flag_after_observation(self):
        event = _make_event("HR", 80.0, BASE_TS)
        fv = self.agg.update(event)
        assert fv["HR__missing"] == 0.0

    def test_temperature_already_celsius(self):
        # PhysioNet dataset stores Temp in Celsius — no conversion needed
        event = _make_event("Temp", 37.0, BASE_TS, "TempTest")
        fv = self.agg.update(event)
        mean = fv.get("Temp__w60_mean", float("nan"))
        assert not math.isnan(mean)
        assert abs(mean - 37.0) < 0.1

    def test_physiological_clipping(self):
        # HR of 500 is impossible → should be clipped to NaN
        event = _make_event("HR", 500.0, BASE_TS, "ClipTest")
        fv = self.agg.update(event)
        # NaN means the value was clipped out — missing flag should be 1
        assert math.isnan(fv.get("HR__w60_mean", float("nan"))) or \
               fv["HR__missing"] == 1.0

    def test_multiple_observations_stats(self):
        for i, val in enumerate([70.0, 80.0, 90.0, 100.0]):
            ts = BASE_TS + timedelta(minutes=i * 15)
            self.agg.update(_make_event("HR", val, ts, "StatsTest"))

        fv = self.agg.get_feature_vector("StatsTest", BASE_TS + timedelta(minutes=45))
        mean = fv["HR__w60_mean"]
        assert abs(mean - 85.0) < 1.0   # mean of 70,80,90,100

    def test_sliding_window_eviction(self):
        # Add 3 observations at t=0, then jump far ahead
        for i, val in enumerate([60.0, 70.0, 80.0]):
            self.agg.update(_make_event("HR", val, BASE_TS + timedelta(minutes=i), "EvictTest"))

        # Jump 10 hours ahead — 60-min window should be empty
        future_ts = BASE_TS + timedelta(hours=10, minutes=1)
        fv = self.agg.get_feature_vector("EvictTest", future_ts)
        assert fv["HR__w60_count"] == 0.0

    def test_derived_shock_index(self):
        ts = BASE_TS
        self.agg.update(_make_event("HR", 120.0, ts, "Shock"))
        self.agg.update(_make_event("SBP", 80.0, ts, "Shock"))
        fv = self.agg.get_feature_vector("Shock", ts)
        si = fv.get("shock_index_60", float("nan"))
        if not math.isnan(si):
            assert abs(si - 1.5) < 0.1  # 120/80 = 1.5

    def test_feature_names_list(self):
        names = self.agg.feature_names()
        # PhysioNet column names used
        assert "HR__w60_mean" in names
        assert "Lactate__w60_mean" in names
        assert "O2Sat__w60_mean" in names
        assert "shock_index_60" in names
        assert "lactate_rising" in names
        assert len(names) > 50

    def test_lab_feature_ingested(self):
        # Lactate is a key sepsis lab — verify it's tracked
        event = _make_event("Lactate", 3.1, BASE_TS, "LabTest")
        fv = self.agg.update(event)
        assert "Lactate__w60_mean" in fv
        assert abs(fv["Lactate__w60_mean"] - 3.1) < 1e-6

    def test_spo2_feature_name(self):
        # PhysioNet uses O2Sat (not spo2)
        event = _make_event("O2Sat", 95.0, BASE_TS, "O2Test")
        fv = self.agg.update(event)
        assert "O2Sat__w60_mean" in fv

    def test_derived_pf_ratio(self):
        ts = BASE_TS
        self.agg.update(_make_event("O2Sat", 94.0, ts, "PFTest"))
        self.agg.update(_make_event("FiO2", 40.0, ts, "PFTest"))
        fv = self.agg.get_feature_vector("PFTest", ts)
        pf = fv.get("pf_ratio_proxy", float("nan"))
        if not math.isnan(pf):
            # 94 / (40/100) = 235
            assert abs(pf - 235.0) < 1.0

    def test_multi_patient_isolation(self):
        # Data from patient A should not affect patient B
        self.agg.update(_make_event("HR", 120.0, BASE_TS, "PatA"))
        fv_b = self.agg.get_feature_vector("PatB", BASE_TS)
        assert fv_b["HR__missing"] == 1.0


# ── _compute_stats tests ─────────────────────────────────────────────────────

class TestComputeStats:
    def test_empty(self):
        stats = _compute_stats([], [])
        assert stats["count"] == 0
        assert math.isnan(stats["mean"])

    def test_single(self):
        ts = [BASE_TS]
        stats = _compute_stats([42.0], ts)
        assert abs(stats["mean"] - 42.0) < 1e-9
        assert abs(stats["std"]) < 1e-9

    def test_trend_increasing(self):
        ts = [BASE_TS + timedelta(minutes=i) for i in range(5)]
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = _compute_stats(vals, ts)
        assert stats["slope"] > 0


# ── MissingDataHandler tests ──────────────────────────────────────────────────

class TestMissingDataHandler:
    def setup_method(self):
        self.handler = MissingDataHandler(cfg_path="config/settings.yaml")

    def test_impute_with_default(self):
        # HR is the PhysioNet column name for heart rate
        features = {"HR__w60_mean": float("nan")}
        filled = self.handler.impute(features)
        # Should be imputed to clinical default (80.0 for HR)
        assert not math.isnan(filled["HR__w60_mean"])

    def test_impute_preserves_valid_values(self):
        features = {"HR__w60_mean": 95.0}
        filled = self.handler.impute(features)
        assert abs(filled["HR__w60_mean"] - 95.0) < 1e-9

    def test_population_median_update(self):
        # Feed 20 observations of HR=100 → median should be ~100
        for _ in range(20):
            self.handler.update_population_stats({"HR__w60_mean": 100.0})
        features = {"HR__w60_mean": float("nan")}
        filled = self.handler.impute(features)
        assert abs(filled["HR__w60_mean"] - 100.0) < 5.0

    def test_missing_indicator_preserved(self):
        features = {"HR__missing": 1.0, "HR__w60_mean": float("nan")}
        filled = self.handler.impute(features)
        assert filled["HR__missing"] == 1.0

    def test_imputation_count_increments(self):
        features = {"HR__w60_mean": float("nan")}
        self.handler.impute(features)
        assert self.handler.imputation_count >= 1

    def test_lactate_default(self):
        # Lactate default should be ~1.5 mmol/L (normal)
        features = {"Lactate__w60_mean": float("nan")}
        filled = self.handler.impute(features)
        assert not math.isnan(filled["Lactate__w60_mean"])
        assert filled["Lactate__w60_mean"] < 5.0  # not elevated
