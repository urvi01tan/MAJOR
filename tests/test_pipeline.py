"""
End-to-end pipeline tests (no dataset required).
Uses synthetic events generated in-memory.
Dataset: PhysioNet/CinC Challenge 2019 feature names (HR, SBP, O2Sat, etc.)
"""

from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _synthetic_event(
    feature: str = "HR",
    value: float = 85.0,
    patient_id: str = "P001",
    ts: datetime | None = None,
) -> dict:
    return {
        "patient_id": patient_id,
        "timestamp": ts or datetime.utcnow(),
        "source": "vital",
        "feature_name": feature,
        "value": value,
        "unit": "",
        "sepsis_label": 0,
        "iculos": 1,
    }


def _make_online_model_mock():
    """Lightweight mock of OnlineModel that doesn't require River."""
    m = MagicMock()
    m.predict.return_value = 0.4
    m.rolling_auroc = 0.75
    m.n_predictions = 0
    m.n_updates = 0
    m.version = 1
    m.model_type = "logistic_regression"
    m.recent_error_rate = 0.2
    m.enqueue_for_update.return_value = None
    m.provide_label.return_value = None
    m.get_feature_weights.return_value = {}
    m.save_state.return_value = None
    m.to_dict.return_value = {}
    return m


def _make_baseline_mock():
    m = MagicMock()
    m.predict.return_value = 0.35
    m.auroc = 0.72
    m.feature_names = []
    m.version = "baseline_v1"
    m.get_feature_importance.return_value = {}
    m.to_dict.return_value = {}
    return m


class TestWindowAggregatorE2E:
    """End-to-end feature computation test (no mocking)."""

    def test_feature_vector_latency(self):
        """Feature vector computation should be < 50ms per event."""
        from src.features.window_aggregator import WindowAggregator
        agg = WindowAggregator(cfg_path="config/settings.yaml")

        # Prime with a few observations
        ts = datetime.utcnow()
        for i in range(10):
            agg.update(_synthetic_event("HR", 80.0 + i,
                                        ts=ts + timedelta(minutes=i * 10)))

        # Measure feature vector computation time
        start = time.perf_counter()
        for _ in range(100):
            agg.get_feature_vector("P001", ts + timedelta(minutes=10))
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.05, f"Feature vector too slow: {elapsed*1000:.1f}ms"

    def test_multi_patient_isolation(self):
        """Features for one patient must not affect another."""
        from src.features.window_aggregator import WindowAggregator
        agg = WindowAggregator(cfg_path="config/settings.yaml")

        ts = datetime.utcnow()
        agg.update(_synthetic_event("HR", 120.0, patient_id="SICK", ts=ts))
        agg.update(_synthetic_event("HR", 60.0, patient_id="STABLE", ts=ts))

        sick_fv = agg.get_feature_vector("SICK", ts)
        stable_fv = agg.get_feature_vector("STABLE", ts)

        # PhysioNet column name: HR, window: w60 (smallest window = 60 min)
        assert abs(sick_fv["HR__w60_mean"] - 120.0) < 1e-3
        assert abs(stable_fv["HR__w60_mean"] - 60.0) < 1e-3


class TestEnsembleSelectorE2E:
    """Test ensemble selector logic (mocked models)."""

    def _make_selector(self, online_auroc: float = 0.75):
        from src.models.ensemble import EnsembleSelector
        online = _make_online_model_mock()
        online.rolling_auroc = online_auroc
        baseline = _make_baseline_mock()
        selector = EnsembleSelector(online, baseline, cfg_path="config/settings.yaml")
        return selector, online, baseline

    def test_uses_online_model_when_healthy(self):
        selector, online, baseline = self._make_selector(online_auroc=0.75)
        # baseline.auroc=0.72, rollback_gap=0.05 → threshold=0.67
        # online=0.75 > 0.67 → use online
        result = selector.predict("P001", {})
        assert result["model_used"] == "online"

    def test_rollback_when_online_degrades(self):
        selector, online, baseline = self._make_selector(online_auroc=0.55)
        # baseline.auroc=0.72, threshold=0.67
        # online=0.55 < 0.67 → rollback to baseline
        selector.predict("P001", {})  # trigger check
        assert selector.active_model == "baseline"
        assert selector.rollback_count >= 1

    def test_alert_with_cooldown(self):
        from src.models.ensemble import EnsembleSelector
        online = _make_online_model_mock()
        online.predict.return_value = 0.80  # above threshold
        online.rolling_auroc = 0.75
        baseline = _make_baseline_mock()
        baseline.predict.return_value = 0.80
        selector = EnsembleSelector(online, baseline, cfg_path="config/settings.yaml")

        ts = datetime.utcnow()
        r1 = selector.predict("P999", {}, now=ts)
        assert r1["alert"] is True

        # Same patient 5 minutes later → should NOT alert (cooldown=30min)
        r2 = selector.predict("P999", {}, now=ts + timedelta(minutes=5))
        assert r2["alert"] is False

        # 35 minutes later → should alert again
        r3 = selector.predict("P999", {}, now=ts + timedelta(minutes=35))
        assert r3["alert"] is True

    def test_force_rollback(self):
        selector, _, _ = self._make_selector(online_auroc=0.80)
        assert selector.active_model == "online"
        selector.force_rollback()
        assert selector.active_model == "baseline"

    def test_restore_online(self):
        selector, _, _ = self._make_selector(online_auroc=0.80)
        selector.force_rollback()
        selector.restore_online()
        assert selector.active_model == "online"


class TestDriftDetectorIntegration:
    def test_no_crash_on_nan_input(self):
        from src.drift.detector import DriftDetector
        det = DriftDetector(cfg_path="config/settings.yaml")
        events = det.update(float("nan"))
        assert events == []  # NaN should be handled gracefully

    def test_status_fields_present(self):
        from src.drift.detector import DriftDetector
        det = DriftDetector(cfg_path="config/settings.yaml")
        det.update(0.5)
        s = det.status()
        for key in ["n_updates", "total_drift_events", "in_warning", "ph_current_sum"]:
            assert key in s, f"Missing key: {key}"
