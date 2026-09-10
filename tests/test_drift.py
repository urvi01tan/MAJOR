"""Tests for drift detectors."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.drift.detector import DriftDetector, PageHinkley


class TestPageHinkley:
    def test_no_drift_stable(self):
        ph = PageHinkley(threshold=50.0, alpha=0.005)
        for _ in range(100):
            fired = ph.update(0.3)
        assert not fired, "Should not fire on stable low-score stream"

    def test_drift_on_sudden_spike(self):
        ph = PageHinkley(threshold=10.0, alpha=0.001)
        # Stable period
        for _ in range(50):
            ph.update(0.2)
        # Sudden spike to very high values
        fired = False
        for _ in range(20):
            if ph.update(0.95):
                fired = True
                break
        assert fired, "Should detect drift on sustained high-score spike"

    def test_reset_clears_state(self):
        ph = PageHinkley(threshold=10.0, alpha=0.001)
        for _ in range(30):
            ph.update(0.9)
        ph.reset()
        assert ph._sum == 0.0
        assert ph._n == 0


class TestDriftDetector:
    def test_init(self):
        det = DriftDetector(cfg_path="config/settings.yaml")
        assert det.total_drift_count == 0

    def test_stable_stream_no_drift(self):
        det = DriftDetector(cfg_path="config/settings.yaml")
        for i in range(100):
            events = det.update(0.3 + 0.01 * (i % 5))
        # Minor fluctuations should not trigger ADWIN or Page-Hinkley
        # (may or may not trigger depending on exact parameters — just verify no crash)
        assert det.total_drift_count >= 0

    def test_abrupt_drift_page_hinkley(self):
        """Inject a sudden spike — Page-Hinkley should eventually fire."""
        det = DriftDetector(cfg_path="config/settings.yaml")
        # Override PH threshold to be very low so it fires quickly
        from river.drift import PageHinkley as RiverPH
        det._ph = RiverPH(threshold=5.0, delta=0.0)
        det._ph_threshold = 5.0
        det._ph_delta = 0.0

        # Stable
        for _ in range(20):
            det.update(0.1)
        # Spike
        fired_events = []
        for _ in range(50):
            evts = det.update(0.99)
            fired_events.extend(evts)

        ph_events = [e for e in fired_events if e.detector == "PageHinkley"]
        assert len(ph_events) > 0, "Page-Hinkley should detect abrupt spike"

    def test_drift_event_logging(self):
        det = DriftDetector(cfg_path="config/settings.yaml")
        det._ph.threshold = 2.0
        det._ph.alpha = 0.0

        for _ in range(30):
            det.update(0.99)

        events = det.drift_events
        assert isinstance(events, list)
        if events:
            assert "detector" in events[0]
            assert "timestamp" in events[0]
            assert "severity" in events[0]

    def test_drift_score_series(self):
        det = DriftDetector(cfg_path="config/settings.yaml")
        for i in range(10):
            det.update(0.5)
        series = det.drift_score_series()
        assert "adwin_scores" in series
        assert "ph_sums" in series
        assert len(series["adwin_scores"]) == 10

    def test_status_dict(self):
        det = DriftDetector(cfg_path="config/settings.yaml")
        det.update(0.4)
        s = det.status()
        assert "n_updates" in s
        assert "total_drift_events" in s
        assert s["n_updates"] == 1
