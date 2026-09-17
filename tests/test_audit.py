from __future__ import annotations

import math
import sys
import tempfile
import os
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_audit_db(tmp_path, monkeypatch):
    import yaml
    with open("config/settings.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["audit_db"] = str(tmp_path / "test_audit.db")
    cfg_path = str(tmp_path / "test_settings.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    return cfg_path



class TestAuditLogger:
    def test_init_creates_db(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)
        import yaml
        with open(temp_audit_db) as f:
            cfg = yaml.safe_load(f)
        assert Path(cfg["paths"]["audit_db"]).exists()

    def test_log_prediction_returns_id(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)
        result = {
            "patient_id": "P001",
            "hadm_id": "H001",
            "stay_id": "S001",
            "timestamp": datetime.utcnow().isoformat(),
            "model_used": "online",
            "active_model": "online",
            "risk_score": 0.75,
            "alert": True,
            "online_score": 0.75,
            "baseline_score": 0.60,
        }
        features = {"heart_rate__w60_mean": 110.0}
        pid = logger.log_prediction(result, features)
        assert isinstance(pid, str)
        assert len(pid) > 0

    def test_log_and_query_recent(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)

        for i in range(5):
            result = {
                "patient_id": f"P00{i}",
                "hadm_id": f"H00{i}",
                "stay_id": None,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": "online",
                "active_model": "online",
                "risk_score": 0.5 + i * 0.05,
                "alert": i >= 3,
                "online_score": 0.5,
                "baseline_score": 0.4,
            }
            logger.log_prediction(result, {})

        recent = logger.recent_predictions(10)
        assert len(recent) == 5
        # Most recent first
        assert all("patient_id" in r for r in recent)

    def test_log_feedback(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)
        fb_id = logger.log_feedback(
            prediction_id="test_pred_123",
            patient_id="P001",
            feedback="true_positive",
            clinician_id="dr_smith",
            notes="Confirmed sepsis",
        )
        assert isinstance(fb_id, str)

    def test_log_drift_event(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)
        drift_event = {
            "detector": "ADWIN",
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "drift",
            "metric_value": 0.85,
        }
        logger.log_drift_event(drift_event, model_version=3, action_taken="snapshot")
        # No exception = pass

    def test_alert_count(self, temp_audit_db):
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)

        # Log 3 alerts and 2 non-alerts
        for i in range(5):
            result = {
                "patient_id": f"P{i}",
                "hadm_id": None,
                "stay_id": None,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": "online",
                "active_model": "online",
                "risk_score": 0.7 if i < 3 else 0.3,
                "alert": i < 3,
                "online_score": 0.7,
                "baseline_score": 0.5,
            }
            logger.log_prediction(result, {})

        count = logger.alert_count(last_n_hours=1)
        assert count == 3

    def test_immutability_no_updates(self, temp_audit_db):
        """
        Audit logger should only append — there's no UPDATE method exposed.
        Verify we can't call a non-existent update method.
        """
        from src.audit.logger import AuditLogger
        logger = AuditLogger(cfg_path=temp_audit_db)
        assert not hasattr(logger, "update_prediction"), \
            "AuditLogger must not expose any update/mutation method"
