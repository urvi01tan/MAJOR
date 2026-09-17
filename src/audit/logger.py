from __future__ import annotations

import hashlib
import json
import logging
import uuid
import warnings
from datetime import datetime

from pathlib import Path
from typing import Any

import yaml

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(64), unique=True, nullable=False, index=True)
    patient_id = Column(String(32), nullable=False, index=True)
    hadm_id = Column(String(32))
    stay_id = Column(String(32))
    timestamp = Column(DateTime, nullable=False)
    model_version = Column(String(32))
    model_type = Column(String(64))
    model_used = Column(String(16))       # "online" | "baseline"
    risk_score = Column(Float)
    alert_fired = Column(Integer, default=0)  # 0/1
    online_score = Column(Float)
    baseline_score = Column(Float)
    feature_hash = Column(String(64))    # SHA256 of feature dict (reproducibility)
    features_json = Column(Text)         # full feature vector (compressed JSON)
    explanation_json = Column(Text)      # explainability output
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelUpdateRecord(Base):
    __tablename__ = "model_updates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    update_id = Column(String(64), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    model_version = Column(Integer)
    model_type = Column(String(64))
    n_samples_in_update = Column(Integer)
    trigger = Column(String(64))     # "auto_periodic" | "batch" | "clinician_feedback"
    rolling_auroc = Column(Float)
    n_total_updates = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftEventRecord(Base):
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    detector = Column(String(32))     # "ADWIN" | "DDM" | "PageHinkley"
    severity = Column(String(16))     # "warning" | "drift"
    metric_value = Column(Float)
    details_json = Column(Text)
    model_version_at_event = Column(Integer)
    action_taken = Column(String(64)) # "snapshot" | "rollback" | "logged_only"
    created_at = Column(DateTime, default=datetime.utcnow)


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(64), unique=True, nullable=False)
    prediction_id = Column(String(64), index=True)
    patient_id = Column(String(32))
    clinician_id = Column(String(64), default="anonymous")
    feedback = Column(String(16))   # "true_positive" | "false_positive" | "missed"
    notes = Column(Text)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditMetaRecord(Base):
    __tablename__ = "audit_meta"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(64), unique=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# ID generation helpers
# ---------------------------------------------------------------------------

def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _prediction_id(patient_id: str, ts: datetime) -> str:
    """Generate a guaranteed-unique prediction ID (UUID4 + patient + ts)."""
    return _sha256(f"{patient_id}_{ts.isoformat()}_{uuid.uuid4().hex}")


def _update_id(n_updates: int, ts: datetime) -> str:
    return _sha256(f"upd_{n_updates}_{ts.isoformat()}")


def _drift_id(detector: str, ts: datetime) -> str:
    return _sha256(f"drift_{detector}_{ts.isoformat()}")


def _feedback_id(prediction_id: str, ts: datetime) -> str:
    return _sha256(f"fb_{prediction_id}_{ts.isoformat()}")


# ---------------------------------------------------------------------------
# Audit Logger
# ---------------------------------------------------------------------------

class AuditLogger:
    """
    Thread-safe, append-only audit logger backed by SQLite.

    Usage:
        logger = AuditLogger()
        logger.log_prediction(result, features, explanation)
        logger.log_feedback(prediction_id, patient_id, "true_positive")
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        cfg = _load_cfg(cfg_path)
        db_path = Path(cfg["paths"]["audit_db"])
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        self._init_meta()
        log.info(f"[AuditLogger] Audit DB initialised at {db_path}")

    def _init_meta(self) -> None:
        with self._Session() as sess:
            existing = sess.query(AuditMetaRecord).filter_by(key="schema_version").first()
            if not existing:
                sess.add(AuditMetaRecord(key="schema_version", value=SCHEMA_VERSION))
                sess.add(AuditMetaRecord(key="created_at", value=datetime.utcnow().isoformat()))
                sess.commit()

    # ------------------------------------------------------------------
    def log_prediction(
        self,
        result: dict[str, Any],
        features: dict[str, float],
        explanation: dict[str, Any] | None = None,
    ) -> str:
        """Log a prediction result. Returns the prediction_id."""
        ts = datetime.fromisoformat(result["timestamp"]) if isinstance(result["timestamp"], str) else result["timestamp"]
        pid = _prediction_id(result["patient_id"], ts)

        # Hash features for reproducibility (don't store full vector if privacy concern)
        feat_json = json.dumps({k: v for k, v in features.items() if v == v}, separators=(",", ":"))
        feat_hash = hashlib.sha256(feat_json.encode()).hexdigest()[:32]

        record = PredictionRecord(
            prediction_id=pid,
            patient_id=result["patient_id"],
            hadm_id=result.get("hadm_id"),
            stay_id=result.get("stay_id"),
            timestamp=ts,
            model_version=str(result.get("active_model", "")),
            model_type=result.get("model_used", ""),
            model_used=result.get("model_used", ""),
            risk_score=result.get("risk_score"),
            alert_fired=int(result.get("alert", False)),
            online_score=result.get("online_score"),
            baseline_score=result.get("baseline_score"),
            feature_hash=feat_hash,
            features_json=feat_json,
            explanation_json=json.dumps(explanation, separators=(",", ":")) if explanation else None,
        )

        with self._Session() as sess:
            try:
                sess.add(record)
                sess.commit()
            except Exception as exc:
                log.error(f"[AuditLogger] Failed to log prediction: {exc}")
                sess.rollback()

        return pid

    # ------------------------------------------------------------------
    def log_model_update(
        self,
        model,
        trigger: str = "auto_periodic",
        n_samples: int = 1,
    ) -> None:
        upd_id = _update_id(model.n_updates, datetime.utcnow())
        record = ModelUpdateRecord(
            update_id=upd_id,
            timestamp=datetime.utcnow(),
            model_version=model.version,
            model_type=model.model_type,
            n_samples_in_update=n_samples,
            trigger=trigger,
            rolling_auroc=model.rolling_auroc,
            n_total_updates=model.n_updates,
        )
        with self._Session() as sess:
            try:
                sess.add(record)
                sess.commit()
            except Exception:
                sess.rollback()

    # ------------------------------------------------------------------
    def log_drift_event(
        self, drift_event_dict: dict, model_version: int, action_taken: str = "logged_only"
    ) -> None:
        ts = datetime.fromisoformat(drift_event_dict["timestamp"])
        evt_id = _drift_id(drift_event_dict["detector"], ts)
        record = DriftEventRecord(
            event_id=evt_id,
            timestamp=ts,
            detector=drift_event_dict["detector"],
            severity=drift_event_dict["severity"],
            metric_value=drift_event_dict.get("metric_value"),
            details_json=json.dumps(drift_event_dict),
            model_version_at_event=model_version,
            action_taken=action_taken,
        )
        with self._Session() as sess:
            try:
                sess.add(record)
                sess.commit()
            except Exception:
                sess.rollback()

    # ------------------------------------------------------------------
    def log_feedback(
        self,
        prediction_id: str,
        patient_id: str,
        feedback: str,
        clinician_id: str = "anonymous",
        notes: str = "",
    ) -> str:
        ts = datetime.utcnow()
        fb_id = _feedback_id(prediction_id, ts)
        record = FeedbackRecord(
            feedback_id=fb_id,
            prediction_id=prediction_id,
            patient_id=patient_id,
            clinician_id=clinician_id,
            feedback=feedback,
            notes=notes,
            timestamp=ts,
        )
        with self._Session() as sess:
            try:
                sess.add(record)
                sess.commit()
            except Exception:
                sess.rollback()
        return fb_id

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def recent_predictions(self, n: int = 100) -> list[dict]:
        with self._Session() as sess:
            rows = (
                sess.query(PredictionRecord)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "prediction_id": r.prediction_id,
                    "patient_id": r.patient_id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "risk_score": r.risk_score,
                    "alert_fired": bool(r.alert_fired),
                    "model_used": r.model_used,
                    "online_score": r.online_score,
                    "baseline_score": r.baseline_score,
                }
                for r in rows
            ]

    def alert_count(self, last_n_hours: int = 24) -> int:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=last_n_hours)
        with self._Session() as sess:
            return sess.query(PredictionRecord).filter(
                PredictionRecord.alert_fired == 1,
                PredictionRecord.timestamp >= cutoff,
            ).count()

    def export_predictions_csv(self, output_path: str) -> None:
        import pandas as pd
        with self._Session() as sess:
            rows = sess.query(PredictionRecord).all()
        data = [
            {
                "prediction_id": r.prediction_id,
                "patient_id": r.patient_id,
                "timestamp": r.timestamp,
                "risk_score": r.risk_score,
                "alert_fired": r.alert_fired,
                "model_used": r.model_used,
                "model_version": r.model_version,
                "online_score": r.online_score,
                "baseline_score": r.baseline_score,
                "feature_hash": r.feature_hash,
            }
            for r in rows
        ]
        pd.DataFrame(data).to_csv(output_path, index=False)
        log.info(f"[AuditLogger] Exported {len(data)} predictions to {output_path}")

    def recent_feedback(self, n: int = 50) -> list[dict]:
        with self._Session() as sess:
            rows = (
                sess.query(FeedbackRecord)
                .order_by(FeedbackRecord.timestamp.desc())
                .limit(n)
                .all()
            )
            return [
                {
                    "feedback_id": r.feedback_id,
                    "prediction_id": r.prediction_id,
                    "patient_id": r.patient_id,
                    "feedback": r.feedback,
                    "clinician_id": r.clinician_id,
                    "notes": r.notes,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]

    def shift_stats(self) -> dict:
        with self._Session() as sess:
            n_pred = sess.query(PredictionRecord).count()
            n_alerts = sess.query(PredictionRecord).filter(PredictionRecord.alert_fired == 1).count()
            n_feedback = sess.query(FeedbackRecord).count()
            tp = sess.query(FeedbackRecord).filter(FeedbackRecord.feedback == "true_positive").count()
            fp = sess.query(FeedbackRecord).filter(FeedbackRecord.feedback == "false_positive").count()
        return {
            "n_predictions": n_pred,
            "n_alerts": n_alerts,
            "n_feedback": n_feedback,
            "true_positives": tp,
            "false_positives": fp,
            "alert_rate": (n_alerts / n_pred) if n_pred else 0.0,
            "ppv": (tp / (tp + fp)) if (tp + fp) else None,
        }
