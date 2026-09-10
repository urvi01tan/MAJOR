"""
Offline Baseline Model Training — PhysioNet CinC 2019
======================================================
Trains a scikit-learn Pipeline (StandardScaler + LogisticRegression)
on the PhysioNet/CinC Challenge 2019 dataset as the validated static
safety baseline.

Dataset: kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
  - ~40,000 patients, one PSV file per patient
  - Each row = 1 hour of ICU data
  - SepsisLabel column = 0 or 1 (label is inline — no separate derivation)

Steps:
  1. Load PSV files from data/physionet2019/training/
  2. Build feature matrix using WindowAggregator (retrospective replay)
  3. Labels come directly from SepsisLabel column
  4. Train / evaluate on 80/20 patient split
  5. Save model + metadata to models/baseline_model.pkl

Usage:
  python scripts/train_baseline.py
  python scripts/train_baseline.py --n-patients 2000 --model xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# --- ensure src is importable ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.missing_handler import MissingDataHandler
from src.features.window_aggregator import WindowAggregator
from src.ingestion.stream_simulator import PhysioNetDataLoader, StreamSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(
    patients: list[tuple[str, pd.DataFrame]],
    sample_every_n_hours: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) from patient DataFrames.

    For each patient, we take a snapshot of the rolling feature vector
    every `sample_every_n_hours` hours to avoid class imbalance from
    the majority of non-sepsis hours.

    Args:
        patients: list of (patient_id, hourly_dataframe) tuples
        sample_every_n_hours: subsample rate to keep feature matrix manageable

    Returns:
        X: feature DataFrame
        y: sepsis label Series
    """
    aggregator = WindowAggregator()
    imputer = MissingDataHandler()

    feature_rows: list[dict] = []
    labels: list[int] = []

    for patient_id, df in patients:
        # Build a mini stream for this patient and aggregate features
        for hour_index, (_, row) in enumerate(df.iterrows()):
            sepsis_label = int(row.get("SepsisLabel", 0))

            # Emit each non-NaN feature column as an event
            from src.ingestion.stream_simulator import (
                ALL_FEATURE_COLS, _ICU_ADMIT_BASE, _COL_SOURCE
            )
            from datetime import timedelta

            ts = _ICU_ADMIT_BASE + timedelta(hours=hour_index)
            for col in ALL_FEATURE_COLS:
                if col not in row.index:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                event = {
                    "patient_id": patient_id,
                    "timestamp": ts,
                    "source": _COL_SOURCE.get(col, "unknown"),
                    "feature_name": col,
                    "value": float(val),
                    "unit": "",
                    "sepsis_label": sepsis_label,
                    "iculos": hour_index,
                }
                aggregator.update(event)

            # Subsample: snapshot every N hours
            if hour_index % sample_every_n_hours == 0:
                fv_raw = aggregator.get_feature_vector(patient_id, ts)
                if fv_raw:
                    fv = imputer.impute_and_update(fv_raw)
                    feature_rows.append(fv)
                    labels.append(sepsis_label)

    if not feature_rows:
        raise RuntimeError(
            "No feature rows generated. Check PSV files in data/physionet2019/training/"
        )

    X = pd.DataFrame(feature_rows).fillna(0.0)
    y = pd.Series(labels, name="sepsis_label")
    log.info(
        f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features, "
        f"sepsis prevalence: {y.mean():.1%}"
    )
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "logistic_regression"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            log.error("xgboost not installed. Run: pip install xgboost")
            sys.exit(1)

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        model.fit(X_train, y_train)

    else:  # logistic_regression (default)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=0.1,
                solver="saga",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(X_train, y_train)

    # Evaluation
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auroc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    sensitivity = report.get("1", {}).get("recall", float("nan"))
    specificity = report.get("0", {}).get("recall", float("nan"))

    log.info(f"Test AUROC:    {auroc:.4f}")
    log.info(f"Sensitivity:   {sensitivity:.3f}")
    log.info(f"Specificity:   {specificity:.3f}")
    log.info("\n" + classification_report(y_test, y_pred))

    meta = {
        "model_type": model_type,
        "dataset": "PhysioNet/CinC Challenge 2019",
        "kaggle_url": "kaggle.com/datasets/salikhussaini49/prediction-of-sepsis",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_auroc": auroc,
        "test_sensitivity": sensitivity,
        "test_specificity": specificity,
        "trained_at": datetime.utcnow().isoformat(),
        "version": "baseline_v1",
    }
    return model, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PhysioNet CinC 2019 offline baseline model"
    )
    parser.add_argument(
        "--n-patients", type=int, default=None,
        help="Limit to N patients for faster training (default: all ~40K)"
    )
    parser.add_argument(
        "--sample-every", type=int, default=3,
        help="Take feature snapshot every N hours per patient (default: 3)"
    )
    parser.add_argument(
        "--model", choices=["logistic_regression", "xgboost"],
        default="logistic_regression"
    )
    parser.add_argument("--cfg", default="config/settings.yaml")
    args = parser.parse_args()

    cfg = _load_cfg(args.cfg)
    training_dir = cfg["paths"]["training_dir"]

    log.info("=" * 60)
    log.info("PhysioNet CinC 2019 — Baseline Model Training")
    log.info("=" * 60)

    # Load patient data
    log.info(f"Loading PSV files from {training_dir} …")
    loader = PhysioNetDataLoader(training_dir)
    patients = loader.load_all(n_patients=args.n_patients)
    log.info(f"Loaded {len(patients):,} patients.")

    # Split patients (not rows) for train/test to avoid data leakage
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(patients))
    split = int(0.8 * len(patients))
    train_patients = [patients[i] for i in indices[:split]]
    test_patients = [patients[i] for i in indices[split:]]

    # Build feature matrices separately
    log.info("Building training feature matrix …")
    X_train_full, y_train_full = build_feature_matrix(
        train_patients, sample_every_n_hours=args.sample_every
    )
    log.info("Building test feature matrix …")
    X_test_full, y_test_full = build_feature_matrix(
        test_patients, sample_every_n_hours=args.sample_every
    )

    # Align columns
    all_cols = list(X_train_full.columns)
    X_test_full = X_test_full.reindex(columns=all_cols, fill_value=0.0)

    X = pd.concat([X_train_full, X_test_full], ignore_index=True)
    y = pd.concat([y_train_full, y_test_full], ignore_index=True)

    # Train
    log.info(f"Training {args.model} …")
    model, meta = train_model(X, y, model_type=args.model)

    # Save
    out_path = Path(cfg["paths"]["baseline_model"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": all_cols,
            "version": meta["version"],
            "trained_at": meta["trained_at"],
            "dataset": meta["dataset"],
        }, f)
    log.info(f"Baseline model saved → {out_path}")

    meta_path = Path(cfg["paths"]["baseline_meta"])
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Metadata saved → {meta_path}")
    log.info("✅  Training complete!")


if __name__ == "__main__":
    main()
