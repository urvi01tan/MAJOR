
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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.missing_handler import MissingDataHandler
from src.features.window_aggregator import WindowAggregator
from src.ingestion.stream_simulator import PhysioNetDataLoader, StreamSimulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


VITAL_COLS  = ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
LAB_COLS    = ["BaseExcess","HCO3","FiO2","pH","PaCO2","SaO2","AST","BUN",
               "Alkalinephos","Calcium","Chloride","Creatinine","Bilirubin_direct",
               "Glucose","Lactate","Magnesium","Phosphate","Potassium",
               "Bilirubin_total","TroponinI","Hct","Hgb","PTT","WBC",
               "Fibrinogen","Platelets"]
DEMO_COLS   = ["Age","Gender","Unit1","Unit2","HospAdmTime","ICULOS"]
ALL_COLS    = VITAL_COLS + LAB_COLS + DEMO_COLS

# Rolling window sizes in hours (each row = 1 hour in the PSV files)
WINDOWS_H   = [1, 4, 8]


def _fast_features_for_patient(pid: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a feature matrix for a single patient using pandas rolling windows.
    Much faster than the event-by-event streaming approach (~100x speedup).

    For each clinical column and each window size we compute:
      mean, std, min, max, last (most recent non-NaN value), slope proxy (last - first in window)
    Plus: missing indicator, staleness (hours since last obs), and derived features.
    """
    # Keep only columns that exist in this file
    cols = [c for c in ALL_COLS if c in df.columns]
    data = df[cols].copy()

    feature_frames: list[pd.DataFrame] = []

    for col in cols:
        s = data[col].copy()
        for w in WINDOWS_H:
            roll = s.rolling(window=w, min_periods=1)
            feat_prefix = f"{col}__w{w*60}"
            # Slope: use vectorised diff over window (fast, no Python callbacks)
            slope_approx = s.diff(w).fillna(s.diff(1)).fillna(0.0) / max(w, 1)
            wdf = pd.DataFrame({
                f"{feat_prefix}_mean":  roll.mean(),
                f"{feat_prefix}_std":   roll.std().fillna(0.0),
                f"{feat_prefix}_min":   roll.min(),
                f"{feat_prefix}_max":   roll.max(),
                f"{feat_prefix}_count": roll.count(),
                f"{feat_prefix}_slope": slope_approx,
            })
            feature_frames.append(wdf)

        # Missing indicator and staleness
        obs_mask = s.notna()
        feature_frames.append(pd.DataFrame({
            f"{col}__missing": (~obs_mask).astype(float),
            f"{col}__staleness_h": s.isna().astype(int).groupby(
                obs_mask.cumsum()
            ).cumsum().astype(float),
        }))

    X = pd.concat(feature_frames, axis=1).copy()  # defragment before adding derived cols

    # ── Derived features ──────────────────────────────────────────────────
    hr  = X.get("HR__w60_mean",  pd.Series(float("nan"), index=X.index))
    sbp = X.get("SBP__w60_mean", pd.Series(float("nan"), index=X.index))
    dbp = X.get("DBP__w60_mean", pd.Series(float("nan"), index=X.index))
    spo2= X.get("O2Sat__w60_mean", pd.Series(float("nan"), index=X.index))
    fio2= X.get("FiO2__w60_mean",  pd.Series(float("nan"), index=X.index))
    lac = X.get("Lactate__w240_slope", pd.Series(float("nan"), index=X.index))

    X["shock_index_60"]    = hr / sbp.replace(0, float("nan"))
    X["pulse_pressure_60"] = sbp - dbp
    X["pf_ratio_proxy"]    = spo2 / (fio2 / 100.0).replace(0, float("nan"))
    X["lactate_rising"]    = (lac > 0).astype(float)

    y = df["SepsisLabel"].fillna(0).astype(int) if "SepsisLabel" in df.columns \
        else pd.Series(0, index=df.index)

    return X, y


def build_feature_matrix(
    patients: list[tuple[str, pd.DataFrame]],
    sample_every_n_hours: int = 3,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build train/test feature matrix using fast pandas rolling windows.
    Samples every `sample_every_n_hours` rows per patient.
    """
    from tqdm import tqdm

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for patient_id, df in tqdm(patients, desc="  Building features", unit="pt", ncols=80):
        try:
            X_p, y_p = _fast_features_for_patient(patient_id, df)
            # Sample every N rows
            idx = list(range(0, len(X_p), sample_every_n_hours))
            all_X.append(X_p.iloc[idx])
            all_y.append(y_p.iloc[idx])
        except Exception as exc:
            log.warning(f"  [skip] {patient_id}: {exc}")

    if not all_X:
        raise RuntimeError("No feature rows generated. Check PSV files.")

    X = pd.concat(all_X, ignore_index=True).fillna(0.0)
    y = pd.concat(all_y, ignore_index=True)

    log.info(
        f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features | "
        f"Sepsis prevalence: {y.mean():.1%}"
    )
    return X, y




def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "logistic_regression",
):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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

    else:  
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

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    auroc = roc_auc_score(y_test, y_proba)
    try:
        auprc = average_precision_score(y_test, y_proba)
    except Exception:
        auprc = float("nan")
    report = classification_report(y_test, y_pred, output_dict=True)

    sensitivity = report.get("1", {}).get("recall", float("nan"))
    specificity = report.get("0", {}).get("recall", float("nan"))
    precision = report.get("1", {}).get("precision", float("nan"))

    log.info(f"Test AUROC:    {auroc:.4f}")
    log.info(f"Test AUPRC:    {auprc:.4f}")
    log.info(f"Sensitivity:   {sensitivity:.3f}")
    log.info(f"Specificity:   {specificity:.3f}")
    log.info(f"Precision:     {precision:.3f}")
    log.info("\n" + classification_report(y_test, y_pred))

    # ── Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    tn, fp_count, fn_count, tp_count = cm.ravel()
    log.info("Confusion Matrix:")
    log.info(f"  TN={tn:,}  FP={fp_count:,}")
    log.info(f"  FN={fn_count:,}  TP={tp_count:,}")
    log.info(f"  PPV (precision): {tp_count/(tp_count+fp_count):.3f}" if (tp_count+fp_count) > 0 else "  PPV: N/A")
    log.info(f"  NPV: {tn/(tn+fn_count):.3f}" if (tn+fn_count) > 0 else "  NPV: N/A")

    meta = {
        "model_type": model_type,
        "dataset": "PhysioNet/CinC Challenge 2019",
        "kaggle_url": "kaggle.com/datasets/salikhussaini49/prediction-of-sepsis",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_auroc": float(auroc),
        "test_auprc": float(auprc) if auprc == auprc else None,
        "test_sensitivity": float(sensitivity) if sensitivity == sensitivity else None,
        "test_specificity": float(specificity) if specificity == specificity else None,
        "test_precision": float(precision) if precision == precision else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp_count), "fn": int(fn_count), "tp": int(tp_count)},
        "sepsis_prevalence_train": float(y_train.mean()),
        "sepsis_prevalence_test": float(y_test.mean()),
        "trained_at": datetime.utcnow().isoformat(),
        "version": "baseline_v1",
    }
    return model, meta


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
    log.info(f"Loading PSV files from {training_dir} …")
    loader = PhysioNetDataLoader(training_dir)
    patients = loader.load_all(n_patients=args.n_patients)
    log.info(f"Loaded {len(patients):,} patients.")
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

    all_cols = list(X_train_full.columns)
    X_test_full = X_test_full.reindex(columns=all_cols, fill_value=0.0)

    log.info(f"Training {args.model} on patient-level split (no leakage) …")
    model, meta = train_model(
        X_train_full, y_train_full, X_test_full, y_test_full, model_type=args.model
    )
    meta["n_patients_train"] = len(train_patients)
    meta["n_patients_test"] = len(test_patients)

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

    # ── Feature importance JSON ────────────────────────────────────────
    fi_path_str = cfg.get("paths", {}).get("feature_importance", "models/feature_importance.json")
    fi_path = Path(fi_path_str)
    try:
        inner = model
        if hasattr(inner, "steps"):
            inner = inner.steps[-1][1]
        if hasattr(inner, "coef_"):
            fi = {f: float(c) for f, c in zip(all_cols, inner.coef_[0])}
        elif hasattr(inner, "feature_importances_"):
            fi = {f: float(v) for f, v in zip(all_cols, inner.feature_importances_)}
        else:
            fi = {}
        if fi:
            with open(fi_path, "w") as f:
                json.dump(fi, f, indent=2)
            log.info(f"Feature importances saved → {fi_path} ({len(fi)} features)")
    except Exception as exc:
        log.warning(f"Could not save feature importances: {exc}")

    log.info("✅  Training complete!")


if __name__ == "__main__":
    main()
