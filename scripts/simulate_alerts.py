"""
Simulate real-time patient deterioration streams with clinical alert triggering,
explainability generation, and database logging.
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("AlertRunner")


def main():
    from src.pipeline import EWSPipeline
    from src.ingestion.stream_simulator import (
        StreamSimulator,
        PhysioNetDataLoader,
        ALL_FEATURE_COLS,
        _COL_SOURCE,
        _ICU_ADMIT_BASE,
    )

    print("\n" + "=" * 80)
    print(" 🏥 SEPSISGUARD EWS — PATIENT INFERENCE & ALERT GENERATOR")
    print("=" * 80 + "\n")

    log.info("Initializing Early Warning Score (EWS) Pipeline...")
    pipeline = EWSPipeline()
    pipeline.start()

    # Set sensitivity threshold to 0.40 for sensitive alert triggering
    pipeline.set_alert_threshold(0.40)
    log.info("Alert threshold adjusted to 40% for clinical sensitivity.")

    loader = PhysioNetDataLoader(pipeline.cfg["paths"]["training_dir"])
    patient_files = list(Path(pipeline.cfg["paths"]["training_dir"]).glob("*.psv"))

    sepsis_patients = []
    log.info("Finding deteriorating sepsis patient profiles from dataset...")
    for f in patient_files[:1000]:
        try:
            df = pd.read_csv(f, sep="|")
            if "SepsisLabel" in df.columns and df["SepsisLabel"].max() == 1:
                sepsis_patients.append((f.stem, df))
                if len(sepsis_patients) >= 12:
                    break
        except Exception:
            continue

    log.info(f"Loaded {len(sepsis_patients)} confirmed deteriorating sepsis patients.")

    # Load labels
    labels = {pid: 1 for pid, _ in sepsis_patients}
    pipeline.load_labels(labels)

    alerts_triggered = []
    events_count = 0
    t0 = time.time()

    print("\n" + "─" * 80)
    print(" ▶ STREAMING PATIENT TELEMETRY & RUNNING ENSEMBLE INFERENCE")
    print("─" * 80 + "\n")

    for pid, df in sepsis_patients:
        # Replay each hourly observation for the deteriorating patient
        for hour_idx, (_, row) in enumerate(df.iterrows()):
            ts = datetime.now() - timedelta(hours=len(df) - hour_idx)
            for col in ALL_FEATURE_COLS:
                if col not in row.index or pd.isna(row[col]):
                    continue
                ev = {
                    "patient_id": pid,
                    "timestamp": ts,
                    "source": _COL_SOURCE.get(col, "vitals"),
                    "feature_name": col,
                    "value": float(row[col]),
                    "unit": "",
                    "sepsis_label": int(row.get("SepsisLabel", 0)),
                    "iculos": int(row.get("ICULOS", hour_idx)),
                }
                res = pipeline.process_event(ev)
                events_count += 1
                if res and res.get("alert"):
                    alerts_triggered.append(res)
                    print_alert_badge(res)

    # Also run synthetic high-risk generators to test extreme deterioration edge cases
    log.info("Injecting synthetic deteriorating ICU cases...")
    sim = StreamSimulator()
    synthetic_events = sim.generate_random_events(n_patients=6, high_risk_count=4)
    for ev in synthetic_events:
        res = pipeline.process_event(ev)
        events_count += 1
        if res and res.get("alert"):
            alerts_triggered.append(res)
            print_alert_badge(res)

    pipeline.shutdown()
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print(" 📊 EXECUTION & ALERT SUMMARY")
    print("=" * 80)
    print(f"  Total Telemetry Events Processed : {events_count:,}")
    print(f"  Total Alerts Fired               : {len(alerts_triggered)}")
    print(f"  Patients Triggering Alerts       : {len({a['patient_id'] for a in alerts_triggered})}")
    print(f"  Execution Time                   : {elapsed:.2f}s")
    print(f"  Audit Database                   : logs/audit.db (Saved & Ready)")
    print("=" * 80 + "\n")


def print_alert_badge(res: dict):
    pid = res["patient_id"]
    score = res.get("risk_score", 0.0)
    sev = res.get("severity", "HIGH").upper()
    news2 = res.get("news2", "—")
    sofa = res.get("sofa", "—")
    model = res.get("model_used", "Ensemble")
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))

    print(f"\n┌────────────────────────────────────────────────────────────────────────┐")
    print(f"│ 🚨 ALERT DISPATCHED | Patient: {pid:<10} | Severity: {sev:<10}       │")
    print(f"├────────────────────────────────────────────────────────────────────────┤")
    print(f"│ Sepsis Risk Score : [{bar}] {score:6.1%} (Model: {model})")
    print(f"│ Clinical Scores   : NEWS2={news2}  |  SOFA={sofa}")
    
    exp = res.get("explanation")
    if exp and isinstance(exp, dict):
        if "summary" in exp:
            print(f"│ Clinical Summary  : {exp['summary']}")
        top_f = exp.get("top_features", [])
        if top_f:
            print(f"│ Key Determinants  :")
            for f in top_f[:3]:
                desc = f.get("plain_english", f.get("feature", ""))
                c = f.get("contribution", 0.0)
                print(f"│   • {desc} (Impact: {c:+.3f})")
    print(f"└────────────────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
