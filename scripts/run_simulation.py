"""
Headless simulation runner — feeds the EWSPipeline with real patient data
so the Streamlit dashboard has live data to display.

Run this in a separate terminal while the dashboard is open:
    py -3 scripts/run_simulation.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Headless EWS simulation")
    parser.add_argument("--n-patients", type=int, default=80,
                        help="Number of ICU patients to simulate (default: 80)")
    parser.add_argument("--speed", type=float, default=0,
                        help="Delay between events in seconds (0 = as fast as possible)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Stop after N events (default: run until all patients done)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"SepsisGuard — Headless Simulation ({args.n_patients} patients)")
    log.info("=" * 60)

    from src.pipeline import EWSPipeline
    from src.ingestion.stream_simulator import StreamSimulator

    log.info("Initialising EWS pipeline …")
    pipeline = EWSPipeline()
    pipeline.start()

    status = pipeline.status()
    log.info(f"Baseline AUROC: {status.get('baseline_auroc', 0):.4f}")
    log.info(f"Alert threshold: {status.get('alert_threshold', 0.65):.0%}")

    log.info(f"Loading {args.n_patients} patient records …")
    sim = StreamSimulator()
    sim.load_data(n_patients=args.n_patients)

    try:
        labels = sim.get_patient_labels()
        pipeline.load_labels(labels)
        log.info(f"Labels loaded: {sum(labels.values())} sepsis patients")
    except Exception as exc:
        log.warning(f"Could not load labels: {exc}")

    log.info("Starting stream … Press Ctrl+C to stop.\n")

    n_events = 0
    n_alerts = 0
    n_patients_seen: set[str] = set()
    t0 = time.time()

    try:
        for event in sim.stream(n_patients=args.n_patients, realtime=False):
            result = pipeline.process_event(event)
            n_events += 1
            n_patients_seen.add(event["patient_id"])

            if result and result.get("alert"):
                n_alerts += 1
                pid = result["patient_id"]
                score = result["risk_score"]
                sev = result.get("severity", "—")
                news2 = result.get("news2", "—")
                sofa = result.get("sofa", "—")
                log.info(
                    f"🚨 ALERT  Patient={pid:12s}  Risk={score:.1%}  "
                    f"Severity={sev:8s}  NEWS2={news2}  SOFA={sofa}"
                )

            # Progress every 500 events
            if n_events % 500 == 0:
                elapsed = time.time() - t0
                rate = n_events / elapsed
                log.info(
                    f"   Progress: {n_events:,} events | "
                    f"{len(n_patients_seen)} patients | "
                    f"{n_alerts} alerts | "
                    f"{rate:.0f} events/sec"
                )

            if args.max_events and n_events >= args.max_events:
                log.info(f"Reached --max-events={args.max_events}, stopping.")
                break

            if args.speed > 0:
                time.sleep(args.speed)

    except KeyboardInterrupt:
        log.info("\nInterrupted by user.")

    elapsed = time.time() - t0
    final = pipeline.status()
    log.info("\n" + "=" * 60)
    log.info("SIMULATION COMPLETE")
    log.info("=" * 60)
    log.info(f"  Events processed : {final['n_events_processed']:,}")
    log.info(f"  Predictions made : {final['n_predictions']:,}")
    log.info(f"  Alerts fired     : {final['n_alerts']:,}")
    log.info(f"  Patients tracked : {final['n_patients_tracked']:,}")
    log.info(f"  Online AUROC     : {final['online_auroc']:.4f}")
    log.info(f"  Baseline AUROC   : {final['baseline_auroc']:.4f}")
    log.info(f"  Drift events     : {final['total_drift_events']:,}")
    log.info(f"  Elapsed          : {elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
