"""
Stream Simulation CLI — PhysioNet CinC 2019
============================================
Entry point to run the full pipeline in simulation mode,
replaying the PhysioNet/CinC Challenge 2019 dataset as a real-time stream.

Dataset: kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

Usage:
  python scripts/simulate_stream.py
  python scripts/simulate_stream.py --n-patients 500 --speed 60
  python scripts/simulate_stream.py --n-patients 200 --no-realtime
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.stream_simulator import StreamSimulator
from src.pipeline import EWSPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/simulate_stream.log"),
    ],
)
log = logging.getLogger(__name__)


def print_result(result: dict) -> None:
    """Print a formatted prediction result to console."""
    alert_str = "🚨 ALERT" if result.get("alert") else "  ·"
    score = result.get("risk_score", 0.0)
    bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    model_tag = result.get("model_used", "?")[0].upper()  # O or B
    print(
        f"{alert_str} Pt {result['patient_id'][:8]:>8} "
        f"[{bar}] {score:.1%} [{model_tag}] "
        f"AUROC={result.get('online_rolling_auroc', 0):.3f}"
    )
    if result.get("explanation") and result.get("alert"):
        exp = result["explanation"]
        print(f"    📋 {exp.get('summary', '')}")
        for feat in exp.get("top_features", [])[:3]:
            print(f"       • {feat['plain_english']} (contribution={feat['contribution']:+.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate real-time patient deterioration stream (PhysioNet CinC 2019)"
    )
    parser.add_argument(
        "--n-patients", type=int, default=None,
        help="Number of patients to simulate (default: all ~40K)"
    )
    parser.add_argument(
        "--speed", type=int, default=None,
        help="Speed multiplier override (e.g. 60 = 1 simulated hour per 60s)"
    )
    parser.add_argument(
        "--no-realtime", action="store_true",
        help="Disable real-time throttling (max speed)"
    )
    parser.add_argument("--cfg", default="config/settings.yaml")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  Real-Time Patient Deterioration EWS — PhysioNet CinC 2019")
    log.info("=" * 70)

    # Override speed if provided
    if args.speed:
        import yaml
        with open(args.cfg) as f:
            cfg = yaml.safe_load(f)
        cfg["streaming"]["speed_multiplier"] = args.speed
        with open(args.cfg, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        log.info(f"Speed multiplier set to {args.speed}×")

    # Initialise pipeline
    log.info("Initialising EWS pipeline …")
    pipeline = EWSPipeline(args.cfg)
    pipeline.start()

    # Labels are inline in the stream events (sepsis_label field)
    # Pipeline reads them directly — no separate label map needed

    # Set up simulator
    simulator = StreamSimulator(args.cfg)
    simulator.load_data(n_patients=args.n_patients)

    # Feed labels from loaded patient data into pipeline for delayed-label learning
    try:
        patient_labels = simulator.get_patient_labels()
        pipeline.load_labels(patient_labels)
        log.info(
            f"Loaded {len(patient_labels):,} patient labels "
            f"({sum(patient_labels.values()):,} sepsis positive)."
        )
    except Exception as exc:
        log.warning(f"Could not load labels: {exc}. Online model will run without labels.")

    # Run
    n_processed = 0
    n_alerts = 0
    start_wall = time.time()

    log.info("▶  Streaming started. Ctrl+C to stop.\n")
    print(
        f"{'ALERT':>8}  {'Patient':>8}  {'Risk Score':>24}  "
        f"{'Model':>5}  {'Rolling AUROC':>13}"
    )
    print("-" * 75)

    try:
        for event in simulator.stream(
            n_patients=args.n_patients,
            realtime=not args.no_realtime,
        ):
            result = pipeline.process_event(event)
            if result:
                n_processed += 1
                if result.get("alert"):
                    n_alerts += 1
                # Print every 20th prediction (to avoid flooding console)
                if n_processed % 20 == 0 or result.get("alert"):
                    print_result(result)

                # Print pipeline status every 1000 events
                if n_processed % 1000 == 0:
                    status = pipeline.status()
                    elapsed = time.time() - start_wall
                    log.info(
                        f"── Status @ {n_processed:,} predictions "
                        f"({elapsed:.0f}s elapsed) ──"
                    )
                    log.info(f"  Active model: {status['active_model']}")
                    log.info(f"  Online AUROC: {status['online_auroc']:.4f}")
                    log.info(f"  Drift events: {status['total_drift_events']}")
                    log.info(f"  Rollbacks:    {status['rollback_count']}")
                    log.info(f"  Alerts fired: {n_alerts}")

    except KeyboardInterrupt:
        log.info("\n⏹  Interrupted by user.")

    finally:
        pipeline.shutdown()
        elapsed = time.time() - start_wall

        import yaml
        with open(args.cfg) as f:
            cfg = yaml.safe_load(f)

        log.info(f"\n{'='*60}")
        log.info(f"Simulation complete in {elapsed:.1f}s")
        log.info(f"Events processed:  {pipeline.status()['n_events_processed']:,}")
        log.info(f"Predictions made:  {n_processed:,}")
        log.info(f"Alerts fired:      {n_alerts:,}")
        log.info(f"Model snapshots:   {pipeline.registry.version_count()}")
        log.info(f"Drift events:      {pipeline.drift_detector.total_drift_count}")
        log.info(f"Audit DB:          {cfg['paths']['audit_db']}")
        log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
