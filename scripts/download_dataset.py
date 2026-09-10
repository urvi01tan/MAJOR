#!/usr/bin/env python3
"""
Download PhysioNet/CinC Challenge 2019 — Early Prediction of Sepsis
====================================================================
Dataset: kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

This script downloads the dataset via the Kaggle API (free account only).
No PhysioNet credentials required.

Requirements:
  pip install kaggle
  # Then place your Kaggle API key at: ~/.kaggle/kaggle.json
  # Get it from: https://www.kaggle.com/account → "Create New Token"

Usage:
  python scripts/download_dataset.py
  python scripts/download_dataset.py --out-dir data/physionet2019
  python scripts/download_dataset.py --manual   # print manual download instructions
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "salikhussaini49/prediction-of-sepsis"
DEFAULT_OUT_DIR = "data/physionet2019"
TRAINING_SUBDIR = "training"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_kaggle_cli() -> bool:
    """Return True if kaggle CLI is installed and on PATH."""
    return shutil.which("kaggle") is not None


def _check_kaggle_credentials() -> bool:
    """Return True if ~/.kaggle/kaggle.json exists."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def _print_manual_instructions(out_dir: str) -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       Manual Dataset Download Instructions                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Go to: https://www.kaggle.com/datasets/salikhussaini49/          ║
║            prediction-of-sepsis                                      ║
║                                                                      ║
║  2. Click the [Download] button (requires free Kaggle account)       ║
║                                                                      ║
║  3. Unzip the downloaded file into:                                  ║
║       """ + out_dir + """/training/                                  ║
║                                                                      ║
║  4. Verify: the directory should contain ~40,000 .psv files:         ║
║       p000001.psv, p000002.psv, ...                                  ║
║                                                                      ║
║  Dataset size: ~170 MB compressed, ~650 MB unzipped                  ║
║  Patients:     ~40,336 ICU stays                                     ║
║  Features:     40 columns (8 vitals + 26 labs + 6 demographics)      ║
║  Label:        SepsisLabel (0/1) — per-hour                          ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def _setup_kaggle_api() -> bool:
    """
    Try to install kaggle CLI via pip if not present.
    Returns True if kaggle is available after setup.
    """
    if _check_kaggle_cli():
        return True
    print("📦  kaggle CLI not found. Installing via pip …")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "kaggle", "-q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ❌  pip install kaggle failed:\n{result.stderr}")
        return False
    print("  ✅  kaggle installed.")
    return _check_kaggle_cli()


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_dataset(out_dir: str = DEFAULT_OUT_DIR) -> bool:
    """
    Download the PhysioNet CinC 2019 dataset from Kaggle.
    Returns True on success.
    """
    out_path = Path(out_dir)
    training_path = out_path / TRAINING_SUBDIR

    # Check if already downloaded
    existing_psv = list(training_path.glob("*.psv"))
    if len(existing_psv) > 100:
        print(
            f"✅  Dataset already present: {len(existing_psv):,} PSV files in {training_path}\n"
            f"    Delete {training_path} to re-download."
        )
        return True

    # Ensure kaggle CLI is available
    if not _setup_kaggle_api():
        print("\n⚠️  Could not set up Kaggle CLI.")
        _print_manual_instructions(out_dir)
        return False

    # Check credentials
    if not _check_kaggle_credentials():
        print("""
⚠️  Kaggle API credentials not found at ~/.kaggle/kaggle.json

To set up:
  1. Go to https://www.kaggle.com/account
  2. Click "Create New API Token" → downloads kaggle.json
  3. Move it to ~/.kaggle/kaggle.json
  4. Run: chmod 600 ~/.kaggle/kaggle.json  (Linux/Mac)
""")
        _print_manual_instructions(out_dir)
        return False

    # Download
    out_path.mkdir(parents=True, exist_ok=True)
    zip_path = out_path / "prediction-of-sepsis.zip"

    print(f"\n⬇️   Downloading {KAGGLE_DATASET} …")
    print(f"     Destination: {out_path}\n")

    result = subprocess.run(
        ["kaggle", "datasets", "download", KAGGLE_DATASET,
         "-p", str(out_path), "--force"],
        text=True,
    )

    if result.returncode != 0:
        print("❌  Kaggle download failed.")
        _print_manual_instructions(out_dir)
        return False

    # Find the downloaded zip
    zips = list(out_path.glob("*.zip"))
    if not zips:
        print("❌  Download completed but no zip file found.")
        return False

    zip_path = zips[0]
    print(f"\n📦  Extracting {zip_path.name} …")
    training_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract PSV files into training subdir
        psv_members = [m for m in zf.namelist() if m.endswith(".psv")]
        if not psv_members:
            # Try extracting everything and look for PSV files
            zf.extractall(out_path)
        else:
            for member in psv_members:
                # Extract flat (strip any directory prefix)
                filename = Path(member).name
                source = zf.open(member)
                dest = training_path / filename
                with open(dest, "wb") as f:
                    f.write(source.read())

    # Verify
    psv_files = list(training_path.glob("*.psv"))
    if len(psv_files) < 100:
        # PSVs may have been extracted to a subdirectory — find and move them
        found = list(out_path.rglob("*.psv"))
        for f in found:
            if f.parent != training_path:
                f.rename(training_path / f.name)
        psv_files = list(training_path.glob("*.psv"))

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    n = len(psv_files)
    if n < 100:
        print(f"⚠️  Only {n} PSV files found. Expected ~40,000.")
        _print_manual_instructions(out_dir)
        return False

    print(f"""
✅  Dataset downloaded successfully!
    📂  Location:  {training_path}
    📄  Patients:  {n:,} PSV files
    💡  Next step: python scripts/train_baseline.py
""")
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PhysioNet/CinC 2019 Sepsis dataset from Kaggle"
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print manual download instructions and exit",
    )
    args = parser.parse_args()

    if args.manual:
        _print_manual_instructions(args.out_dir)
        return

    success = download_dataset(args.out_dir)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
