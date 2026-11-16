#!/usr/bin/env python3

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


KAGGLE_DATASET = "salikhussaini49/prediction-of-sepsis"
DEFAULT_OUT_DIR = "data/physionet2019"


def download_dataset(out_dir=DEFAULT_OUT_DIR):

    out_path = Path(out_dir)
    training_path = out_path / "training"

    # Already downloaded?
    psv_files = list(training_path.glob("*.psv"))

    if len(psv_files) > 100:
        print(f" Dataset already exists: {len(psv_files)} patients")
        return True

    # Create output folder
    out_path.mkdir(parents=True, exist_ok=True)

    # Download dataset from Kaggle
    print("⬇Downloading dataset...")

    result = subprocess.run([
        "kaggle",
        "datasets",
        "download",
        KAGGLE_DATASET,
        "-p", str(out_path),
        "--force"
    ])

    if result.returncode != 0:
        print("Download failed.")
        return False

    # Find ZIP file
    zip_files = list(out_path.glob("*.zip"))

    if not zip_files:
        print("ZIP file not found.")
        return False

    zip_file = zip_files[0]

    # Create training folder
    training_path.mkdir(exist_ok=True)

    # Extract PSV files
    print("Extracting dataset...")

    with zipfile.ZipFile(zip_file, "r") as z:

        for file in z.namelist():

            if file.endswith(".psv"):
                filename = Path(file).name

                with z.open(file) as source:
                    with open(training_path / filename, "wb") as destination:
                        destination.write(source.read())

    # Delete ZIP file
    zip_file.unlink()

    # Verify
    psv_files = list(training_path.glob("*.psv"))

    print(f"Download complete!")
    print(f"Location: {training_path}")
    print(f"Patients: {len(psv_files)}")

    if len(psv_files) < 100:
        print(" Dataset seems incomplete.")
        return False

    return True


def main():

    parser = argparse.ArgumentParser(
        description="Download PhysioNet 2019 Sepsis Dataset"
    )

    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR
    )

    args = parser.parse_args()

    success = download_dataset(args.out_dir)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()