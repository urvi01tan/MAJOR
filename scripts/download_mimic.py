#!/usr/bin/env python3
"""
Download full MIMIC-IV dataset from PhysioNet using credentials.

Requirements:
  - PhysioNet account with MIMIC-IV data use agreement completed
  - pip install requests tqdm

Usage:
  python scripts/download_mimic.py --username YOUR_USERNAME --version 2.2
  python scripts/download_mimic.py --username YOUR_USERNAME --tables hosp icu
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# MIMIC-IV table manifest (PhysioNet paths)
# ---------------------------------------------------------------------------
MIMIC_VERSION = "2.2"
BASE_URL = "https://physionet.org/files/mimiciv/{version}/{module}/{table}.csv.gz"

# Minimum required tables for this system (download only what we need)
REQUIRED_TABLES = {
    "hosp": [
        "patients",
        "admissions",
        "diagnoses_icd",
        "labevents",
        "d_labitems",
        "microbiologyevents",
        "prescriptions",
    ],
    "icu": [
        "icustays",
        "chartevents",
        "d_items",
        "outputevents",
        "inputevents",
        "procedureevents",
    ],
}


def download_file(url: str, dest_path: Path, session: requests.Session) -> bool:
    """Stream-download a file with a progress bar. Returns True on success."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists")
        return True

    try:
        resp = session.get(url, stream=True, timeout=60)
        if resp.status_code == 403:
            print(f"  [ERROR] 403 Forbidden – check your credentials and data use agreement for {url}")
            return False
        if resp.status_code == 404:
            print(f"  [WARN]  404 Not Found – skipping {url}")
            return True  # Non-fatal; some tables may not exist in all versions
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True

    except requests.RequestException as exc:
        print(f"  [ERROR] Download failed for {url}: {exc}")
        return False


def build_session(username: str, password: str) -> requests.Session:
    session = requests.Session()
    session.auth = (username, password)
    return session


def verify_credentials(session: requests.Session) -> bool:
    """Quick credential check against a small known file."""
    test_url = f"https://physionet.org/files/mimiciv/{MIMIC_VERSION}/README.md"
    resp = session.get(test_url, timeout=30)
    return resp.status_code == 200


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full MIMIC-IV dataset from PhysioNet"
    )
    parser.add_argument("--username", required=True, help="PhysioNet username")
    parser.add_argument(
        "--version", default=MIMIC_VERSION, help=f"MIMIC-IV version (default: {MIMIC_VERSION})"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(REQUIRED_TABLES.keys()),
        default=list(REQUIRED_TABLES.keys()),
        help="Modules to download (default: all required)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/mimic",
        help="Output directory (default: data/mimic)",
    )
    args = parser.parse_args()

    password = getpass.getpass(f"PhysioNet password for '{args.username}': ")
    session = build_session(args.username, password)

    print("\n🔑  Verifying credentials …")
    if not verify_credentials(session):
        print(
            "❌  Credential check failed. Ensure:\n"
            "  1. Username and password are correct.\n"
            "  2. You have completed the MIMIC-IV data use agreement at physionet.org.\n"
            "  3. Your account has been approved for MIMIC-IV access."
        )
        sys.exit(1)
    print("✅  Credentials valid.\n")

    out_dir = Path(args.out_dir)
    failed = []

    for module in args.tables:
        tables = REQUIRED_TABLES.get(module, [])
        print(f"📦  Module: {module}  ({len(tables)} tables)")
        for table in tables:
            url = BASE_URL.format(version=args.version, module=module, table=table)
            dest = out_dir / module / f"{table}.csv.gz"
            ok = download_file(url, dest, session)
            if not ok:
                failed.append(url)
        print()

    if failed:
        print(f"⚠️  {len(failed)} file(s) failed to download:")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(
            f"✅  All tables downloaded to '{out_dir}'\n"
            f"    Next step: python scripts/train_baseline.py"
        )


if __name__ == "__main__":
    main()
