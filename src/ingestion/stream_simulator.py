from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import yaml


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


VITAL_COLS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
]

LAB_COLS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets",
]

DEMOGRAPHIC_COLS = [
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
]

ALL_FEATURE_COLS = VITAL_COLS + LAB_COLS + DEMOGRAPHIC_COLS

_COL_SOURCE: dict[str, str] = {
    **{c: "vital" for c in VITAL_COLS},
    **{c: "lab" for c in LAB_COLS},
    **{c: "demographic" for c in DEMOGRAPHIC_COLS},
}
_ICU_ADMIT_BASE = datetime(2024, 1, 1, 0, 0, 0)


class PhysioNetDataLoader:
    def __init__(self, training_dir: str | Path):
        self.training_dir = Path(training_dir)
        if not self.training_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found: {self.training_dir}\n"
                "Run: python scripts/download_dataset.py"
            )

    def list_patient_files(
        self,
        n_patients: int | None = None,
        seed: int = 42,
    ) -> list[Path]:
        """Return list of PSV file paths, optionally sampled to n_patients."""
        files = sorted(self.training_dir.glob("*.psv"))
        if not files:
            raise FileNotFoundError(
                f"No .psv files found in {self.training_dir}\n"
                "Run: python scripts/download_dataset.py"
            )
        if n_patients is not None and n_patients < len(files):
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(files), size=n_patients, replace=False)
            files = [files[i] for i in sorted(indices)]
        print(f"[PhysioNetDataLoader] {len(files):,} patient files selected.")
        return files

    @staticmethod
    def load_patient(psv_path: Path) -> pd.DataFrame:
        """Load a single patient PSV file into a DataFrame."""
        df = pd.read_csv(psv_path, sep="|")
        # Ensure SepsisLabel exists
        if "SepsisLabel" not in df.columns:
            df["SepsisLabel"] = 0
        return df

    def load_all(
        self,
        n_patients: int | None = None,
        seed: int = 42,
    ) -> list[tuple[str, pd.DataFrame]]:
        """
        Load all selected patient files.
        Returns list of (patient_id, dataframe) tuples.
        """
        files = self.list_patient_files(n_patients, seed)
        patients = []
        for f in files:
            patient_id = f.stem  # e.g. "p000001"
            try:
                df = self.load_patient(f)
                patients.append((patient_id, df))
            except Exception as exc:
                print(f"[PhysioNetDataLoader] Warning: could not load {f.name}: {exc}")
        print(f"[PhysioNetDataLoader] Loaded {len(patients):,} patients.")
        return patients


# ---------------------------------------------------------------------------
# Main StreamSimulator
# ---------------------------------------------------------------------------

class StreamSimulator:
    """
    Replays PhysioNet CinC 2019 PSV data as a real-time event stream.

    Each hourly row is expanded into individual feature events so the
    downstream pipeline sees: one event per (patient, hour, feature).
    This matches the original MIMIC-style streaming API exactly.

    Usage:
        sim = StreamSimulator(cfg_path="config/settings.yaml")
        for event in sim.stream(n_patients=500):
            pipeline.process_event(event)
    """

    def __init__(self, cfg_path: str = "config/settings.yaml"):
        self.cfg = _load_cfg(cfg_path)
        self.training_dir = Path(self.cfg["paths"]["training_dir"])
        self.speed_multiplier = self.cfg["streaming"]["speed_multiplier"]
        self._patients: list[tuple[str, pd.DataFrame]] = []

    # ------------------------------------------------------------------
    def load_data(self, n_patients: int | None = None) -> "StreamSimulator":
        """Pre-load all patient PSV files."""
        loader = PhysioNetDataLoader(self.training_dir)
        self._patients = loader.load_all(n_patients=n_patients)
        return self

    # ------------------------------------------------------------------
    def _row_to_events(
        self,
        patient_id: str,
        row: pd.Series,
        hour_index: int,
    ) -> list[dict]:
        """
        Expand one hourly row into individual feature events.
        Returns one event dict per non-NaN feature column.
        """
        # Synthetic timestamp: ICU admit + hour offset
        ts = _ICU_ADMIT_BASE + timedelta(hours=hour_index)
        sepsis_label = int(row.get("SepsisLabel", 0))
        iculos = int(row.get("ICULOS", hour_index))

        events = []
        for col in ALL_FEATURE_COLS:
            if col not in row.index:
                continue
            val = row[col]
            if pd.isna(val):
                continue  # skip NaN — missing_handler will impute
            events.append({
                "patient_id": patient_id,
                "timestamp": ts,
                "source": _COL_SOURCE.get(col, "unknown"),
                "feature_name": col,
                "value": float(val),
                "unit": "",
                "sepsis_label": sepsis_label,
                "iculos": iculos,
            })
        return events

    # ------------------------------------------------------------------
    def stream(
        self,
        n_patients: int | None = None,
        realtime: bool = True,
    ) -> Generator[dict, None, None]:
        """
        Yield events one by one in chronological order across all patients.

        Events are interleaved by (patient_id, hour) — i.e. we process
        all patients' hour-1 rows, then all hour-2 rows, etc., producing
        a realistic multi-patient concurrent stream.

        Args:
            n_patients: Limit simulation to N patients.
            realtime:   If True, throttle to speed_multiplier × real-time.
        """
        if not self._patients:
            self.load_data(n_patients=n_patients)

        # Find maximum number of hourly rows across all patients
        max_hours = max(len(df) for _, df in self._patients)

        wall_start = time.time()
        sim_hour = 0  # simulated hours elapsed

        for hour_index in range(max_hours):
            for patient_id, df in self._patients:
                if hour_index >= len(df):
                    continue
                row = df.iloc[hour_index]
                for event in self._row_to_events(patient_id, row, hour_index):
                    if realtime:
                        # Each simulated hour → (3600 / speed_multiplier) wall seconds
                        target_wall = (hour_index * 3600.0) / self.speed_multiplier
                        wall_elapsed = time.time() - wall_start
                        sleep_time = target_wall - wall_elapsed
                        if sleep_time > 0.001:
                            time.sleep(min(sleep_time, 0.5))
                    yield event

            sim_hour = hour_index + 1

    # ------------------------------------------------------------------
    def stream_batches(
        self,
        batch_size: int | None = None,
        n_patients: int | None = None,
        realtime: bool = True,
    ) -> Generator[list[dict], None, None]:
        """Yield events in mini-batches for more efficient processing."""
        bs = batch_size or self.cfg["streaming"]["batch_size"]
        batch: list[dict] = []
        for event in self.stream(n_patients=n_patients, realtime=realtime):
            batch.append(event)
            if len(batch) >= bs:
                yield batch
                batch = []
        if batch:
            yield batch

    # ------------------------------------------------------------------
    def get_patient_labels(self) -> dict[str, int]:
        """
        Return {patient_id: max_sepsis_label} — 1 if patient ever had sepsis.
        Useful for offline baseline training.
        """
        return {
            pid: int(df["SepsisLabel"].max())
            for pid, df in self._patients
            if "SepsisLabel" in df.columns
        }
