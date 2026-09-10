"""
Model Registry
==============
Versioned model snapshot store for the online incremental model.

Features:
  - Automatic periodic snapshots every N predictions
  - Manual snapshot on demand
  - Rollback to any previous version
  - Registry manifest stored as JSON (human-readable audit trail)
  - Optional MLflow integration (feature-flagged)
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)


def _load_cfg(cfg_path: str = "config/settings.yaml") -> dict:
    with open(cfg_path) as f:
        return yaml.safe_load(f)


class ModelRegistry:
    """
    Manages versioned snapshots of the online model.

    Usage:
        registry = ModelRegistry(online_model)
        # Auto-snapshot every N predictions:
        registry.maybe_snapshot()
        # Manual snapshot:
        registry.snapshot(reason="pre_drift_event")
        # Rollback:
        registry.rollback(version=3)
    """

    MANIFEST_FILE = "manifest.json"

    def __init__(
        self,
        online_model,
        cfg_path: str = "config/settings.yaml",
    ):
        self.cfg = _load_cfg(cfg_path)
        self.model = online_model
        self.snapshot_dir = Path(self.cfg["paths"]["model_snapshots"])
        self.snapshot_interval = self.cfg["online_model"]["snapshot_every_n_predictions"]
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._manifest: list[dict] = []
        self._load_manifest()
        self._last_snapshot_n = 0

    # ------------------------------------------------------------------
    def _load_manifest(self) -> None:
        manifest_path = self.snapshot_dir / self.MANIFEST_FILE
        if manifest_path.exists():
            with open(manifest_path) as f:
                self._manifest = json.load(f)
            log.info(f"[Registry] Loaded manifest with {len(self._manifest)} versions.")
        else:
            self._manifest = []

    def _save_manifest(self) -> None:
        manifest_path = self.snapshot_dir / self.MANIFEST_FILE
        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

    # ------------------------------------------------------------------
    def maybe_snapshot(self) -> bool:
        """
        Snapshot the model if N predictions have passed since last snapshot.
        Returns True if a snapshot was taken.
        """
        n = self.model.n_predictions
        if n - self._last_snapshot_n >= self.snapshot_interval:
            self.snapshot(reason="auto_periodic")
            return True
        return False

    def snapshot(self, reason: str = "manual") -> dict:
        """
        Save the current model state as a new versioned snapshot.
        Returns the snapshot manifest entry.
        """
        version = len(self._manifest) + 1
        ts = datetime.utcnow()
        filename = f"model_v{version:04d}_{ts.strftime('%Y%m%dT%H%M%S')}.pkl"
        path = self.snapshot_dir / filename

        self.model.version = version
        self.model.save_state(path)

        entry = {
            "version": version,
            "filename": filename,
            "path": str(path),
            "timestamp": ts.isoformat(),
            "reason": reason,
            "n_predictions": self.model.n_predictions,
            "n_updates": self.model.n_updates,
            "rolling_auroc": self.model.rolling_auroc,
            "model_type": self.model.model_type,
        }
        self._manifest.append(entry)
        self._save_manifest()
        self._last_snapshot_n = self.model.n_predictions

        log.info(
            f"[Registry] Snapshot v{version} saved → {filename} "
            f"(reason={reason}, auroc={self.model.rolling_auroc:.3f})"
        )
        return entry

    # ------------------------------------------------------------------
    def rollback(self, version: int | None = None) -> dict:
        """
        Rollback model to a previous version.
        If version is None, rolls back to the previous snapshot.
        Returns the manifest entry of the restored version.
        """
        if not self._manifest:
            raise RuntimeError("No snapshots available for rollback.")

        if version is None:
            entry = self._manifest[-2] if len(self._manifest) >= 2 else self._manifest[-1]
        else:
            entries = [e for e in self._manifest if e["version"] == version]
            if not entries:
                raise ValueError(f"Version {version} not found in registry.")
            entry = entries[0]

        self.model.load_state(entry["path"])
        log.warning(
            f"[Registry] ⏪  Rolled back to v{entry['version']} "
            f"from {entry['timestamp']} (reason was: {entry['reason']})"
        )

        # Record rollback event in manifest
        rollback_entry = {
            "version": len(self._manifest) + 1,
            "filename": entry["filename"],
            "path": entry["path"],
            "timestamp": datetime.utcnow().isoformat(),
            "reason": f"rollback_to_v{entry['version']}",
            "n_predictions": self.model.n_predictions,
            "n_updates": self.model.n_updates,
            "rolling_auroc": self.model.rolling_auroc,
            "model_type": self.model.model_type,
        }
        self._manifest.append(rollback_entry)
        self._save_manifest()
        return entry

    # ------------------------------------------------------------------
    def list_versions(self) -> list[dict]:
        return self._manifest.copy()

    def latest_version(self) -> dict | None:
        return self._manifest[-1] if self._manifest else None

    def version_count(self) -> int:
        return len(self._manifest)
