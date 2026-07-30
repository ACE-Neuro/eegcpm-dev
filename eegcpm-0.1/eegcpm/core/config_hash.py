"""
Config-hash machinery (per S10 in the spec).

The 5 golden configs are pinned at content-hash. Any change to a
golden config triggers the CI golden-config-hash test. The
parity-run test asserts that each stage's recorded `config_hash`
(in the STATE_SCHEMA) equals the corresponding GOLDEN_HASHES entry
on BOTH local and HPC.

This module exposes:
  - GOLDEN_CONFIG_PATHS: the 5 paths the parity run names explicitly
  - GOLDEN_HASHES: SHA-256 of each, computed at config-load time
  - hash_config(path) -> str: SHA-256 of the file contents
  - check_golden_config_hashes(local_dir, hpc_dir) -> None
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional


# The 5 golden configs (per S10; mirrored in §3.s). These are the
# ONLY configs the parity run is allowed to use.
GOLDEN_CONFIG_PATHS: Dict[str, str] = {
    "preprocessing": "examples/configs/preprocessing/hbn_langer.yaml",
    "drowsiness":    "examples/configs/features/drowsiness_metrics.yaml",
    "specparam":     "examples/configs/features/specparam_resting_ec.yaml",
    "connectivity":  "examples/configs/features/connectivity_resting_ec.yaml",
    "prediction":    "examples/configs/prediction/cpm_d_factor.yaml",
}


def hash_config(path: Path) -> str:
    """SHA-256 of the file contents (bytes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_golden_hashes(
    root: Optional[Path] = None,
) -> Dict[str, str]:
    """Compute the SHA-256 of each golden config relative to `root`
    (defaults to the repo root)."""
    root = Path(root) if root is not None else _repo_root()
    return {
        stage: hash_config(root / rel)
        for stage, rel in GOLDEN_CONFIG_PATHS.items()
    }


def _repo_root() -> Path:
    """Return the eegcpm-0.1 package root."""
    return Path(__file__).resolve().parent.parent.parent


def check_golden_config_hashes(
    local_state: Dict[str, Dict[str, str]],
    hpc_state: Dict[str, Dict[str, str]],
    expected_hashes: Optional[Dict[str, str]] = None,
) -> None:
    """S10: for every stage, the recorded config_hash on BOTH local
    and HPC must equal the corresponding GOLDEN_HASHES entry. This
    closes the live ENG-006 gap: the parity run is bound to the
    golden pipeline, not to a different one.
    """
    if expected_hashes is None:
        expected_hashes = load_golden_hashes()
    for stage, expected in expected_hashes.items():
        local_hash = local_state.get(stage, {}).get("config_hash")
        hpc_hash = hpc_state.get(stage, {}).get("config_hash")
        if local_hash != expected:
            raise ValueError(
                f"Local {stage} config_hash ({local_hash}) != golden "
                f"({expected}); parity run used a non-golden config."
            )
        if hpc_hash != expected:
            raise ValueError(
                f"HPC {stage} config_hash ({hpc_hash}) != golden "
                f"({expected}); parity run used a non-golden config."
            )
        if local_hash != hpc_hash:
            raise ValueError(
                f"{stage}: local ({local_hash}) != hpc ({hpc_hash})"
            )
