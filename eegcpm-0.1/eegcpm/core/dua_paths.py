"""
DUA-aware paths (per S03 + spec §3.l).

EEGCPMPaths is the centralized path manager. The DUA root
separation:
  - `derivatives_root`: /share/...  (open, aggregate outputs only)
  - `dua_root`: HPC home, mode 0700  (DUA-gated, per-subject artifacts)

`_assert_dua_root_isolated` is reordered (per S03):
  1. resolve (canonicalize symlinks)
  2. refuse if under /share (raises; does not return early)
  3. assert mode 0o700 UNCONDITIONALLY (reachable in both branches)
  4. os.chmod after mkdir (umask-safe)

`get_prediction_dir` is idempotent (per ENG-003 + R3-eng-003):
exist_ok=True with content validation against config_hash. The
R3-eng-003 resolution is: IDEMPOTENT WINS. If the existing dir's
config_hash matches the current run, return as-is; if it differs,
raise PermissionError.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Optional


class EEGCPMPaths:
    """Centralized path management with DUA-root separation.

    S03: DUA root mode-0700 check is REACHABLE in both branches
    (not dead code after a return). R3-eng-003: get_prediction_dir
    is idempotent with content validation.
    """

    def __init__(
        self,
        project_root: Path,
        eegcpm_root: Optional[Path] = None,
        dua_root: Optional[Path] = None,
        current_run_config_hash: Optional[str] = None,
    ):
        self.project_root = Path(project_root)
        self.eegcpm_root = Path(eegcpm_root) if eegcpm_root else self.project_root / "eegcpm"
        self.derivatives_root = self.project_root / "derivatives"
        # DUA root for per-subject artifacts; defaults to HPC home
        self.dua_root = Path(dua_root) if dua_root else Path.home() / "data_raw" / "phenotypic"
        self._current_run_config_hash = current_run_config_hash
        self._assert_dua_root_isolated()

    def _assert_dua_root_isolated(self) -> None:
        """S03: resolve -> refuse /share -> assert mode 0o700 unconditionally
        -> os.chmod after mkdir. The mode check is REACHABLE in both
        branches now (not dead code after a return)."""
        # S03 step 1: resolve (canonicalize symlinks)
        dua_resolved = self.dua_root.resolve()
        # S03 step 2: refuse if under /share (raises; does not return
        # early — both branches reach step 3)
        share_root = Path("/share").resolve()
        under_share = False
        try:
            dua_resolved.relative_to(share_root)
            under_share = True
        except ValueError:
            under_share = False
        if under_share:
            raise PermissionError(
                f"DUA root {dua_resolved} is under /share; refusing. "
                f"DUA-gated artifacts must live under HPC home with mode 0700."
            )
        # S03 step 3: assert mode 0o700 UNCONDITIONALLY (reachable
        # in both branches now)
        if not dua_resolved.exists():
            dua_resolved.mkdir(parents=True, mode=0o700)
            os.chmod(dua_resolved, 0o700)   # belt-and-braces
        mode = stat.S_IMODE(dua_resolved.stat().st_mode)
        if mode != 0o700:
            raise PermissionError(
                f"DUA root {dua_resolved} mode is {oct(mode)}, expected 0o700. "
                f"Run `chmod 700 {dua_resolved}` and retry."
            )

    def get_prediction_dir(self, model_name: str) -> Path:
        """Idempotent prediction-dir resolver. R3-eng-003: IDEMPOTENT
        WINS. If the directory exists, validate that its content
        matches the current run's identity (config_hash); if mismatch,
        raise; if empty or matching, return as-is. This makes the
        dispatcher's `--from <stage>` resume safe.
        """
        dua = self.dua_root / "prediction" / model_name
        dua.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(dua, 0o700)   # S03: explicit chmod (umask-safe)
        # Content validation: if non-empty, the manifest must match
        # the current run identity.
        manifest = dua / "manifest.yaml"
        if manifest.exists() and self._current_run_config_hash is not None:
            try:
                import yaml
                with open(manifest) as f:
                    existing = yaml.safe_load(f)
                existing_hash = (existing or {}).get("config_hash")
                if existing_hash and existing_hash != self._current_run_config_hash:
                    raise PermissionError(
                        f"Prediction dir {dua} contains artifacts from "
                        f"a different run (existing config_hash "
                        f"{existing_hash}, current "
                        f"{self._current_run_config_hash}). Refusing to "
                        f"overwrite; rename the existing dir or use a "
                        f"new model_name."
                    )
            except ImportError:
                # yaml not available; skip content validation
                pass
        return dua

    def get_aggregates_dir(self, model_name: str) -> Path:
        """Aggregate-only outputs -> /share."""
        share = self.derivatives_root / "aggregates" / model_name
        share.mkdir(parents=True, exist_ok=True)
        return share
