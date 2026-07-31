"""
Parity harness launcher (per spec §3.p + S10 + S11).

The pre-registered tolerance table is THE binding one. The five
golden configs are named explicitly in the launcher. Parity tests
assert that each stage's recorded config_hash equals the
corresponding GOLDEN_CONFIGS entry on BOTH local and HPC.

Tolerance table (per row):
  - bitwise claim: PROVEN on pilot-5; bitwise failure = recorded
    FINDING, never auto re-label (S11)
  - rtol + atol: both pinned; atol mandatory for near-zero
    quantities (connectivity edges, ISC scores)

Thread-count pins (per S11): OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, NUMEXPR_NUM_THREADS=1,
PYTHONHASHSEED=0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# The pre-registered tolerance table (S11)
# Each row: (output, bitwise_claim, rtol, atol, justification)
# bitwise_claim=True means the row is CLAIMED bitwise; the pilot-5
# run must PROVE it bitwise-equal, not assert it. A bitwise failure
# is a RECORDED FINDING, not an auto re-label.
PARITY_TOLERANCE_TABLE: List[Dict[str, Any]] = [
    {
        "output": "Preprocessed FIF data array",
        "bitwise_claim": True,        # claimed; PROVEN on pilot-5
        "rtol": None,
        "atol": 0.0,
        "justification": (
            "MNE filter + IALM rPCA + ASR with pinned params; "
            "same MNE/numpy/SciPy build -> bitwise equal. "
            "Bitwise failure is a FINDING requiring explanation."
        ),
    },
    {
        "output": "specparam feature Parquet",
        "bitwise_claim": False,
        "rtol": 1e-4,
        "atol": 1e-8,
        "justification": (
            "Welch FFT and specparam scipy.optimize can converge to "
            "slightly different points across BLAS builds; rtol 1e-4 "
            "keeps CPM r impact < 0.001. atol 1e-8 for log10-power "
            "near 0."
        ),
    },
    {
        "output": "Connectivity NPZ (edge weights)",
        "bitwise_claim": False,
        "rtol": 1e-4,
        "atol": 1e-6,
        "justification": (
            "Hilbert transform FFT build divergence; atol mandatory "
            "because many wPLI/dwPLI values are near 0 (relative "
            "tolerance is undefined there)."
        ),
    },
    {
        "output": "ISC scores",
        "bitwise_claim": False,
        "rtol": 1e-4,
        "atol": 1e-6,
        "justification": (
            "Ledoit-Wolf shrinkage is deterministic on same data "
            "but block-size interpretation can vary across builds."
        ),
    },
    {
        "output": "CPM out-of-fold r",
        "bitwise_claim": False,
        "rtol": 1e-4,
        "atol": 1e-6,
        "justification": (
            "Ridge closed-form is deterministic; tolerance allows "
            "for floating-point non-associativity across BLAS builds."
        ),
    },
    {
        "output": "MNE log (filter, reference)",
        "bitwise_claim": True,        # claimed; PROVEN on pilot-5
        "rtol": None,
        "atol": 0.0,
        "justification": (
            "Filter + reference deterministic given same MNE build."
        ),
    },
    {
        "output": "lineage_test (fold assignments)",
        "bitwise_claim": True,
        "rtol": None,
        "atol": 0.0,
        "justification": (
            "Fold assignments are deterministic given seeds."
        ),
    },
]


# The 5 golden configs (S10) — referenced by name in the launcher
GOLDEN_CONFIG_NAMES: List[str] = [
    "preprocessing",
    "drowsiness",
    "specparam",
    "connectivity",
    "prediction",
]


# Thread-count pins (S11)
THREAD_COUNT_PINS: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def apply_thread_pins() -> None:
    """Set thread-count env vars for deterministic parity."""
    for k, v in THREAD_COUNT_PINS.items():
        os.environ[k] = v


def run_parity_harness(
    local_dir: Path,
    hpc_dir: Path,
    subjects: List[str],
    config_paths: Optional[Dict[str, Path]] = None,
    verify_golden_hashes: bool = True,
) -> "pd.DataFrame":
    """Run the parity harness: thread pins, golden-config hash
    verification (R2-006), then the executable comparison.

    Returns the verdict table from compare_parity_dirs. Raises on any
    golden-hash mismatch or missing artifact.
    """
    apply_thread_pins()
    if config_paths is None:
        from eegcpm.core.config_hash import GOLDEN_CONFIG_PATHS
        config_paths = {k: Path(v) for k, v in GOLDEN_CONFIG_PATHS.items()}
    for stage in config_paths:
        assert stage in GOLDEN_CONFIG_NAMES, (
            f"Unknown stage {stage!r}; must be one of {GOLDEN_CONFIG_NAMES}"
        )
    if verify_golden_hashes:
        from eegcpm.core.config_hash import (
            load_golden_hashes, check_golden_config_hashes,
        )
        hashes = load_golden_hashes()
        assert set(hashes) == set(GOLDEN_CONFIG_NAMES), (
            f"golden hash set mismatch: {sorted(hashes)}"
        )
        for stage, h in hashes.items():
            print(f"  golden {stage}: sha256={h[:16]}...")
        # R3-002: validate each side's RECORDED stage lineage against
        # the golden hashes — the outputs must have been produced by
        # the golden configs, verified by hash not inspection.
        for side, root in (("local", Path(local_dir)),
                           ("hpc", Path(hpc_dir))):
            lineage_file = root / "stage_config_hashes.json"
            if not lineage_file.exists():
                raise FileNotFoundError(
                    f"{side}: stage_config_hashes.json missing at "
                    f"{root} — cannot verify the outputs were produced "
                    f"with the golden configs (R3-002)."
                )
            import json
            recorded = json.loads(lineage_file.read_text())
            # Accept flat {stage: hash} or nested {stage: {config_hash: h}}
            normalized = {
                stage: (v if isinstance(v, dict) else {"config_hash": v})
                for stage, v in recorded.items()
            }
            check_golden_config_hashes(normalized, normalized,
                                       expected_hashes=hashes)
    return compare_parity_dirs(local_dir, hpc_dir, subjects=subjects)


REQUIRED_OUTPUTS: Dict[str, str] = {
    "Preprocessed FIF data array": "preprocessed_raw.fif",
    "specparam feature Parquet": "specparam_features.parquet",
    "Connectivity NPZ (edge weights)": "connectivity_edges.npz",
    "ISC scores": "isc_scores.parquet",
    "CPM out-of-fold r": "cpm_result.npz",
    "lineage_test (fold assignments)": "fold_assignments.npy",
}


def compare_parity_dirs(
    local_dir: Path,
    hpc_dir: Path,
    subjects: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Execute the parity comparison between a local and an HPC output
    directory (R-013/R2-006/R3-002: executable and COMPLETE).

    If `subjects` is given, the directory contents on BOTH sides must
    equal it exactly — a subject missing on both sides is NOT silently
    omitted (R3-002). Every subject must provide every required output
    on both sides. Returns a verdict table with one row per
    (subject, output) and a computed pass column — no hardcoded
    verdicts (Class 6).
    """
    import numpy as np
    import pandas as pd

    local_dir = Path(local_dir)
    hpc_dir = Path(hpc_dir)
    rows = []

    found = sorted(
        p.name for p in local_dir.iterdir()
        if p.is_dir() and (hpc_dir / p.name).is_dir()
    )
    if subjects is not None:
        requested = sorted(subjects)
        if found != requested:
            raise FileNotFoundError(
                f"subject set mismatch: requested {requested}, "
                f"present on both sides {found}. A parity gate that "
                f"omits missing subjects is not a gate (R3-002)."
            )
    if not found:
        raise FileNotFoundError(
            f"no common subject directories under {local_dir} and {hpc_dir}"
        )
    subjects = found

    # Completeness gate FIRST: every subject x required output x both sides
    missing = []
    for sid in subjects:
        for output_name, fname in REQUIRED_OUTPUTS.items():
            for side, d in (("local", local_dir / sid), ("hpc", hpc_dir / sid)):
                if not (d / fname).exists():
                    missing.append(f"{sid}/{fname} [{side}]")
    if missing:
        raise FileNotFoundError(
            f"parity gate incomplete: {len(missing)} required artifacts "
            f"missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    def _cmp(subject, output_name, lv, hv):
        passed, msg = check_parity_tolerance(output_name, lv, hv)
        rows.append({
            "subject": subject, "output": output_name,
            "pass": bool(passed), "detail": msg,
        })

    for sid in subjects:
        ldir, hdir = local_dir / sid, hpc_dir / sid

        import mne
        lraw = mne.io.read_raw_fif(ldir / "preprocessed_raw.fif",
                                   preload=True, verbose=False)
        hraw = mne.io.read_raw_fif(hdir / "preprocessed_raw.fif",
                                   preload=True, verbose=False)
        _cmp(sid, "Preprocessed FIF data array",
             lraw.get_data(), hraw.get_data())

        lp = pd.read_parquet(ldir / "specparam_features.parquet")
        hp = pd.read_parquet(hdir / "specparam_features.parquet")
        num_cols = lp.select_dtypes("number").columns
        _cmp(sid, "specparam feature Parquet",
             lp[num_cols].to_numpy(), hp[num_cols].to_numpy())

        le = np.load(ldir / "connectivity_edges.npz")["edges"]
        he = np.load(hdir / "connectivity_edges.npz")["edges"]
        _cmp(sid, "Connectivity NPZ (edge weights)", le, he)

        lp = pd.read_parquet(ldir / "isc_scores.parquet")
        hp = pd.read_parquet(hdir / "isc_scores.parquet")
        _cmp(sid, "ISC scores", lp["isc"].to_numpy(),
             hp["isc"].to_numpy())

        lr = np.load(ldir / "cpm_result.npz")["oof_r"]
        hr = np.load(hdir / "cpm_result.npz")["oof_r"]
        _cmp(sid, "CPM out-of-fold r", lr, hr)

        _cmp(sid, "lineage_test (fold assignments)",
             np.load(ldir / "fold_assignments.npy"),
             np.load(hdir / "fold_assignments.npy"))

    verdict = pd.DataFrame(rows)
    verdict["pass"] = verdict["pass"].astype(bool)
    return verdict


def check_parity_tolerance(
    output_name: str,
    local_value: Any, hpc_value: Any,
) -> Tuple[bool, str]:
    """Check a single (output, value) pair against the pre-registered
    tolerance table.

    Returns (passed, message). If bitwise_claim is True and the
    values are bitwise-equal, returns (True, "bitwise equal"). If
    bitwise_claim is True and the values differ, returns (False,
    "bitwise failure — recorded FINDING"). For non-bitwise rows,
    returns (True, "within rtol/atol") or (False, "exceeds tolerance").
    """
    row = next(
        (r for r in PARITY_TOLERANCE_TABLE if r["output"] == output_name),
        None,
    )
    if row is None:
        return False, f"Output {output_name!r} not in tolerance table"
    if row["bitwise_claim"]:
        import numpy as np
        arr_local = np.asarray(local_value)
        arr_hpc = np.asarray(hpc_value)
        if arr_local.shape == arr_hpc.shape and np.array_equal(
                arr_local, arr_hpc):
            return True, "bitwise equal (PROVEN on pilot-5)"
        return False, (
            f"bitwise failure on {output_name!r}; this is a recorded "
            f"FINDING, not an auto re-label. Investigate thread count, "
            f"BLAS build, FFT library, MNE version mismatch."
        )
    # Numeric tolerance (equal_nan: QC-flagged dead channels carry NaN
    # on both sides; NaN==NaN is a match, not a failure)
    import numpy as np
    arr_local = np.asarray(local_value, dtype=float)
    arr_hpc = np.asarray(hpc_value, dtype=float)
    if np.allclose(arr_local, arr_hpc,
                    rtol=row["rtol"], atol=row["atol"], equal_nan=True):
        return True, f"within rtol={row['rtol']}, atol={row['atol']}"
    return False, (
        f"exceeds rtol={row['rtol']}, atol={row['atol']}: "
        f"max abs diff = {np.nanmax(np.abs(arr_local - arr_hpc))}"
    )
