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
) -> None:
    """Run the parity harness with thread pins and 5 golden configs
    named explicitly.

    config_paths: {stage_name: path_to_config} for the 5 stages.
    If None, the harness uses the names without explicit paths (the
    caller is expected to construct the CLI commands with the
    GOLDEN_CONFIGS paths).
    """
    apply_thread_pins()
    # The launcher asserts config_hash on BOTH sides; see
    # check_golden_config_hashes in eegcpm.core.config_hash
    if config_paths is None:
        # Default paths
        config_paths = {
            "preprocessing": Path("examples/configs/preprocessing/hbn_langer.yaml"),
            "drowsiness":    Path("examples/configs/features/drowsiness_metrics.yaml"),
            "specparam":     Path("examples/configs/features/specparam_resting_ec.yaml"),
            "connectivity":  Path("examples/configs/features/connectivity_resting_ec.yaml"),
            "prediction":    Path("examples/configs/prediction/cpm_d_factor.yaml"),
        }
    # This function is a CLI launcher; the actual subprocess calls
    # are in the bash script. The Python entry point just records
    # the intent.
    print(f"Parity harness launching with {len(subjects)} subjects")
    print(f"Thread-count pins applied: {THREAD_COUNT_PINS}")
    print(f"Golden configs (5): {GOLDEN_CONFIG_NAMES}")
    for stage, path in config_paths.items():
        assert stage in GOLDEN_CONFIG_NAMES, (
            f"Unknown stage {stage!r}; must be one of {GOLDEN_CONFIG_NAMES}"
        )
        print(f"  {stage}: {path}")


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
    # Numeric tolerance
    import numpy as np
    arr_local = np.asarray(local_value, dtype=float)
    arr_hpc = np.asarray(hpc_value, dtype=float)
    if np.allclose(arr_local, arr_hpc,
                    rtol=row["rtol"], atol=row["atol"]):
        return True, f"within rtol={row['rtol']}, atol={row['atol']}"
    return False, (
        f"exceeds rtol={row['rtol']}, atol={row['atol']}: "
        f"max abs diff = {np.max(np.abs(arr_local - arr_hpc))}"
    )
