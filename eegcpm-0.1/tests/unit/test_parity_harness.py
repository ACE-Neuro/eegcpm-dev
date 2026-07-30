"""Tests for the parity harness (per spec §3.p + S10 + S11)."""

import os
from pathlib import Path

import numpy as np
import pytest

from eegcpm.core.config_hash import (
    GOLDEN_CONFIG_PATHS,
    check_golden_config_hashes,
    hash_config,
    load_golden_hashes,
)
from eegcpm.core.parity_harness import (
    GOLDEN_CONFIG_NAMES,
    PARITY_TOLERANCE_TABLE,
    THREAD_COUNT_PINS,
    apply_thread_pins,
    check_parity_tolerance,
    run_parity_harness,
)


# --------------------------------------------------------------- golden configs (S10)

def test_golden_config_names_5():
    """5 golden configs named explicitly (S10)."""
    assert len(GOLDEN_CONFIG_NAMES) == 5
    assert set(GOLDEN_CONFIG_NAMES) == {
        "preprocessing", "drowsiness", "specparam", "connectivity", "prediction",
    }


def test_golden_config_paths_keys_match_names():
    """GOLDEN_CONFIG_PATHS keys must match GOLDEN_CONFIG_NAMES."""
    assert set(GOLDEN_CONFIG_PATHS.keys()) == set(GOLDEN_CONFIG_NAMES)


# --------------------------------------------------------------- tolerance table (S11)

def test_tolerance_table_has_bitwise_claim_per_row():
    """Every row has a bitwise_claim field."""
    for row in PARITY_TOLERANCE_TABLE:
        assert "bitwise_claim" in row
        assert isinstance(row["bitwise_claim"], bool)


def test_tolerance_table_near_zero_rows_have_atol():
    """Rows whose quantities are near 0 (connectivity edges, ISC
    scores) MUST have atol specified (rtol is undefined near 0)."""
    near_zero = {"Connectivity NPZ (edge weights)", "ISC scores"}
    for row in PARITY_TOLERANCE_TABLE:
        if row["output"] in near_zero:
            assert row["atol"] is not None and row["atol"] > 0, (
                f"{row['output']!r} is near-zero and MUST have atol; "
                f"got atol={row['atol']}"
            )


def test_tolerance_table_bitwise_rows_have_no_tolerance():
    """Bitwise rows have rtol=None, atol=0."""
    for row in PARITY_TOLERANCE_TABLE:
        if row["bitwise_claim"]:
            assert row["rtol"] is None
            assert row["atol"] == 0.0


# --------------------------------------------------------------- check_parity_tolerance

def test_check_parity_tolerance_bitwise_match_passes():
    """Bitwise claim + bitwise equal -> PASS."""
    passed, msg = check_parity_tolerance(
        "Preprocessed FIF data array",
        local_value=np.array([1.0, 2.0, 3.0]),
        hpc_value=np.array([1.0, 2.0, 3.0]),
    )
    assert passed is True
    assert "bitwise equal" in msg


def test_check_parity_tolerance_bitwise_mismatch_records_finding():
    """Bitwise claim + bitwise mismatch -> FAIL with 'recorded FINDING'."""
    passed, msg = check_parity_tolerance(
        "Preprocessed FIF data array",
        local_value=np.array([1.0, 2.0, 3.0]),
        hpc_value=np.array([1.0, 2.0, 3.000001]),
    )
    assert passed is False
    assert "recorded FINDING" in msg
    # The message contains the literal phrase "not an auto re-label";
    # verify the meaning (the re-label is forbidden) by checking
    # the message forbids the auto-relabel behavior.
    assert "not an auto re-label" in msg


def test_check_parity_tolerance_within_rtol_atol_passes():
    """Numeric row within tolerance -> PASS."""
    passed, msg = check_parity_tolerance(
        "CPM out-of-fold r",
        local_value=np.array([0.10]),
        hpc_value=np.array([0.10 + 1e-6]),
    )
    assert passed is True


def test_check_parity_tolerance_near_zero_atol_required():
    """Connectivity NPZ near-zero -> atol 1e-6 catches the difference."""
    passed, msg = check_parity_tolerance(
        "Connectivity NPZ (edge weights)",
        local_value=np.array([0.0, 0.0, 0.0]),
        hpc_value=np.array([0.0, 5e-7, 0.0]),  # 5e-7 is within atol 1e-6
    )
    assert passed is True


def test_check_parity_tolerance_near_zero_fails_outside_atol():
    """Connectivity NPZ near-zero with diff > atol -> FAIL."""
    passed, msg = check_parity_tolerance(
        "Connectivity NPZ (edge weights)",
        local_value=np.array([0.0, 0.0, 0.0]),
        hpc_value=np.array([0.0, 1e-5, 0.0]),  # 1e-5 > atol 1e-6
    )
    assert passed is False


def test_check_parity_tolerance_unknown_output_fails():
    passed, msg = check_parity_tolerance(
        "unknown output",
        local_value=np.array([1.0]),
        hpc_value=np.array([1.0]),
    )
    assert passed is False
    assert "not in tolerance table" in msg


# --------------------------------------------------------------- thread pins (S11)

def test_thread_count_pins_complete():
    """All 5 thread-count env vars are pinned."""
    assert set(THREAD_COUNT_PINS.keys()) == {
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED",
    }
    for v in THREAD_COUNT_PINS.values():
        assert v == "1" or v == "0"


def test_apply_thread_pins_sets_env():
    """apply_thread_pins sets the env vars."""
    # Clean state
    for k in THREAD_COUNT_PINS:
        os.environ.pop(k, None)
    apply_thread_pins()
    for k, v in THREAD_COUNT_PINS.items():
        assert os.environ.get(k) == v, (
            f"{k} not set; got {os.environ.get(k)!r}, expected {v!r}"
        )


# --------------------------------------------------------------- harness launcher

def test_run_parity_harness_prints_5_golden_configs(tmp_path, capsys):
    """The harness launcher prints the 5 golden configs explicitly."""
    run_parity_harness(
        local_dir=tmp_path, hpc_dir=tmp_path,
        subjects=["S1", "S2", "S3"],
    )
    out = capsys.readouterr().out
    assert "Golden configs (5)" in out
    for stage in GOLDEN_CONFIG_NAMES:
        assert stage in out


def test_run_parity_harness_rejects_unknown_stage(tmp_path):
    """If config_paths includes an unknown stage, the harness raises."""
    bad_paths = {
        "unknown_stage": Path("/tmp/bad.yaml"),
        "preprocessing": Path("examples/configs/preprocessing/hbn_langer.yaml"),
    }
    with pytest.raises(AssertionError, match="Unknown stage"):
        run_parity_harness(
            local_dir=tmp_path, hpc_dir=tmp_path,
            subjects=["S1"],
            config_paths=bad_paths,
        )


# --------------------------------------------------------------- golden hash (S10)

def test_check_golden_config_hashes_passes_on_match():
    expected = {"preprocessing": "abc", "drowsiness": "def",
                "specparam": "ghi", "connectivity": "jkl",
                "prediction": "mno"}
    local = {k: {"config_hash": v} for k, v in expected.items()}
    hpc = {k: {"config_hash": v} for k, v in expected.items()}
    # No raise
    check_golden_config_hashes(local, hpc, expected_hashes=expected)


def test_check_golden_config_hashes_raises_on_local_mismatch():
    expected = {"preprocessing": "abc", "drowsiness": "def",
                "specparam": "ghi", "connectivity": "jkl",
                "prediction": "mno"}
    local = {k: {"config_hash": v} for k, v in expected.items()}
    local["preprocessing"]["config_hash"] = "WRONG"
    hpc = {k: {"config_hash": v} for k, v in expected.items()}
    with pytest.raises(ValueError, match="Local preprocessing"):
        check_golden_config_hashes(local, hpc, expected_hashes=expected)
