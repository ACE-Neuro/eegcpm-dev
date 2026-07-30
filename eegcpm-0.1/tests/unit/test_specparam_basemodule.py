"""R-010: specparam BaseModule contract test.

The SpecparamFeatureModule MUST:
  - implement validate_input (rejects non-2D, too-short inputs)
  - implement process returning a ModuleResult (NOT a plain dict)
  - emit the exact SPECPARAM_COLUMNS schema in outputs
  - include subject_id, session, condition, bandpowers, hashes,
    timestamps in every row
"""

import time
from pathlib import Path

import mne
import numpy as np
import pytest

from eegcpm.modules.features.specparam_features import (
    PINNED_WELCH,
    REALIZED_SEGMENT_COUNTS,
    SPECPARAM_COLUMNS,
    SpecparamFeatureModule,
)


def _make_module(tmp_path):
    config = {
        "aperiodic_mode": "fixed",
        "subject_id": "sub-001",
        "session": "01",
        "condition": "resting_ec",
        "input_file_sha256": "abc123",
    }
    return SpecparamFeatureModule(config=config, output_dir=tmp_path)


def test_basemodule_validate_input_accepts_valid_2d(tmp_path):
    """validate_input accepts a valid (n_channels, n_times) array."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(20, 5000) * 1e-6
    assert module.validate_input(data) is True


def test_basemodule_validate_input_rejects_non_array(tmp_path):
    """validate_input rejects non-array input."""
    module = _make_module(tmp_path)
    assert module.validate_input([1, 2, 3]) is False
    assert module.validate_input("not an array") is False
    assert module.validate_input(None) is False


def test_basemodule_validate_input_rejects_wrong_ndim(tmp_path):
    """validate_input rejects 1D or 3D arrays."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    assert module.validate_input(rng.randn(5000)) is False   # 1D
    assert module.validate_input(rng.randn(5, 5, 5)) is False  # 3D


def test_basemodule_validate_input_rejects_too_short(tmp_path):
    """validate_input rejects arrays with < 2 channels or < 256 samples."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    assert module.validate_input(rng.randn(1, 1000)) is False   # 1 ch
    assert module.validate_input(rng.randn(5, 100)) is False    # too few samples


def test_process_returns_module_result(tmp_path):
    """process returns a ModuleResult (BaseModule contract)."""
    from eegcpm.pipeline.base import ModuleResult
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(20, 5000) * 1e-6
    result = module.process(data, sfreq=500.0)
    assert isinstance(result, ModuleResult)
    assert result.success is True
    assert result.module_name == "specparam_features"


def test_process_emits_exact_specparam_columns_schema(tmp_path):
    """Every emitted row has the SPECPARAM_COLUMNS keys."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(5, 5000) * 1e-6
    result = module.process(data, sfreq=500.0)
    features = result.outputs["features"]
    assert len(features) == 5
    for row in features:
        for col in SPECPARAM_COLUMNS:
            assert col in row, (
                f"Column {col!r} missing from emitted row; "
                f"got columns: {list(row.keys())}"
            )


def test_process_includes_subject_session_condition(tmp_path):
    """Every emitted row has subject_id, session, condition."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(3, 5000) * 1e-6
    result = module.process(data, subject="sub-002",
                              session="02", condition="resting_eo")
    for row in result.outputs["features"]:
        assert row["subject_id"] == "sub-002"
        assert row["session"] == "02"
        assert row["condition"] == "resting_eo"


def test_process_includes_bandpowers(tmp_path):
    """Every emitted row has bandpower_delta/theta/alpha/beta/gamma."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(5, 5000) * 1e-6
    result = module.process(data, sfreq=500.0)
    for row in result.outputs["features"]:
        for bp in ("bandpower_delta", "bandpower_theta", "bandpower_alpha",
                    "bandpower_beta", "bandpower_gamma"):
            assert bp in row, f"Missing bandpower: {bp}"
            assert isinstance(row[bp], float)


def test_process_includes_hashes_and_timestamps(tmp_path):
    """Every emitted row has input_file_sha256 and extraction_timestamp."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(3, 5000) * 1e-6
    t_before = time.time()
    result = module.process(data, sfreq=500.0)
    t_after = time.time()
    for row in result.outputs["features"]:
        assert row["input_file_sha256"] == "abc123"
        assert t_before <= row["extraction_timestamp"] <= t_after


def test_process_metadata_includes_pinned_welch(tmp_path):
    """ModuleResult.metadata exposes the pinned Welch params."""
    module = _make_module(tmp_path)
    rng = np.random.RandomState(0)
    data = rng.randn(3, 5000) * 1e-6
    result = module.process(data, sfreq=500.0)
    assert result.metadata["pinned_welch"] == dict(PINNED_WELCH)
    assert result.metadata["realized_segment_counts"] == dict(REALIZED_SEGMENT_COUNTS)
    assert result.metadata["aperiodic_mode"] == "fixed"
    assert result.metadata["n_rows"] == 3
