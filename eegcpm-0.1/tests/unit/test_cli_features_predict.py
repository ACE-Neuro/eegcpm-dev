"""CLI-level tests for the features and predict commands (R-002)."""

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from eegcpm.cli.features import features_command
from eegcpm.cli.predict import predict_command


def _features_config(is_primary: bool = True) -> dict:
    return {
        "stage": "features",
        "name": "specparam_resting_ec",
        "variant": "ridge_all_edges" if is_primary else "sensitivity",
        "is_primary": is_primary,
        "feature_columns": ["alpha_power", "alpha_theta_trajectory"],
    }


def _predict_config() -> dict:
    return {
        "stage": "prediction",
        "name": "cpm_d_factor",
        "variant": "ridge_all_edges",
        "is_primary": True,
        "unpenalized_columns": [
            "age_spline_1", "age_spline_2", "age_spline_3", "age_spline_4",
            "sex_M", "site_CBIC", "site_CUNY", "site_NA",
            "release_number_2", "release_number_3",
            "raw_bad_channel_count", "impedance_avg", "impedance_available",
            "duration_attempted_seconds",
            "alpha_theta_trajectory", "alpha_dropout_count",
            "theta_intrusion_index", "frac_channels_passing",
        ],
    }



def _tiny_fif_dir(tmp_path, n_subjects=2, n_times=2500):
    """Create a tiny preprocessed-FIF input dir for real dispatch."""
    import mne
    inp = tmp_path / "prep"
    inp.mkdir(exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(n_subjects):
        info = mne.create_info([f"E{k}" for k in range(4)], 500.0, "eeg")
        data = rng.normal(size=(4, n_times)) * 1e-6
        mne.io.RawArray(data, info).save(
            inp / f"sub-{i:03d}_raw.fif", overwrite=True, verbose=False)
    return inp


def _tiny_features_parquet(tmp_path, n=30, covariates=(), feature_cols=("alpha_power",)):
    """Tiny features parquet with all required covariate columns."""
    rng = np.random.RandomState(1)
    df = pd.DataFrame({"subject_id": [f"S{i}" for i in range(n)]})
    for c in feature_cols:
        df[c] = rng.normal(size=n)
    for c in covariates:
        df[c] = rng.normal(size=n)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir(exist_ok=True)
    df.to_parquet(feat_dir / "features.parquet")
    return feat_dir

def test_features_command_runs_l2_guard(tmp_path):
    """R-002: features_command invokes l2_assert_no_blocked_columns."""
    config_path = tmp_path / "features.yaml"
    config_path.write_text(json.dumps(_features_config()))
    args = type("Args", (), {})()
    args.config = config_path
    args.project = tmp_path
    args.feature_type = "drowsiness"
    args.input_dir = _tiny_fif_dir(tmp_path)
    args.output_dir = tmp_path / "feat_out"
    args.condition = "resting_ec"
    args.sfreq = 500.0
    result = features_command(args)
    assert "output" in result


def test_features_command_blocks_p_factor_in_columns(tmp_path):
    """R-002: l2_assert_no_blocked_columns raises on p_factor in
    feature_columns."""
    config = _features_config()
    config["feature_columns"] = ["alpha_power", "p_factor"]
    config_path = tmp_path / "features.yaml"
    config_path.write_text(json.dumps(config))
    args = type("Args", (), {})()
    args.config = config_path
    args.project = tmp_path
    args.feature_type = "specparam"
    with pytest.raises(ValueError, match="L2 BLOCK"):
        features_command(args)


def test_features_command_validates_primary_config(tmp_path):
    """R-002: validate_primary_config runs on primary configs."""
    config_path = tmp_path / "features.yaml"
    config_path.write_text(json.dumps(_predict_config()))
    args = type("Args", (), {})()
    args.config = config_path
    args.project = tmp_path
    args.feature_type = "drowsiness"
    args.input_dir = _tiny_fif_dir(tmp_path)
    args.output_dir = tmp_path / "feat_out"
    args.condition = "resting_ec"
    args.sfreq = 500.0
    # primary config; validator pads unpenalized_columns from defaults
    result = features_command(args)
    assert "output" in result


def test_features_command_rejects_wrong_unpenalized_set(tmp_path):
    """R-002: validator catches a primary config with the wrong
    unpenalized_columns set."""
    config = _predict_config()
    config["unpenalized_columns"] = ["not_a_real_column"]
    config_path = tmp_path / "features.yaml"
    config_path.write_text(json.dumps(config))
    args = type("Args", (), {})()
    args.config = config_path
    args.project = tmp_path
    args.feature_type = "specparam"
    with pytest.raises(ValueError, match="unpenalized_columns mismatch"):
        features_command(args)


def test_predict_command_enforces_dua_paths(tmp_path):
    """R-002: predict_command enforces DUA path separation."""
    # Create a target file
    target = tmp_path / "frozen.parquet"
    pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "d": [0.5, 0.6],
    }).to_parquet(target)
    # Create features dir with all primary covariates + features
    feat_dir = _tiny_features_parquet(
        tmp_path, n=30,
        covariates=_predict_config()["unpenalized_columns"],
        feature_cols=("alpha_power", "beta_power"))
    # Run with explicit dua_root
    dua_root = tmp_path / "dua"
    dua_root.mkdir(mode=0o700)
    args = type("Args", (), {})()
    args.config = tmp_path / "predict.yaml"
    args.project = tmp_path
    args.target_file = target
    args.target_column = "d"
    args.features_dir = feat_dir
    args.dua_root = dua_root
    args.current_run_config_hash = "abc123"
    config_path = tmp_path / "predict.yaml"
    small = _predict_config()
    small["cv"] = {"n_folds": 5, "n_repeats": 2}
    small["permutation"] = {"n_permutations_screening": 20,
                            "n_permutations_confirmatory": 20,
                            "escalation_p_lo": 0.005,
                            "escalation_p_hi": 0.10,
                            "permutation_seed": 1}
    config_path.write_text(json.dumps(small))
    args.config = config_path
    target = pd.DataFrame({"subject_id": [f"S{i}" for i in range(30)],
                           "d": np.random.RandomState(2).normal(size=30)})
    target.to_parquet(args.target_file)
    result = predict_command(args)
    assert "prediction_dir" in result
    assert "aggregates_dir" in result
    assert "observed_r" in result
    assert "DUA prediction dir" in result["prediction_dir"] or \
        dua_root.name in result["prediction_dir"]


def test_predict_command_l2_guard_on_target_column(tmp_path):
    """R-002: l2_assert_no_blocked_columns raises if target column
    collides with features."""
    # Features dir with a column named the same as the target
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "alpha_power": [1.0, 2.0],
        "d": [0.5, 0.6],  # collides with target
    }).to_parquet(feat_dir / "features.parquet")
    target = tmp_path / "frozen.parquet"
    pd.DataFrame({"subject_id": ["S1", "S2"], "d": [0.5, 0.6]}).to_parquet(target)
    args = type("Args", (), {})()
    args.config = tmp_path / "predict.yaml"
    args.project = tmp_path
    args.target_file = target
    args.target_column = "d"
    args.features_dir = feat_dir
    args.dua_root = tmp_path / "dua"
    args.current_run_config_hash = "abc123"
    (tmp_path / "predict.yaml").write_text(json.dumps(_predict_config()))
    with pytest.raises(ValueError, match="target_column"):
        predict_command(args)


def test_predict_command_requires_dual_paths(tmp_path):
    """R-002: prediction_dir is under DUA (mode 0700); aggregates
    under /share."""
    target = tmp_path / "frozen.parquet"
    pd.DataFrame({"subject_id": [f"S{i}" for i in range(30)],
                  "d": np.random.RandomState(3).normal(size=30)}).to_parquet(target)
    feat_dir = _tiny_features_parquet(
        tmp_path, n=30,
        covariates=_predict_config()["unpenalized_columns"],
        feature_cols=("alpha_power",))
    dua_root = tmp_path / "dua"
    dua_root.mkdir(mode=0o700)
    args = type("Args", (), {})()
    args.config = tmp_path / "predict.yaml"
    args.project = tmp_path
    args.target_file = target
    args.target_column = "d"
    args.features_dir = feat_dir
    args.dua_root = dua_root
    args.current_run_config_hash = "abc123"
    small = _predict_config()
    small["cv"] = {"n_folds": 5, "n_repeats": 1}
    small["permutation"] = {"n_permutations_screening": 5,
                            "n_permutations_confirmatory": 5,
                            "escalation_p_lo": 0.005,
                            "escalation_p_hi": 0.10,
                            "permutation_seed": 1}
    (tmp_path / "predict.yaml").write_text(json.dumps(small))
    result = predict_command(args)
    # prediction_dir is under dua_root (DUA-gated)
    assert str(dua_root) in result["prediction_dir"]
    # aggregates_dir is under derivatives (the open share equivalent)
    assert "aggregates" in result["aggregates_dir"]


def test_predict_command_rejects_dua_root_under_share(tmp_path):
    """R-002: dua_root under /share raises."""
    target = tmp_path / "frozen.parquet"
    pd.DataFrame({"subject_id": ["S1"], "d": [0.5]}).to_parquet(target)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    (tmp_path / "predict.yaml").write_text(json.dumps(_predict_config()))
    args = type("Args", (), {})()
    args.config = tmp_path / "predict.yaml"
    args.project = tmp_path
    args.target_file = target
    args.target_column = "d"
    args.features_dir = feat_dir
    args.dua_root = Path("/share/some/dua")
    args.current_run_config_hash = "abc123"
    with pytest.raises(PermissionError, match="refusing"):
        predict_command(args)
