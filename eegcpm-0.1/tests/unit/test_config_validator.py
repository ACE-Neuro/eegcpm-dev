"""Tests for the config validator (per S05 + R1 in the spec)."""

import numpy as np
import pytest

from eegcpm.core.config_validator import (
    PRIMARY_VARIANTS,
    is_primary_config,
    validate_primary_config,
)
from eegcpm.core.covariate_lists import (
    POST_CLEANING_QC_SENSITIVITY,
    PRIMARY_UNPENALIZED_PROBED,
)


def _make_cfg(**overrides):
    """Build a minimal primary config namespace."""
    cfg = type("Cfg", (), {})()
    cfg.stage = "prediction"
    cfg.name = "cpm_d_factor"
    cfg.variant = "ridge_all_edges"
    cfg.is_primary = True
    cfg.unpenalized_columns = list(PRIMARY_UNPENALIZED_PROBED)
    cfg.covariate_adjustment_mode = "residualize_features_train_only"
    cfg.alpha_grid = list(np.logspace(-2, 6, 17))
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_is_primary_config_true_for_is_primary_true():
    cfg = _make_cfg()
    assert is_primary_config(cfg) is True


def test_is_primary_config_true_for_primary_variant():
    cfg = _make_cfg(is_primary=False)
    assert is_primary_config(cfg) is True


def test_is_primary_config_false_for_non_primary():
    cfg = _make_cfg(is_primary=False, variant="age_deviance_exploratory")
    assert is_primary_config(cfg) is False


def test_validate_primary_config_reachable():
    """R1: when is_primary=True and variant=ridge_all_edges (the
    actual primary config), the validator MUST raise on a forbidden
    column. This is the reachability check that was missing."""
    cfg = _make_cfg(unpenalized_columns=list(PRIMARY_UNPENALIZED_PROBED) + ["emg_proxy"])
    with pytest.raises(ValueError,
                        match="POST_CLEANING_QC_SENSITIVITY members"):
        validate_primary_config(cfg)


def test_validate_primary_config_skip_for_non_primary():
    """R1: secondary check — non-primary configs are skipped
    (validator returns without raising)."""
    cfg = _make_cfg(is_primary=False, variant="age_deviance_exploratory",
                    unpenalized_columns=["age_residual"])
    # No raise: validator returns early
    validate_primary_config(cfg)


def test_validate_primary_config_reachable_with_variant_only():
    """R1: even if is_primary is missing, variant=ridge_all_edges
    fires the gate (PRIMARY_VARIANTS redundancy)."""
    cfg = _make_cfg(is_primary=False, variant="ridge_all_edges",
                    unpenalized_columns=list(PRIMARY_UNPENALIZED_PROBED) + ["emg_proxy"])
    with pytest.raises(ValueError,
                        match="POST_CLEANING_QC_SENSITIVITY members"):
        validate_primary_config(cfg)


def test_validate_primary_config_alpha_grid_wrong_length():
    cfg = _make_cfg(alpha_grid=list(np.logspace(-2, 6, 16)))  # 16, not 17
    with pytest.raises(ValueError, match="expected 17"):
        validate_primary_config(cfg)


def test_validate_primary_config_alpha_grid_wrong_values():
    cfg = _make_cfg(alpha_grid=[1.0] * 17)  # 17 values but wrong
    with pytest.raises(ValueError, match="does not match np.logspace"):
        validate_primary_config(cfg)


def test_validate_primary_config_unknown_mode():
    cfg = _make_cfg(covariate_adjustment_mode="unknown_mode")
    with pytest.raises(ValueError, match="Unknown covariate_adjustment_mode"):
        validate_primary_config(cfg)


def test_validate_primary_config_in_model_unpenalized_requires_qr_math():
    cfg = _make_cfg(covariate_adjustment_mode="in_model_unpenalized")
    with pytest.raises(ValueError, match="qr_projection_math"):
        validate_primary_config(cfg)


def test_validate_primary_config_in_model_unpenalized_with_qr_math():
    cfg = _make_cfg(covariate_adjustment_mode="in_model_unpenalized",
                    qr_projection_math="QR on covs then penalized solve")
    # No raise
    validate_primary_config(cfg)


def test_validate_primary_config_missing_columns():
    """If a PRIMARY column is missing, the validator raises."""
    cols = list(PRIMARY_UNPENALIZED_PROBED)
    cols.remove("frac_channels_passing")
    cfg = _make_cfg(unpenalized_columns=cols)
    with pytest.raises(ValueError, match="unpenalized_columns mismatch"):
        validate_primary_config(cfg)


def test_validate_primary_config_post_cleaning_forbidden():
    """POST_CLEANING_QC_SENSITIVITY members are forbidden in primary."""
    cfg = _make_cfg(
        unpenalized_columns=list(PRIMARY_UNPENALIZED_PROBED) + ["emg_proxy"])
    with pytest.raises(ValueError,
                        match="POST_CLEANING_QC_SENSITIVITY members"):
        validate_primary_config(cfg)
