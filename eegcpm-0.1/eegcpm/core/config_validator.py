"""
Config validator (per S05 + R1 in the spec).

R1: gate on `is_primary: true` OR `variant in PRIMARY_VARIANTS`
(NOT `cfg.variant == "primary"` — that was the unreachable gate).

The primary config declares `variant: ridge_all_edges`, so the old
gate would have returned early for every primary config, making
the S05 reachability check pass for any input. The corrected gate
uses an explicit `is_primary: true` field, OR membership in
PRIMARY_VARIANTS.

We also enforce (per R1 follow-up):
  (a) unpenalized_columns set-equals PRIMARY_UNPENALIZED_PROBED
      (no rel1_*, no members of POST_CLEANING_QC_SENSITIVITY);
  (b) covariate_adjustment_mode is in {residualize_features_train_only,
      residualize_target_train_only};
  (c) in_model_unpenalized requires a `qr_projection_math:`
      config field naming the projection method;
  (d) alpha_grid has exactly 17 values and matches
      np.logspace(-2, 6, 17) to 1e-12.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from .covariate_lists import (
    POST_CLEANING_QC_SENSITIVITY,
    PRIMARY_UNPENALIZED_PROBED,
)


PRIMARY_VARIANTS: set = {"ridge_all_edges"}


def is_primary_config(cfg: Any) -> bool:
    """R1: a config is 'primary' if either is_primary is True OR
    variant is in PRIMARY_VARIANTS. We use BOTH for redundancy."""
    return (
        getattr(cfg, "is_primary", False) is True
        or getattr(cfg, "variant", None) in PRIMARY_VARIANTS
    )


def validate_primary_config(cfg: Any) -> None:
    """For any primary config, assert the four invariants in S05/R1.

    Non-primary configs are skipped (the validator returns without
    raising; this is the documented behavior of the gate).
    """
    if not is_primary_config(cfg):
        return

    # R1: check POST_CLEANING_QC_SENSITIVITY FIRST so the more
    # specific error fires when the same column is BOTH missing from
    # PRIMARY_UNPENALIZED_PROBED AND is a sensitivity member. This
    # gives the implementer the correct diagnostic.
    forbidden_present: List[str] = sorted(
        set(cfg.unpenalized_columns) & set(POST_CLEANING_QC_SENSITIVITY)
    )
    if forbidden_present:
        raise ValueError(
            f"POST_CLEANING_QC_SENSITIVITY members {forbidden_present} "
            f"are forbidden in primary config; they are sensitivity-only."
        )
    # (a) unpenalized_columns set-equals PRIMARY_UNPENALIZED_PROBED
    if set(cfg.unpenalized_columns) != set(PRIMARY_UNPENALIZED_PROBED):
        raise ValueError(
            f"unpenalized_columns mismatch.\n"
            f"  expected (PRIMARY + probed): "
            f"{sorted(PRIMARY_UNPENALIZED_PROBED)}\n"
            f"  got: {sorted(cfg.unpenalized_columns)}\n"
            f"  forbidden (POST_CLEANING_QC_SENSITIVITY): "
            f"{sorted(POST_CLEANING_QC_SENSITIVITY)}"
        )
    # (b) covariate_adjustment_mode
    allowed_modes = {
        "residualize_features_train_only",
        "residualize_target_train_only",
    }
    if cfg.covariate_adjustment_mode not in allowed_modes:
        if cfg.covariate_adjustment_mode == "in_model_unpenalized":
            if not hasattr(cfg, "qr_projection_math"):
                raise ValueError(
                    "in_model_unpenalized requires "
                    "`qr_projection_math:` config field."
                )
        else:
            raise ValueError(
                f"Unknown covariate_adjustment_mode: "
                f"{cfg.covariate_adjustment_mode!r}"
            )
    # (d) alpha_grid has 17 values and matches the rule
    expected_grid: List[float] = list(np.logspace(-2, 6, 17))
    if len(cfg.alpha_grid) != 17:
        raise ValueError(
            f"alpha_grid has {len(cfg.alpha_grid)} values; expected 17."
        )
    if not np.allclose(cfg.alpha_grid, expected_grid, atol=1e-12):
        raise ValueError(
            f"alpha_grid does not match np.logspace(-2, 6, 17); "
            f"got {cfg.alpha_grid[:3]}...; rule is binding."
        )
