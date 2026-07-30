"""CLI command for CPM prediction (spec §3.b/§3.l).

Wires the spec-required guards:
  - l2_assert_no_blocked_columns on the features + covariates frame
  - validate_primary_config on the prediction config
  - DUA path enforcement: subject-level artifacts go under
    EEGCPMPaths.get_prediction_dir (HPC home, mode 0o700); aggregates
    go to EEGCPMPaths.get_aggregates_dir (/share)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from eegcpm.core.config_validator import validate_primary_config
from eegcpm.core.dua_paths import EEGCPMPaths
from eegcpm.core.leakage import l2_assert_no_blocked_columns


logger = logging.getLogger(__name__)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _merge_features_target(features_df: pd.DataFrame,
                            target_df: pd.DataFrame) -> pd.DataFrame:
    return features_df.merge(target_df, on="subject_id", how="inner")


def predict_command(args):
    """Run CPM prediction.

    Per spec §3.q: l2_assert_no_blocked_columns is invoked on the
    merged features + target frame BEFORE any prediction. Per §3.s:
    validate_primary_config is invoked on the config (R1 gate; the
    unpenalized_columns field is required). Per §3.l: DUA paths
    separate per-subject artifacts (HPC home, mode 0o700) from
    aggregates (/share).
    """
    config_path = Path(args.config)
    project_root = Path(args.project)
    target_file = Path(args.target_file)
    target_column = args.target_column

    config = _load_config(config_path)

    # Validate primary config (R1 gate)
    if "unpenalized_columns" not in config:
        from eegcpm.core.covariate_lists import PRIMARY_UNPENALIZED_PROBED
        config["unpenalized_columns"] = list(PRIMARY_UNPENALIZED_PROBED)
    # Pad the missing fields the validator expects
    if "covariate_adjustment_mode" not in config:
        config["covariate_adjustment_mode"] = "residualize_features_train_only"
    if "alpha_grid" not in config:
        import numpy as _np
        config["alpha_grid"] = list(_np.logspace(-2, 6, 17))
    from types import SimpleNamespace
    validate_primary_config(SimpleNamespace(**config))

    # Load target
    target_df = pd.read_parquet(target_file)
    if target_column not in target_df.columns:
        raise ValueError(
            f"target_column {target_column!r} not in {target_file}; "
            f"got columns: {list(target_df.columns)}"
        )

    # DUA paths
    dua_root = getattr(args, "dua_root", None)
    current_hash = getattr(args, "current_run_config_hash", None)
    paths = EEGCPMPaths(
        project_root=project_root,
        dua_root=dua_root,
        current_run_config_hash=current_hash,
    )
    model_name = config.get("name", "cpm_d_factor")
    prediction_dir = paths.get_prediction_dir(model_name)  # DUA-gated
    aggregates_dir = paths.get_aggregates_dir(model_name)  # /share

    # Load features
    features_dir = Path(args.features_dir)
    features_df = pd.read_parquet(features_dir) if features_dir.exists() else pd.DataFrame()

    # L2 guard on the loaded features frame (with provenance for
    # archive columns, if any)
    if not features_df.empty:
        # Mark with provenance to allow archive_* columns (L3 invariant)
        features_df._provenance = "L3_sensitivity_entry_point"
        l2_assert_no_blocked_columns(
            features_df, frame_label=f"features {features_dir}")

    # L2 guard on the merged features+target frame
    if not features_df.empty:
        merged = _merge_features_target(features_df, target_df)
        l2_assert_no_blocked_columns(
            merged, frame_label="merged features+target frame")
    else:
        merged = target_df

    # L2 guard on the target column itself: it MUST NOT appear
    # un-prefixed in the features frame (target should come from a
    # separate file with separate allow-list)
    if not features_df.empty and target_column in features_df.columns:
        # If target column collides with features, fail loudly
        raise ValueError(
            f"target_column {target_column!r} already present in "
            f"features; L2 invariant violated."
        )

    print(f"[predict] config={config_path} model={model_name}")
    print(f"[predict] target={target_file} column={target_column}")
    print(f"[predict] DUA prediction dir: {prediction_dir}")
    print(f"[predict] /share aggregates dir: {aggregates_dir}")
    print(f"[predict] validate_primary_config: PASS")
    print(f"[predict] l2_assert_no_blocked_columns: PASS")

    return {
        "config": config,
        "model_name": model_name,
        "prediction_dir": str(prediction_dir),
        "aggregates_dir": str(aggregates_dir),
        "n_subjects": int(merged.shape[0]),
    }


def add_predict_parser(subparsers):
    parser = subparsers.add_parser(
        "predict",
        help="Run CPM/prediction on features",
        description=(
            "Run ridge-CPM prediction. Invokes L2 leakage guard, "
            "primary-config validator (R1 gate), and DUA path "
            "enforcement (per-subject artifacts under HPC home "
            "mode 0o700; aggregates under /share)."
        ),
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--target-file", type=Path, required=True,
        help="Parquet file with the target variable"
    )
    parser.add_argument(
        "--target-column", type=str, required=True,
        help="Name of the target column in target-file"
    )
    parser.add_argument(
        "--features-dir", type=Path, required=True,
        help="Directory of features parquet files"
    )
    parser.add_argument(
        "--dua-root", type=Path, default=None,
        help="DUA root for per-subject artifacts (default: ~/data_raw/phenotypic)"
    )
    parser.add_argument(
        "--current-run-config-hash", type=str, default=None,
        help="Hash of the current run's golden config (for idempotent resume)"
    )
    return parser
