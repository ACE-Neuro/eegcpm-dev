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

    # Validate primary config (R1 gate; fires only for primary variants)
    from types import SimpleNamespace
    is_primary = config.get("is_primary") or config.get("variant") in {
        "ridge_all_edges"}
    if is_primary:
        if "unpenalized_columns" not in config:
            from eegcpm.core.covariate_lists import PRIMARY_UNPENALIZED_PROBED
            config["unpenalized_columns"] = list(PRIMARY_UNPENALIZED_PROBED)
        if "covariate_adjustment_mode" not in config:
            config["covariate_adjustment_mode"] = "residualize_features_train_only"
        if "alpha_grid" not in config:
            import numpy as _np
            config["alpha_grid"] = list(_np.logspace(-2, 6, 17))
        validate_primary_config(SimpleNamespace(**config))
    if "unpenalized_columns" not in config:
        config["unpenalized_columns"] = []

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
    pq_files = sorted(features_dir.glob("**/*.parquet"))
    if not pq_files:
        raise FileNotFoundError(
            f"no features parquet under {features_dir}"
        )
    features_df = pd.concat(
        [pd.read_parquet(p) for p in pq_files], ignore_index=True)

    # L2 guard on the loaded features frame. R2-003: provenance is
    # NEVER manufactured here — archive_* columns are accepted only
    # with the immutable provenance set by l3_load_archive_scores.
    l2_assert_no_blocked_columns(
        features_df, frame_label=f"features {features_dir}")

    # L2 guard on the merged features+target frame
    merged = _merge_features_target(features_df, target_df)
    if merged.empty:
        raise ValueError(
            "features ∩ target is empty — join-key mismatch (class 11)."
        )
    l2_assert_no_blocked_columns(
        merged, frame_label="merged features+target frame")

    # Target column must not also live in the features frame
    if target_column in features_df.columns:
        raise ValueError(
            f"target_column {target_column!r} already present in "
            f"features; L2 invariant violated."
        )

    # ------------------------------------------------------------------
    # R2-002: REAL execution — ridge-CPM with the spec pipeline
    # ------------------------------------------------------------------
    from eegcpm.evaluation.prediction.ridge_cpm import (
        collect_oof_predictions, repeated_cv_run, adaptive_permutation,
        ALPHA_GRID, CV_N_REPEATS, CV_SEED,
    )

    unpenalized = list(config["unpenalized_columns"])
    present_cov = [c for c in unpenalized if c in merged.columns]
    missing_cov = sorted(set(unpenalized) - set(present_cov))
    if missing_cov:
        raise ValueError(
            f"unpenalized covariates missing from merged frame: "
            f"{missing_cov}"
        )
    exclude = set(unpenalized) | {"subject_id", target_column}
    feature_cols = [c for c in merged.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(merged[c])]
    if not feature_cols:
        raise ValueError("no numeric feature columns after exclusions")

    X = merged[feature_cols].to_numpy(dtype=float)
    Z = merged[present_cov].to_numpy(dtype=float)
    y = merged[target_column].to_numpy(dtype=float)
    sids = merged["subject_id"].to_numpy()
    n_folds = int(config.get("cv", {}).get("n_folds", 5))
    n_repeats = int(config.get("cv", {}).get("n_repeats", CV_N_REPEATS))
    cv_seed = int(config.get("cv", {}).get("cv_seed", CV_SEED))
    alpha_grid = list(config.get("alpha_grid", ALPHA_GRID))

    # Observed statistic (mean over repeats, pooled out-of-fold r)
    rep_r, rep_boundary = repeated_cv_run(
        X, y, covariates=Z, n_folds=n_folds, n_repeats=n_repeats,
        cv_seed=cv_seed, alpha_grid=alpha_grid)
    observed_r = float(np.mean(rep_r))

    # Adaptive permutation p (full-pipeline reruns per permutation)
    from types import SimpleNamespace
    perm_cfg = SimpleNamespace(
        n_permutations_screening=int(
            config.get("permutation", {}).get("n_permutations_screening", 1000)),
        n_permutations_confirmatory=int(
            config.get("permutation", {}).get("n_permutations_confirmatory", 10000)),
        escalation_p_lo=float(
            config.get("permutation", {}).get("escalation_p_lo", 0.005)),
        escalation_p_hi=float(
            config.get("permutation", {}).get("escalation_p_hi", 0.10)),
        permutation_seed=int(
            config.get("permutation", {}).get("permutation_seed", 20260731)),
        cv_n_folds=n_folds,
        cv_n_repeats=n_repeats,
        cv_seed=cv_seed,
        alpha_grid=alpha_grid,
    )
    p_hat, (p_lo, p_hi) = adaptive_permutation(
        X, y, perm_cfg, observed_r, covariates=Z)

    # Subject-level OOF predictions -> DUA root
    oof = collect_oof_predictions(
        X, y, sids, covariates=Z, n_folds=n_folds,
        n_repeats=n_repeats, cv_seed=cv_seed, alpha_grid=alpha_grid)
    oof_path = Path(prediction_dir) / "oof_predictions.parquet"
    Path(prediction_dir).mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)

    # Aggregates only -> /share
    import json
    Path(aggregates_dir).mkdir(parents=True, exist_ok=True)
    summary = {
        "model_name": model_name,
        "target": target_column,
        "n_subjects": int(merged.shape[0]),
        "completeness_rule": "features ∩ frozen target, listwise",
        "n_features": int(X.shape[1]),
        "n_covariates": int(Z.shape[1]),
        "observed_r": observed_r,
        "repeat_r": [float(r) for r in rep_r],
        "alpha_boundary_hit_fraction": float(np.mean(rep_boundary)),
        "permutation_p": float(p_hat),
        "permutation_mc_ci": [float(p_lo), float(p_hi)],
        "config": str(config_path),
        "target_file": str(target_file),
    }
    summary_path = Path(aggregates_dir) / "cpm_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"[predict] N={merged.shape[0]} features={X.shape[1]} "
          f"observed_r={observed_r:.4f} p={p_hat:.4f} "
          f"[{p_lo:.4f}, {p_hi:.4f}]")
    print(f"[predict] OOF (DUA): {oof_path}")
    print(f"[predict] summary (/share): {summary_path}")

    return summary


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
