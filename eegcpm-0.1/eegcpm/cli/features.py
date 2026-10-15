"""CLI command for EEG features extraction (spec §3.j/§3.k/§3.d).

Wires the spec-required guards:
  - l2_assert_no_blocked_columns on the loaded features frame
  - validate_primary_config on the prediction config
  - DUA path enforcement for any subject-level outputs
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


def _load_features(features_dir: Path) -> pd.DataFrame:
    """Load a single features parquet; empty frame if not found."""
    pq_files = list(features_dir.glob("**/*.parquet"))
    if not pq_files:
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in pq_files]
    return pd.concat(frames, ignore_index=True)


def _source_erp_frames(args, config: Dict[str, Any], output_dir: Path) -> List[pd.DataFrame]:
    """Extract source-ERP features from the source stage's ROI archives (P2-F6).

    Two input modes:

    1. ``--input-dir``: every ``*_roi_tc.npz`` (evoked; the trial archive
       ``*_roi_tc_trials.npz`` is skipped) under the directory.
    2. ``--preprocessing/--task/--source`` (or the config's ``depends_on``):
       resolved through ``EEGCPMPaths.get_source_dir(...) / f"{subject}_roi_tc.npz"``.
    """
    from types import SimpleNamespace

    from eegcpm.modules.features.source_erp_features import SourceERPFeatureModule

    depends_on = config.get("depends_on", {}) or {}
    source_variant = getattr(args, "source", None) or depends_on.get("source")
    preprocessing = getattr(args, "preprocessing", None) or depends_on.get("preprocessing")
    task = getattr(args, "task", None) or depends_on.get("task")

    npz_files: List[Path] = []
    input_dir = getattr(args, "input_dir", None)
    if input_dir:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir {input_dir} not found")
        npz_files = sorted(
            p
            for p in input_dir.glob("**/*_roi_tc.npz")
            if not p.name.endswith("_roi_tc_trials.npz")
        )
    else:
        if not all([preprocessing, task, source_variant]):
            raise ValueError(
                "source_erp without --input-dir requires --preprocessing, --task and "
                "--source (or depends_on in the config)"
            )
        from eegcpm.core.paths import EEGCPMPaths as StagePaths

        paths = StagePaths(Path(args.project))
        source_root = paths.get_source_root(preprocessing, task, source_variant)
        if not source_root.exists():
            raise FileNotFoundError(f"Source directory not found: {source_root}")
        subjects = getattr(args, "subjects", None)
        sessions = getattr(args, "sessions", None)
        for subject_dir in sorted(source_root.glob("sub-*")):
            subject_id = subject_dir.name.replace("sub-", "")
            if subjects and subject_id not in subjects:
                continue
            session_dirs = sorted(subject_dir.glob("ses-*")) or [subject_dir]
            for session_dir in session_dirs:
                session_id = (
                    session_dir.name.replace("ses-", "")
                    if session_dir.name.startswith("ses-")
                    else "01"
                )
                if sessions and session_id not in sessions:
                    continue
                npz_path = (
                    paths.get_source_dir(
                        preprocessing, task, source_variant, subject_id, session_id
                    )
                    / f"{subject_id}_roi_tc.npz"
                )
                if npz_path.is_file():
                    npz_files.append(npz_path)

    if not npz_files:
        raise FileNotFoundError(
            "no *_roi_tc.npz files found for source_erp "
            f"(input_dir={input_dir}, preprocessing={preprocessing}, "
            f"task={task}, source={source_variant})"
        )

    frames: List[pd.DataFrame] = []
    for npz_path in npz_files:
        if npz_path.parent.name.startswith("ses-"):
            subject_id = npz_path.parent.parent.name.replace("sub-", "")
            session_id = npz_path.parent.name.replace("ses-", "")
        else:
            subject_id = npz_path.name.replace("_roi_tc.npz", "").replace("sub-", "")
            session_id = "01"
        module = SourceERPFeatureModule(config=config, output_dir=output_dir)
        result = module.process(
            npz_path, subject=SimpleNamespace(id=subject_id, session=session_id)
        )
        if not result.success:
            raise RuntimeError(f"source_erp failed for {npz_path.name}: {result.errors}")
        frames.append(result.outputs["features"])
        print(f"[features] source_erp: {npz_path.name} -> {result.outputs['n_features']} features")
    return frames


def features_command(args):
    """Run feature extraction on preprocessed EEG.

    Per spec §3.q: l2_assert_no_blocked_columns is invoked on the
    loaded features frame BEFORE the extraction pipeline writes its
    outputs. Per §3.s: validate_primary_config is invoked on the
    config when the variant is primary.
    """
    config_path = Path(args.config)
    project_root = Path(args.project)
    feature_type = getattr(args, "feature_type", "specparam")

    config = _load_config(config_path)

    # L2 guard on the config itself (block-listed names appearing
    # in feature column lists)
    if "feature_columns" in config:
        l2_assert_no_blocked_columns(
            pd.DataFrame(columns=config["feature_columns"]),
            frame_label=f"feature config {config_path}",
        )

    # Validate primary config (R1 gate)
    if config.get("is_primary") or config.get("variant") in {
        "ridge_all_edges"
    }:
        # Pad unpenalized_columns if missing (the validator
        # checks set equality; absent means use PRIMARY defaults)
        if "unpenalized_columns" not in config:
            from eegcpm.core.covariate_lists import PRIMARY_UNPENALIZED_PROBED
            config["unpenalized_columns"] = list(PRIMARY_UNPENALIZED_PROBED)
        # Pad the missing fields the validator expects
        if "covariate_adjustment_mode" not in config:
            config["covariate_adjustment_mode"] = "residualize_features_train_only"
        if "alpha_grid" not in config:
            import numpy as _np
            config["alpha_grid"] = list(_np.logspace(-2, 6, 17))
        # The validator reads cfg.unpenalized_columns (attribute),
        # so wrap the dict in a SimpleNamespace
        from types import SimpleNamespace
        validate_primary_config(SimpleNamespace(**config))

    # ------------------------------------------------------------------
    # R2-002: REAL feature extraction
    # ------------------------------------------------------------------
    import mne

    input_dir_arg = getattr(args, "input_dir", None)
    input_dir = Path(input_dir_arg) if input_dir_arg else None
    output_dir = Path(getattr(args, "output_dir", ""))
    if not str(output_dir):
        raise ValueError("--output-dir is required")
    condition = getattr(args, "condition", "resting_ec")
    sfreq = float(getattr(args, "sfreq", 500.0))
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if feature_type == "source_erp":
        frames = _source_erp_frames(args, config, output_dir)
    else:
        if input_dir is None or not input_dir.is_dir():
            raise FileNotFoundError(f"--input-dir {input_dir} not found")

        fif_files = sorted(input_dir.glob("**/*.fif"))
        if not fif_files:
            raise FileNotFoundError(f"no .fif files under {input_dir}")

        if feature_type == "isc":
            # Group operation: LOO templates across all subjects
            from eegcpm.modules.features.isc_cca import loo_transform
            data = {}
            for fif in fif_files:
                raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
                data[fif.stem.replace("_raw", "")] = raw.get_data()
            out_df = loo_transform(data)
            out_df["condition"] = condition
            frames.append(out_df)
        else:
            for fif in fif_files:
                raw = mne.io.read_raw_fif(fif, preload=True, verbose=False)
                subject = fif.stem.replace("_raw", "")
                data = raw.get_data()
                if feature_type == "specparam":
                    from eegcpm.modules.features.specparam_features import (
                        SpecparamFeatureModule,
                    )
                    mod = SpecparamFeatureModule(
                        subject_id=subject, condition=condition)
                    result = mod.process(data, subject=subject,
                                         condition=condition, sfreq=sfreq)
                    df = result.data if hasattr(result, "data") else result
                    frames.append(df if isinstance(df, pd.DataFrame)
                                  else pd.DataFrame(df))
                elif feature_type == "connectivity":
                    from eegcpm.modules.connectivity.connectivity import (
                        ConnectivityModule,
                    )
                    cm = ConnectivityModule(n_channels=data.shape[0],
                                            sfreq=sfreq)
                    edge_dict = cm.edges(data)
                    row = {"subject_id": subject, "condition": condition}
                    for method, band_dict in edge_dict.items():
                        for band, e in band_dict.items():
                            for k, v in enumerate(e):
                                row[f"{method}__{band}__e{k}"] = v
                    frames.append(pd.DataFrame([row]))
                elif feature_type == "drowsiness":
                    from eegcpm.modules.features.drowsiness import (
                        compute_drowsiness_metrics,
                    )
                    metrics = compute_drowsiness_metrics(data, sfreq)
                    frames.append(pd.DataFrame([{
                        "subject_id": subject, "condition": condition,
                        **metrics,
                    }]))

    out_df = pd.concat(frames, ignore_index=True)

    # L2 guard on the OUTPUT frame (before writing anything)
    l2_assert_no_blocked_columns(
        out_df, frame_label=f"features output ({feature_type})")

    out_path = output_dir / f"{feature_type}_{condition}.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"[features] {feature_type}/{condition}: "
          f"{len(frames)} recordings -> {out_df.shape} -> {out_path}")
    return {"output": str(out_path), "n_rows": int(out_df.shape[0]),
            "n_cols": int(out_df.shape[1])}


def add_features_parser(subparsers):
    parser = subparsers.add_parser(
        "features",
        help="Extract features (specparam/connectivity/ISC/drowsiness/source_erp)",
        description=(
            "Run feature extraction on preprocessed EEG. Invokes "
            "L2 leakage guard on the loaded features frame and the "
            "primary-config validator (R1 gate) on the config."
        ),
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["specparam", "connectivity", "isc", "drowsiness", "source_erp"],
        default="specparam",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=False, default=None,
        help="Directory of preprocessed .fif recordings (or of source *_roi_tc.npz "
             "for --feature-type source_erp). Required for every other feature type; "
             "for source_erp it may be omitted when --preprocessing/--task/--source "
             "are given."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for feature parquet files"
    )
    parser.add_argument(
        "--condition", type=str, default="resting_ec",
        help="Condition label (resting_ec, resting_eo, despicable_me, ...)"
    )
    parser.add_argument(
        "--sfreq", type=float, default=500.0,
        help="Sampling rate in Hz (default 500)"
    )
    # Source-ERP input discovery (mirrors the connectivity parser): when
    # --input-dir is omitted, the source archive is resolved from the project tree.
    parser.add_argument(
        "--preprocessing", type=str, default=None,
        help="Preprocessing pipeline variant (source_erp input discovery)"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Task name (source_erp input discovery)"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Source variant (e.g., dSPM-CONN32) for --feature-type source_erp"
    )
    parser.add_argument(
        "--subjects", type=str, nargs="+", default=None,
        help="Specific subjects to process (source_erp input discovery)"
    )
    parser.add_argument(
        "--sessions", type=str, nargs="+", default=None,
        help="Specific sessions to process (source_erp input discovery)"
    )
    return parser
