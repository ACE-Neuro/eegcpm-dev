"""Prediction - predict questionnaire scores from EEG connectivity features."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eegcpm.core.paths import EEGCPMPaths
from eegcpm.evaluation.prediction.workflow import (
    build_connectivity_feature_table,
    cross_validate_regression,
    load_behavior_table,
    prepare_target_table,
    save_prediction_outputs,
    scan_connectivity_options,
)
from eegcpm.ui.session_persistence import restore_project_from_storage


MODEL_LABELS = {
    "spls_lite": "sPLS-lite (feature selection + PLS)",
    "pls": "PLS regression",
    "elasticnet": "ElasticNet",
    "svr": "SVR (linear)",
    "xgboost": "XGBoost regressor (optional package)",
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
}


def _default_project_root() -> Path | None:
    restore_project_from_storage()
    if "eegcpm_root" in st.session_state:
        return Path(st.session_state.eegcpm_root).parent
    return None


def _read_behavior_input(uploaded_file, behavior_path: str) -> pd.DataFrame | None:
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(uploaded_file)
        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(uploaded_file, sep="\t")
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(uploaded_file)
        st.error(f"Unsupported uploaded file type: {suffix}")
        return None
    if behavior_path:
        path = Path(behavior_path).expanduser()
        if not path.exists():
            st.warning(
                "Behavior file is not accessible from the machine running Streamlit: "
                f"`{path}`. If this is an HPC path, either run Streamlit on HPC, "
                "upload the file here for a local run, or keep the path for generated HPC scripts."
            )
            return None
        return load_behavior_table(path)
    return None


def _infer_column(columns: list[str], candidates: list[str]) -> int:
    lowered = {col.lower(): idx for idx, col in enumerate(columns)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return 0


def _make_local_script(config: dict) -> str:
    return f"""#!/bin/bash
# EEGCPM prediction run
# Save this script next to your project or run it from any shell with EEGCPM installed.

python - <<'PY'
from pathlib import Path
import json
from eegcpm.evaluation.prediction.workflow import (
    load_behavior_table, prepare_target_table, scan_connectivity_options,
    build_connectivity_feature_table, cross_validate_regression, save_prediction_outputs,
)

config = json.loads(r'''{json.dumps(config, indent=2)}''')
project_root = Path(config["project_root"])
behavior = load_behavior_table(Path(config["behavior_path"]))
target = prepare_target_table(
    behavior,
    subject_column=config["subject_column"],
    target_column=config.get("target_column"),
    build_columns=config.get("build_columns"),
    target_name=config["target_name"],
)
features = build_connectivity_feature_table(
    project_root=project_root,
    pipeline=config["pipeline"],
    subjects=config["subjects"],
    tasks=config["tasks"],
    variants=config["variants"],
    conditions=config["conditions"],
    windows=config["windows"],
    methods=config["methods"],
    bands=config["bands"],
    statistics=config["statistics"],
    clip_negative_dwpli=config["clip_negative_dwpli"],
)
merged = target.merge(features, on="subject", how="inner").set_index("subject")
X = merged.drop(columns=[config["target_name"]])
y = merged[config["target_name"]]
metrics, predictions = cross_validate_regression(
    X, y, models=config["models"], cv_method=config["cv_method"],
    n_splits=config["n_splits"], n_repeats=config["n_repeats"],
    random_state=config["random_state"],
)
run_dir = save_prediction_outputs(
    project_root / "derivatives" / "prediction" / config["target_name"],
    config, features, target, metrics, predictions,
)
print(metrics)
print(f"Saved prediction run: {{run_dir}}")
PY
"""


def _make_slurm_script(config: dict, hpc: dict) -> str:
    hpc_project = hpc.get("project_root") or config["project_root"]
    eegcpm_root = hpc.get("eegcpm_root") or "/path/to/eegcpm-0.1"
    venv = hpc.get("venv") or f"{eegcpm_root}/venv/bin/activate"
    email = hpc.get("email", "")
    email_line = f"#SBATCH --mail-user={email}" if email else "# #SBATCH --mail-user=your_email@eduhk.hk"
    slurm_config = dict(config)
    slurm_config["project_root"] = hpc_project
    if slurm_config.get("hpc_behavior_path"):
        slurm_config["behavior_path"] = slurm_config["hpc_behavior_path"]
    return f"""#!/bin/bash
#SBATCH --job-name=eegcpm_predict
#SBATCH --partition={hpc.get('partition', 'shared_cpu')}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={hpc.get('cpus', 4)}
#SBATCH --mem={hpc.get('mem', '16G')}
#SBATCH --time={hpc.get('time', '02:00:00')}
#SBATCH --output={hpc_project}/logs/prediction_%A.out
#SBATCH --error={hpc_project}/logs/prediction_%A.err
#SBATCH --mail-type=END,FAIL
{email_line}

set -e
mkdir -p "{hpc_project}/logs"
source "{venv}"
cd "{eegcpm_root}"

{_make_local_script(slurm_config)}
"""


def main():
    st.set_page_config(page_title="Prediction - EEGCPM", page_icon="🎯", layout="wide")
    st.title("🎯 Prediction: ASRS From EEG Connectivity")
    st.markdown(
        "Predict continuous questionnaire scores, such as ASRS-Inattention and ASRS full "
        "score, from source-level EEG connectivity matrices."
    )

    project_root = _default_project_root()
    if project_root is None:
        st.warning("No project configured. Please select a project on the home page first.")
        return

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.info(f"Project root: `{project_root}`")

    paths = EEGCPMPaths(project_root)
    connectivity_root = paths.derivatives_root / "connectivity"
    if not connectivity_root.exists():
        st.info("No connectivity results found. Run connectivity first.")
        return

    pipelines = sorted([p.name for p in connectivity_root.iterdir() if p.is_dir()])
    if not pipelines:
        st.info("No connectivity pipelines found.")
        return
    pipeline_index = pipelines.index("optimized") if "optimized" in pipelines else 0
    with top_right:
        pipeline = st.selectbox("Preprocessing pipeline", pipelines, index=pipeline_index)

    options = scan_connectivity_options(project_root, pipeline)
    if not options["tasks"]:
        st.info(f"No connectivity files found for pipeline '{pipeline}'.")
        return

    behavior_loaded = False
    subject_column = "subject"
    target_column = "asrs_inattention"
    build_columns: list[str] | None = None
    target_name = "asrs_inattention"
    target_table = pd.DataFrame(columns=["subject", target_name])

    st.markdown("### Configure Prediction")
    target_tab, feature_tab, model_tab = st.tabs([
        "1. Target Data", "2. Connectivity Features", "3. Models & Validation"
    ])

    with target_tab:
        st.header("Behavioral Target")
        st.caption(
            "Use a cleaned CSV if possible. The file path must be accessible from the "
            "machine running Streamlit. For HPC-only paths, use script generation or run Streamlit on HPC."
        )
        uploaded = st.file_uploader("Upload behavior CSV/TSV/Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])
        default_behavior_path = paths.eegcpm_root / "behavior" / "asrs_scores.csv"
        behavior_path = st.text_input(
            "Or app-accessible behavior file path",
            value=st.session_state.get(
                "prediction_behavior_path",
                str(default_behavior_path) if default_behavior_path.exists() else "",
            ),
            placeholder="/path/to/behavior_scores.csv",
        )
        if behavior_path:
            st.session_state["prediction_behavior_path"] = behavior_path

        hpc_behavior_path = st.text_input(
            "HPC behavior file path for generated SLURM scripts",
            value=st.session_state.get("prediction_hpc_behavior_path", behavior_path),
            placeholder="/path/to/behavior_scores.csv",
            help="Used only in generated HPC scripts. It does not need to exist on your local computer.",
        )
        if hpc_behavior_path:
            st.session_state["prediction_hpc_behavior_path"] = hpc_behavior_path

        behavior = _read_behavior_input(uploaded, behavior_path)
        if behavior is None:
            st.info(
                "Behavior data is not loaded yet. You can still configure features, models, "
                "and HPC scripts below. For local runs, upload the file or enter a path accessible "
                "to this Streamlit session."
            )
            st.markdown("**Manual column names for generated scripts**")
            subject_column = st.text_input("Subject ID column name", value="subject")
            target_column = st.text_input("Target score column name", value="asrs_inattention")
            target_name = st.text_input(
                "Output target name", value="asrs_inattention"
            )
        else:
            behavior_loaded = True
            st.dataframe(behavior.head(10), width="stretch")
            columns = list(behavior.columns)
            subject_idx = _infer_column(columns, ["subject", "subject_id", "participant_id", "Q13"])
            subject_column = st.selectbox("Subject ID column", columns, index=subject_idx)

            target_mode = st.radio(
                "Target mode",
                ["Use existing score column", "Build sum score from selected columns"],
                horizontal=True,
            )
            target_column = None
            build_columns = None
            if target_mode == "Use existing score column":
                target_idx = _infer_column(
                    columns,
                    ["ASRS Part A", "ASRA Part A", "asrs_inattention", "ASRS Total Score"],
                )
                target_column = st.selectbox("Target score column", columns, index=target_idx)
                target_name = st.text_input(
                    "Output target name",
                    value="asrs_inattention" if "part a" in target_column.lower() else "asrs_full",
                )
            else:
                default_items = [col for col in columns if col.startswith("ASRS")][:9]
                build_columns = st.multiselect(
                    "Columns to sum", columns, default=default_items
                )
                target_name = st.text_input("Output target name", value="asrs_inattention")

            target_table = prepare_target_table(
                behavior,
                subject_column=subject_column,
                target_column=target_column,
                build_columns=build_columns,
                target_name=target_name,
            )
            st.success(f"Loaded {len(target_table)} subjects with valid target values.")
            st.dataframe(target_table.head(10), width="stretch")

    with feature_tab:
        st.header("Connectivity Feature Selection")
        col1, col2 = st.columns(2)
        with col1:
            selected_tasks = st.multiselect("Tasks", options["tasks"], default=options["tasks"][:1])
            selected_variants = st.multiselect(
                "Source variants",
                options["variants"],
                default=[v for v in options["variants"] if "prestim500" in v][:1] or options["variants"][:1],
            )
            selected_methods = st.multiselect(
                "Connectivity methods", options["methods"], default=[m for m in ["wpli", "dwpli"] if m in options["methods"]]
            )
            selected_bands = st.multiselect(
                "Frequency bands", options["bands"], default=[b for b in ["theta"] if b in options["bands"]] or options["bands"][:1]
            )
        with col2:
            selected_windows = st.multiselect(
                "Time windows", options["windows"], default=[w for w in ["baseline"] if w in options["windows"]] or options["windows"][:1]
            )
            selected_conditions = st.multiselect("Conditions", options["conditions"], default=[])
            selected_statistics = st.multiselect(
                "Statistics", options["statistics"], default=[s for s in ["mean"] if s in options["statistics"]] or options["statistics"][:1]
            )
            clip_negative_dwpli = st.checkbox("Clip negative dwPLI values to 0", value=True)

        if behavior_loaded:
            available_subjects = sorted(set(options["subjects"]) & set(target_table["subject"]))
        else:
            available_subjects = options["subjects"]
        subject_mode = st.radio("Subjects", ["All matched subjects", "Select subjects"], horizontal=True)
        if subject_mode == "Select subjects":
            selected_subjects = st.multiselect("Selected subjects", available_subjects, default=available_subjects[:20])
        else:
            selected_subjects = available_subjects

        metric_label = "Matched subjects with connectivity + target" if behavior_loaded else "Connectivity subjects"
        st.metric(metric_label, len(selected_subjects))
        if not behavior_loaded:
            st.warning("Subject matching with ASRS will be checked after behavior data is loaded.")
        elif not selected_subjects:
            st.error("No matched subjects found. Check subject ID formatting and selected tasks/variants.")

    with model_tab:
        st.header("Models & Validation")
        default_models = ["spls_lite", "elasticnet", "svr"]
        selected_models = st.multiselect(
            "Regression models",
            list(MODEL_LABELS.keys()),
            default=[m for m in default_models if m in MODEL_LABELS],
            format_func=lambda x: MODEL_LABELS[x],
        )
        st.caption("sPLS-lite is a Python approximation: univariate feature selection followed by PLS. True mixOmics sPLS can be added later through R integration.")

        cv_method = st.selectbox(
            "Validation",
            ["repeated_kfold", "kfold", "leave_one_out"],
            format_func=lambda x: {
                "repeated_kfold": "Repeated k-fold CV",
                "kfold": "Single k-fold CV",
                "leave_one_out": "Leave-one-out CV",
            }[x],
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            n_splits = st.number_input("Folds", min_value=2, max_value=20, value=5)
        with col2:
            n_repeats = st.number_input("Repeats", min_value=1, max_value=100, value=20)
        with col3:
            random_state = st.number_input("Random seed", min_value=0, value=42)
        st.info("Primary metric: Pearson's r. Run permutation testing later for promising models.")

    config = {
        "project_root": str(project_root),
        "behavior_path": behavior_path,
        "hpc_behavior_path": hpc_behavior_path,
        "pipeline": pipeline,
        "subject_column": subject_column,
        "target_column": target_column,
        "build_columns": build_columns,
        "target_name": target_name,
        "subjects": selected_subjects,
        "tasks": selected_tasks,
        "variants": selected_variants,
        "conditions": selected_conditions,
        "windows": selected_windows,
        "methods": selected_methods,
        "bands": selected_bands,
        "statistics": selected_statistics,
        "clip_negative_dwpli": clip_negative_dwpli,
        "models": selected_models,
        "cv_method": cv_method,
        "n_splits": int(n_splits),
        "n_repeats": int(n_repeats),
        "random_state": int(random_state),
    }

    st.markdown("---")
    st.header("📝 Generate Scripts")
    st.markdown(f"""
    **Configuration Summary**:
    - Pipeline: `{pipeline}`
    - Target: `{target_name}`
    - Subjects: {len(selected_subjects)}
    - Tasks: `{', '.join(selected_tasks) if selected_tasks else 'none'}`
    - Variants: `{', '.join(selected_variants) if selected_variants else 'none'}`
    - Models: `{', '.join(selected_models) if selected_models else 'none'}`
    - Validation: `{cv_method}`
    """)

    script_tab1, script_tab2 = st.tabs(["🖥️ Local Script", "🏛️ HPC/SLURM Script"])

    with script_tab1:
        st.markdown("### Bash script for local execution")
        if not behavior_path:
            st.warning("Local script generation requires a behavior file path accessible to the local runtime.")

        local_script = _make_local_script(config)
        st.code(local_script, language="bash")

        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="⬇️ Download",
                data=local_script,
                file_name=f"prediction_{target_name}.sh",
                mime="text/plain",
                width="stretch",
            )

        st.markdown(f"""
        **Usage:**
        ```bash
        chmod +x prediction_{target_name}.sh
        ./prediction_{target_name}.sh
        ```

        **Notes:**
        - Use this when Streamlit and the data paths are on the same machine.
        - For local Mac runs, upload/copy the behavior CSV locally or use an accessible local path.
        - Outputs are saved to `derivatives/prediction/{target_name}/`.
        """)

        st.markdown("---")
        st.subheader("Optional: Run interactively now")
        col1, col2 = st.columns(2)
        with col1:
            run_local = st.button("Run locally now", type="primary")
        with col2:
            build_features_only = st.button("Preview feature table")

        if (run_local or build_features_only) and not behavior_loaded:
            st.error("Load behavior data before previewing features or running local prediction.")
        elif run_local or build_features_only:
            with st.spinner("Building feature table..."):
                feature_table = build_connectivity_feature_table(
                    project_root=project_root,
                    pipeline=pipeline,
                    subjects=selected_subjects,
                    tasks=selected_tasks,
                    variants=selected_variants,
                    conditions=selected_conditions,
                    windows=selected_windows,
                    methods=selected_methods,
                    bands=selected_bands,
                    statistics=selected_statistics,
                    clip_negative_dwpli=clip_negative_dwpli,
                )
            merged = target_table.merge(feature_table, on="subject", how="inner").set_index("subject")
            feature_cols = [col for col in merged.columns if col != target_name]
            st.write(f"Feature matrix: {merged.shape[0]} subjects x {len(feature_cols)} features")
            st.dataframe(merged.iloc[:10, : min(12, merged.shape[1])], width="stretch")

            if run_local:
                if not selected_models:
                    st.error("Select at least one model.")
                elif len(feature_cols) == 0:
                    st.error("No features were built. Check selected feature options.")
                else:
                    with st.spinner("Running cross-validation..."):
                        metrics, predictions = cross_validate_regression(
                            merged[feature_cols],
                            merged[target_name],
                            models=selected_models,
                            cv_method=cv_method,
                            n_splits=int(n_splits),
                            n_repeats=int(n_repeats),
                            random_state=int(random_state),
                        )
                    st.subheader("Model Metrics")
                    st.dataframe(metrics.sort_values("pearson_r", ascending=False), width="stretch")

                    output_root = project_root / "derivatives" / "prediction" / target_name
                    run_dir = save_prediction_outputs(
                        output_root, config, feature_table, target_table, metrics, predictions
                    )
                    st.success(f"Saved prediction run: {run_dir}")

                    best = metrics.sort_values("pearson_r", ascending=False).iloc[0]
                    best_preds = predictions[predictions["model"] == best["model"]]
                    mean_preds = best_preds.groupby("subject", as_index=False).agg(
                        observed=("observed", "first"), predicted=("predicted", "mean")
                    )
                    st.subheader(f"Observed vs Predicted: {best['model']}")
                    st.scatter_chart(mean_preds, x="observed", y="predicted")

    with script_tab2:
        st.markdown("### SLURM script for HPC clusters")
        st.caption(
            "Use this when the connectivity files and behavior CSV are on HPC. "
            "The generated script uses the HPC behavior path, not the local uploaded file."
        )

        with st.expander("HPC Settings", expanded=True):
            hpc_project = st.text_input("HPC project root", value=str(project_root))
            hpc_eegcpm = st.text_input("HPC EEGCPM root", value="", placeholder="/path/to/eegcpm-0.1")
            hpc_venv = st.text_input("HPC venv activate", value=f"{hpc_eegcpm}/venv/bin/activate")
            hpc_partition = st.selectbox("Partition", ["shared_cpu", "shared_gpu_l40", "shared_gpu_h20"])
            hpc_time = st.text_input("Time", value="02:00:00")
            hpc_mem = st.text_input("Memory", value="16G")
            hpc_cpus = st.number_input("CPUs", min_value=1, max_value=16, value=4)
            hpc_email = st.text_input("Email", value=st.session_state.get("hpc_email", ""))

        if not hpc_behavior_path:
            st.warning("Set the HPC behavior file path in Target Data before generating the SLURM script.")

        hpc_script = _make_slurm_script(
            config,
            {
                "project_root": hpc_project,
                "eegcpm_root": hpc_eegcpm,
                "venv": hpc_venv,
                "partition": hpc_partition,
                "time": hpc_time,
                "mem": hpc_mem,
                "cpus": int(hpc_cpus),
                "email": hpc_email,
            },
        )
        st.code(hpc_script, language="bash")

        col1, col2 = st.columns([3, 1])
        with col2:
            st.download_button(
                label="⬇️ Download",
                data=hpc_script,
                file_name=f"prediction_{target_name}_slurm.sh",
                mime="text/plain",
                width="stretch",
            )

        st.markdown(f"""
        **Usage:**
        ```bash
        # 1. Upload/download the script to HPC if needed

        # 2. SSH to HPC and submit
        cd {hpc_project}
        sbatch prediction_{target_name}_slurm.sh

        # 3. Monitor job status
        squeue -u $USER

        # 4. Check job summary
        sacct -j <JOB_ID> --brief

        # 5. Cancel if needed
        scancel <JOB_ID>
        ```

        **Notes:**
        - The behavior file should exist on HPC at `{hpc_behavior_path or 'set in Target Data'}`.
        - Outputs are saved to `{hpc_project}/derivatives/prediction/{target_name}/`.
        - Prediction runs use all selected subjects together for cross-validation.
        """)

    pred_root = project_root / "derivatives" / "prediction"
    if pred_root.exists():
        st.markdown("---")
        st.header("Previous Prediction Runs")
        runs = sorted([p for p in pred_root.glob("*/*") if p.is_dir()], reverse=True)
        if runs:
            selected_run = st.selectbox("Prediction run", runs, format_func=lambda p: str(p.relative_to(pred_root)))
            metrics_file = selected_run / "model_metrics.csv"
            config_file = selected_run / "prediction_config.json"
            if metrics_file.exists():
                st.dataframe(pd.read_csv(metrics_file), width="stretch")
            if config_file.exists():
                st.json(json.loads(config_file.read_text()), expanded=False)


if __name__ == "__main__":
    main()
