#!/bin/bash
#SBATCH --job-name=eegcpm_predict
#SBATCH --partition=shared_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/share/ps_clivewong/25SusAttn/logs/prediction_%A.out
#SBATCH --error=/share/ps_clivewong/25SusAttn/logs/prediction_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@eduhk.hk

set -e
mkdir -p "/share/ps_clivewong/25SusAttn/logs"
source "/share/ps_clivewong/eegcpm-dev/eegcpm-0.1/venv/bin/activate"
cd "/share/ps_clivewong/eegcpm-dev/eegcpm-0.1"

#!/bin/bash
# EEGCPM prediction run
# Save this script next to your project or run it from any shell with EEGCPM installed.

python - <<'PY'
from pathlib import Path
import json
from eegcpm.evaluation.prediction.workflow import (
    load_behavior_table, prepare_target_table, scan_connectivity_options,
    build_connectivity_feature_table, cross_validate_regression, save_prediction_outputs,
)

config = json.loads(r'''{
  "project_root": "/share/ps_clivewong/25SusAttn",
  "behavior_path": "/share/behavior.csv",
  "subject_column": "subject",
  "target_name": "asrs_inattention",
  "pipeline": "optimized",
  "subjects": [
    "001",
    "002"
  ],
  "tasks": [
    "cptvis"
  ],
  "variants": [
    "a"
  ],
  "conditions": [
    "c"
  ],
  "windows": [
    "w"
  ],
  "methods": [
    "m"
  ],
  "bands": [
    "b"
  ],
  "statistics": [
    "s"
  ],
  "clip_negative_dwpli": true,
  "models": [
    "ridge"
  ],
  "cv_method": "kfold",
  "n_splits": 5,
  "n_repeats": 1,
  "random_state": 0,
  "hpc_behavior_path": "/share/behavior.csv"
}''')
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
print(f"Saved prediction run: {run_dir}")
PY

