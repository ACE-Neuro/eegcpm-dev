# EEGCPM Configuration Files

This directory contains all configuration files for the EEGCPM pipeline, organized by processing stage.

## Directory Structure

```
config/
├── preprocessing/          # EEG preprocessing configurations
├── connectivity/           # Connectivity analysis configurations
├── features/              # Feature engineering configurations
├── prediction/            # Predictive modeling configurations
│   ├── trial_level/       # Trial-by-trial predictions
│   └── subject_level/     # Subject-level predictions (CPM)
└── pipelines/             # Full end-to-end pipeline configs
```

## Stage Descriptions

### 1. Preprocessing (`preprocessing/`)
EEG preprocessing configurations including:
- Filtering (highpass, lowpass, notch)
- Bad channel detection and interpolation
- ASR (Artifact Subspace Reconstruction)
- ICA (Independent Component Analysis)
- Re-referencing

**Example configs:**
- `asr_pipeline.yaml` - Full pipeline with ASR + ICA (recommended)
- `ica_only.yaml` - ICA without ASR
- `minimal.yaml` - Basic filtering only

### 2. Connectivity (`connectivity/`)
Functional connectivity analysis configurations:
- Methods: PLV, wPLI, coherence, correlation
- Frequency bands: delta, theta, alpha, beta, gamma
- ROI parcellations: CONN 32-ROI, custom atlases

**Example configs:**
- `plv_alpha_beta.yaml` - Phase-locking value in alpha/beta bands
- `wpli_broadband.yaml` - Weighted PLI across all bands

### 3. Features (`features/`)
Feature extraction configurations:
- ERP time windows and components
- Time-frequency analysis
- Graph theoretical metrics
- Dimensionality reduction

**Example configs:**
- `erp_windows.yaml` - Task-specific ERP components
- `timefreq_power.yaml` - Spectral power features
- `connectivity_graph.yaml` - Graph metrics from connectivity matrices

### 4. Prediction (`prediction/`)

#### Trial-Level (`prediction/trial_level/`)
Single-trial classification/regression:
- LSTM networks
- CNN architectures
- SVM/SVR models

**Example configs:**
- `lstm_trial.yaml` - LSTM for trial classification
- `cnn_trial.yaml` - CNN for EEG classification

#### Subject-Level (`prediction/subject_level/`)
Between-subject prediction (CPM framework):
- Ridge regression (classic CPM)
- Ensemble models
- Deep learning approaches

**Example configs:**
- `cpm_ridge.yaml` - Classic CPM with ridge regression
- `ensemble.yaml` - Ensemble of multiple models

### 5. Full Pipelines (`pipelines/`)
End-to-end pipeline configurations that combine multiple stages:

**Example configs:**
- `full_cpm_pipeline.yaml` - Complete CPM: Preprocessing → Connectivity → Prediction
- `erp_prediction.yaml` - ERP-based: Preprocessing → ERP Features → ML
- `custom_experiment.yaml` - Custom multi-stage pipeline

## Config File Format

All config files use YAML format with the following structure:

```yaml
# Metadata
stage: preprocessing  # Which processing stage
name: asr_pipeline   # Config identifier
version: "1.0"       # Version for reproducibility

# Stage-specific settings
preprocessing:
  filter:
    highpass_freq: 0.5
    lowpass_freq: 40.0
  asr:
    enabled: true
    cutoff: 20
  ica:
    enabled: true
    n_components: rank-1  # Auto-determine based on data rank
    method: picard

# QC settings (optional)
qc:
  generate_qc: true
  dpi: 100
```

## Usage in UI

The Streamlit UI automatically scans these directories and presents configs in a hierarchical dropdown:

1. **Preprocessing page**: Shows configs from `preprocessing/`
2. **Connectivity page**: Shows configs from `connectivity/`
3. **Features page**: Shows configs from `features/`
4. **Prediction page**: Shows configs from `prediction/trial_level/` or `prediction/subject_level/`

## Creating New Configs

### Option 1: Via UI (Recommended)
Use the **Pipeline Config** page in the Streamlit UI to create and edit configs interactively.

### Option 2: Manual YAML
1. Copy an existing config as a template
2. Modify parameters as needed
3. Save in the appropriate subdirectory
4. Refresh the UI to see the new config

## Best Practices

1. **Naming**: Use descriptive names that indicate the config purpose
   - ✅ `asr_pipeline.yaml`, `ica_only.yaml`, `highpass_1hz.yaml`
   - ❌ `config1.yaml`, `test.yaml`, `new.yaml`

2. **Versioning**: Include version field for reproducibility
   ```yaml
   version: "1.0"  # Increment when making breaking changes
   ```

3. **Documentation**: Add comments explaining non-obvious parameters
   ```yaml
   asr:
     cutoff: 20  # EEGLAB: aggressive 3, default 5, conservative 20
                 # (GUI default 20; lower = more aggressive)
   ```

4. **Defaults**: Keep a `default.yaml` in each directory as a fallback

5. **Validation**: Test new configs on a small dataset first

## Migration from Old Structure

Old flat configs have been moved:
- `asr_pipeline_config.yaml` → `preprocessing/asr_pipeline.yaml`
- `cli_test_config.yaml` → `preprocessing/cli_test.yaml`

The UI automatically uses the new hierarchical structure.
