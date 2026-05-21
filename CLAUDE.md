# CLAUDE.md - EEGCPM Development Guide

This document provides essential context for Claude Code (or any AI assistant) working on the EEGCPM codebase.

---

## Project Overview

EEGCPM (EEG Connectome Predictive Modeling) is a Python toolbox for end-to-end EEG analysis pipelines, from raw data to behavioral prediction.

**Primary use case**: Predict behavioral/cognitive scores from EEG functional connectivity patterns.

---

## Directory Structure

```
/Users/clive/eegcpm/                    # Development root (NOT published)
├── README.md                           # Team status and roadmap
├── CLAUDE.md                           # This file - AI development guide
├── docs/                               # User documentation
│   ├── SETUP.md                        # Quick start and environment
│   └── WORKFLOWS.md                    # Multi-run workflows
├── planning/                           # Architecture documentation
│   ├── ARCHITECTURE.md                 # Stage-first, modules, paths
│   ├── TASK_SYSTEM.md                  # Task config, filters, binning
│   ├── UI_ARCHITECTURE.md              # Streamlit pages, data flow
│   ├── QC_SYSTEM.md                    # Quality control system
│   ├── BAD_CHANNEL_DETECTION.md        # Clustering analysis
│   └── EXCLUSION_CRITERIA.md           # Preprocessing exclusion rules
├── implementation/                     # Historical records
│   ├── CHANGELOG.md                    # Development history
│   └── EEGLAB_COMPARISON.md            # EEGLAB vs MNE comparison
│
└── eegcpm-0.1/                         # PUBLISHABLE PACKAGE
    ├── README.md                       # User documentation
    ├── pyproject.toml                  # Package config
    ├── eegcpm/                         # Source code
    ├── tests/                          # Test suite (253 passing)
    └── config/                         # Example configs
```

**IMPORTANT**: Keep `eegcpm-0.1/` clean - no AI/development artifacts. This folder will be published.

---

## Package Architecture

For detailed architecture, see `planning/ARCHITECTURE.md`.

### Module Structure

```
eegcpm/
├── core/               # Foundation (models, config, paths, validation)
├── data/               # Data loading and templates (CONN ROIs)
├── pipeline/           # Execution framework (executor, checkpoint, state)
├── modules/            # Analysis modules
│   ├── preprocessing/  # Filtering, ICA, ASR, bad channels
│   ├── epochs/         # Event segmentation
│   ├── source/         # Source reconstruction
│   ├── connectivity/   # PLV, wPLI, coherence
│   └── qc/             # Quality control
├── evaluation/         # Prediction (CPM, models, metrics)
├── plugins/            # Dataset importers (HBN)
├── ui/                 # Streamlit interface
└── utils/              # Logging, helpers
```

### Key Design Patterns

**1. Module System**: All processing steps inherit from `BaseModule`
- `RawDataModule`, `EpochsModule`, `SourceModule` specializations
- Returns `ModuleResult` with data, metadata, warnings, figures

**2. Stage-First Architecture**: Organize by processing stage, not pipeline
- Stage 1: Preprocessing (variants: standard, minimal, robust)
- Stage 2: Epochs (depends on preprocessing)
- Stage 3: Source (depends on epochs)
- Stage 4: Features (depends on source)
- Stage 5: Prediction (depends on features)

**3. Centralized Paths**: `EEGCPMPaths` class manages all directory structures

**4. Pydantic Configuration**: Type-safe configs with validation

For details, see `planning/ARCHITECTURE.md`.

---

## Project Structure (Stage-First)

```
project_root/
├── bids/                          # BIDS input data
├── derivatives/                   # Stage-first outputs
│   ├── preprocessing/{pipeline}/sub-{ID}/ses-{session}/task-{task}/run-{run}/
│   ├── epochs/{preprocessing}/{task}/sub-{ID}/ses-{session}/
│   ├── source/{preprocessing}/{task}/variant-{method}-{template}/sub-{ID}/...
│   ├── features/{preprocessing}/{task}/{source-variant}/{feature-type}/
│   └── prediction/{model-name}/
└── eegcpm/                        # Workspace
    ├── configs/                   # Configuration files
    └── .eegcpm/state.db          # Workflow tracking
```

**Key Rules**:
1. All outputs → `derivatives/{stage}/{variant}/sub-{ID}/...`
2. All configs → `eegcpm/configs/{stage}/`
3. State tracking → `eegcpm/.eegcpm/state.db`
4. Subject folders → Always `sub-{ID}` prefix (BIDS compliance)
5. QC reports → Co-located with data (not separate `qc/` folder)

---

## Path Management

```python
from eegcpm.core.paths import EEGCPMPaths

paths = EEGCPMPaths(project_root="/Volumes/Work/data/hbn")

# Get stage-specific paths
preproc_dir = paths.get_preprocessing_dir(
    pipeline="standard",
    subject="NDARAA306NT2",
    session="01",
    task="contdet",
    run="1"
)
```

For details, see `planning/ARCHITECTURE.md`.

---

## Quality Control System

QC reports are co-located with outputs at each stage. No separate `qc/` folders.

**QC Modules**:
- **RawDataQC**: PSD, variance, correlation
- **PreprocessedQC**: Before/after comparison, ICA diagnostics
- **EpochsQC**: Rejection stats, trial counts, ERPs
- **SourceQC**: ROI coverage, SNR
- **ConnectivityQC**: Connectivity matrices, network stats

For details, see `planning/QC_SYSTEM.md` and `planning/BAD_CHANNEL_DETECTION.md`.

---

## Development Environment

**Python**: System Python 3.12.6 at `/Library/Frameworks/Python.framework/Versions/3.12/`

**Installation**:
```bash
cd /Users/clive/eegcpm/eegcpm-0.1
pip install -e ".[dev]"
```

**Environment Activation**: ✅ None needed - commands work immediately

**Important**: Changes to `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/` are live instantly (editable install)

For full setup details, see `docs/SETUP.md`.

---

## Common Commands

### CLI
```bash
# Check processing status
eegcpm status --project /path/to/project

# Run preprocessing
eegcpm preprocess \
  --config configs/preprocessing/standard.yaml \
  --project /path/to/project \
  --task rest

# Extract epochs
eegcpm epochs \
  --config configs/epochs/default.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Source reconstruction
eegcpm source-reconstruct \
  --config configs/source/dspm.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Connectivity analysis
eegcpm connectivity \
  --config configs/connectivity/default.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Streamlit UI
streamlit run eegcpm/ui/app.py --server.port 8502

# Tests
python3 -m pytest tests/ -v --override-ini="addopts="
```

### Clear Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

---

## Naming Conventions

| Convention | Format | Example |
|------------|--------|---------|
| **Stage directories** | Single word | `preprocessing`, `epochs` |
| **Pipeline variants** | Single word | `standard`, `minimal` |
| **Subject IDs** | sub-{ID} | `sub-NDARAB678VYW` |
| **Session IDs** | ses-{N} | `ses-01` |
| **Task names** | task-{name} | `task-rest`, `task-contdet` |
| **Run IDs** | run-{N} | `run-01`, `run-1` |
| **QC reports** | {subject}_{stage}_qc.html | `sub-001_preprocessed_qc.html` |

**Important**: Always use `sub-` prefix for subject directories (BIDS compliance).

---

## Multi-Agent Workflow

### Agent Roles

| Agent | Model | Use For |
|-------|-------|---------|
| **Architect** | Opus 4 | Architecture decisions, code review, approval |
| **Developer** | Sonnet | Feature implementation, refactoring, testing |
| **Coder** | Codex | Complex algorithms, performance-critical code |
| **Scout** | Sonnet:web | API lookups, documentation research |
| **Assistant** | Haiku | Formatting, simple tasks, batch operations |

### Spawning Agents

```python
# For complex implementation
Task(
    subagent_type="general-purpose",
    model="sonnet",
    prompt="Implement feature X following spec...",
    description="Implement feature X"
)

# For codebase exploration
Task(
    subagent_type="Explore",
    prompt="Find all files implementing bad channel detection",
    description="Locate bad channel code"
)
```

### Workflow Protocol

1. **Discussion**: Understand requirements, propose architecture
2. **Planning**: Break into tasks, assign agents
3. **Implementation**: Parallel execution with tests
4. **Review**: Architect reviews and approves
5. **Integration**: Merge components, run full test suite

---

## Context Management

### Avoid Background Task Accumulation

**Bad** (tracked by Claude Code):
```bash
python script.py 2>&1 | tee /tmp/log.log &
```

**Good** (detached):
```bash
nohup python script.py > /tmp/log.log 2>&1 &
tail -50 /tmp/log.log  # Check manually
```

### Use Agents for Large Tasks

Agents work in separate contexts and return condensed summaries:
- Exploration → `Task(subagent_type='Explore')`
- Research → `Task(subagent_type='claude-code-guide')`
- Implementation → `Task(subagent_type='general-purpose')`

### Todo List Management

- In-memory storage by Claude Code
- Automatically visible
- Use `TodoWrite` tool for multi-step task tracking

---

## Configuration System

All configuration via Pydantic models and YAML files.

**Example** (`configs/preprocessing/standard.yaml`):
```yaml
stage: preprocessing
pipeline: standard

steps:
  - name: montage
    params:
      type: GSN-HydroCel-129

  - name: filter
    params:
      l_freq: 0.5
      h_freq: 40.0

  - name: ica
    params:
      method: fastica
      n_components: rank-1
      l_freq_fit: 1.0
```

For task configuration details, see `planning/TASK_SYSTEM.md`.

---

## UI Architecture

Streamlit multipage application in `eegcpm/ui/`.

**Pages** (ordered by numeric prefix):
- 0: Preprocessing Config - Configure preprocessing pipeline
- 1: Task Config - Configure epoch parameters
- 2: Batch Preprocessing - Process multiple subjects
- 3: Single Preprocessing - Single subject processing
- 4: QC Browser - View quality control reports
- 6: Epochs Interactive - Interactive epoch exploration
- 7: Epochs Batch - Batch epoch processing
- 8: Group Summary - Group-level summaries
- 9: Trial Sorting - Epoch organization
- 10: Source Reconstruction - Source analysis
- 11: Connectivity - Connectivity analysis

For details, see `planning/UI_ARCHITECTURE.md`.

---

## Current Status

**Working** (253 tests passing):
- ✅ Preprocessing (filtering, ICA, ASR, bad channels)
- ✅ QC System (raw, preprocessed, epochs, source, connectivity QC)
- ✅ Epochs (event extraction, rejection, ERPs)
- ✅ Source reconstruction (dSPM, sLORETA with CONN ROIs)
- ✅ Connectivity (PLV, wPLI, dwPLI, coherence, PDC, DTF - 13 methods)
- ✅ Features (ERP features, bandpower, connectivity matrices)
- ✅ Prediction (CPM, ML models, permutation testing)
- ✅ CLI and Streamlit UI
- ✅ Stage-first architecture
- ✅ Path management and dependency tracking

**Planned**:
- REST API
- GPU acceleration
- Deep learning models (neural module)

---

## Code Style

- Line length: 100 chars
- Format: Black
- Linting: Ruff (E, F, I, W rules)
- Type hints: Required for public APIs
- Docstrings: Google style

---

## MNE-Python Integration

Key MNE objects:
- `mne.io.Raw` - continuous EEG
- `mne.Epochs` - epoched data
- `mne.SourceEstimate` - source-level data
- `mne.Info` - measurement metadata

Documentation: https://mne.tools/stable/

---

## CONN 32-ROI Parcellation

Default parcellation for source reconstruction (defined in `data/conn_rois.py`):

| Network | Count | Example ROIs |
|---------|-------|-------------|
| DefaultMode | 4 | MPFC, PCC, LP(L/R) |
| SensoriMotor | 3 | Lateral(L/R), Superior |
| Visual | 4 | Medial, Occipital, Lateral(L/R) |
| Salience | 7 | ACC, AInsula(L/R), RPFC(L/R), SMG(L/R) |
| DorsalAttention | 4 | FEF(L/R), IPS(L/R) |
| FrontoParietal | 4 | LPFC(L/R), PPC(L/R) |
| Language | 4 | IFG(L/R), pSTG(L/R) |
| Cerebellar | 2 | Anterior, Posterior |

---

## Additional Documentation

**User Documentation**:
- `docs/SETUP.md` - Environment setup and quick start
- `docs/WORKFLOWS.md` - Multi-run workflows

**Architecture**:
- `planning/ARCHITECTURE.md` - Stage-first architecture, modules, paths
- `planning/TASK_SYSTEM.md` - Task configuration, filtering, binning
- `planning/UI_ARCHITECTURE.md` - Streamlit UI structure
- `planning/QC_SYSTEM.md` - Quality control system
- `planning/BAD_CHANNEL_DETECTION.md` - Bad channel clustering

**Historical**:
- `implementation/CHANGELOG.md` - Development history
- `implementation/EEGLAB_COMPARISON.md` - EEGLAB vs MNE comparison

---

**For detailed information, always refer to the specific documentation files above.**
