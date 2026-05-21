# EEGCPM - EEG Connectome Predictive Modeling Toolbox

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-253%20passed-brightgreen.svg)]()

A comprehensive Python toolbox for EEG preprocessing, source reconstruction, connectivity analysis, and predictive modeling. Implements the Connectome Predictive Modeling (CPM) framework for predicting behavioral/cognitive outcomes from EEG functional connectivity.

## Features

### Analysis Pipeline (Fully Implemented)
- **Quality Control**: Raw data QC with diagnostic plots, HTML reports, batch processing
- **Preprocessing**: Band-pass filtering, ICA artifact removal (Infomax, PICARD), ASR, bad channel detection
- **Epochs**: Event-based segmentation, rejection (threshold, autoreject), baseline correction
- **Source Reconstruction**: dSPM, sLORETA, eLORETA with CONN 32-ROI parcellation
- **Connectivity**: 13 methods including PLV, wPLI, dwPLI, coherence, PDC, DTF

### Feature Extraction (Modules Implemented, CLI/UI in Development)
- **ERP Features**: Peak amplitudes, latencies, mean amplitudes
- **Spectral Features**: Band power (absolute/relative), frequency-specific power
- **Connectivity Features**: Network matrices, graph metrics
- **Time-Frequency**: Morlet wavelets, multitaper, band power

### Prediction Framework (Modules Implemented, CLI/UI in Development)
- **CPM**: Edge selection, positive/negative network models
- **Models**: Ridge regression, SVM, elastic net, ensemble
- **Evaluation**: Cross-validation, permutation testing, feature importance
- **Deep Learning**: Planned for v0.2.0

### Infrastructure
- **HPC Integration**: SLURM batch processing with checkpoint/resume
- **GUI**: Streamlit-based web interface
- **BIDS Compatible**: Works with BIDS-formatted datasets
- **Multi-Format Support**: FIF, BrainVision, EDF, BDF, EEGLAB, NeuroScan, CURRY

## Installation

```bash
# Clone the repository
git clone https://github.com/eegcpm/eegcpm.git
cd eegcpm/eegcpm-0.1

# Basic installation
pip install -e .

# With Streamlit UI
pip install -e ".[ui]"

# With deep learning support
pip install -e ".[neural]"

# Full installation (all optional dependencies)
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from eegcpm import Config
from eegcpm.core.project import scan_bids_directory, AnalysisProject
from eegcpm.pipeline.executor import create_standard_pipeline

# Load existing BIDS dataset
project = scan_bids_directory("/path/to/bids/data")

# Or create a new analysis project
analysis = AnalysisProject(
    source_dir="/path/to/bids/data",
    analysis_dir="/path/to/analysis",
)

# Configure pipeline
config = Config()
config.preprocessing.l_freq = 0.5
config.preprocessing.h_freq = 40.0
config.preprocessing.ica_method = "picard"
config.source.method = "sLORETA"
config.connectivity.methods = ["correlation", "plv"]

# Create and run pipeline
pipeline = create_standard_pipeline(project, config)
results = pipeline.run(resume=True)  # Resume from checkpoint if interrupted

# Check results
summary = pipeline.get_summary()
print(f"Processed {summary['successful']}/{summary['total_subjects']} subjects")
```

### Quality Control

```python
from pathlib import Path
from eegcpm.modules.qc import RawQC, run_raw_qc_batch

# Single subject QC
raw = mne.io.read_raw_fif("sub-001_eeg.fif", preload=True)
qc = RawQC(Path("./qc_output"), line_freq=60.0)
result = qc.compute(raw, "sub-001")
qc.generate_html_report(result, Path("./qc_output/sub-001_qc.html"))

# Batch QC with index page
fif_files = list(Path("/data/bids").glob("sub-*/ses-*/eeg/*_eeg.fif"))
results, index_path = run_raw_qc_batch(fif_files, Path("./qc_output"))
print(f"View QC reports at: file://{index_path}")
```

### Streamlit Web Interface

```bash
streamlit run eegcpm/ui/app.py
```

### Command Line Interface

```bash
# Check processing status
eegcpm status --project /path/to/project

# Run preprocessing pipeline
eegcpm preprocess \
  --config configs/preprocessing/standard.yaml \
  --project /path/to/project \
  --task rest

# Extract epochs from preprocessed data
eegcpm epochs \
  --config configs/epochs/default.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Run source reconstruction
eegcpm source-reconstruct \
  --config configs/source/dspm.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Compute connectivity
eegcpm connectivity \
  --config configs/connectivity/default.yaml \
  --project /path/to/project \
  --preprocessing standard \
  --task rest

# Resume failed/incomplete workflows
eegcpm resume --project /path/to/project

# Import QC metrics to state database
eegcpm import-qc --project /path/to/project --qc-json qc_results.json

# For detailed help on any command
eegcpm <command> --help
```

## Configuration

All settings are managed through Pydantic configuration classes:

```python
from eegcpm import Config
from eegcpm.core.config import RejectionConfig

config = Config()

# Preprocessing
config.preprocessing.l_freq = 1.0          # High-pass filter
config.preprocessing.h_freq = 45.0         # Low-pass filter
config.preprocessing.ica_method = "picard" # ICA algorithm
config.preprocessing.use_asr = True        # Artifact Subspace Reconstruction

# Epoch rejection (use presets or custom)
config.epochs.rejection = RejectionConfig.strict()  # or .lenient() or .adaptive()

# Source reconstruction
config.source.method = "sLORETA"
config.source.parcellation = "conn_networks"  # 32 ROIs

# Connectivity
config.connectivity.methods = ["plv", "wpli", "correlation"]
config.connectivity.frequency_bands = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# Save/load configuration
config.to_yaml("my_config.yaml")
config = Config.from_yaml("my_config.yaml")
```

## Project Structure

EEGCPM uses a BIDS-inspired directory structure:

```
project_root/
├── sub-001/
│   └── ses-01/
│       └── eeg/
│           ├── sub-001_ses-01_task-oddball_eeg.fif
│           └── sub-001_ses-01_task-oddball_events.tsv
├── derivatives/
│   └── pipeline-baseline/
│       ├── config.yaml
│       ├── qc/
│       ├── preprocessed/
│       ├── epochs/
│       ├── source/
│       ├── connectivity/
│       └── features/
└── project.json
```

## CONN 32-ROI Parcellation

Default parcellation uses the CONN Toolbox network ROIs (32 nodes across 8 canonical networks):

| Network | ROIs | Description |
|---------|------|-------------|
| **Default Mode** | MPFC, LP(L/R), PCC | Self-referential, memory |
| **SensoriMotor** | Lateral(L/R), Superior | Motor planning/execution |
| **Visual** | Medial, Occipital, Lateral(L/R) | Visual processing |
| **Salience** | ACC, AInsula(L/R), RPFC(L/R), SMG(L/R) | Attention switching |
| **Dorsal Attention** | FEF(L/R), IPS(L/R) | Top-down attention |
| **FrontoParietal** | LPFC(L/R), PPC(L/R) | Executive control |
| **Language** | IFG(L/R), pSTG(L/R) | Language processing |
| **Cerebellar** | Anterior, Posterior | Motor coordination |

## Pipeline Modules

The analysis pipeline consists of modular, chainable steps:

```
Raw EEG → Preprocessing → Epochs → Source → Connectivity → Features → Prediction
            ↓              ↓         ↓           ↓            ↓           ↓
         Filtering      Segment   sLORETA     PLV/wPLI    Vectorize     CPM
           ICA          Reject    32 ROIs    per band    Edge select   Ridge
           ASR          Baseline
```

Each module:
- Validates input data
- Processes with configurable parameters
- Saves outputs and checkpoints
- Reports success/failure with timing

## Testing

```bash
# Run all tests
python -m pytest tests/ -v --override-ini="addopts="

# Run specific test module
python -m pytest tests/unit/test_core.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=eegcpm --cov-report=html
```

Current test coverage: **253 tests passing**

## Dependencies

### Core Requirements
- Python >= 3.9
- MNE-Python >= 1.0
- NumPy >= 1.20
- SciPy >= 1.7
- Pandas >= 1.3
- Scikit-learn >= 1.0
- Pydantic >= 2.0
- PyYAML >= 6.0

### Optional
- Streamlit >= 1.20 (UI)
- PyTorch >= 2.0 (deep learning)
- Autoreject >= 0.4 (adaptive rejection)
- NetworkX >= 2.8 (graph analysis)

## Citation

If you use EEGCPM in your research, please cite:

```bibtex
@software{eegcpm2024,
  title = {EEGCPM: EEG Connectome Predictive Modeling Toolbox},
  year = {2024},
  url = {https://github.com/eegcpm/eegcpm}
}
```

## Related Work

- [MNE-Python](https://mne.tools/) - Core EEG/MEG analysis
- [CONN Toolbox](https://web.conn-toolbox.org/) - ROI parcellation reference
- [CPM](https://github.com/YaleMRRC/CPM) - Original CPM implementation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass
5. Submit a pull request
