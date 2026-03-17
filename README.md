# EEGCPM Development Repository

![Private](https://img.shields.io/badge/repo-private-red)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue)
![Tests](https://img.shields.io/badge/tests-253%20passing-brightgreen)

## Project Overview

**EEGCPM** (EEG Connectome Predictive Modeling) is a Python toolbox for end-to-end EEG analysis pipelines, from raw BIDS data to behavioral prediction using functional connectivity patterns.

**Primary Use Case**: Predict behavioral/cognitive scores from EEG functional connectivity using the CPM framework.

**Target Audience**: Neuroscience researchers working with BIDS datasets (e.g., HBN) who need reproducible EEG pipelines.

---

## Current Status

### Fully Implemented

| Stage | CLI Command | UI Pages | Key Features |
|-------|-------------|----------|--------------|
| **Preprocessing** | `eegcpm preprocess` | 0, 2, 3 | Filtering, ICA, ASR, bad channel detection |
| **Epochs** | `eegcpm epochs` | 1, 6, 7, 9 | Event segmentation, rejection, run combination |
| **Source Reconstruction** | `eegcpm source-reconstruct` | 10 | dSPM, sLORETA, eLORETA with CONN 32-ROI |
| **Connectivity** | `eegcpm connectivity` | 11 | 13 methods (PLV, wPLI, dwPLI, coherence, PDC, DTF, etc.) |
| **QC System** | `eegcpm import-qc` | 4, 8 | HTML reports at all stages |
| **Utilities** | `eegcpm status`, `resume` | Home | Project status, checkpoint resume |

### Partially Implemented

| Stage | Module Status | CLI | UI | What's Needed |
|-------|--------------|-----|-----|---------------|
| **Features** | Implemented | None | None | Add CLI command + UI page |
| **Prediction** | Implemented | None | None | Add CLI command + UI page |

---

## Quick Stats

| Metric | Value |
|--------|-------|
| Tests | 253 passing |
| CLI Commands | 7 |
| UI Pages | 12 |
| Connectivity Methods | 13 |
| Source Methods | 3 (dSPM, sLORETA, eLORETA) |
| ROI Parcellation | CONN 32-ROI |
| Python Version | 3.9+ (developed on 3.12.6) |

---

## Prioritized Roadmap

### High Priority

1. **Add Features CLI command + UI page**
   - Modules exist: `eegcpm/modules/features/`
   - Need: CLI in `eegcpm/cli/features.py`, UI page `12_features.py`

2. **Add Prediction CLI command + UI page**
   - Modules exist: `eegcpm/evaluation/prediction/`
   - Need: CLI in `eegcpm/cli/prediction.py`, UI page `13_prediction.py`

### Medium Priority

3. REST API for programmatic access
4. GPU acceleration for large datasets
5. Deep learning models (neural module)

### Lower Priority

6. UI enhancements (real-time updates, interactive plots)
7. Naming convention standardization across codebase

---

## For Team Members

### Quick Start

```bash
git clone https://github.com/clivehywong/eegcpm-dev.git
cd eegcpm-dev/eegcpm-0.1
pip install -e ".[dev]"
pytest tests/ -v  # Should see 253 passing
```

### Run the UI

```bash
streamlit run eegcpm/ui/app.py --server.port 8502
```

### Development Workflow

1. Create feature branch: `feature/description`
2. Make changes, add tests
3. Run tests: `pytest tests/ -v`
4. Push and create Pull Request
5. Wait for review (1 approval required)
6. Merge after approval

Branch protection is enabled on `main`.

---

## Repository Structure

```
eegcpm-dev/                    # Private development repo
├── README.md                  # This file - team status/roadmap
├── CLAUDE.md                  # Main development guide (comprehensive)
├── eegcpm-0.1/               # Publishable package
│   ├── eegcpm/               # Source code
│   │   ├── cli/              # 7 CLI commands
│   │   ├── modules/          # Analysis modules
│   │   ├── ui/               # 12 Streamlit pages
│   │   ├── evaluation/       # Prediction framework
│   │   └── ...
│   ├── tests/                # 253 tests
│   └── README.md             # User documentation
├── planning/                  # Architecture docs (6 files)
├── implementation/            # Development history
├── docs/                      # User documentation
└── .github/                   # Team workflow docs
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [CLAUDE.md](./CLAUDE.md) | Main development guide (comprehensive) |
| [.github/CONTRIBUTING.md](./.github/CONTRIBUTING.md) | Contribution workflow |
| [.github/README.md](./.github/README.md) | New member onboarding |
| [planning/ARCHITECTURE.md](./planning/ARCHITECTURE.md) | Stage-first architecture |
| [planning/QC_SYSTEM.md](./planning/QC_SYSTEM.md) | Quality control system |
| [planning/TASK_SYSTEM.md](./planning/TASK_SYSTEM.md) | Task configuration |
| [planning/UI_ARCHITECTURE.md](./planning/UI_ARCHITECTURE.md) | UI structure |
| [implementation/CHANGELOG.md](./implementation/CHANGELOG.md) | Development history |
| [eegcpm-0.1/README.md](./eegcpm-0.1/README.md) | Package user docs |

---

## Tech Stack

- **EEG Analysis**: MNE-Python
- **UI**: Streamlit
- **Testing**: pytest
- **Config**: Pydantic + YAML
- **Key Dependencies**: NumPy, SciPy, pandas, scikit-learn

---

## Code Standards

- **Line length**: 100 characters
- **Formatter**: Black
- **Linter**: Ruff (E, F, I, W rules)
- **Type hints**: Required for public APIs
- **Docstrings**: Google style

---

## Contributing

See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for full details.

**Quick checklist**:
- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Follows code standards
- [ ] Documentation updated if needed

---

## Contact

Questions? Open an issue or contact @clivehywong

---

**Repository**: https://github.com/clivehywong/eegcpm-dev (Private)
