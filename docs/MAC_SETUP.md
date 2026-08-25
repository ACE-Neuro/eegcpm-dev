# EEGCPM macOS Quick Start

> For Linux, see `docs/LINUX_SETUP.md`. For HPC/SLURM batch jobs, see `docs/HPC_GUIDE.md`.

## Environment (No Activation Needed)

**Python**: System Python 3.12.6 at `/Library/Frameworks/Python.framework/Versions/3.12/`
**Activation**: ✅ None - commands work immediately from any terminal

```bash
# Verify setup
which python3    # /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
eegcpm --help    # CLI should work
```

---

## Common Commands

### CLI Preprocessing
```bash
cd /Users/clive/eegcpm/eegcpm-0.1

# Single subject
eegcpm preprocess \
  --config /Volumes/Work/data/hbn/eegcpm/configs/preprocessing/standard.yaml \
  --project /Volumes/Work/data/hbn \
  --pipeline standard \
  --subject NDARAA306NT2 \
  --task contdet

# All subjects
eegcpm preprocess \
  --config /Volumes/Work/data/hbn/eegcpm/configs/preprocessing/standard.yaml \
  --project /Volumes/Work/data/hbn \
  --pipeline standard \
  --task contdet
```

### Streamlit UI
```bash
cd /Users/clive/eegcpm/eegcpm-0.1
streamlit run eegcpm/ui/app.py --server.port 8502
# Open: http://localhost:8502
```

### Testing
```bash
cd /Users/clive/eegcpm/eegcpm-0.1
python3 -m pytest tests/ -v --override-ini="addopts="
```

---

## Directory Structure (Stage-First Architecture)

**IMPORTANT**: All processing outputs go to `derivatives/` (NOT `eegcpm/pipelines/`!)

```
/Volumes/Work/data/hbn/
├── bids/                          # Input data (BIDS format)
│
├── derivatives/                   # ✅ ALL OUTPUTS (stage-first)
│   ├── preprocessing/             # Stage 1: Preprocessing
│   │   └── standard/              # Pipeline variant
│   │       └── sub-{ID}/          # ⚠️ Note: "sub-" prefix required
│   │           └── ses-{session}/
│   │               └── task-{task}/
│   │                   └── run-{run}/
│   │                       ├── *_preprocessed_raw.fif
│   │                       ├── *_ica.fif
│   │                       ├── *_preprocessed_qc.html
│   │                       └── *_qc_metrics.json
│   │
│   ├── epochs/                    # Stage 2: Epoching
│   │   └── {preprocessing}/       # Which preprocessing was used
│   │       └── {task}/
│   │           └── sub-{ID}/
│   │               └── ses-{session}/
│   │
│   ├── source/                    # Stage 3: Source reconstruction
│   │   └── {preprocessing}/
│   │       └── {task}/
│   │           └── variant-{method}-{template}/
│   │
│   ├── features/                  # Stage 4: Feature extraction
│   └── prediction/                # Stage 5: Prediction
│
└── eegcpm/                        # ✅ WORKSPACE ONLY (configs & state)
    ├── configs/                   # Configuration files
    │   ├── preprocessing/
    │   │   └── standard.yaml
    │   ├── source/
    │   └── features/
    │
    └── .eegcpm/                   # Workflow tracking
        └── state.db               # Processing state database
```

### Path Architecture Rules:
1. **All outputs** → `derivatives/{stage}/{variant}/sub-{ID}/...`
2. **All configs** → `eegcpm/configs/{stage}/`
3. **State tracking** → `eegcpm/.eegcpm/state.db`
4. **Subject folders** → Always use `sub-{ID}` prefix (BIDS compliance)
5. **Never** write to `eegcpm/pipelines/` (old architecture, removed)

---

## Development Workflow

1. **Edit code** in `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/`
2. **Changes are live** immediately (editable install)
3. **No reinstall needed** (unless changing `pyproject.toml` entry points)

---

## Key Files

- **Package source**: `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/`
- **Tests**: `/Users/clive/eegcpm/eegcpm-0.1/tests/`
- **CLI entry points**: `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/cli/`
- **UI pages**: `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/ui/pages/`
- **Path management**: `/Users/clive/eegcpm/eegcpm-0.1/eegcpm/core/paths.py`
- **Full docs**: `/Users/clive/eegcpm/CLAUDE.md`

---

## Troubleshooting

### Clear Python cache
```bash
cd /Users/clive/eegcpm/eegcpm-0.1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

### Verify installation
```bash
pip show eegcpm
# Should show: Editable project location: /Users/clive/eegcpm/eegcpm-0.1
```

### Re-install if needed
```bash
cd /Users/clive/eegcpm/eegcpm-0.1
pip install -e ".[dev]"
```
