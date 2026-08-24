# EEGCPM Linux Setup

Quick start for installing EEGCPM on Linux (tested on `/share/ps_clivewong/eegcpm-dev`).

For the macOS setup, see `docs/MAC_SETUP.md`. For HPC/SLURM batch jobs, see `docs/HPC_GUIDE.md`.

---

## Requirements

- Python >=3.9 (>=3.12 recommended, matches dev environment)
- `git`, `pip`
- ~2 GB free disk for MNE data templates fetched on first run

Choose **one** of the two install paths below.

---

## Option A: venv (lightweight, no Anaconda required)

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To re-enter later:

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
source .venv/bin/activate
```

---

## Option B: Conda (matches HPC guide)

```bash
conda create -p /share/ps_clivewong/eegcpm-dev/.conda/envs/eegcpm python=3.12 -y
conda activate /share/ps_clivewong/eegcpm-dev/.conda/envs/eegcpm
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
pip install -e ".[dev]"
```

On HPC with `module`, prepend:

```bash
source /usr/share/modules/init/profile.sh
module load anaconda/25.1.1   # or your installed version
```

---

## Choosing Extras

`pyproject.toml:28-72` defines the extras. Install only what you need:

| Extra   | Adds                                          |
|---------|-----------------------------------------------|
| `dev`   | pytest, black, ruff, mypy (default for tests) |
| `ui`    | streamlit, plotly                             |
| `asr`   | eegprep, eeglabio (ASR cleaning)              |
| `neural`| torch (deep learning models)                  |
| `all`   | everything above + autoreject, nibabel, networkx |

Examples:

```bash
pip install -e ".[dev]"          # CLI + tests + lint
pip install -e ".[ui,dev]"       # add Streamlit UI
pip install -e ".[all]"          # maximal install (heavy)
```

---

## Verify Installation

```bash
which eegcpm        # should print .venv or conda-env path, not /usr/bin
pip show eegcpm     # Editable project location: /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
eegcpm --help       # CLI works
python3 -m pytest tests/ -v --override-ini="addopts="   # 263 tests should pass
```

If `eegcpm --help` is not found, your env is not active — re-run `source .venv/bin/activate` (venv) or `conda activate ...` (conda).

---

## Common Commands

> All paths assume project root `/share/ps_clivewong/eegcpm-dev` and an active env.

### CLI

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1

# Check status
eegcpm status --project /share/ps_clivewong/eegcpm-dev

# Preprocess a single subject
eegcpm preprocess \
  --config /share/ps_clivewong/eegcpm-dev/eegcpm/configs/preprocessing/standard.yaml \
  --project /share/ps_clivewong/eegcpm-dev \
  --pipeline standard \
  --subject NDARAA306NT2 \
  --task contdet

# All subjects, single task
eegcpm preprocess \
  --config /share/ps_clivewong/eegcpm-dev/eegcpm/configs/preprocessing/standard.yaml \
  --project /share/ps_clivewong/eegcpm-dev \
  --pipeline standard \
  --task contdet
```

### Streamlit UI

Requires the `ui` extra: `pip install -e ".[ui,dev]"`.

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
streamlit run eegcpm/ui/app.py --server.port 8502
# Open: http://localhost:8502
```

### Tests

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
python3 -m pytest tests/ -v --override-ini="addopts="
```

### Lint / Format

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1

# Lint
ruff check eegcpm/ tests/

# Format check
black --check eegcpm/ tests/

# Type check
mypy eegcpm/
```

---

## Project Layout (Stage-First)

```
/share/ps_clivewong/eegcpm-dev/
├── CLAUDE.md                 # AI/developer guide
├── docs/                     # user documentation
├── planning/                 # architecture notes
└── eegcpm-0.1/               # PUBLISHABLE PACKAGE
    ├── eegcpm/               # source code (changes live instantly)
    ├── tests/                # 263 tests
    ├── config/               # example configs
    └── pyproject.toml
```

`CLAUDE.md` is the authoritative AI-dev guide; see `planning/ARCHITECTURE.md` for stage-first design.

---

## Troubleshooting

### `eegcpm: command not found`

Env is not active. Re-run `source .venv/bin/activate` (venv) or `conda activate /path/to/env`.

### Permission denied on `/share/...`

`/share/ps_clivewong` is a network share — installation via pip works, but some editing tools may need elevated permissions. Check `ls -la /share/ps_clivewong/eegcpm-dev`.

### MNE complains about missing data path

First run downloads template brains (~1.5 GB) to `~/mne_data`. Pre-fetch if quota is tight:

```bash
python3 -c "import mne; mne.datasets.fetch_fsaverage(verbose=True)"
```

### Clear Python cache

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

### Re-install (e.g. after `pyproject.toml` changes)

```bash
cd /share/ps_clivewong/eegcpm-dev/eegcpm-0.1
pip install -e ".[dev]" --force-reinstall --no-deps
pip install -e ".[dev]"   # then reinstall deps normally
```

---

## See Also

- `docs/MAC_SETUP.md` — macOS setup
- `docs/HPC_GUIDE.md` — HPC + SLURM batch jobs
- `docs/WORKFLOWS.md` — multi-run / multi-pipeline workflows
- `../CLAUDE.md` — full developer + AI guide
- `../planning/ARCHITECTURE.md` — stage-first architecture
