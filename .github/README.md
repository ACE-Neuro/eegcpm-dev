# Team Onboarding Guide

Welcome to the EEGCPM development team! 👋

## Quick Start (For New Team Members)

### 1. Accept GitHub Invitation

Check your email for an invitation to access this private repository.

### 2. Clone the Repository

```bash
git clone https://github.com/clivehywong/eegcpm-dev.git
cd eegcpm-dev
```

### 3. Install the Package

```bash
cd eegcpm-0.1
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
# Run tests (should see 263 passing)
pytest tests/ -v --override-ini="addopts="

# Try CLI
eegcpm --help

# Try UI (optional)
streamlit run eegcpm/ui/app.py --server.port 8502
```

### 5. Read Documentation

**Must read** (in order):
1. `../CLAUDE.md` - AI development guide (comprehensive overview)
2. `../planning/ARCHITECTURE.md` - System architecture
3. `CONTRIBUTING.md` - How to contribute code

**Reference**:
- `../docs/SETUP.md` - Environment setup details
- `../planning/QC_SYSTEM.md` - Quality control system
- `../eegcpm-0.1/README.md` - User-facing documentation

---

## Development Workflow Summary

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes, test
pytest tests/ -v

# 3. Commit and push
git commit -m "Add my feature"
git push -u origin feature/my-feature

# 4. Create Pull Request on GitHub
# 5. Wait for review and approval
# 6. Merge!
```

See `CONTRIBUTING.md` for full details.

---

## Project Overview

**EEGCPM** = EEG Connectome Predictive Modeling

A comprehensive Python toolbox for:
- EEG preprocessing (filtering, ICA, ASR)
- Quality control (automated QC reports)
- Epochs extraction (event-based segmentation)
- Source reconstruction (dSPM, sLORETA)
- Connectivity analysis (13 methods)
- Feature extraction & predictive modeling (CPM)

---

## Repository Structure

```
eegcpm-dev/                    # Private development repo
├── eegcpm-0.1/               # Main package (will be published separately)
│   ├── eegcpm/               # Source code
│   │   ├── cli/              # 7 CLI commands
│   │   ├── modules/          # Analysis modules
│   │   ├── ui/               # 11 Streamlit pages
│   │   └── ...
│   └── tests/                # 263 tests
├── planning/                  # Architecture documentation
├── implementation/            # Development history
└── docs/                      # User documentation
```

---

## Team Communication

- **GitHub Issues**: For bugs, feature requests
- **Pull Requests**: For code reviews
- **Discussions**: Use GitHub Discussions (if enabled)
- **Questions**: Contact @clivehywong

---

## Key Technologies

- **Python**: 3.9+
- **EEG Analysis**: MNE-Python
- **UI**: Streamlit
- **Testing**: pytest
- **Dependencies**: NumPy, SciPy, pandas, scikit-learn

---

## Common Tasks

### Add a New Analysis Module

1. Create module in `eegcpm-0.1/eegcpm/modules/`
2. Add tests in `eegcpm-0.1/tests/unit/`
3. Add CLI command in `eegcpm-0.1/eegcpm/cli/`
4. Add UI page in `eegcpm-0.1/eegcpm/ui/pages/`
5. Update documentation

### Fix a Bug

1. Create branch: `fix/bug-description`
2. Write test that fails (reproduces bug)
3. Fix bug (test now passes)
4. Submit PR

### Update Documentation

1. Update relevant `.md` files
2. If architecture changes, update `planning/ARCHITECTURE.md`
3. If new features, update `eegcpm-0.1/README.md`

---

## Getting Help

1. **Read `CLAUDE.md`** - Most questions answered here
2. **Check `planning/`** - Architecture details
3. **Ask on PR/Issue** - Team will help
4. **Contact lead** - @clivehywong

---

**Happy coding!** 🚀
