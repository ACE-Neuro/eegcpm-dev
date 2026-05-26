# Contributing to EEGCPM Development

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/clivehywong/eegcpm-dev.git
cd eegcpm-dev
```

### 2. Install Dependencies

```bash
cd eegcpm-0.1
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests
pytest tests/ -v --override-ini="addopts="

# Should see: 263 tests passing
```

---

## Development Workflow

### Working on a New Feature

```bash
# 1. Create a new branch from main
git checkout main
git pull
git checkout -b feature/your-feature-name

# 2. Make your changes
# ... edit files ...

# 3. Run tests
pytest tests/ -v

# 4. Commit changes
git add .
git commit -m "Description of your changes

- Bullet point of what changed
- Another change
"

# 5. Push your branch
git push -u origin feature/your-feature-name
```

### Creating a Pull Request

1. Go to: https://github.com/clivehywong/eegcpm-dev/pulls
2. Click **"New pull request"**
3. Select your branch: `feature/your-feature-name` → `main`
4. Fill in description:
   - What does this PR do?
   - What files changed?
   - Any testing done?
5. Request review from @clivehywong or team lead
6. Wait for approval before merging

---

## Code Standards

### Python Style

- Follow **PEP 8** style guide
- Line length: **100 characters** (not 80)
- Use **Google-style docstrings**

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value
    """
    return True
```

### Testing Requirements

- All new features **must have tests**
- Tests go in `eegcpm-0.1/tests/unit/` or `tests/`
- Run full test suite before submitting PR
- Aim for >80% code coverage

### Documentation

- Update `CLAUDE.md` if architecture changes
- Add docstrings to all public functions/classes
- Update relevant files in `planning/` for major changes

---

## Project Structure

```
eegcpm-dev/
├── CLAUDE.md              # AI development guide - READ THIS FIRST
├── docs/                  # User documentation
├── planning/              # Architecture docs
│   ├── ARCHITECTURE.md    # Core architecture
│   ├── QC_SYSTEM.md       # Quality control
│   └── ...
├── implementation/        # Historical records
└── eegcpm-0.1/           # Main package
    ├── eegcpm/           # Source code
    │   ├── cli/          # Command-line interface
    │   ├── modules/      # Analysis modules
    │   └── ui/           # Streamlit UI
    └── tests/            # Test suite
```

---

## Key Commands

### Running the CLI

```bash
# Check status
eegcpm status --project /path/to/project

# Run preprocessing
eegcpm preprocess --config configs/preprocessing/standard.yaml --project /path/to/project

# See all commands
eegcpm --help
```

### Running the UI

```bash
streamlit run eegcpm/ui/app.py --server.port 8502
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/unit/test_epochs.py -v

# With coverage
pytest tests/ --cov=eegcpm --cov-report=html
```

---

## Common Issues

### Import Errors

If you get `ModuleNotFoundError: No module named 'eegcpm'`:

```bash
cd eegcpm-0.1
pip install -e ".[dev]"
```

### Test Failures

Some tests may fail if MNE sample data is not downloaded:
```bash
python -c "import mne; mne.datasets.sample.data_path()"
```

### Git Conflicts

If you have merge conflicts:
```bash
git pull origin main
# Resolve conflicts in your editor
git add .
git commit -m "Resolve merge conflicts"
```

---

## Need Help?

- **Read first**: `CLAUDE.md` (AI development guide)
- **Architecture questions**: See `planning/ARCHITECTURE.md`
- **Bugs**: Open an issue on GitHub
- **Questions**: Ask @clivehywong or team lead

---

## Pull Request Checklist

Before submitting a PR, verify:

- [ ] Code follows PEP 8 style
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] No debug print statements or commented code
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

---

## Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/fixes

Examples:
- `feature/add-deep-learning`
- `fix/epochs-rejection-bug`
- `docs/update-readme-examples`
- `refactor/cleanup-qc-modules`

---

**Thank you for contributing to EEGCPM!** 🎉
