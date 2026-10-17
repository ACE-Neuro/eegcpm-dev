"""Tests for batch preprocessing script generation functions.

The generators live in :mod:`eegcpm.pipeline.slurm` (there is exactly ONE HPC
implementation); the Streamlit page only imports and previews them.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock only streamlit (not eegcpm.ui.utils — that's a real package other tests need)
sys.modules.setdefault('streamlit', MagicMock())

from eegcpm.pipeline.slurm import generate_local_script, generate_slurm_script  # noqa: E402

_PAGE_PATH = Path(__file__).parent.parent.parent / "eegcpm" / "ui" / "pages" / "2_batch_preprocessing.py"


@pytest.fixture
def base_config():
    """Minimal batch config for testing."""
    return {
        'pipeline_name': 'standard',
        'subjects': ['NDARAA306NT2', 'NDARAB678VYW', 'NDARAC123XYZ'],
        'tasks': ['rest', 'contdet'],
        'config_file': '/local/path/config/preprocessing/standard.yaml',
        'parallel_jobs': 10,
        'force': False,
        'hpc': {
            'bids_root': '/share/ps_clivewong/25SusAttn/bids',
            'eegcpm_root': '/share/ps_clivewong/eegcpm-dev/eegcpm-0.1',
            'conda_env': 'eegcpm',
            'email': 'user@eduhk.hk',
            'partition': 'shared_cpu',
            'time': '02:00:00',
            'mem': '16G',
            'cpus': 4,
        }
    }


@pytest.fixture
def bids_root():
    return Path('/Volumes/Work/data/hbn/bids')


@pytest.fixture
def eegcpm_root():
    return Path('/Users/clive/eegcpm/eegcpm-0.1')


class TestGenerateSlurm:
    """Tests for SLURM script generation."""

    def test_uses_hpc_paths(self, base_config, bids_root, eegcpm_root):
        """Script uses HPC paths, not local paths."""
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)

        assert '/share/ps_clivewong/25SusAttn/bids' in script
        assert '/share/ps_clivewong/eegcpm-dev/eegcpm-0.1' in script
        assert '/Volumes/Work' not in script

    def test_uses_placeholders_when_hpc_empty(self, base_config, bids_root, eegcpm_root):
        """Placeholder paths when HPC settings not configured."""
        base_config['hpc'] = {'bids_root': '', 'eegcpm_root': ''}
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)

        assert '/path/to/your/bids/on/hpc' in script
        assert '/path/to/eegcpm-0.1/on/hpc' in script

    def test_array_and_tasks(self, base_config, bids_root, eegcpm_root):
        """Array size matches subjects and task loop is correct."""
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)

        assert '--array=0-2%10' in script
        assert 'TASKS=("rest" "contdet")' in script
        assert 'for TASK in "${TASKS[@]}"' in script

    def test_force_flag(self, base_config, bids_root, eegcpm_root):
        """--force appears only when enabled."""
        script_no_force = generate_slurm_script(base_config, bids_root, eegcpm_root)
        assert '--force' not in script_no_force

        base_config['force'] = True
        script_force = generate_slurm_script(base_config, bids_root, eegcpm_root)
        assert '--force' in script_force

    def test_email_handling(self, base_config, bids_root, eegcpm_root):
        """Email included when set, commented when empty."""
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)
        assert '#SBATCH --mail-user=user@eduhk.hk' in script

        base_config['hpc']['email'] = ''
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)
        assert '# #SBATCH --mail-user=' in script

    def test_header_block_is_pinned(self, base_config, bids_root, eegcpm_root):
        """The #SBATCH header block (order included) is byte-stable."""
        script = generate_slurm_script(base_config, bids_root, eegcpm_root)

        assert script.startswith(
            "#!/bin/bash\n"
            "#SBATCH --job-name=eegcpm_standard\n"
            "#SBATCH --partition=shared_cpu\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --cpus-per-task=4\n"
            "#SBATCH --mem=16G\n"
            "#SBATCH --time=02:00:00\n"
            "#SBATCH --array=0-2%10\n"
            "#SBATCH --output=/share/ps_clivewong/25SusAttn/derivatives/preprocessing/"
            "standard/slurm_%A_%a.out\n"
            "#SBATCH --error=/share/ps_clivewong/25SusAttn/derivatives/preprocessing/"
            "standard/slurm_%A_%a.err\n"
            "#SBATCH --mail-type=END,FAIL\n"
            "#SBATCH --mail-user=user@eduhk.hk\n"
            "\n"
            "# EEGCPM SLURM Batch Preprocessing\n"
        )


class TestGenerateLocal:
    """Tests for local script generation."""

    def test_local_script(self, base_config, bids_root, eegcpm_root):
        """Local script uses local paths and loops through tasks/subjects."""
        script = generate_local_script(base_config, bids_root, eegcpm_root)

        assert '/Volumes/Work/data/hbn/bids' in script
        assert 'TASKS=("rest" "contdet")' in script
        assert 'NDARAA306NT2' in script
        assert 'NDARAC123XYZ' in script


class TestNoInlineGenerators:
    """The pages must not carry their own copy of an sbatch generator."""

    def _function_names(self, page: Path) -> set:
        tree = ast.parse(page.read_text(encoding="utf-8"))
        return {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }

    def test_batch_page_has_no_local_generators(self):
        names = self._function_names(_PAGE_PATH)
        assert "generate_slurm_script" not in names, (
            "ui/pages/2_batch_preprocessing.py must import generate_slurm_script "
            "from eegcpm.pipeline.slurm instead of defining its own copy"
        )
        assert "generate_local_script" not in names

    def test_pages_contain_no_sbatch_headers(self):
        """No page may inline a #SBATCH header (one implementation only).

        ``_archived/`` is deliberately out of scope: archives are history, not
        code that runs (the archived mock HPC page must not be revived as a
        second copy of the unified module).
        """
        pages_dir = _PAGE_PATH.parent
        offenders = []
        for page in sorted(pages_dir.glob("*.py")):
            for lineno, line in enumerate(
                page.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if "#SBATCH" in line:
                    offenders.append(f"{page.name}:{lineno}")
        assert offenders == [], (
            "inline #SBATCH headers in UI pages: "
            + ", ".join(offenders)
            + " — use eegcpm.pipeline.slurm.generate_*_script instead"
        )
