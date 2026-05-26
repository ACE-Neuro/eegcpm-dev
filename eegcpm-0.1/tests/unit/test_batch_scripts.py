"""Tests for batch preprocessing script generation functions."""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Mock only streamlit (not eegcpm.ui.utils — that's a real package other tests need)
sys.modules.setdefault('streamlit', MagicMock())

# Load the page module directly
_page_path = Path(__file__).parent.parent.parent / "eegcpm" / "ui" / "pages" / "2_batch_preprocessing.py"
_spec = importlib.util.spec_from_file_location("batch_page", _page_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

generate_local_script = _mod.generate_local_script
generate_slurm_script = _mod.generate_slurm_script


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


class TestGenerateLocal:
    """Tests for local script generation."""

    def test_local_script(self, base_config, bids_root, eegcpm_root):
        """Local script uses local paths and loops through tasks/subjects."""
        script = generate_local_script(base_config, bids_root, eegcpm_root)

        assert '/Volumes/Work/data/hbn/bids' in script
        assert 'TASKS=("rest" "contdet")' in script
        assert 'NDARAA306NT2' in script
        assert 'NDARAC123XYZ' in script
