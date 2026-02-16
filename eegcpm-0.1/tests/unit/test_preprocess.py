"""Tests for preprocessing module."""

import pytest
import tempfile
from pathlib import Path

import numpy as np
import mne

from eegcpm.cli.preprocess import _find_input_files


class TestFindInputFiles:
    """Test _find_input_files function for multi-format file detection."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample raw object for testing."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "EEG 003"],
            sfreq=256.0,
            ch_types="eeg",
        )
        data = np.random.randn(3, 256 * 10)  # 10 seconds of data
        return mne.io.RawArray(data, info)

    def create_bids_structure(self, tmpdir, subject_id, task, has_session=True):
        """Helper to create BIDS directory structure."""
        if has_session:
            eeg_dir = Path(tmpdir) / f"sub-{subject_id}" / "ses-01" / "eeg"
        else:
            eeg_dir = Path(tmpdir) / f"sub-{subject_id}" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        return eeg_dir

    def test_find_fif_single_file(self, sample_raw):
        """Test finding single FIF file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create single FIF file
            fif_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.fif"
            sample_raw.save(fif_file, overwrite=True, verbose=False)

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "FIF"
            assert len(result["files"]) == 1
            assert result["files"][0] == fif_file

    def test_find_fif_multiple_runs(self, sample_raw):
        """Test finding multiple FIF runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create multiple run files
            run_files = []
            for run_num in [1, 2, 3]:
                fif_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_run-{run_num}_eeg.fif"
                sample_raw.save(fif_file, overwrite=True, verbose=False)
                run_files.append(fif_file)

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "FIF"
            assert len(result["files"]) == 3
            # Files should be sorted
            assert result["files"] == sorted(run_files)

    def test_find_brainvision_files(self):
        """Test finding BrainVision files (.vhdr)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create BrainVision file (just .vhdr for detection)
            vhdr_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.vhdr"
            vhdr_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "BrainVision"
            assert len(result["files"]) == 1
            assert result["files"][0] == vhdr_file

    def test_find_edf_files(self):
        """Test finding EDF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create EDF file
            edf_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.edf"
            edf_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "EDF"
            assert len(result["files"]) == 1

    def test_find_bdf_files(self):
        """Test finding BDF files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create BDF file
            bdf_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.bdf"
            bdf_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "BDF"
            assert len(result["files"]) == 1

    def test_find_eeglab_files(self):
        """Test finding EEGLAB files (.set)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create EEGLAB file
            set_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.set"
            set_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "EEGLAB"
            assert len(result["files"]) == 1

    def test_find_neuroscan_files(self):
        """Test finding NeuroScan files (.cnt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create NeuroScan file
            cnt_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.cnt"
            cnt_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "NeuroScan"
            assert len(result["files"]) == 1

    def test_find_curry_files(self):
        """Test finding CURRY files (.cdt)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create CURRY file
            cdt_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.cdt"
            cdt_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "CURRY"
            assert len(result["files"]) == 1

    def test_format_priority_order(self, sample_raw):
        """Test that formats are detected in priority order (FIF first)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create both FIF and EDF files
            fif_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.fif"
            sample_raw.save(fif_file, overwrite=True, verbose=False)

            edf_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.edf"
            edf_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            # Should prefer FIF over EDF
            assert result["file_type"] == "FIF"
            assert result["files"][0] == fif_file

    def test_no_session_directory(self):
        """Test finding files in BIDS structure without session directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task, has_session=False)

            # Create file without session in path
            fif_file = eeg_dir / f"sub-{subject_id}_task-{task}_eeg.fif"
            fif_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "FIF"
            assert len(result["files"]) == 1

    def test_legacy_bids_directory(self, sample_raw):
        """Test finding files in legacy bids/ subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            # Create legacy structure
            eeg_dir = Path(tmpdir) / "bids" / f"sub-{subject_id}" / "ses-01" / "eeg"
            eeg_dir.mkdir(parents=True, exist_ok=True)

            fif_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.fif"
            sample_raw.save(fif_file, overwrite=True, verbose=False)

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "FIF"
            assert len(result["files"]) == 1

    def test_not_found_case(self):
        """Test handling when no files are found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "NONEXISTENT"
            task = "rest"

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "NOT_FOUND"
            assert len(result["files"]) == 0

    def test_return_value_structure(self):
        """Test that return value has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            vhdr_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.vhdr"
            vhdr_file.touch()

            result = _find_input_files(Path(tmpdir), subject_id, task)

            # Check structure
            assert isinstance(result, dict)
            assert "files" in result
            assert "file_type" in result
            assert isinstance(result["files"], list)
            assert isinstance(result["file_type"], str)

    def test_multiple_runs_brainvision(self):
        """Test finding multiple BrainVision runs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "oddball"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create multiple run files
            run_files = []
            for run_num in [1, 2]:
                vhdr_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_run-{run_num}_eeg.vhdr"
                vhdr_file.touch()
                run_files.append(vhdr_file)

            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "BrainVision"
            assert len(result["files"]) == 2
            assert result["files"] == sorted(run_files)

    def test_single_vs_multiple_runs(self, sample_raw):
        """Test that multiple runs take priority over single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create both single file and run files
            single_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.fif"
            sample_raw.save(single_file, overwrite=True, verbose=False)

            run_files = []
            for run_num in [1, 2]:
                run_file = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_run-{run_num}_eeg.fif"
                sample_raw.save(run_file, overwrite=True, verbose=False)
                run_files.append(run_file)

            result = _find_input_files(Path(tmpdir), subject_id, task)

            # Should find multiple runs, not single file
            assert len(result["files"]) == 2
            assert result["files"] == sorted(run_files)

    def test_different_task_not_found(self):
        """Test that files with different task names are not found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "TEST001"
            task = "rest"
            eeg_dir = self.create_bids_structure(tmpdir, subject_id, task)

            # Create file with different task name
            other_task_file = eeg_dir / f"sub-{subject_id}_ses-01_task-oddball_eeg.fif"
            other_task_file.touch()

            # Search for "rest" task
            result = _find_input_files(Path(tmpdir), subject_id, task)

            assert result["file_type"] == "NOT_FOUND"
            assert len(result["files"]) == 0
