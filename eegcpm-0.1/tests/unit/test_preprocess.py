"""Tests for preprocessing module."""

import pytest
import tempfile
from pathlib import Path

import numpy as np
import mne

from eegcpm.cli.preprocess import _find_input_files, read_raw_eeg
from eegcpm.data.bids_utils import find_eeg_run_files


@pytest.fixture
def sample_raw():
    """Module-level sample raw object for sharing across test classes."""
    info = mne.create_info(
        ch_names=["EEG 001", "EEG 002", "EEG 003"],
        sfreq=256.0,
        ch_types="eeg",
    )
    data = np.random.randn(3, 256 * 10)
    return mne.io.RawArray(data, info)


class TestFindInputFiles:
    """Test _find_input_files function for multi-format file detection."""

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


class TestFindEegRunFiles:
    """Tests for find_eeg_run_files in bids_utils."""

    def make_subject_dir(self, tmpdir, subject_id, sessions=None):
        """Create BIDS subject directory, optionally with session subdirectories."""
        subject_dir = Path(tmpdir) / f"sub-{subject_id}"
        if sessions:
            for ses in sessions:
                (subject_dir / f"ses-{ses}" / "eeg").mkdir(parents=True, exist_ok=True)
        else:
            (subject_dir / "eeg").mkdir(parents=True, exist_ok=True)
        return subject_dir

    def test_single_session_single_file(self):
        """Test finding a single file within one session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id, sessions=["01"])
            f = subject_dir / "ses-01" / "eeg" / f"sub-{subject_id}_ses-01_task-{task}_eeg.vhdr"
            f.touch()

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert result["file_type"] == "BrainVision"
            assert result["files"] == [f]

    def test_single_session_run_files(self):
        """Test finding run-based files (_run-*) within one session."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id, sessions=["01"])
            eeg_dir = subject_dir / "ses-01" / "eeg"
            run_files = []
            for run in ["01", "02"]:
                f = eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_run-{run}_eeg.vhdr"
                f.touch()
                run_files.append(f)

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert len(result["files"]) == 2
            assert result["files"] == sorted(run_files)

    def test_multiple_sessions_collects_all_files(self):
        """Test that files are collected across all sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id, sessions=["01", "02"])
            expected = []
            for ses in ["01", "02"]:
                f = subject_dir / f"ses-{ses}" / "eeg" / f"sub-{subject_id}_ses-{ses}_task-{task}_eeg.vhdr"
                f.touch()
                expected.append(f)

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert len(result["files"]) == 2
            assert set(result["files"]) == set(expected)

    def test_no_session_single_file(self):
        """Test finding a single file with no session directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id)
            f = subject_dir / "eeg" / f"sub-{subject_id}_task-{task}_eeg.vhdr"
            f.touch()

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert result["file_type"] == "BrainVision"
            assert result["files"] == [f]

    def test_no_session_run_files(self):
        """Test finding run-based files with no session directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id)
            eeg_dir = subject_dir / "eeg"
            run_files = []
            for run in ["01", "02", "03"]:
                f = eeg_dir / f"sub-{subject_id}_task-{task}_run-{run}_eeg.vhdr"
                f.touch()
                run_files.append(f)

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert len(result["files"]) == 3
            assert result["files"] == sorted(run_files)

    def test_returns_none_when_not_found(self):
        """Test returns None when no matching files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_dir = Path(tmpdir) / "sub-NONE"
            subject_dir.mkdir()

            result = find_eeg_run_files(subject_dir, "NONE", "rest")

            assert result is None

    def test_format_priority(self):
        """Test that FIF takes priority over other formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = self.make_subject_dir(tmpdir, subject_id, sessions=["01"])
            eeg_dir = subject_dir / "ses-01" / "eeg"
            (eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.fif").touch()
            (eeg_dir / f"sub-{subject_id}_ses-01_task-{task}_eeg.vhdr").touch()

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result["file_type"] == "FIF"

    def test_skips_session_without_eeg_dir(self):
        """Test that sessions missing an eeg/ subdirectory are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id, task = "001", "rest"
            subject_dir = Path(tmpdir) / f"sub-{subject_id}"
            (subject_dir / "ses-01").mkdir(parents=True)  # no eeg/ inside
            eeg_dir = subject_dir / "ses-02" / "eeg"
            eeg_dir.mkdir(parents=True)
            f = eeg_dir / f"sub-{subject_id}_ses-02_task-{task}_eeg.vhdr"
            f.touch()

            result = find_eeg_run_files(subject_dir, subject_id, task)

            assert result is not None
            assert result["files"] == [f]

    def test_wrong_task_not_found(self):
        """Test that files for a different task are not returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subject_id = "001"
            subject_dir = self.make_subject_dir(tmpdir, subject_id, sessions=["01"])
            eeg_dir = subject_dir / "ses-01" / "eeg"
            (eeg_dir / f"sub-{subject_id}_ses-01_task-oddball_eeg.vhdr").touch()

            result = find_eeg_run_files(subject_dir, subject_id, task="rest")

            assert result is None


class TestReadRawEeg:
    """Tests for read_raw_eeg function."""

    def test_unsupported_extension_raises_value_error(self):
        """Test that an unsupported extension raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fake_file = Path(tmpdir) / "test_eeg.xyz"
            fake_file.touch()

            with pytest.raises(ValueError, match="Unsupported EEG file extension"):
                read_raw_eeg(fake_file)

    def test_reads_fif_file(self, sample_raw):
        """Test that a valid FIF file is read successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fif_file = Path(tmpdir) / "test_eeg.fif"
            sample_raw.save(fif_file, overwrite=True, verbose=False)

            raw = read_raw_eeg(fif_file)

            assert raw is not None
            assert len(raw.ch_names) == len(sample_raw.ch_names)
