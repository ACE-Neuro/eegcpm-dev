"""Tests for eegcpm.ui.utils.bids_scanner."""

from pathlib import Path

from eegcpm.ui.utils.bids_scanner import scan_sessions


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_scan_sessions_with_session_dirs(tmp_path: Path) -> None:
    """Sessions are returned (without ses- prefix) when ses-* dirs exist."""
    _touch(tmp_path / "sub-001" / "ses-01" / "eeg" / "sub-001_ses-01_task-rest_eeg.set")
    _touch(tmp_path / "sub-001" / "ses-02" / "eeg" / "sub-001_ses-02_task-rest_eeg.set")

    assert scan_sessions(tmp_path, "001") == ["01", "02"]


def test_scan_sessions_sessionless_layout(tmp_path: Path) -> None:
    """Sessionless BIDS (sub-XXX/eeg/ directly) returns a single empty session.

    This regression test guards against the UI failing with
    'No sessions found for subject XXX' on valid sessionless datasets.
    """
    _touch(tmp_path / "sub-001" / "eeg" / "sub-001_task-rest_eeg.set")

    assert scan_sessions(tmp_path, "001") == [""]


def test_scan_sessions_missing_subject(tmp_path: Path) -> None:
    """Returns [] when the subject directory does not exist."""
    assert scan_sessions(tmp_path, "999") == []


def test_scan_sessions_subject_dir_without_eeg(tmp_path: Path) -> None:
    """Subject dir exists but has no ses-* and no eeg/ -> empty list (no fallback)."""
    (tmp_path / "sub-001" / "ecg").mkdir(parents=True)

    assert scan_sessions(tmp_path, "001") == []
