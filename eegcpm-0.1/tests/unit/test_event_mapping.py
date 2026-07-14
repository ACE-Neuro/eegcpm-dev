"""Tests for event_mapping module — BIDS events.tsv loading and lookup."""

import pytest
from pathlib import Path

import numpy as np

from eegcpm.data.event_mapping import (
    load_events_from_bids_tsv,
    find_events_tsv,
    load_event_mapping_from_bids,
    translate_event_codes,
)


# ---------------------------------------------------------------------------
# load_events_from_bids_tsv
# ---------------------------------------------------------------------------

class TestLoadEventsFromBidsTsv:
    """Tests for load_events_from_bids_tsv()."""

    def _write_events_tsv(self, path: Path, rows: list[tuple], columns: str | None = None):
        """Helper: write a minimal BIDS events.tsv."""
        if columns is None:
            columns = "onset\tduration\ttrial_type"
        with open(path, "w") as f:
            f.write(columns + "\n")
            for row in rows:
                f.write("\t".join(str(v) for v in row) + "\n")

    def test_basic_load(self, tmp_path):
        """Load a basic events.tsv with onset and trial_type columns."""
        tsv = tmp_path / "sub-001_ses-01_task-cptaud_events.tsv"
        self._write_events_tsv(tsv, [
            (1.0, 0.0, "target"),
            (2.0, 0.0, "nontarget"),
            (3.0, 0.0, "target"),
        ])

        events, event_id = load_events_from_bids_tsv(tsv, sfreq=500.0)

        assert events is not None
        assert events.shape == (3, 3)
        assert events.dtype == int
        assert events[0, 0] == 500   # 1.0 s * 500 Hz
        assert events[1, 0] == 1000  # 2.0 s * 500 Hz
        assert events[2, 0] == 1500  # 3.0 s * 500 Hz
        assert event_id == {"nontarget": 1, "target": 2}

    def test_filter_by_phase(self, tmp_path):
        """Filter rows where phase == 'main'."""
        tsv = tmp_path / "events.tsv"
        self._write_events_tsv(tsv, [
            (0.5, 0.0, "target", "practice"),
            (1.0, 0.0, "target", "main"),
            (2.0, 0.0, "nontarget", "main"),
        ], columns="onset\tduration\ttrial_type\tphase")

        events, event_id = load_events_from_bids_tsv(
            tsv, sfreq=500.0, filter_col="phase", filter_val="main"
        )

        assert events is not None
        assert events.shape[0] == 2
        assert 0.5 * 500 not in events[:, 0]  # practice trial excluded

    def test_max_time_clip(self, tmp_path):
        """Exclude events beyond max_time."""
        tsv = tmp_path / "events.tsv"
        self._write_events_tsv(tsv, [
            (1.0, 0.0, "target"),
            (5.0, 0.0, "nontarget"),
            (8.0, 0.0, "target"),
        ])

        events, event_id = load_events_from_bids_tsv(tsv, sfreq=500.0, max_time=6.0)

        assert events is not None
        assert events.shape[0] == 2
        assert 8.0 * 500 not in events[:, 0]

    def test_file_not_found(self, tmp_path):
        """Returns (None, None) when file does not exist."""
        missing = tmp_path / "nonexistent.tsv"
        events, event_id = load_events_from_bids_tsv(missing, sfreq=500.0)
        assert events is None
        assert event_id is None

    def test_missing_onset_column(self, tmp_path):
        """Returns (None, None) when onset column is missing."""
        tsv = tmp_path / "events.tsv"
        with open(tsv, "w") as f:
            f.write("duration\ttrial_type\n")
            f.write("0.0\ttarget\n")

        events, event_id = load_events_from_bids_tsv(tsv, sfreq=500.0)
        assert events is None
        assert event_id is None

    def test_missing_trial_type_column(self, tmp_path):
        """Returns (None, None) when trial_type column is missing."""
        tsv = tmp_path / "events.tsv"
        with open(tsv, "w") as f:
            f.write("onset\tduration\n")
            f.write("1.0\t0.0\n")

        events, event_id = load_events_from_bids_tsv(tsv, sfreq=500.0)
        assert events is None
        assert event_id is None

    def test_custom_trial_type_col(self, tmp_path):
        """Use a custom column name for trial type."""
        tsv = tmp_path / "events.tsv"
        with open(tsv, "w") as f:
            f.write("onset\tduration\tevent_type\n")
            f.write("1.0\t0.0\tgo\n")
            f.write("2.0\t0.0\togo\n")

        events, event_id = load_events_from_bids_tsv(
            tsv, sfreq=500.0, trial_type_col="event_type"
        )

        assert events is not None
        assert events.shape[0] == 2
        assert "go" in event_id

    def test_empty_after_filter(self, tmp_path):
        """Returns (None, None) when no rows survive filtering."""
        tsv = tmp_path / "events.tsv"
        self._write_events_tsv(tsv, [
            (1.0, 0.0, "target", "practice"),
        ], columns="onset\tduration\ttrial_type\tphase")

        events, event_id = load_events_from_bids_tsv(
            tsv, sfreq=500.0, filter_col="phase", filter_val="main"
        )

        assert events is None
        assert event_id is None


# ---------------------------------------------------------------------------
# find_events_tsv
# ---------------------------------------------------------------------------

class TestFindEventsTsv:
    """Tests for find_events_tsv()."""

    def _create_bids_eeg(self, tmp_path: Path, subject: str, session: str, task: str, run: str | None = None):
        """Create the BIDS eeg directory and return it."""
        eeg_dir = tmp_path / f"sub-{subject}" / f"ses-{session}" / "eeg"
        eeg_dir.mkdir(parents=True, exist_ok=True)
        if run:
            tsv = eeg_dir / f"sub-{subject}_ses-{session}_task-{task}_run-{run}_events.tsv"
        else:
            tsv = eeg_dir / f"sub-{subject}_ses-{session}_task-{task}_events.tsv"
        tsv.touch()
        return tsv

    def test_with_run(self, tmp_path):
        """Find events.tsv when run number is specified."""
        expected = self._create_bids_eeg(tmp_path, "001", "01", "cptaud", run="1")
        result = find_events_tsv(tmp_path, subject="001", session="01", task="cptaud", run="1")
        assert result == expected

    def test_without_run(self, tmp_path):
        """Find events.tsv when no run number is given."""
        expected = self._create_bids_eeg(tmp_path, "001", "01", "cptaud")
        result = find_events_tsv(tmp_path, subject="001", session="01", task="cptaud")
        assert result == expected

    def test_fallback_no_run_when_run_missing(self, tmp_path):
        """Fall back to run-less events.tsv when the run-specific one is missing."""
        expected = self._create_bids_eeg(tmp_path, "001", "01", "cptaud")
        result = find_events_tsv(tmp_path, subject="001", session="01", task="cptaud", run="2")
        assert result == expected

    def test_not_found(self, tmp_path):
        """Return empty Path when no file exists."""
        result = find_events_tsv(tmp_path, subject="001", session="01", task="cptaud")
        assert result == Path()


# ---------------------------------------------------------------------------
# load_event_mapping_from_bids  (existing function, same file — bonus coverage)
# ---------------------------------------------------------------------------

class TestLoadEventMappingFromBids:
    """Tests for load_event_mapping_from_bids()."""

    def test_basic_mapping(self, tmp_path):
        """Build trial_type → value mapping."""
        tsv = tmp_path / "events.tsv"
        with open(tsv, "w") as f:
            f.write("onset\tduration\ttrial_type\tvalue\n")
            f.write("1.0\t0.0\ttarget\t1\n")
            f.write("2.0\t0.0\tnontarget\t2\n")

        mapping = load_event_mapping_from_bids(tsv)
        assert mapping == {"target": "1", "nontarget": "2"}

    def test_file_not_found(self, tmp_path):
        """Return empty dict for missing file."""
        assert load_event_mapping_from_bids(tmp_path / "nonexistent.tsv") == {}

    def test_missing_columns(self, tmp_path):
        """Return empty dict when trial_type/value columns are absent."""
        tsv = tmp_path / "events.tsv"
        tsv.write_text("onset\tduration\n1.0\t0.0\n")
        assert load_event_mapping_from_bids(tsv) == {}


# ---------------------------------------------------------------------------
# translate_event_codes  (existing function, same file — bonus coverage)
# ---------------------------------------------------------------------------

class TestTranslateEventCodes:
    """Tests for translate_event_codes()."""

    def test_translate_semantic_to_numeric(self):
        """Translate semantic names using mapping."""
        mapping = {"target": "1", "nontarget": "2"}
        result = translate_event_codes(["target", "nontarget"], mapping)
        assert result == ["1", "2"]

    def test_keep_numeric_unchanged(self):
        """Leave numeric codes as-is when not in mapping."""
        mapping = {"target": "1"}
        result = translate_event_codes(["target", "99"], mapping)
        assert result == ["1", "99"]

    def test_empty_list(self):
        """Handle empty event code list."""
        assert translate_event_codes([], {}) == []

    def test_unmapped_semantic_kept(self):
        """Keep semantic name unchanged when not in mapping."""
        result = translate_event_codes(["unknown"], {"target": "1"})
        assert result == ["unknown"]
