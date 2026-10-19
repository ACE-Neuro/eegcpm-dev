"""Tests for the page-9 data layer (Pass 2).

The Streamlit page itself is exercised manually; these tests cover the pure
functions it relies on — metadata resolution (FIF preferred, CSV fallback) and
the selection export (epochs + ``selection.json``), which are the parts where a
silent mistake would corrupt every downstream analysis.
"""

import importlib.util
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

PAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "eegcpm" / "ui" / "pages" / "9_trial_sorting.py"
)


@pytest.fixture(scope="module")
def page():
    spec = importlib.util.spec_from_file_location("page9_trial_sorting", PAGE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_type": ["target", "nontarget"] * (n // 2),
            "key_resp.corr": [1.0, 0.0] * (n // 2),
            "key_resp.rt": np.linspace(0.3, 0.9, n),
            "trial_index": np.arange(n),
        }
    )


def _epochs(metadata: pd.DataFrame = None, n: int = 12) -> mne.EpochsArray:
    info = mne.create_info([f"EEG{i}" for i in range(3)], 100.0, ch_types=["eeg"] * 3)
    data = np.zeros((n, 3, 11))
    events = np.column_stack(
        [
            np.arange(1, n + 1) * 25,
            np.zeros(n, dtype=int),
            [1 if i % 2 == 0 else 2 for i in range(n)],
        ]
    )
    return mne.EpochsArray(
        data, info, events=events, tmin=-0.05,
        event_id={"target": 1, "nontarget": 2},
        metadata=metadata, verbose=False,
    )


@pytest.fixture()
def session_dir(tmp_path):
    d = tmp_path / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
    d.mkdir(parents=True)
    return d


class TestLoadCombinedEpochs:
    def test_prefers_fif_metadata_over_csv(self, page, session_dir):
        metadata = _metadata()
        _epochs(metadata).save(session_dir / "001_epo.fif", overwrite=True, verbose=False)
        # A CSV that disagrees with the FIF must NOT win.
        pd.DataFrame({"trial_type": ["bogus"] * len(metadata)}).to_csv(
            session_dir / "metadata.csv", index=False
        )

        epochs, loaded, source, error = page.load_combined_epochs(session_dir)

        assert error is None
        assert source.startswith("epochs FIF")
        assert epochs.metadata is not None
        assert len(loaded) == len(metadata)
        assert set(loaded["trial_type"]) == {"target", "nontarget"}

    def test_falls_back_to_pass1_metadata_csv(self, page, session_dir):
        metadata = _metadata()
        _epochs(metadata=None, n=len(metadata)).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )
        metadata.to_csv(session_dir / "001_epochs_metadata.csv", index=False)

        epochs, loaded, source, error = page.load_combined_epochs(session_dir)

        assert error is None
        assert source.startswith("CSV")
        assert "001_epochs_metadata.csv" in source
        assert len(loaded) == len(metadata)
        # attached to the in-memory epochs so later selections keep provenance
        assert epochs.metadata is not None
        assert len(epochs.metadata) == len(epochs)

    def test_falls_back_to_legacy_metadata_csv(self, page, session_dir):
        metadata = _metadata()
        _epochs(metadata=None, n=len(metadata)).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )
        metadata.to_csv(session_dir / "metadata.csv", index=False)

        _epochs_out, loaded, source, error = page.load_combined_epochs(session_dir)
        assert error is None
        assert "metadata.csv" in source
        assert len(loaded) == len(metadata)

    def test_mismatched_csv_is_reported_not_attached(self, page, session_dir):
        _epochs(metadata=None, n=12).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )
        _metadata(n=10).to_csv(session_dir / "001_epochs_metadata.csv", index=False)

        epochs, loaded, source, error = page.load_combined_epochs(session_dir)

        assert epochs is not None  # page still usable for the Condition tab
        assert loaded is None
        assert source is None
        assert "does not match the epochs" in error
        assert "10 rows vs 12 epochs" in error
        assert epochs.metadata is None

    def test_no_metadata_returns_none_without_error(self, page, session_dir):
        _epochs(metadata=None, n=12).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )
        epochs, loaded, source, error = page.load_combined_epochs(session_dir)
        assert len(epochs) == 12
        assert loaded is None
        assert source is None
        assert error is None

    def test_multiple_epoch_files_refused(self, page, session_dir):
        _epochs(metadata=None, n=12).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )
        _epochs(metadata=None, n=12).save(
            session_dir / "stale_epo.fif", overwrite=True, verbose=False
        )
        epochs, _loaded, _source, error = page.load_combined_epochs(session_dir)
        assert epochs is None
        assert "Multiple epoch files" in error

    def test_no_epoch_files(self, page, session_dir):
        epochs, _loaded, _source, error = page.load_combined_epochs(session_dir)
        assert epochs is None
        assert "No epoch files" in error


class TestAttachMetadata:
    def test_attach_valid(self, page):
        epochs = _epochs(metadata=None, n=12)
        metadata = _metadata()
        attached = page.attach_metadata(epochs, metadata, "upload.csv")
        assert len(attached) == 12
        assert epochs.metadata is attached

    def test_attach_mismatch_raises(self, page):
        epochs = _epochs(metadata=None, n=12)
        with pytest.raises(ValueError, match="10 rows but the epochs have 12"):
            page.attach_metadata(epochs, _metadata(n=10), "upload.csv")


class TestExportSelection:
    def test_writes_epochs_and_record(self, page, tmp_path):
        from eegcpm.core.task_config import SelectionRecord

        epochs = _epochs(metadata=_metadata(), n=12)
        record = SelectionRecord(
            spec={"conditions": ["target"]},
            n_before=12,
            n_after=6,
            mask_hash="a" * 64,
            source_epochs_hash="b" * 64,
            timestamp="2026-10-20T00:00:00+00:00",
        )
        out_dir = tmp_path / "selections" / "ui_filter" / "sub-001" / "ses-01"

        result = page.export_selection(
            out_dir,
            name="ui_filter",
            subject_id="001",
            session="01",
            task="demo",
            entries=[("selection", epochs, [record])],
            source_epochs_file="/x/001_epo.fif",
        )

        assert result["files"] == ["001_epo.fif"]
        saved = mne.read_epochs(out_dir / "001_epo.fif", preload=False, verbose=False)
        assert len(saved) == 12
        assert saved.metadata is not None

        payload = json.loads((out_dir / "selection.json").read_text())
        assert payload["name"] == "ui_filter"
        assert payload["created_by"] == "ui:9_trial_sorting"
        assert payload["records"][0]["label"] == "selection"
        assert payload["records"][0]["n_after"] == 6

    def test_refuses_existing_without_overwrite(self, page, tmp_path):
        epochs = _epochs(metadata=_metadata(), n=12)
        kwargs = dict(
            name="sel",
            subject_id="001",
            session="01",
            task="demo",
            entries=[("selection", epochs, [])],
            source_epochs_file="/x/001_epo.fif",
        )
        page.export_selection(tmp_path / "out", **kwargs)
        with pytest.raises(FileExistsError, match="already contains epochs output"):
            page.export_selection(tmp_path / "out", **kwargs)
        # explicit overwrite succeeds
        page.export_selection(tmp_path / "out", overwrite=True, **kwargs)

    def test_invalid_name_rejected_by_page_helper(self, page):
        assert page._valid_selection_name("ui_filter-1")
        assert not page._valid_selection_name("bad name")
        assert not page._valid_selection_name("../escape")
