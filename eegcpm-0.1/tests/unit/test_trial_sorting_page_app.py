"""End-to-end Streamlit smoke test for page 9 (Pass 2).

Runs the real page under Streamlit's ``AppTest`` against a throwaway project:
load epochs (metadata from the FIF), apply a filter, sort, split and export a
selection. This is the regression net for the page's widget wiring, which the
pure-function tests in ``test_trial_sorting_page.py`` cannot cover.
"""

import sys
import types
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import yaml

# `tests/unit/test_batch_scripts.py` installs a MagicMock for ``streamlit`` in
# ``sys.modules`` at import time. pytest imports every test module during
# collection, so when this module is collected after it, ``streamlit`` resolves
# to the mock and the real package (plus ``streamlit.components``) cannot be
# imported. Drop the mock so the real package is imported for this module (and
# for any module collected after it).
_streamlit_module = sys.modules.get("streamlit")
if _streamlit_module is not None and not isinstance(_streamlit_module, types.ModuleType):
    del sys.modules["streamlit"]

pytest.importorskip("streamlit.testing.v1")
pytest.importorskip("streamlit_local_storage")

from streamlit.testing.v1 import AppTest  # noqa: E402

from eegcpm.ui.project_manager import Project, ProjectManager  # noqa: E402

PAGE_PATH = (
    Path(__file__).resolve().parents[2]
    / "eegcpm" / "ui" / "pages" / "9_trial_sorting.py"
)
N_TRIALS = 24


def _write_task_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "task_name": "demo",
                "description": "demo",
                "task_type": "event-related",
                "conditions": [
                    {"name": "target", "event_codes": ["target"]},
                    {"name": "nontarget", "event_codes": ["nontarget"]},
                ],
                "response_mapping": {
                    "response_column": "key_resp.corr",
                    "categories": {"correct": 1, "incorrect": 0},
                    "rt_column": "key_resp.rt",
                },
                "binning": [
                    {
                        "name": "rt_quartiles",
                        "method": "quantile",
                        "n_bins": 4,
                        "column": "key_resp.rt",
                        "labels": ["a", "b", "c", "d"],
                    }
                ],
            },
            sort_keys=False,
        )
    )


def _build_project(root: Path, *, with_metadata: bool = True) -> Path:
    _write_task_config(root / "eegcpm" / "configs" / "tasks" / "demo.yaml")
    (root / "bids").mkdir(parents=True, exist_ok=True)
    (root / "derivatives" / "preprocessing" / "pipe").mkdir(parents=True, exist_ok=True)
    session = root / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
    session.mkdir(parents=True)

    n = N_TRIALS
    info = mne.create_info([f"EEG{i}" for i in range(3)], 100.0, "eeg")
    data = np.random.RandomState(0).normal(scale=1e-6, size=(n, 3, 11))
    events = np.column_stack(
        [
            np.arange(1, n + 1) * 25,
            np.zeros(n, dtype=int),
            [1 if i % 2 == 0 else 2 for i in range(n)],
        ]
    )
    metadata = pd.DataFrame(
        {
            "trial_type": ["target", "nontarget"] * (n // 2),
            "key_resp.corr": [1.0, 0.0] * (n // 2),
            "key_resp.rt": np.linspace(0.3, 0.9, n),
        }
    )
    mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.05,
        event_id={"target": 1, "nontarget": 2},
        metadata=metadata if with_metadata else None,
        verbose=False,
    ).save(session / "001_epo.fif", overwrite=True, verbose=False)
    return root


def _app(root: Path, tmp_path: Path) -> AppTest:
    at = AppTest.from_file(str(PAGE_PATH), default_timeout=120)
    at.session_state["current_project_name"] = "demo"
    at.session_state["bids_root"] = str(root / "bids")
    at.session_state["eegcpm_root"] = str(root / "eegcpm")
    pm = ProjectManager(tmp_path / "projects.yaml")
    pm.projects = [Project("demo", str(root / "bids"), str(root / "eegcpm"))]
    at.session_state["project_manager"] = pm
    return at


def _load(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.sidebar.button[0].click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _button(at: AppTest, label: str):
    matches = [b for b in at.button if label in b.label]
    assert matches, f"button {label!r} not found; have {[b.label for b in at.button]}"
    return matches[0]


class TestPageNineAppTest:
    def test_loads_epochs_with_fif_metadata(self, tmp_path):
        at = _load(_app(_build_project(tmp_path / "proj"), tmp_path))

        assert [t.label for t in at.tabs] == [
            "🏷️ By Condition",
            "✅ By Response",
            "📊 By Bins",
            "🎯 Filter",
            "↕️ Sort / Split",
            "💾 Export",
        ]
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Total Epochs"] == str(N_TRIALS)
        assert any("metadata: epochs FIF" in s.value for s in at.success)
        # No "metadata missing" warning when the FIF carries metadata.
        assert not [w for w in at.warning if "No trial metadata" in w.value]

    def test_filter_sort_split_and_export(self, tmp_path):
        root = _build_project(tmp_path / "proj")
        at = _load(_app(root, tmp_path))

        # Filter: keep only the 12 target trials.
        at.multiselect[0].select("target").run()
        _button(at, "Apply filter").click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert any("12 / 24 trials kept" in s.value for s in at.success)
        assert "filtered" in at.session_state
        assert len(at.session_state["filtered"]["epochs"]) == 12

        # Sort + split.
        _button(at, "Sort trials").click().run()
        assert any(s.value.startswith("✓ Sorted") for s in at.success)
        _button(at, "Split trials").click().run()
        assert any(s.value.startswith("✓ Split 24 trials into 2 parts")
                   for s in at.success)

        # Export the filtered selection.
        export_radio = [
            r for r in at.radio if "Filtered Selection" in r.options
        ][0]
        export_radio.set_value("Filtered Selection").run()
        _button(at, "Export Selection").click().run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert any(s.value.startswith("✓ Saved") for s in at.success)
        assert not at.error

        out_dir = (
            root / "derivatives" / "epochs" / "pipe" / "demo"
            / "selections" / "filtered" / "sub-001" / "ses-01"
        )
        assert (out_dir / "001_epo.fif").is_file()
        import json

        record = json.loads((out_dir / "selection.json").read_text())
        assert record["name"] == "filtered"
        assert record["n_outputs"] == 1
        assert record["records"][0]["label"] == "selection"
        assert record["records"][0]["n_after"] == 12
        assert len(record["records"][0]["mask_hash"]) == 64

        saved = mne.read_epochs(out_dir / "001_epo.fif", preload=False, verbose=False)
        assert len(saved) == 12
        assert set(saved.metadata["trial_type"]) == {"target"}

    def test_page_warns_and_offers_upload_without_metadata(self, tmp_path):
        at = _load(_app(_build_project(tmp_path / "proj", with_metadata=False), tmp_path))

        assert any("No trial metadata found" in w.value for w in at.warning)
        assert at.file_uploader, "no CSV upload offered"
        # The page stays usable: the condition tab still works from event ids.
        assert at.dataframe, "condition table missing"
        assert not at.exception, [str(e.value) for e in at.exception]
