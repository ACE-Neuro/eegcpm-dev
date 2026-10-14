"""Output discovery + stale-output cleanup (B1 rev1 — N2 / N3).

B1 renamed the preprocessed output to
``{subject}{_ses-X}{_run-Y}_preprocessed_raw.fif``. Every consumer must
therefore glob the per-run directory through the shared helper
``find_preprocessed_raw`` instead of hard-coding
``{subject_id}_preprocessed_raw.fif``:

- :class:`~eegcpm.pipeline.run_processor.RunProcessor` (both write and
  re-discovery through ``load_processed_runs``),
- :class:`~eegcpm.pipeline.epoch_combiner.EpochCombiner`,
- ``aggregate_reports`` and the ``epochs`` CLI.

A run that is about to be (re)processed must first have its stale
outputs cleared (``clear_run_outputs``): a forced re-run that fails must
not leave the previous FIF + success sidecar behind.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pytest
import yaml

from eegcpm.core.paths import clear_run_outputs, find_preprocessed_raw
from eegcpm.pipeline.epoch_combiner import EpochCombinationResult, EpochCombiner
from eegcpm.pipeline.run_processor import RunProcessingResult, RunProcessor

PIPELINE = "testpipe"


# ------------------------------------------------------- helper semantics

def test_find_preprocessed_raw_none_when_absent(tmp_path):
    assert find_preprocessed_raw(tmp_path) is None
    assert find_preprocessed_raw(tmp_path / "does_not_exist") is None


def test_find_preprocessed_raw_single_match(tmp_path):
    fif = tmp_path / "sub-01_ses-01_run-1_preprocessed_raw.fif"
    fif.write_bytes(b"")
    assert find_preprocessed_raw(tmp_path) == fif


def test_find_preprocessed_raw_rejects_ambiguity(tmp_path):
    """Two candidate FIFs must raise: picking one would silently
    consume a stale output."""
    (tmp_path / "sub-01_ses-01_run-1_preprocessed_raw.fif").write_bytes(b"")
    (tmp_path / "01_preprocessed_raw.fif").write_bytes(b"")
    with pytest.raises(RuntimeError, match="Multiple preprocessed raw FIF"):
        find_preprocessed_raw(tmp_path)


def test_clear_run_outputs_removes_only_run_artifacts(tmp_path):
    stale = [
        "sub-01_ses-01_run-1_preprocessed_raw.fif",
        "sub-01_ses-01_run-1_ica.fif",
        "sub-01_ses-01_run-1_preprocessing.json",
        "sub-01_ses-01_run-1_preprocessing_FAILED.json",
        "sub-01_ses-01_run-1_preprocessing_EXCLUDED.json",
        "sub-01_ses-01_run-1_qc_metrics.json",
        "sub-01_ses-01_run-1_preprocessed_qc.html",
        "sub-01_ses-01_run-1_preprocessed_qc.json",
    ]
    for name in stale:
        (tmp_path / name).write_bytes(b"")

    keep = ["processing_state.json", "notes.txt"]
    for name in keep:
        (tmp_path / name).write_bytes(b"")
    # A nested run directory must not be touched (non-recursive).
    nested = tmp_path / "run-2"
    nested.mkdir()
    (nested / "sub-01_ses-01_run-2_preprocessed_raw.fif").write_bytes(b"")

    clear_run_outputs(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(keep + ["run-2"])
    assert (nested / "sub-01_ses-01_run-2_preprocessed_raw.fif").exists()


# ------------------------------------------------- run_processor discovery

@pytest.fixture
def bids_tree(tmp_path):
    """Minimal BIDS tree with one saved synthetic run."""
    eeg_dir = tmp_path / "sub-01" / "ses-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    info = mne.create_info([f"EEG{i:03d}" for i in range(1, 9)], 250.0, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((8, 2500)) * 1e-6, info)
    raw.save(eeg_dir / "sub-01_ses-01_task-T_run-1_eeg.fif",
             overwrite=True, verbose=False)
    return tmp_path


def _config(steps):
    return {
        "stage": "preprocessing",
        "name": "test",
        "steps": [
            {"name": "filter", "params": {"l_freq": 1.0, "h_freq": 40.0}},
            *steps,
        ],
    }


def test_run_processor_output_found_by_load_processed_runs(bids_tree):
    """A successful run is written AND re-discovered by
    `load_processed_runs` (the same subject/session/run identity)."""
    output_root = bids_tree / "derivatives" / "preprocessing" / PIPELINE
    processor = RunProcessor(
        bids_root=bids_tree,
        output_root=output_root,
        config=_config([]),
        state_manager=None,
        verbose=False,
    )
    results = processor.process_subject_task(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)

    assert len(results) == 1 and results[0].success is True
    run_dir = output_root / "sub-01" / "ses-01" / "task-T" / "run-1"
    fifs = list(run_dir.glob("*_preprocessed_raw.fif"))
    assert len(fifs) == 1, f"expected one FIF, got {fifs}"
    assert fifs[0].name == "01_ses-01_run-1_preprocessed_raw.fif"

    loaded = processor.load_processed_runs(
        subject_id="01", task="T", session="01", pipeline=PIPELINE)
    assert len(loaded) == 1, "the processed run was not re-discovered"
    assert loaded[0].run == "1"
    assert loaded[0].success is True
    assert Path(loaded[0].output_path) == run_dir


# --------------------------------------------------- epoch_combiner lookup

def test_epoch_combiner_finds_preprocessed_fif(tmp_path, monkeypatch):
    """`combine_with_selection` must find the FIF written under the
    `{subject}_ses-X_run-Y` stem (it used to hard-code the bare
    subject-only filename and silently select no runs)."""
    run_dir = tmp_path / "sub-01" / "ses-01" / "task-T" / "run-1"
    run_dir.mkdir(parents=True)
    fif = run_dir / "01_ses-01_run-1_preprocessed_raw.fif"
    fif.write_bytes(b"")

    captured = {}

    def fake_combine_runs(self, raw_files, run_ids, subject_id, session, task,
                          pipeline="preprocessing"):
        captured["raw_files"] = list(raw_files)
        captured["run_ids"] = list(run_ids)
        return EpochCombinationResult(success=True, n_runs_combined=len(raw_files))

    monkeypatch.setattr(EpochCombiner, "combine_runs", fake_combine_runs)

    combiner = EpochCombiner(config={}, output_dir=tmp_path / "combined",
                             verbose=False)
    result = combiner.combine_with_selection(
        run_results=[RunProcessingResult(
            run="1", success=True, output_path=run_dir)],
        selection={"1": True},
        subject_id="01",
        session="01",
        task="T",
        generate_qc=False,
    )

    assert result.success is True
    assert captured["raw_files"] == [fif]
    assert captured["run_ids"] == ["1"]
