"""Tests for the ``eegcpm epochs-select`` CLI (Pass 2).

The command consumes epochs produced by the ``epochs`` stage, applies a named
selection (filters / sort / split / bins) and writes it under
``derivatives/epochs/{preprocessing}/{task}/selections/{name}/``.

Every test builds a throwaway project with a real (tiny) epochs FIF that
carries per-trial metadata — the same shape Pass 1's epochs stage writes.
"""

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import yaml

from eegcpm.cli.epochs_select import (
    add_epochs_select_parser,
    epochs_select_command,
    find_source_epochs,
    parse_rt_bounds,
    validate_selection_name,
)

N_PER_CONDITION = 30
CONDITIONS = ["target", "nontarget", "distractor"]
TRIAL_SCALE = 1e-6


# ---------------------------------------------------------------------------
# synthetic project
# ---------------------------------------------------------------------------


def _metadata(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = N_PER_CONDITION * len(CONDITIONS)
    trial_type = np.repeat(np.array(CONDITIONS), N_PER_CONDITION)
    corr = rng.choice([1.0, 0.0, np.nan], size=n, p=[0.6, 0.3, 0.1])
    rt = rng.uniform(0.2, 1.0, size=n)
    rt[np.isnan(corr)] = np.nan
    return pd.DataFrame(
        {
            "trial_type": trial_type,
            "key_resp.corr": corr,
            "key_resp.rt": rt,
            "difficulty": rng.uniform(0.0, 1.0, size=n),
            "trial_index": np.arange(n),
        }
    )


def _make_epochs(metadata: pd.DataFrame, *, with_metadata: bool = True) -> mne.EpochsArray:
    n = len(metadata)
    info = mne.create_info([f"EEG{i:03d}" for i in range(4)], 100.0, ch_types=["eeg"] * 4)
    data = np.full((n, 4, 11), TRIAL_SCALE, dtype=float)
    data *= np.arange(n)[:, None, None]  # trial i is constant i * TRIAL_SCALE
    codes = {"target": 1, "nontarget": 2, "distractor": 3}
    events = np.column_stack(
        [
            np.arange(1, n + 1) * 25,
            np.zeros(n, dtype=int),
            [codes[str(t)] for t in metadata["trial_type"]],
        ]
    )
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.05,
        event_id=codes,
        metadata=metadata if with_metadata else None,
        verbose=False,
    )


def _write_task_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "task_name": "demo",
        "description": "Synthetic demo task for epochs-select tests",
        "task_type": "event-related",
        "tmin": -0.05,
        "tmax": 0.1,
        "baseline": [-0.05, 0.0],
        "conditions": [
            {"name": c, "event_codes": [c]} for c in CONDITIONS
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
                "labels": ["fast", "medium_fast", "medium_slow", "slow"],
            }
        ],
    }
    with open(path, "w") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)


@pytest.fixture()
def project(tmp_path):
    """A throwaway project: task config + one 90-trial epochs FIF for sub-001."""
    root = tmp_path / "proj"
    _write_task_config(root / "eegcpm" / "configs" / "tasks" / "demo.yaml")
    session_dir = (
        root / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
    )
    session_dir.mkdir(parents=True)
    _make_epochs(_metadata()).save(
        session_dir / "001_epo.fif", overwrite=True, verbose=False
    )
    return root


def _args(project: Path, **overrides) -> argparse.Namespace:
    defaults = dict(
        project=project,
        preprocessing="pipe",
        task="demo",
        subjects=["001"],
        name="testsel",
        conditions=None,
        response=None,
        accuracy=None,
        rt=None,
        where=None,
        sort_by=None,
        descending=False,
        split_by=None,
        split_column=None,
        split_parts=2,
        bin_spec=None,
        average=False,
        force=False,
        dry_run=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _out_dir(project: Path, name: str = "testsel") -> Path:
    return (
        project / "derivatives" / "epochs" / "pipe" / "demo"
        / "selections" / name / "sub-001" / "ses-01"
    )


def _read_epochs(path: Path) -> mne.Epochs:
    return mne.read_epochs(path, preload=False, verbose=False)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_validate_name_accepts_safe_alphabet(self):
        for name in ["stop_fail", "rt-quartiles", "Sel01", "-_-"]:
            assert validate_selection_name(name) == name

    @pytest.mark.parametrize("name", ["", "bad name", "../escape", "a/b", "a.b", "x\n"])
    def test_validate_name_rejects_unsafe(self, name):
        with pytest.raises(ValueError, match="Invalid selection name"):
            validate_selection_name(name)

    def test_parse_rt_bounds_open_tokens(self):
        assert parse_rt_bounds(None) == (None, None)
        assert parse_rt_bounds(["none", "0.9"]) == (None, 0.9)
        assert parse_rt_bounds(["0.2", "None"]) == (0.2, None)
        assert parse_rt_bounds(["0.2", "0.9"]) == (0.2, 0.9)

    def test_parse_rt_bounds_rejects_garbage_and_inverted(self):
        with pytest.raises(ValueError, match="Invalid RT bound"):
            parse_rt_bounds(["fast", "1.0"])
        with pytest.raises(ValueError, match="greater than"):
            parse_rt_bounds(["1.0", "0.2"])

    def test_find_source_epochs_refuses_ambiguous(self, project):
        session_dir = (
            project / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
        )
        _make_epochs(_metadata()).save(
            session_dir / "stale_epo.fif", overwrite=True, verbose=False
        )
        with pytest.raises(RuntimeError, match="Multiple"):
            find_source_epochs(session_dir)

    def test_task_config_falls_back_to_packaged_copy(self, tmp_path):
        """A project without mirrored configs still resolves packaged ones."""
        from eegcpm.cli.epochs_select import resolve_task_config
        from eegcpm.core.paths import EEGCPMPaths

        paths = EEGCPMPaths(tmp_path / "no-such-project")
        config, path = resolve_task_config(paths, "cptvis")
        assert config.task_name == "cptvis"
        assert path.parent.name == "tasks"


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_prints_plan_and_writes_nothing(self, project, capsys):
        rc = epochs_select_command(
            _args(project, conditions=["target"], dry_run=True)
        )
        out = capsys.readouterr().out

        assert rc == 0
        assert "Dry run" in out
        assert "would write to" in out
        # 90 trials -> 30 target trials, reported in the plan
        assert "90 → 30 trials" in out
        # nothing on disk
        assert not _out_dir(project).exists()
        assert not (project / "derivatives" / "epochs" / "pipe" / "demo" / "selections").exists()


# ---------------------------------------------------------------------------
# selection outputs
# ---------------------------------------------------------------------------


class TestSelectionOutput:
    def test_writes_epochs_record_and_averages(self, project, capsys):
        rc = epochs_select_command(
            _args(project, conditions=["target"], average=True)
        )
        capsys.readouterr()  # silence command output
        assert rc == 0

        out_dir = _out_dir(project)
        epochs_path = out_dir / "001_epo.fif"
        record_path = out_dir / "selection.json"
        assert epochs_path.is_file()
        assert record_path.is_file()
        assert list(out_dir.glob("*_ave.fif")), "no averages written"

        epochs = _read_epochs(epochs_path)
        assert len(epochs) == N_PER_CONDITION
        assert epochs.metadata is not None
        assert set(epochs.metadata["trial_type"]) == {"target"}

        # expected averages: overall + the one condition present
        names = sorted(p.name for p in out_dir.glob("*_ave.fif"))
        assert names == ["001_ave.fif", "001_target_ave.fif"]

        payload = json.loads(record_path.read_text())
        assert payload["name"] == "testsel"
        assert payload["task"] == "demo"
        assert payload["preprocessing"] == "pipe"
        assert payload["n_source_trials"] == len(_metadata())
        assert payload["n_output_trials"] == N_PER_CONDITION
        assert payload["spec"]["conditions"] == ["target"]
        assert payload["records"], "selection.json has no SelectionRecord entries"
        record = payload["records"][0]
        for key in (
            "spec", "n_before", "n_after", "mask_hash",
            "source_epochs_hash", "timestamp",
        ):
            assert key in record
        assert record["label"] == "selection"
        assert record["n_after"] == N_PER_CONDITION
        assert len(record["mask_hash"]) == 64
        assert Path(payload["source_epochs_file"]).name == "001_epo.fif"

    def test_output_lands_in_selections_subdir(self, project):
        base_dir = (
            project / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
        )
        before = sorted(p.name for p in base_dir.glob("*_epo.fif"))

        assert epochs_select_command(_args(project, conditions=["target"])) == 0

        assert _out_dir(project).is_dir()
        assert _out_dir(project).parent.parent.parent.name == "selections"
        # the source epochs directory is untouched (still exactly one _epo.fif)
        assert sorted(p.name for p in base_dir.glob("*_epo.fif")) == before == ["001_epo.fif"]

    def test_metadata_alignment_survives_selection(self, project):
        """Selected epochs keep data/metadata alignment (trial_index marker)."""
        assert epochs_select_command(
            _args(project, where="difficulty > 0.5", sort_by="key_resp.rt")
        ) == 0

        epochs = _read_epochs(_out_dir(project) / "001_epo.fif")
        markers = np.rint(epochs.get_data()[:, 0, 0] / TRIAL_SCALE).astype(int)
        np.testing.assert_array_equal(markers, epochs.metadata["trial_index"].to_numpy())

        # sorted ascending by RT (NaN last): finite RTs are non-decreasing
        rt = epochs.metadata["key_resp.rt"].to_numpy(dtype=float)
        finite = rt[np.isfinite(rt)]
        assert np.all(np.diff(finite) >= 0)

    def test_split_by_position_writes_one_file_per_part(self, project):
        rc = epochs_select_command(
            _args(project, split_by="position", split_parts=2)
        )
        assert rc == 0

        out_dir = _out_dir(project)
        parts = sorted(out_dir.glob("*_epo.fif"))
        assert [p.name for p in parts] == ["001_half_1_epo.fif", "001_half_2_epo.fif"]
        assert [len(_read_epochs(p)) for p in parts] == [45, 45]

        payload = json.loads((out_dir / "selection.json").read_text())
        assert payload["n_outputs"] == 2
        assert payload["records"][0]["spec"]["split_by"] == "position"

    def test_bin_spec_writes_bins(self, project):
        rc = epochs_select_command(_args(project, bin_spec="rt_quartiles"))
        assert rc == 0

        out_dir = _out_dir(project)
        bins = sorted(p.name for p in out_dir.glob("*_epo.fif"))
        assert bins == [
            "001_fast_epo.fif",
            "001_medium_fast_epo.fif",
            "001_medium_slow_epo.fif",
            "001_slow_epo.fif",
        ]
        total = sum(len(_read_epochs(p)) for p in out_dir.glob("*_epo.fif"))
        # trials without an RT drop out of the quantile binning (10% are NaN)
        assert 0 < total <= len(_metadata())

    def test_response_and_accuracy_filters(self, project):
        rc = epochs_select_command(
            _args(project, conditions=["target"], accuracy=0)
        )
        assert rc == 0
        epochs = _read_epochs(_out_dir(project) / "001_epo.fif")
        assert len(epochs) > 0
        assert set(epochs.metadata["trial_type"]) == {"target"}
        assert set(epochs.metadata["key_resp.corr"]) == {0.0}

    def test_rt_filter(self, project):
        rc = epochs_select_command(_args(project, rt=["0.4", "0.6"]))
        assert rc == 0
        epochs = _read_epochs(_out_dir(project) / "001_epo.fif")
        rt = epochs.metadata["key_resp.rt"].to_numpy(dtype=float)
        assert len(rt) > 0
        assert np.all((rt >= 0.4) & (rt <= 0.6))


# ---------------------------------------------------------------------------
# refusals (rc=1, no silent success)
# ---------------------------------------------------------------------------


class TestRefusals:
    def test_bad_name_refused_before_any_write(self, project, capsys):
        rc = epochs_select_command(_args(project, name="bad name"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Invalid selection name" in out
        assert not (project / "derivatives" / "epochs" / "pipe" / "demo" / "selections").exists()

    def test_empty_selection_refused(self, project, capsys):
        rc = epochs_select_command(_args(project, where="difficulty > 10"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Selection produced no trials" in out
        assert not _out_dir(project).exists()

    def test_unknown_condition_refused(self, project, capsys):
        rc = epochs_select_command(_args(project, conditions=["stop_fail"]))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Unknown condition" in out
        assert not _out_dir(project).exists()

    def test_unknown_bin_spec_refused(self, project, capsys):
        rc = epochs_select_command(_args(project, bin_spec="nope"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Unknown bin spec" in out

    def test_missing_metadata_refused(self, project, capsys):
        session_dir = (
            project / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
        )
        # Replace the source FIF with one that has no metadata.
        (session_dir / "001_epo.fif").unlink()
        _make_epochs(_metadata(), with_metadata=False).save(
            session_dir / "001_epo.fif", overwrite=True, verbose=False
        )

        rc = epochs_select_command(_args(project, conditions=["target"]))
        out = capsys.readouterr().out
        assert rc == 1
        assert "metadata not attached" in out
        assert "re-run the epochs stage" in out
        assert not _out_dir(project).exists()

    def test_no_epochs_refused(self, project, capsys):
        session_dir = (
            project / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
        )
        (session_dir / "001_epo.fif").unlink()
        rc = epochs_select_command(_args(project, conditions=["target"]))
        out = capsys.readouterr().out
        assert rc == 1
        assert "No '*_epo.fif'" in out

    def test_multiple_epochs_refused(self, project, capsys):
        session_dir = (
            project / "derivatives" / "epochs" / "pipe" / "demo" / "sub-001" / "ses-01"
        )
        _make_epochs(_metadata()).save(
            session_dir / "999_epo.fif", overwrite=True, verbose=False
        )
        rc = epochs_select_command(_args(project, conditions=["target"]))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Multiple '*_epo.fif'" in out

    def test_existing_output_refused_then_force_overwrites(self, project, capsys):
        assert epochs_select_command(_args(project, conditions=["target"])) == 0
        first = json.loads((_out_dir(project) / "selection.json").read_text())
        epochs_path = _out_dir(project) / "001_epo.fif"
        assert epochs_path.is_file()

        rc = epochs_select_command(_args(project, conditions=["nontarget"]))
        out = capsys.readouterr().out
        assert rc == 1
        assert "already contains epochs output" in out
        # untouched: the first selection survives a refused overwrite
        assert json.loads((_out_dir(project) / "selection.json").read_text()) == first

        rc = epochs_select_command(
            _args(project, conditions=["nontarget"], force=True)
        )
        assert rc == 0, capsys.readouterr().out
        epochs = _read_epochs(epochs_path)
        assert set(epochs.metadata["trial_type"]) == {"nontarget"}

    def test_unknown_response_category_refused(self, project, capsys):
        rc = epochs_select_command(_args(project, response="maybe"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "Unknown response category" in out

    def test_missing_task_config_refused(self, project, capsys):
        rc = epochs_select_command(_args(project, task="not_a_task"))
        out = capsys.readouterr().out
        assert rc == 1
        assert "not found" in out.lower()

    def test_split_column_requires_split_by(self, project, capsys):
        rc = epochs_select_command(_args(project, split_column="key_resp.rt"))
        assert rc == 1
        assert "--split-column requires --split-by" in capsys.readouterr().out

    def test_descending_requires_sort_by(self, project, capsys):
        rc = epochs_select_command(_args(project, descending=True))
        assert rc == 1
        assert "--descending requires --sort-by" in capsys.readouterr().out

    def test_split_and_bin_are_mutually_exclusive(self, project, capsys):
        rc = epochs_select_command(
            _args(project, split_by="position", bin_spec="rt_quartiles")
        )
        assert rc == 1
        assert "mutually exclusive" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


class TestCliWiring:
    def test_main_exits_one_on_refusal(self, project, monkeypatch, capsys):
        from eegcpm.cli.main import main as cli_main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eegcpm", "epochs-select",
                "--project", str(project),
                "--preprocessing", "pipe",
                "--task", "demo",
                "--subjects", "001",
                "--name", "bad name",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 1
        assert "Invalid selection name" in capsys.readouterr().out

    def test_main_exits_zero_on_success(self, project, monkeypatch, capsys):
        from eegcpm.cli.main import main as cli_main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eegcpm", "epochs-select",
                "--project", str(project),
                "--preprocessing", "pipe",
                "--task", "demo",
                "--subjects", "001",
                "--name", "wired",
                "--conditions", "target",
            ],
        )
        cli_main()  # no SystemExit on success
        capsys.readouterr()
        assert (_out_dir(project, "wired") / "001_epo.fif").is_file()

    def test_parser_validates_accuracy_choices(self):
        parser = argparse.ArgumentParser(prog="eegcpm")
        subparsers = parser.add_subparsers(dest="command")
        add_epochs_select_parser(subparsers)

        with pytest.raises(SystemExit) as exc:
            parser.parse_args(
                [
                    "epochs-select", "--project", ".", "--preprocessing", "p",
                    "--task", "t", "--name", "n", "--accuracy", "7",
                ]
            )
        assert exc.value.code == 2
