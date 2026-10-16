"""CLI argument validation that must fail in argparse (exit 2), not at runtime.

Plan §B7 tasks 11 and 14:

- **features (B4-K6):** ``--input-dir`` was optional for every
  ``--feature-type``, so a non-``source_erp`` type without it (or a
  ``source_erp`` run without a resolvable ``--source`` variant) failed with a
  ``FileNotFoundError`` traceback instead of a usage message;
- **source-reconstruct (B5-N5):** an explicitly requested task config that is
  missing/unreadable must exit 1 (not warn and fall back to per-event
  processing), and a batch with failed sessions must exit non-zero.
"""

import argparse
import sys
from pathlib import Path

import pytest

from eegcpm.cli.features import add_features_parser, validate_features_args
from eegcpm.cli.main import main as cli_main


def _features_parser():
    parser = argparse.ArgumentParser(prog="eegcpm")
    subparsers = parser.add_subparsers(dest="command")
    return parser, add_features_parser(subparsers)


def _features_args(**overrides):
    defaults = dict(
        project=Path("."),
        config=Path("c.yaml"),
        output_dir=Path("out"),
        feature_type="specparam",
        input_dir=None,
        preprocessing=None,
        task=None,
        source=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestFeaturesArgparseValidation:
    def test_missing_input_dir_for_specparam_exits_two(self):
        parser, features_parser = _features_parser()
        with pytest.raises(SystemExit) as exc:
            validate_features_args(features_parser, _features_args(feature_type="specparam"))
        assert exc.value.code == 2

    @pytest.mark.parametrize(
        "feature_type", ["specparam", "connectivity", "isc", "drowsiness"]
    )
    def test_input_dir_required_for_non_source_types(self, feature_type):
        parser, features_parser = _features_parser()
        with pytest.raises(SystemExit) as exc:
            validate_features_args(
                features_parser, _features_args(feature_type=feature_type)
            )
        assert exc.value.code == 2

    def test_input_dir_present_passes(self):
        parser, features_parser = _features_parser()
        validate_features_args(
            features_parser, _features_args(input_dir=Path("inp"))
        )

    def test_source_erp_without_source_variant_exits_two(self):
        parser, features_parser = _features_parser()
        with pytest.raises(SystemExit) as exc:
            validate_features_args(
                features_parser,
                _features_args(feature_type="source_erp", preprocessing="hbn"),
            )
        assert exc.value.code == 2

    def test_source_erp_with_full_discovery_passes(self):
        parser, features_parser = _features_parser()
        validate_features_args(
            features_parser,
            _features_args(
                feature_type="source_erp",
                preprocessing="hbn",
                task="RestingState",
                source="dSPM-CONN32",
            ),
        )

    def test_source_erp_with_input_dir_passes(self):
        parser, features_parser = _features_parser()
        validate_features_args(
            features_parser,
            _features_args(feature_type="source_erp", input_dir=Path("inp")),
        )

    def test_cli_exits_two_with_usage(self, capsys, monkeypatch, tmp_path):
        """End to end through `main()`: exit code 2 and a usage message."""
        config = tmp_path / "c.yaml"
        config.write_text("stage: features\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eegcpm", "features",
                "--project", str(tmp_path),
                "--config", str(config),
                "--feature-type", "specparam",
                "--output-dir", str(tmp_path / "out"),
            ],
        )
        with pytest.raises(SystemExit) as exc:
            cli_main()
        assert exc.value.code == 2
        assert "usage:" in capsys.readouterr().err


class TestSourceReconstructCliFailures:
    def test_missing_explicit_task_config_exits_one(self, tmp_path, monkeypatch, capsys):
        """B5-N5: a requested-but-missing task config fails the run (exit 1)."""
        from eegcpm.cli import source_reconstruct as src_cli

        config = tmp_path / "source.yaml"
        config.write_text(
            "stage: source\nmethod: dSPM\ndepends_on:\n  preprocessing: hbn\n"
            "  task: RestingState\n"
        )
        # Minimal epochs tree so the CLI gets past the directory checks.
        epochs_dir = (
            tmp_path / "derivatives" / "epochs" / "hbn" / "RestingState"
            / "sub-01" / "ses-01"
        )
        epochs_dir.mkdir(parents=True)

        args = argparse.Namespace(
            project=tmp_path,
            config=config,
            preprocessing="hbn",
            task="RestingState",
            task_config="does_not_exist",
            subjects=None,
            sessions=None,
        )
        with pytest.raises(SystemExit) as exc:
            src_cli.source_reconstruct_command(args)
        assert exc.value.code == 1
        assert "not found" in capsys.readouterr().err

    def test_failed_session_exits_non_zero(self, tmp_path, monkeypatch, capsys):
        """A session with no epoch file is counted failed → exit 1 (B5-N5)."""
        from eegcpm.cli import source_reconstruct as src_cli

        config = tmp_path / "source.yaml"
        config.write_text(
            "stage: source\nmethod: dSPM\ndepends_on:\n  preprocessing: hbn\n"
            "  task: RestingState\n"
        )
        session_dir = (
            tmp_path / "derivatives" / "epochs" / "hbn" / "RestingState"
            / "sub-01" / "ses-01"
        )
        session_dir.mkdir(parents=True)  # no *_epo.fif inside

        args = argparse.Namespace(
            project=tmp_path,
            config=config,
            preprocessing="hbn",
            task="RestingState",
            task_config="none",
            subjects=None,
            sessions=None,
        )
        with pytest.raises(SystemExit) as exc:
            src_cli.source_reconstruct_command(args)
        assert exc.value.code == 1


class TestConnectivityTaskConfigReporting:
    """B7-A1: the connectivity CLI reports task configs it could not read.

    The auto-detect loop used to ``except Exception: continue`` (the file
    vanished without a trace) and the load failure only printed. Both are now
    narrowed, logged and listed in a "Task config problems" summary.
    """

    @staticmethod
    def _args(tmp_path, task_config=None):
        config = tmp_path / "conn.yaml"
        config.write_text(
            "stage: connectivity\nmethods: [plv]\nfrequency_bands: {}\n"
            "time_windows: []\ndepends_on:\n  preprocessing: hbn\n"
            "  task: RestingState\n  source: dSPM-CONN32\n"
        )
        return argparse.Namespace(
            project=tmp_path,
            config=config,
            preprocessing="hbn",
            task="RestingState",
            source="dSPM-CONN32",
            space="source",
            subjects=None,
            sessions=None,
            task_config=task_config,
            output_pipeline=None,
        )

    def test_unreadable_config_is_reported_not_dropped(self, tmp_path, capsys):
        from eegcpm.cli.connectivity import connectivity_command

        tasks_dir = tmp_path / "eegcpm" / "configs" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "broken.yaml").write_text("task_name: [unclosed\n")

        # No source directory in the tmp project: the command stops there, but
        # the dropped config must already be reported.
        with pytest.raises(FileNotFoundError):
            connectivity_command(self._args(tmp_path))
        out = capsys.readouterr().out
        assert "Task config problems" in out
        assert "broken.yaml" in out
        assert "unreadable" in out

    def test_unloadable_explicit_config_is_reported(self, tmp_path, capsys):
        from eegcpm.cli.connectivity import connectivity_command

        tasks_dir = tmp_path / "eegcpm" / "configs" / "tasks"
        tasks_dir.mkdir(parents=True)
        # Valid YAML, but not a TaskConfig (missing required fields).
        (tasks_dir / "half.yaml").write_text("task_name: RestingState\n")

        with pytest.raises(FileNotFoundError):
            connectivity_command(self._args(tmp_path, task_config="half"))
        out = capsys.readouterr().out
        assert "Could not load task config" in out
        assert "Task config problems" in out
        assert "half.yaml" in out
