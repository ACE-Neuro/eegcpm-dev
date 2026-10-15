"""B5 source hardening + provenance (P2-F10/F11/F12/F13, B4-K5/K1/N4/N5).

Covers plan §B5:

- task 1/5 — rank logging and the provenance JSON (every key of the plan's list,
  the per-ROI flip summary, ``excluded_reasons``, ``config_hash``) plus the
  reloadable ``-cov.fif`` / ``-inv.fif`` artifacts;
- task 2 — ``_resolve_method_params`` (eLORETA loose orientation needs
  ``force_equal``) and the single-trial ``lambda2 = 1/snr_trial**2`` spy. Per the
  B4-N5 note the spy is parametrized over the default and a **non-default**
  ``snr_trial``: a missing kwarg falls back to the evoked lambda2, so the default
  1.0 alone would not catch it (the non-default case is the discriminating one);
- task 3 — 3-layer BEM only (``_load_bem``), the forward cache;
- task 4 — the montage pre-flight;
- task 6 — ``_select_epoch_file`` (0 or >1 ``*_epo.fif`` is a loud failure);
- task 7 — condition matching by event-code **value** + numeric fallback;
- task 8 — atomic / stale-safe writes;
- task 9 — the features CLI reports missing inputs instead of skipping them.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np
import pytest
from mne.io.constants import FIFF

import eegcpm.modules.source as source_module
from eegcpm.cli.source_reconstruct import _select_epoch_file
from eegcpm.core.provenance import (
    provenance_path_for_archive,
    sha256_file,
    sha256_json,
    verify_archive_provenance,
)
from eegcpm.modules.source import SourceReconstructionModule

SFREQ = 250.0
T_RANGE = (-0.2, 0.4)
GSN_MONTAGE = "GSN-HydroCel-129"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gsn_info(sfreq=SFREQ) -> mne.Info:
    montage = mne.channels.make_standard_montage(GSN_MONTAGE)
    info = mne.create_info(montage.ch_names, sfreq, "eeg")
    info.set_montage(montage, verbose=False)
    return info


def _tiny_epochs(events, event_id, n_channels=4, sfreq=SFREQ) -> mne.BaseEpochs:
    """Montage-less epochs for condition-selection tests (no forward needed)."""
    data = np.zeros((len(events), n_channels, 50))
    info = mne.create_info(
        [f"EEG{i:03d}" for i in range(n_channels)], sfreq, "eeg"
    )
    return mne.EpochsArray(
        data,
        info,
        events=np.asarray(events, dtype=int),
        event_id=event_id,
        tmin=-0.2,
        verbose=False,
    )


def _source_epochs(info, n_epochs=12, seed=0, tmin=T_RANGE[0], tmax=T_RANGE[1]):
    """Baseline-corrected single-condition epochs on the GSN montage."""
    rng = np.random.default_rng(seed)
    n_times = int(round((tmax - tmin) * SFREQ)) + 1
    data = rng.standard_normal((n_epochs, len(info["chs"]), n_times)) * 1e-5
    events = np.column_stack(
        [
            np.arange(n_epochs) * int(SFREQ),
            np.zeros(n_epochs, dtype=int),
            np.ones(n_epochs, dtype=int),
        ]
    )
    epochs = mne.EpochsArray(
        data,
        info.copy(),
        events=events,
        event_id={"stim": 1},
        tmin=tmin,
        baseline=(None, 0),
        verbose=False,
    )
    epochs.set_eeg_reference(projection=True, verbose=False)
    return epochs


@pytest.fixture(scope="module")
def hardening_forward(fsaverage_subjects_dir, tmp_path_factory):
    """GSN-129 info + the module's fsaverage forward, built **through the module**.

    Building it via ``_get_template_forward`` populates the class-level cache, so
    the tests below can run with ``forward=None`` (exercising the pre-flight and
    the cache path) without paying for a second ~10 s forward build.
    """
    subjects_dir = Path(fsaverage_subjects_dir)
    info = _gsn_info()
    module = SourceReconstructionModule(
        {"method": "dSPM", "subjects_dir": str(subjects_dir)},
        tmp_path_factory.mktemp("hardening_forward"),
    )
    forward, _warnings, montage_check = module._get_template_forward(info)
    assert montage_check["status"] == "ok"
    return {
        "info": info,
        "forward": forward,
        "subjects_dir": subjects_dir,
        "montage_check": montage_check,
    }


# ---------------------------------------------------------------------------
# Task 2: inverse parameters
# ---------------------------------------------------------------------------


class TestMethodParams:
    """P2-F12: eLORETA with a loose orientation constraint needs force_equal."""

    def test_eloreta_loose_gets_force_equal(self):
        assert SourceReconstructionModule._resolve_method_params(
            "eLORETA", 0.2, None
        ) == {"force_equal": True}

    def test_non_eloreta_returns_none(self):
        assert SourceReconstructionModule._resolve_method_params("dSPM", 0.2, None) is None
        assert SourceReconstructionModule._resolve_method_params("sLORETA", 0.2, None) is None
        assert SourceReconstructionModule._resolve_method_params("MNE", 0.2, None) is None

    def test_eloreta_fixed_or_free_orientation_needs_no_flag(self):
        # loose=0 (fixed) and loose=1 (free) are not the loose-in-between case.
        assert SourceReconstructionModule._resolve_method_params("eLORETA", 0.0, None) is None
        assert SourceReconstructionModule._resolve_method_params("eLORETA", 1.0, None) is None

    def test_user_params_win_and_pass_through(self):
        user = {"force_equal": False, "eps": 1e-3}
        resolved = SourceReconstructionModule._resolve_method_params("eLORETA", 0.2, user)
        assert resolved == {"force_equal": False, "eps": 1e-3}
        assert user == {"force_equal": False, "eps": 1e-3}  # not mutated
        # non-eLORETA methods pass the user dict through untouched
        assert SourceReconstructionModule._resolve_method_params("dSPM", 0.2, user) is user


# ---------------------------------------------------------------------------
# Task 3/4: BEM + montage pre-flight
# ---------------------------------------------------------------------------


class TestBemAndPreflight:
    def test_load_bem_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="BEM"):
            SourceReconstructionModule._load_bem(tmp_path)

    def test_preflight_missing_montage_raises(self, tmp_path):
        info = mne.create_info(["Fz", "Cz"], SFREQ, "eeg")  # loc[:3] all zero
        with pytest.raises(ValueError, match="montage"):
            SourceReconstructionModule._preflight_montage(info, tmp_path)

    def test_preflight_requires_cardinal_fiducials(self, tmp_path):
        montage = mne.channels.make_dig_montage(
            ch_pos={"Fz": (0.0, 0.1, 0.05), "Cz": (0.0, 0.0, 0.1)},
            coord_frame="head",
        )
        info = mne.create_info(["Fz", "Cz"], SFREQ, "eeg")
        info.set_montage(montage, verbose=False)
        # mne 1.13.2 ``set_montage`` fabricates LPA/Nasion/RPA cardinal points from
        # the channel positions (probed 2026-10-15: ±0.106 m lateral, 0.106 m
        # nasion), so strip them to model a digitisation without fiducials (e.g. a
        # sensor layout copied from a template, or a dig-less FIF import).
        with info._unlock():
            info["dig"] = [
                point
                for point in info["dig"]
                if int(point["kind"]) != int(FIFF.FIFFV_POINT_CARDINAL)
            ]
        with pytest.raises(ValueError, match="fiducials"):
            SourceReconstructionModule._preflight_montage(info, tmp_path)

    def test_preflight_rejects_non_head_dig_frame(self, tmp_path):
        """Every digitised point must be in the head frame (P2-F13)."""
        info = _gsn_info()
        for point in info["dig"]:
            point["coord_frame"] = int(FIFF.FIFFV_COORD_MRI)
        with pytest.raises(ValueError, match="head frame"):
            SourceReconstructionModule._preflight_montage(info, tmp_path)

    @pytest.mark.fsaverage
    def test_preflight_gsn_passes(self, fsaverage_subjects_dir):
        check = SourceReconstructionModule._preflight_montage(
            _gsn_info(), fsaverage_subjects_dir
        )
        assert check["status"] == "ok"
        assert check["n_electrodes"] == 129
        # Probed 2026-10-13: GSN-HydroCel-129 on fsaverage is 7.2 mm median /
        # 18.7 mm max, i.e. inside both thresholds (warn 10 / fail 25).
        assert check["median_mm"] < 10.0
        assert check["max_mm"] < 25.0
        assert check["warnings"] == []

    @pytest.mark.fsaverage
    def test_preflight_shifted_positions_fail(self, fsaverage_subjects_dir):
        info = _gsn_info()
        for channel in info["chs"]:
            channel["loc"][:3] = np.asarray(channel["loc"][:3], dtype=float) + np.array(
                [0.0, 0.0, 0.040]
            )
        with pytest.raises(ValueError, match="alignment"):
            SourceReconstructionModule._preflight_montage(
                info, fsaverage_subjects_dir
            )

    @pytest.mark.fsaverage
    @pytest.mark.slow
    def test_forward_cache_key_is_montage_and_spacing_specific(
        self, hardening_forward, tmp_path
    ):
        subjects_dir = hardening_forward["subjects_dir"]
        module = SourceReconstructionModule(
            {"subjects_dir": str(subjects_dir)}, tmp_path
        )
        key = module._forward_cache_key(hardening_forward["info"], "oct6", subjects_dir)
        assert key in SourceReconstructionModule._FORWARD_CACHE
        other = module._forward_cache_key(hardening_forward["info"], "oct5", subjects_dir)
        assert other != key
        assert other not in SourceReconstructionModule._FORWARD_CACHE

        # Montage half of the key (B5-A7): shifted electrode positions must
        # produce a different key (previously only code-inspected).
        shifted = hardening_forward["info"].copy()
        shifted["chs"][0]["loc"][:3] = shifted["chs"][0]["loc"][:3] + 5e-4
        moved = module._forward_cache_key(shifted, "oct6", subjects_dir)
        assert moved != key


# ---------------------------------------------------------------------------
# Task 2: single-trial lambda2 + pick_ori spy (B4-N5)
# ---------------------------------------------------------------------------


@pytest.mark.fsaverage
@pytest.mark.slow
@pytest.mark.parametrize(
    "snr_trial, expected_lambda2",
    [(1.0, 1.0), (2.0, 0.25)],
    ids=["default_snr_trial", "non_default_snr_trial"],
)
def test_trial_lambda2_pick_ori_and_method_params(
    hardening_forward, tmp_path, monkeypatch, snr_trial, expected_lambda2
):
    """The trial inversion passes ``1/snr_trial**2``, not the evoked lambda2.

    The non-default case (``snr_trial=2.0`` → 0.25) is the one that catches a
    missing kwarg (B4-N5): the default 1.0 equals the fallback value, so only the
    non-default assertion distinguishes "wired" from "silently using the default".
    Both are pinned here.
    """
    original = source_module.apply_inverse_epochs
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(source_module, "apply_inverse_epochs", spy)

    module = SourceReconstructionModule(
        {
            "method": "dSPM",
            "mode": "trials",
            "min_trials": 5,
            "inverse": {"snr": 3.0, "snr_trial": snr_trial},
            "subjects_dir": str(hardening_forward["subjects_dir"]),
            "output": {"generate_qc": False, "save_stc": False},
        },
        tmp_path,
    )
    result = module.process(
        _source_epochs(hardening_forward["info"], n_epochs=8),
        forward=hardening_forward["forward"],
    )
    assert result.success, result.errors
    assert calls, "apply_inverse_epochs was never called"

    for args, kwargs in calls:
        assert args[2] == pytest.approx(expected_lambda2)
        assert kwargs["pick_ori"] == "normal"
        assert kwargs["return_generator"] is True
        assert kwargs["method"] == "dSPM"
        assert kwargs["method_params"] is None

    trials = np.load(tmp_path / "unknown_roi_tc_trials.npz")
    assert trials["stim"].ndim == 3
    assert trials["stim"].shape[0] == 8

    # B5-A2: provenance must not claim template assets when the caller supplied
    # the forward.
    import json

    provenance = json.loads((tmp_path / "unknown_provenance.json").read_text())
    assert provenance["forward_source"] == "caller"
    assert provenance["bem"] is None
    assert provenance["trans"] is None
    assert provenance["spacing"] is None


@pytest.mark.fsaverage
@pytest.mark.slow
def test_eloreta_smoke_run(hardening_forward, tmp_path):
    """eLORETA end-to-end: pre-flight + cached forward + force_equal inverse."""
    module = SourceReconstructionModule(
        {
            "method": "eLORETA",
            "min_trials": 5,
            "subjects_dir": str(hardening_forward["subjects_dir"]),
            "output": {"generate_qc": False, "save_stc": False},
        },
        tmp_path,
    )
    result = module.process(_source_epochs(hardening_forward["info"], n_epochs=8))
    assert result.success, result.errors
    assert result.metadata["method"] == "eLORETA"

    npz = np.load(tmp_path / "unknown_roi_tc.npz")
    assert npz["stim"].ndim == 2
    assert npz["stim"].min() < 0  # signed output survives eLORETA

    provenance = json.loads((tmp_path / "unknown_provenance.json").read_text())
    assert provenance["method_params"] == {"force_equal": True}
    assert provenance["lambda2_evoked"] == pytest.approx(1.0 / 9.0)
    assert provenance["montage_check"]["status"] == "ok"


# ---------------------------------------------------------------------------
# Task 1/5: provenance + reloadable artifacts
# ---------------------------------------------------------------------------

PROVENANCE_KEYS = {
    "stage",
    "module",
    "module_version",
    "created_utc",
    "subject_id",
    "eegcpm",
    "mne",
    "numpy",
    "scipy",
    "method",
    "parcellation",
    "mode",
    "forward_source",
    "spacing",
    "subjects_dir",
    "bem",
    "trans",
    "n_sources",
    "n_channels",
    "n_epochs",
    "sfreq",
    "cov_method",
    "cov_rank",
    "data_rank",
    "loose",
    "depth",
    "pick_ori",
    "label_extraction_mode",
    "lambda2_evoked",
    "lambda2_trial",
    "method_params",
    "nave",
    "nave_confound_warning",
    "conditions_requested",
    "conditions_produced",
    "conditions_missing",
    "conditions_skipped",
    "resolved_event_codes",
    "run_identity_hash",
    "labels",
    "roi_flip_summary",
    "montage_check",
    "epochs_path",
    "epochs_sha256",
    "config_hash",
    "artifacts",
}


@pytest.mark.fsaverage
@pytest.mark.slow
def test_template_asset_key_tracks_mtime_and_size(tmp_path):
    """B5-A4: replacing the template BEM/trans contents in place changes the key.

    The forward cache key used to cover the filenames only, so a replaced BEM
    file kept handing out the stale forward.
    """
    import os
    import time

    from eegcpm.modules.source import SourceReconstructionModule

    bem_dir = tmp_path / "fsaverage" / "bem"
    bem_dir.mkdir(parents=True)
    bem = bem_dir / "fsaverage-5120-5120-5120-bem-sol.fif"
    trans = bem_dir / "fsaverage-trans.fif"
    bem.write_bytes(b"a" * 10)
    trans.write_bytes(b"b" * 10)

    key1 = SourceReconstructionModule._template_asset_key(tmp_path)
    assert key1 == SourceReconstructionModule._template_asset_key(tmp_path)
    assert "missing" not in key1

    bem.write_bytes(b"a" * 20)
    os.utime(bem, (time.time() + 10, time.time() + 10))
    key2 = SourceReconstructionModule._template_asset_key(tmp_path)
    assert key2 != key1

    missing = SourceReconstructionModule._template_asset_key(tmp_path / "nope")
    assert missing.count("missing") == 2


def test_run_identity_hash_combines_config_and_condition_mapping(tmp_path):
    """B5-N3: the identity is config hash + resolved condition mapping.

    Two runs with the same ``SourceConfig`` but different resolved
    ``{condition: codes, matched_event_values}`` mappings are different runs
    and must not share an identity hash.
    """
    from eegcpm.modules.source import SourceReconstructionModule

    module = SourceReconstructionModule({"method": "dSPM"}, tmp_path)
    selection_a = {
        "resolved_event_codes": {
            "stim": {"codes": ["stim"], "matched_event_values": [1]},
        },
    }
    selection_b = {
        "resolved_event_codes": {
            "stim": {"codes": ["stim"], "matched_event_values": [2]},
        },
    }
    hash_a = module.run_identity_hash(selection_a)
    assert hash_a == module.run_identity_hash(selection_a)
    assert hash_a != module.run_identity_hash(selection_b)

    # A config change changes the identity even for the same mapping.
    module_other = SourceReconstructionModule(
        {"method": "dSPM", "inverse": {"snr": 5.0}}, tmp_path)
    assert module_other.config_hash() != module.config_hash()
    assert module_other.run_identity_hash(selection_a) != hash_a


def test_provenance_and_reloadable_artifacts(hardening_forward, tmp_path):
    subjects_dir = hardening_forward["subjects_dir"]
    # Stale artifacts of a previous run: a dropped condition's stc and a trials
    # archive that this evoked run does not produce.
    (tmp_path / "sub-01_old_cond_stc-lh.stc").write_bytes(b"stale")
    (tmp_path / "sub-01_roi_tc_trials.npz").write_bytes(b"stale")

    epochs = _source_epochs(hardening_forward["info"], n_epochs=12)
    epochs_path = tmp_path / "sub-01_task-rest_epo.fif"
    epochs.save(epochs_path, overwrite=True, verbose=False)

    module = SourceReconstructionModule(
        {
            "method": "dSPM",
            "min_trials": 5,
            "subjects_dir": str(subjects_dir),
            "output": {"generate_qc": False},
        },
        tmp_path,
    )
    result = module.process(
        epochs,
        subject=SimpleNamespace(id="sub-01", session="01"),
        epochs_path=epochs_path,
    )
    assert result.success, result.errors

    # Stale outputs of the previous run are gone (no old/new mixing, B4-K5).
    assert not (tmp_path / "sub-01_old_cond_stc-lh.stc").exists()
    assert not (tmp_path / "sub-01_roi_tc_trials.npz").exists()

    provenance_path = tmp_path / "sub-01_provenance.json"
    assert provenance_path.exists()
    provenance = json.loads(provenance_path.read_text())
    missing_keys = PROVENANCE_KEYS - set(provenance)
    assert not missing_keys, f"provenance keys missing: {sorted(missing_keys)}"

    assert provenance["config_hash"] == sha256_json(module.cfg.model_dump(mode="json"))
    assert provenance["forward_source"] == "template"
    assert provenance["method"] == "dSPM"
    assert provenance["pick_ori"] == "normal"
    assert provenance["label_extraction_mode"] == "mean_flip"
    assert provenance["spacing"] == "oct6"
    assert provenance["bem"] == "fsaverage-5120-5120-5120-bem-sol.fif"
    assert provenance["n_channels"] == 129
    assert provenance["n_epochs"] == 12
    assert provenance["lambda2_evoked"] == pytest.approx(1.0 / 9.0)
    assert provenance["lambda2_trial"] == pytest.approx(1.0)
    assert provenance["method_params"] is None
    assert provenance["nave"] == {"stim": 12}
    assert provenance["conditions_produced"] == ["stim"]
    assert provenance["data_rank"] and provenance["cov_rank"]
    assert provenance["epochs_path"] == str(epochs_path)
    assert provenance["epochs_sha256"] == sha256_file(epochs_path)
    assert provenance["montage_check"]["status"] == "ok"
    assert provenance["montage_check"]["median_mm"] < 10.0

    labels = provenance["labels"]
    assert len(labels["roi_names"]) == len(labels["roi_vertex_counts"])
    assert min(labels["roi_vertex_counts"]) >= 3
    assert "Cerebellar.Anterior" in labels["excluded_rois"]
    assert labels["excluded_reasons"]["Cerebellar.Anterior"]
    assert provenance["roi_flip_summary"].keys() == set(labels["roi_names"])
    for name, entry in provenance["roi_flip_summary"].items():
        assert -1.0 <= entry["mean_dot"] <= 1.0, name
        assert entry["n_vertices"] == labels["roi_vertex_counts"][
            labels["roi_names"].index(name)
        ]

    # The archive is registered with its hash and the consumer check accepts it.
    assert provenance["artifacts"]["sub-01_roi_tc.npz"] == sha256_file(
        tmp_path / "sub-01_roi_tc.npz"
    )
    verify_archive_provenance(tmp_path / "sub-01_roi_tc.npz")

    # cov / inv reload with the standard MNE readers.
    cov = mne.read_cov(tmp_path / "sub-01-cov.fif")
    assert set(cov["names"]) == set(epochs.ch_names)
    inv = mne.minimum_norm.read_inverse_operator(tmp_path / "sub-01-inv.fif")
    assert len(inv["src"]) == 2
    assert provenance["n_sources"] == sum(len(s["vertno"]) for s in inv["src"])


def test_config_hash_changes_with_resolved_parameters(tmp_path):
    """``config_hash`` covers resolved parameters, not raw YAML bytes."""
    base = SourceReconstructionModule({"method": "dSPM"}, tmp_path)
    same = SourceReconstructionModule({"method": "dSPM"}, tmp_path)
    other = SourceReconstructionModule(
        {"method": "dSPM", "inverse": {"snr": 3.5}}, tmp_path
    )
    assert base.config_hash() == same.config_hash()
    assert base.config_hash() != other.config_hash()
    assert base.config_hash() == sha256_json(base.cfg.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Task 8: atomic / stale-safe writes
# ---------------------------------------------------------------------------


class TestAtomicWrites:
    def test_failed_npz_write_leaves_nothing_behind(self, tmp_path, monkeypatch):
        target = tmp_path / "sub-01_roi_tc.npz"

        def boom(path, **arrays):
            Path(path).write_bytes(b"partial")  # simulate a truncated write
            raise OSError("disk full")

        monkeypatch.setattr(np, "savez", boom)
        from eegcpm.core.provenance import atomic_write_npz

        with pytest.raises(OSError, match="disk full"):
            atomic_write_npz(target, {"a": np.zeros(3)})
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []  # temp file removed too

    def test_failed_text_write_keeps_previous_target(self, tmp_path):
        target = tmp_path / "prov.json"
        target.write_text("previous")
        from eegcpm.core.provenance import atomic_write_text

        with pytest.raises(TypeError):
            atomic_write_text(target, 12345)  # not a str -> writer raises
        assert target.read_text() == "previous"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["prov.json"]

    def test_remove_stale_outputs_only_touches_this_subject(self, tmp_path):
        (tmp_path / "sub-01_roi_tc.npz").write_bytes(b"old")
        (tmp_path / "sub-01_old_cond_stc-lh.stc").write_bytes(b"old")
        (tmp_path / "sub-01_old_cond_stc-rh.stc").write_bytes(b"old")
        (tmp_path / "sub-02_roi_tc.npz").write_bytes(b"other subject")
        (tmp_path / "unrelated.txt").write_text("keep")

        module = SourceReconstructionModule({}, tmp_path)
        removed = module._remove_stale_outputs("sub-01")

        assert len(removed) == 3
        assert not (tmp_path / "sub-01_roi_tc.npz").exists()
        assert (tmp_path / "sub-02_roi_tc.npz").exists()
        assert (tmp_path / "unrelated.txt").exists()

    @pytest.mark.fsaverage
    @pytest.mark.slow
    def test_failed_write_phase_removes_this_runs_artifacts(
        self, hardening_forward, tmp_path, monkeypatch
    ):
        """A write failure must not leave a partial artifact set (B4-K5).

        The provenance is written last; when it fails, everything this run already
        wrote is removed and ``success`` is False.
        """

        def boom(path, payload):
            raise OSError("no space left on device")

        monkeypatch.setattr(source_module, "atomic_write_json", boom)
        module = SourceReconstructionModule(
            {
                "method": "dSPM",
                "min_trials": 5,
                "subjects_dir": str(hardening_forward["subjects_dir"]),
                "output": {"generate_qc": False},
            },
            tmp_path,
        )
        result = module.process(
            _source_epochs(hardening_forward["info"], n_epochs=8),
            forward=hardening_forward["forward"],
        )
        assert result.success is False
        assert any("no space left" in error for error in result.errors)
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Task 6: CLI epoch-file selection
# ---------------------------------------------------------------------------


class TestSelectEpochFile:
    def test_single_file_returned(self, tmp_path):
        path = tmp_path / "sub-01_task-rest_epo.fif"
        path.write_bytes(b"")
        assert _select_epoch_file(tmp_path) == path

    def test_no_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"no \*_epo\.fif"):
            _select_epoch_file(tmp_path)

    def test_two_files_raise(self, tmp_path):
        (tmp_path / "sub-01_task-a_epo.fif").write_bytes(b"")
        (tmp_path / "sub-01_task-b_epo.fif").write_bytes(b"")
        with pytest.raises(ValueError, match="exactly one"):
            _select_epoch_file(tmp_path)


# ---------------------------------------------------------------------------
# Task 7: condition matching by event-code value
# ---------------------------------------------------------------------------


class TestConditionEventValueMatching:
    """B5 task 7: match ``epochs.events[:, 2]`` values, never hierarchical tags."""

    def test_hierarchical_tag_is_not_overselected(self, tmp_path):
        """A code ``a`` must not drag in epochs tagged ``a/b``.

        MNE's ``epochs[['a']]`` matches hierarchical tags, so with
        ``event_id={'a': 1, 'a/b': 2}`` it selects both (probed 2026-10-15 on
        1.13.2). The value-based selection keeps only the three ``a`` epochs.
        """
        events = [[0, 0, 1], [100, 0, 1], [200, 0, 1], [300, 0, 2], [400, 0, 2]]
        epochs = _tiny_epochs(events, {"a": 1, "a/b": 2})
        module = SourceReconstructionModule(
            {"min_trials": 1},
            tmp_path,
            task_config={"conditions": [{"name": "a", "event_codes": ["a"]}]},
        )
        selected, selection = module._select_condition_epochs(epochs)
        assert selection["nave"] == {"a": 3}
        assert selected["a"].events[:, 2].tolist() == [1, 1, 1]

    @pytest.mark.parametrize("codes", [[7], ["7"]])
    def test_numeric_code_fallback(self, tmp_path, codes):
        """Codes given as numbers (or numeric strings) match by value.

        ``event_id`` keys need not be the numeric code itself (e.g. an event
        named ``Stimulus/S 7``); the fallback mirrors
        ``EventSorter.sort_by_condition``.
        """
        events = [[0, 0, 7], [100, 0, 7], [200, 0, 9]]
        epochs = _tiny_epochs(events, {"Stimulus/S 7": 7, "Stimulus/S 9": 9})
        module = SourceReconstructionModule(
            {"min_trials": 1},
            tmp_path,
            task_config={"conditions": [{"name": "seven", "event_codes": codes}]},
        )
        selected, selection = module._select_condition_epochs(epochs)
        assert selection["nave"] == {"seven": 2}
        assert selected["seven"].events[:, 2].tolist() == [7, 7]

    def test_name_code_matches_by_value(self, tmp_path):
        """The name dialect still works (regression for the B4 behaviour)."""
        events = [[index * 100, 0, 1] for index in range(4)] + [
            [400 + index * 100, 0, 2] for index in range(4)
        ]
        epochs = _tiny_epochs(events, {"1": 1, "2": 2})
        module = SourceReconstructionModule(
            {"min_trials": 1},
            tmp_path,
            task_config={"conditions": [{"name": "one", "event_codes": ["1"]}]},
        )
        _selected, selection = module._select_condition_epochs(epochs)
        assert selection["nave"] == {"one": 4}

    def test_unmatched_codes_are_still_missing(self, tmp_path):
        events = [[0, 0, 1]]
        epochs = _tiny_epochs(events, {"a": 1})
        module = SourceReconstructionModule(
            {"min_trials": 1},
            tmp_path,
            task_config={
                "conditions": [{"name": "absent", "event_codes": ["nope", 42]}]
            },
        )
        selected, selection = module._select_condition_epochs(epochs)
        assert selected == {}
        assert selection["conditions_missing"] == ["absent"]
        assert any("no matching events" in warning for warning in selection["warnings"])


# ---------------------------------------------------------------------------
# Task 9: features CLI reports missing inputs
# ---------------------------------------------------------------------------


def _write_provenance(npz_path: Path, config_hash: str = "c" * 64) -> Path:
    """Minimal provenance sidecar for CLI-level consumer tests."""
    provenance_path = provenance_path_for_archive(npz_path)
    provenance_path.write_text(
        json.dumps(
            {
                "config_hash": config_hash,
                "epochs_sha256": "e" * 64,
                "artifacts": {npz_path.name: sha256_file(npz_path)},
            }
        )
    )
    return provenance_path


def _source_erp_args(project: Path, output_dir: Path, **overrides) -> SimpleNamespace:
    args = SimpleNamespace(
        config=project / "features.yaml",
        project=project,
        feature_type="source_erp",
        input_dir=None,
        output_dir=output_dir,
        condition="resting_ec",
        sfreq=500.0,
        source="dSPM-CONN32",
        preprocessing="standard",
        task="rest",
        subjects=None,
        sessions=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _make_source_tree(project: Path) -> Path:
    """``derivatives/source/standard/rest/variant-dSPM-CONN32/sub-01/ses-01``.

    The archive stem follows the source CLI convention: it passes the
    *stripped* subject id (``01``) to the module, which writes
    ``01_roi_tc.npz`` inside ``sub-01/ses-01/`` — the same stem the features CLI
    discovery looks for.
    """
    import yaml

    root = (
        project
        / "derivatives"
        / "source"
        / "standard"
        / "rest"
        / "variant-dSPM-CONN32"
    )
    session = root / "sub-01" / "ses-01"
    session.mkdir(parents=True)
    npz_path = session / "01_roi_tc.npz"
    np.savez(
        npz_path,
        roi_names=np.array(["ROI_0"], dtype=str),
        stim=np.zeros((1, 11)),
        stim_times=np.arange(-0.2, 0.2 + 1e-9, 0.04),
        sfreq=25.0,
    )
    _write_provenance(npz_path)
    (project / "features.yaml").write_text(
        yaml.safe_dump(
            {
                "stage": "features",
                "variant": "source_erp_test",
                "components": {"N1": {"search_window": [0.0, 0.15], "polarity": "negative"}},
                "metrics": ["peak_amplitude"],
                "generate_qc": False,
            }
        )
    )
    return root


def test_features_cli_reports_missing_sessions(tmp_path, capsys):
    """B5 task 9: absent subject/session inputs are reported and counted."""
    from eegcpm.cli.features import features_command

    root = _make_source_tree(tmp_path)
    (root / "sub-01" / "ses-02").mkdir()  # session without an archive
    (root / "sub-02" / "ses-01").mkdir(parents=True)  # subject without an archive

    result = features_command(_source_erp_args(tmp_path, tmp_path / "features_out"))

    assert result["n_rows"] == 1
    assert result["n_missing"] == 2
    missing_text = " ".join(result["missing"])
    assert "sub-01/ses-02" in missing_text
    assert "sub-02/ses-01" in missing_text
    captured = capsys.readouterr()
    assert "sub-02/ses-01" in captured.err
    assert "missing" in captured.err


def test_features_cli_requested_subject_absent_raises(tmp_path):
    """A requested subject with no source output is a loud failure, not an empty CSV."""
    from eegcpm.cli.features import features_command

    _make_source_tree(tmp_path)
    args = _source_erp_args(
        tmp_path, tmp_path / "features_out", subjects=["99"]
    )
    with pytest.raises(FileNotFoundError, match="sub-99"):
        features_command(args)
    assert not (tmp_path / "features_out" / "source_erp_resting_ec.parquet").exists()
