"""Simulated-dipole ground truth for signed source ROI extraction (P2-F1/F4).

The scientific gate for the source-ERP core: a known signed dipole waveform is injected
at the vertices of one CONN ROI (``SensoriMotor.Superior``, a midline ROI), projected to
129 EGI sensors with the fixed-orientation fsaverage forward model, and recovered through
the real :class:`SourceReconstructionModule` (evoked mode, dSPM). The recovered ROI time
course must:

(1) dominate all other ROIs in the N1/P2 window (0.05-0.25 s);
(2) correlate > 0.8 with the injected waveform and be **negative** at the 0.05-0.15 s
    extremum;
(3) peak within one sample of 0.100 s;
(4) have the same sign for two random seeds;
(5) contain negative values (signed output, i.e. ``pick_ori='normal'`` + ``mean_flip``);
(6) expose valid label metadata (>= 3 vertices per ROI, cerebellar excluded, unique
    names).

If (1) fails the assertion must **not** be weakened: the pre-registered triage is to
check DorsalAttention.FEF leakage (nearest other ROI centre at 34.96 mm), then retry
with the most isolated ROI (DefaultMode.MPFC, nearest other centre at 44.07 mm), and
only then escalate with evidence.

Covariance-estimator guards (P2-F11, moved forward from B5) live at the bottom of this
file; they do not need fsaverage and are deliberately not marked ``fsaverage``.
"""

from pathlib import Path

import mne
import numpy as np
import pytest

from eegcpm.modules.source import SourceReconstructionModule
from eegcpm.modules.source.labels import build_conn_labels

TARGET_ROI = "SensoriMotor.Superior"
SFREQ = 250.0
N_EPOCHS = 40
T_RANGE = (-0.1, 0.4)
SEEDS = (0, 1)
# Nearest other ROI centre for SensoriMotor.Superior (probe 2026-10-14):
# DorsalAttention.FEF_L/R at 34.96 mm; most isolated ROI: DefaultMode.MPFC (44.07 mm).
FEF_LEAKAGE_BASELINE_MM = 34.96


def _gaussian(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


@pytest.fixture(scope="module")
def dipole_setup(fsaverage_subjects_dir):
    """oct6 fsaverage forward (free + fixed) and the injected target waveform."""
    subjects_dir = Path(fsaverage_subjects_dir)
    src = mne.setup_source_space(
        "fsaverage", "oct6", subjects_dir=subjects_dir, add_dist=False, verbose=False
    )
    bem_sol = mne.read_bem_solution(
        subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
        verbose=False,
    )

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    assert "Cz" in montage.ch_names and len(montage.ch_names) == 129
    info = mne.create_info(montage.ch_names, SFREQ, "eeg")
    info.set_montage(montage, verbose=False)

    fwd = mne.make_forward_solution(
        info, "fsaverage", src, bem_sol, eeg=True, meg=False, mindist=5.0, verbose=False
    )
    fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
    )

    # Labels on the source space the inverse operator will use.
    labels, label_info = build_conn_labels(fwd["src"], subjects_dir)
    target_idx = list(label_info["roi_names"]).index(TARGET_ROI)
    target = labels[target_idx]

    n_times = int(round((T_RANGE[1] - T_RANGE[0]) * SFREQ)) + 1
    times = T_RANGE[0] + np.arange(n_times) / SFREQ
    # N1 negative, P2 positive (plan waveform, in "source units").
    w = -1.0 * _gaussian(times, 0.100, 0.015) + 0.6 * _gaussian(times, 0.200, 0.030)

    # sign_flip support for BiHemiLabel is verified here (lh vertices then rh vertices).
    flips = mne.label_sign_flip(target, fwd["src"])
    n_lh_label = len(target.lh.vertices)
    assert len(flips) == n_lh_label + len(target.rh.vertices)

    src_use = fwd["src"]
    n_lh = len(src_use[0]["vertno"])
    stc_data = np.zeros((n_lh + len(src_use[1]["vertno"]), n_times))
    idx_lh = np.searchsorted(src_use[0]["vertno"], target.lh.vertices)
    idx_rh = np.searchsorted(src_use[1]["vertno"], target.rh.vertices)
    stc_data[idx_lh] = flips[:n_lh_label, None] * w[None, :]
    stc_data[n_lh + idx_rh] = flips[n_lh_label:, None] * w[None, :]

    stc_sim = mne.SourceEstimate(
        stc_data,
        vertices=[src_use[0]["vertno"], src_use[1]["vertno"]],
        tmin=T_RANGE[0],
        tstep=1.0 / SFREQ,
        subject="fsaverage",
        verbose=False,
    )
    # Source amplitudes are in Am; the plan waveform (max |w| = 1) triggers MNE's
    # "very large current" RuntimeWarning, which is expected and harmless here.
    evoked = mne.apply_forward(fwd_fixed, stc_sim, info, verbose=False)

    return {
        "fwd": fwd,
        "fwd_fixed": fwd_fixed,
        "labels": labels,
        "label_info": label_info,
        "target_idx": target_idx,
        "times": times,
        "w": w,
        "evoked": evoked,
    }


def _make_epochs(evoked, seed):
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((N_EPOCHS,) + evoked.data.shape) * (
        0.2 * evoked.data.std()
    )
    data = evoked.data[None, :, :] + noise
    events = np.column_stack(
        [
            np.arange(N_EPOCHS) * int(1.0 * SFREQ),
            np.zeros(N_EPOCHS, dtype=int),
            np.ones(N_EPOCHS, dtype=int),
        ]
    )
    epochs = mne.EpochsArray(
        data,
        evoked.info.copy(),
        events=events,
        event_id={"stim": 1},
        tmin=T_RANGE[0],
        baseline=(None, 0),
        verbose=False,
    )
    epochs.set_eeg_reference(projection=True, verbose=False)
    return epochs


@pytest.fixture(scope="module")
def dipole_runs(dipole_setup, tmp_path_factory):
    """Run the real module for both seeds; return the saved ROI time courses."""
    runs = {}
    for seed in SEEDS:
        out_dir = tmp_path_factory.mktemp(f"dipole_seed{seed}")
        module = SourceReconstructionModule({"method": "dSPM"}, out_dir)
        result = module.process(
            _make_epochs(dipole_setup["evoked"], seed), forward=dipole_setup["fwd"]
        )
        assert result.success, result.errors
        npz = np.load(out_dir / "unknown_roi_tc.npz")
        runs[seed] = {
            "result": result,
            "times": npz["stim_times"],
            "roi_names": [str(n) for n in npz["roi_names"]],
            "tc": npz["stim"],
            "roi_vertex_counts": npz["roi_vertex_counts"],
            "roi_hemi": [str(h) for h in npz["roi_hemi"]],
            "excluded_rois": [str(n) for n in npz["excluded_rois"]],
        }
    return runs


def _target_tc(run):
    return run["tc"][run["roi_names"].index(TARGET_ROI)]


@pytest.mark.fsaverage
@pytest.mark.slow
class TestDipoleGroundTruth:
    def test_target_roi_dominates_n1_p2_window(self, dipole_setup, dipole_runs):
        """Assertion (1): the injected ROI has the largest max|tc| in 0.05-0.25 s."""
        times = dipole_setup["times"]
        window = (times >= 0.05) & (times <= 0.25)
        for seed, run in dipole_runs.items():
            assert np.allclose(run["times"], times, rtol=0, atol=1e-12)
            peaks = np.max(np.abs(run["tc"][:, window]), axis=1)
            order = np.argsort(-peaks)
            assert run["roi_names"][order[0]] == TARGET_ROI, (
                f"seed {seed}: dominant ROI is {run['roi_names'][order[0]]} "
                f"(peak {peaks[order[0]]:.3e}); target {TARGET_ROI} peak "
                f"{peaks[order[1]]:.3e}; FEF leakage baseline "
                f"{FEF_LEAKAGE_BASELINE_MM} mm"
            )

    def test_target_waveform_correlation_and_polarity(self, dipole_setup, dipole_runs):
        """Assertion (2): corr(tc, w) > 0.8 and the 0.05-0.15 s extremum is negative."""
        times = dipole_setup["times"]
        w = dipole_setup["w"]
        n1_window = (times >= 0.05) & (times <= 0.15)
        for seed, run in dipole_runs.items():
            tc = _target_tc(run)
            corr = float(np.corrcoef(tc, w)[0, 1])
            assert corr > 0.8, f"seed {seed}: corr(target, w) = {corr:.4f}"
            extremum = tc[n1_window][np.argmax(np.abs(tc[n1_window]))]
            assert extremum < 0, f"seed {seed}: N1 extremum is not negative ({extremum:.3e})"

    def test_target_latency_within_one_sample(self, dipole_setup, dipole_runs):
        """Assertion (3): N1 extremum within 1 sample of 0.100 s."""
        times = dipole_setup["times"]
        n1_window = (times >= 0.05) & (times <= 0.15)
        sample = 1.0 / SFREQ
        for seed, run in dipole_runs.items():
            tc = _target_tc(run)
            lat = times[n1_window][np.argmax(np.abs(tc[n1_window]))]
            assert abs(lat - 0.100) <= sample + 1e-9, f"seed {seed}: N1 latency {lat:.4f} s"

    def test_polarity_consistent_across_seeds(self, dipole_setup, dipole_runs):
        """Assertion (4): the N1 sign at 0.100 s is identical (negative) for both seeds."""
        times = dipole_setup["times"]
        n1_window = (times >= 0.05) & (times <= 0.15)
        signs = {}
        for seed, run in dipole_runs.items():
            tc = _target_tc(run)
            signs[seed] = np.sign(tc[n1_window][np.argmax(np.abs(tc[n1_window]))])
        assert len(set(signs.values())) == 1
        assert signs[SEEDS[0]] < 0

    def test_signed_output(self, dipole_runs):
        """Assertion (5): ROI time courses contain negative values (not a magnitude)."""
        for seed, run in dipole_runs.items():
            assert run["tc"].min() < 0, f"seed {seed}: no negative ROI values"
            assert _target_tc(run).min() < 0

    def test_label_metadata_in_saved_npz(self, dipole_setup, dipole_runs):
        """Assertion (6): vertex counts, cerebellar exclusion, unique names."""
        reference_names = [str(n) for n in dipole_setup["label_info"]["roi_names"]]
        expected_excluded = [str(n) for n in dipole_setup["label_info"]["excluded_rois"]]
        for seed, run in dipole_runs.items():
            assert run["roi_names"] == reference_names
            assert len(set(run["roi_names"])) == len(run["roi_names"])
            assert int(np.min(run["roi_vertex_counts"])) >= 3
            assert len(run["roi_vertex_counts"]) == len(reference_names)
            assert run["excluded_rois"] == expected_excluded
            assert "Cerebellar.Anterior" in run["excluded_rois"]
            assert "Cerebellar.Posterior" in run["excluded_rois"]
            assert TARGET_ROI in run["roi_names"]

    def test_npz_contains_metadata_and_sfreq(self, dipole_runs):
        """npz always carries the label metadata + sfreq (P2-F1)."""
        for run in dipole_runs.values():
            roi_data = run["result"].outputs["roi_data"]
            for key in (
                "roi_names",
                "roi_coords_mni",
                "roi_radius_mm",
                "roi_vertex_counts",
                "roi_hemi",
                "excluded_rois",
                "sfreq",
            ):
                assert key in roi_data
            assert roi_data["sfreq"] == pytest.approx(SFREQ)
            assert len(list(roi_data["roi_hemi"])) == len(run["roi_names"])


class TestSourceCovarianceGuards:
    """P2-F11 guards: estimator runs after the forward and fails loudly.

    No fsaverage needed (a dummy forward object is passed; the guards raise before
    the inverse operator is built), so these tests are deliberately unmarked.
    """

    @staticmethod
    def _epochs(baseline=(None, 0), tmin=-0.2, finite=True):
        rng = np.random.default_rng(7)
        ch_names = [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "T3", "C3", "Cz", "C4", "T4",
            "T5", "P3", "Pz", "P4", "T6",
            "O1", "Oz", "O2",
        ]
        n_times = int(round((0.6 - tmin) * SFREQ))
        data = rng.standard_normal((5, len(ch_names), n_times)) * 1e-5
        if not finite:
            data[0, 0, 0] = np.nan
        info = mne.create_info(ch_names, SFREQ, ["eeg"] * len(ch_names))
        info.set_montage(
            mne.channels.make_standard_montage("standard_1020"), verbose=False
        )
        events = np.column_stack(
            [np.arange(5) * 500, np.zeros(5, dtype=int), np.ones(5, dtype=int)]
        )
        return mne.EpochsArray(
            data, info, events=events, event_id={"stim": 1}, tmin=tmin,
            baseline=baseline, verbose=False,
        )

    @staticmethod
    def _module(tmp_path):
        # A bogus subjects_dir keeps these guard tests independent of the cached
        # fsaverage (the guards raise before any source-space/label use, so no
        # fsaverage is needed and no download can happen).
        return SourceReconstructionModule(
            {"method": "sLORETA", "subjects_dir": str(tmp_path)}, tmp_path
        )

    def test_no_baseline_fails(self, tmp_path):
        module = self._module(tmp_path)
        result = module.process(self._epochs(baseline=None), forward=object())
        assert not result.success
        assert "baseline" in result.errors[0].lower()

    def test_non_finite_data_fails(self, tmp_path):
        module = self._module(tmp_path)
        result = module.process(self._epochs(finite=False), forward=object())
        assert not result.success
        assert "finite" in result.errors[0].lower()

    def test_non_negative_tmin_fails(self, tmp_path):
        # baseline=(0, 0) is the only accepted form for tmin == 0 (mne.baseline
        # rejects (None, 0) as "baseline interval is only one sample").
        module = self._module(tmp_path)
        result = module.process(
            self._epochs(tmin=0.0, baseline=(0.0, 0.0)), forward=object()
        )
        assert not result.success
        assert "pre-stimulus" in result.errors[0].lower()
