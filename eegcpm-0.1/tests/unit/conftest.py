"""Shared fixtures for unit tests.

``fsaverage_subjects_dir`` resolves the *cached* fsaverage only. Tests must never
download data: if the 3-layer BEM solution is missing the dependent tests are
skipped (the CI workflow prefetches fsaverage before running the suite).

``gsn129_dipole_setup`` (B8) is the shared synthetic source-ERP injection setup:
the GSN-HydroCel-129 EEG set, the oct6 fsaverage forward built **through**
``SourceReconstructionModule`` (so the class-level forward cache is reused by the
module under test), the CONN labels, and the plan's ``SensoriMotor.Superior``
dipole waveform projected to the sensors with ``mne.apply_forward``. The B2
ground-truth test's private ``dipole_setup`` is deliberately left untouched (it
predates this fixture); new tests should use this one.

``collect_ignore``: the two specparam test files raise ``RuntimeError`` at
import when the optional backend is not installed (it is deliberately not in
this environment — torch/specparam are out of pipeline scope). Without this,
*every* pytest invocation except the documented one (which lists two explicit
``--ignore`` flags) dies at collection — including the plan's ``pytest -m
chain_e2e``. The ignore is conditional: with specparam installed the files are
collected and run normally.
"""

import importlib.util
import warnings
from pathlib import Path

import mne
import numpy as np
import pytest

collect_ignore = []
if importlib.util.find_spec("specparam") is None:
    collect_ignore += [
        "test_specparam_basemodule.py",
        "test_specparam_features.py",
    ]


@pytest.fixture(scope="session")
def fsaverage_subjects_dir():
    """Subjects dir containing a cached fsaverage, else skip.

    The path is resolved the same way ``mne.datasets.fetch_fsaverage`` resolves it
    (``SUBJECTS_DIR`` config variable, falling back to ``~/mne_data/MNE-fsaverage-data``)
    but without calling the fetch function, which would download.
    """
    subjects_dir = mne.get_config("SUBJECTS_DIR")
    if not subjects_dir:
        subjects_dir = str(Path.home() / "mne_data" / "MNE-fsaverage-data")
    subjects_dir = Path(subjects_dir).expanduser()

    bem_sol = (
        subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    )
    if not bem_sol.exists():
        pytest.skip(f"fsaverage not cached under {subjects_dir}; no download inside tests")
    return subjects_dir


# ---------------------------------------------------------------------------
# B8: shared synthetic source-ERP (dipole) setup
# ---------------------------------------------------------------------------

#: Sampling rate of the B7/B8 synthetic GSN-HydroCel-129 fixture.
GSN129_SFREQ = 250.0
#: Target ROI of the B2 dipole ground truth (midline, oct6).
GSN129_DIPOLE_ROI = "SensoriMotor.Superior"
#: Peak sensor amplitude of the injected dipole ERP (vs the fixture's 15 µV noise).
GSN129_SENSOR_N1_V = 10e-6


def _gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


@pytest.fixture(scope="module")
def gsn129_dipole_setup(fsaverage_subjects_dir, tmp_path_factory):
    """oct6 fsaverage forward + injected dipole waveform for the B8 source-ERP e2e.

    Mirrors the B2 ground-truth recipe (``test_source_dipole_groundtruth.py``)
    with three deliberate differences, all required by the end-to-end test:

    - the forward is built **through** ``SourceReconstructionModule`` on the
      109-channel EEG set the ``hbn_langer`` chain leaves behind (11 neck/face
      channels dropped, the 9 EGI EOG channels retyped by ``ChannelRolesStep``).
      The class-level ``_FORWARD_CACHE`` is therefore populated with exactly the
      forward the module under test will request, so the e2e test pays for one
      ~10 s forward build per session instead of two;
    - the injected waveform is scaled so the sensor-space N1 peaks at
      :data:`GSN129_SENSOR_N1_V` (10 µV, comparable to the fixture's 15 µV 1/f
      noise); the raw B2 waveform (unit peak) would be a physically absurd
      sensor amplitude and trips MNE's large-current warning;
    - the injection is expanded to the fixture's full 129-channel montage with
      **zeros on the EGI EOG and neck/face rows**: only the scalp EEG set carries
      the ERP, so the chain's EOG regression cannot regress the ERP away (the
      same choice as the B7 ``erp_mode='anterior'`` fixture) and the neck/face
      channels — dropped by the chain's first step — stay untouched.

    ``pattern`` is the per-channel projection of the injection onto ``w`` with
    ``Cz`` zeroed — the B7 fixture zeroes Cz *after* the injection, so the
    recording really carries ``p`` with ``p[Cz] = 0``. It is the reference the
    test's average-reference factor is derived from.

    Returns a dict with ``info``, ``fwd``, ``fwd_fixed``, ``labels``,
    ``label_info``, ``target_idx``, ``target_roi``, ``times``, ``w`` (the
    epoch-window waveform, −0.2…0.6 s), ``w_post`` (its 0…0.5 s part, the
    injected waveform), ``injection`` (129 × 126 post-stim samples, V, zeros on
    the EOG/neck-face rows), ``pattern`` (109 EEG gains), ``subjects_dir``,
    ``montage_check``.
    """
    from eegcpm.modules.preprocessing.steps.channel_roles import (
        EGI_129_EOG_CHANNELS,
        EGI_129_NECK_FACE_CHANNELS,
    )
    from eegcpm.modules.source import SourceReconstructionModule
    from eegcpm.modules.source.labels import build_conn_labels

    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    excluded = set(EGI_129_EOG_CHANNELS) | set(EGI_129_NECK_FACE_CHANNELS)
    eeg_names = [name for name in montage.ch_names if name not in excluded]
    assert len(eeg_names) == 109 and "Cz" in eeg_names, (
        "the chain's EEG channel set changed; the shared forward would not match "
        "the epochs the source module receives"
    )
    info = mne.create_info(eeg_names, GSN129_SFREQ, "eeg")
    info.set_montage(montage, verbose=False)

    subjects_dir = Path(fsaverage_subjects_dir)
    module = SourceReconstructionModule(
        {
            "method": "dSPM",
            "subjects_dir": str(subjects_dir),
            "output": {"generate_qc": False, "compute_crosstalk": False},
        },
        tmp_path_factory.mktemp("gsn129_dipole_setup"),
    )
    fwd, _warnings, montage_check = module._get_template_forward(info)
    fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False
    )
    labels, label_info = build_conn_labels(fwd["src"], subjects_dir)
    target_roi = GSN129_DIPOLE_ROI
    target_idx = list(label_info["roi_names"]).index(target_roi)
    target = labels[target_idx]
    flips = mne.label_sign_flip(target, fwd["src"])

    n_lh = len(fwd["src"][0]["vertno"])
    n_lh_label = len(target.lh.vertices)
    times = -0.2 + np.arange(int(round(0.8 * GSN129_SFREQ)) + 1) / GSN129_SFREQ
    w = np.zeros_like(times)
    post = (times >= 0.0) & (times <= 0.5)
    w[post] = (
        -1.0 * _gaussian(times[post], 0.100, 0.015)
        + 0.6 * _gaussian(times[post], 0.200, 0.030)
    )
    # N1 negative, P2 positive; the 1-sample latency assertion is meaningful only
    # if the injected waveform really peaks at 0.100 s.
    n1_window = (times >= 0.05) & (times <= 0.15)
    assert abs(times[n1_window][np.argmin(w[n1_window])] - 0.100) < 1e-9

    idx_lh = np.searchsorted(fwd["src"][0]["vertno"], target.lh.vertices)
    idx_rh = np.searchsorted(fwd["src"][1]["vertno"], target.rh.vertices)
    stc_data = np.zeros((n_lh + len(fwd["src"][1]["vertno"]), times.size))
    stc_data[idx_lh] = flips[:n_lh_label, None] * w[None, :]
    stc_data[n_lh + idx_rh] = flips[n_lh_label:, None] * w[None, :]
    stc = mne.SourceEstimate(
        stc_data,
        vertices=[fwd["src"][0]["vertno"], fwd["src"][1]["vertno"]],
        tmin=float(times[0]),
        tstep=1.0 / GSN129_SFREQ,
        subject="fsaverage",
        verbose=False,
    )
    # ``apply_forward`` logs a "very large current" warning on unit-scale source
    # data; probe at 1 nA and scale the second (real) call, so neither call warns.
    # ``amplitude`` must multiply the *same* probe-scaled waveform: the probe's
    # peak is ``PROBE_SCALE`` × the peak of ``stc``, not of ``stc * PROBE_SCALE``.
    probe_scale = 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        probe = mne.apply_forward(fwd_fixed, stc * probe_scale, info, verbose=False)
    amplitude = probe_scale * GSN129_SENSOR_N1_V / float(np.abs(probe.data).max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        projected = mne.apply_forward(
            fwd_fixed, stc * amplitude, info, verbose=False
        ).data
    assert np.isclose(np.abs(projected).max(), GSN129_SENSOR_N1_V, rtol=1e-6)
    # The fixture adds the injection **at the stim onset**, so only the post-stim
    # window may be passed: carrying the 0.2-s pre-stim padding would shift the
    # injected ERP by 0.2 s (probed 2026-10-16 while writing the test).
    injection_eeg = projected[:, post]
    pattern = (injection_eeg @ w[post]) / (w[post] @ w[post])
    pattern[eeg_names.index("Cz")] = 0.0

    # Expand to the fixture's 129-channel montage: only the scalp EEG set carries
    # the ERP (the EGI EOG channels must not be able to regress it away — the
    # same choice as the B7 ``erp_mode='anterior'`` fixture — and the neck/face
    # channels are dropped by the chain's first step).
    injection = np.zeros((len(montage.ch_names), injection_eeg.shape[1]))
    injection[[montage.ch_names.index(name) for name in eeg_names]] = injection_eeg

    return {
        "info": info,
        "fwd": fwd,
        "fwd_fixed": fwd_fixed,
        "labels": labels,
        "label_info": label_info,
        "target_idx": target_idx,
        "target_roi": target_roi,
        "times": times,
        "w": w,
        "w_post": w[post].copy(),
        "injection": injection,
        "pattern": pattern,
        "subjects_dir": subjects_dir,
        "montage_check": montage_check,
    }
