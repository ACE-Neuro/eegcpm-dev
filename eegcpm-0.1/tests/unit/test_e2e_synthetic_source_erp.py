"""Synthetic end-to-end source-ERP test (plan §B8 task 1 + the Opus B7 amplitude carry).

One synthetic GSN-HydroCel-129 recording — the B7 chain fixture (250 Hz, 150 s,
``first_samp != 0``, ``meas_date`` set, 1/f noise + 60 Hz line noise + blinks on
the EGI EOG channels, ``Cz`` all zeros) — carries an ERP **generated from the B2
dipole**: the plan waveform ``w`` (N1 −1.0 at 0.100 s, P2 +0.6 at 0.200 s) on
every ``SensoriMotor.Superior`` label vertex, multiplied by
``mne.label_sign_flip`` so ``mean_flip`` extraction recovers ``+w``, projected to
the sensors with ``mne.apply_forward`` (10 µV peak) and added at every 2-s
``stim`` annotation (``erp_sensor_data`` in :func:`make_gsn129_recording`; only
the post-stim window is injected, so the N1 really sits at +0.100 s).

The recording goes through the **real** ``hbn_langer`` chain, is epoched with
**plain MNE** — ``mne.Epochs(raw, events, tmin=-0.2, tmax=0.6, baseline=(None, 0),
reject_by_annotation=True)``; the P3 epochs module is deliberately not used —
reconstructed with the real ``SourceReconstructionModule`` (evoked, dSPM, oct6
fsaverage) and consumed by ``SourceERPFeatureModule``. This is the synthetic half
of P2 condition 4; the real-data half is Checkpoint B (plan §B8 task 3).

Assertions (measured values from the 2026-10-16 probe / test run):

1. **chain + epoching sanity** — every chain step ``ok``/``noop`` (the fixture's
   ASR ``repaired_fraction`` is 0.404 with the EEGLAB-standard input high-pass
   — 0.719 before the plan §12.6 decision, implemented 2026-10-16), the ``stim``
   event samples unchanged, all 74 epochs kept (no ``BAD_*`` annotation rejects
   them);
2. **source recovery** — the target ROI's N1 is **negative** and its latency is
   within one sample of the injected 0.100 s (measured −8.28 at 0.100 s), the ROI
   time course correlates with ``w`` (measured r = 0.964; a shifted-template
   control is far below — non-vacuity), the target ROI carries the **largest N1
   of the whole parcellation** (F7: 8.28 vs 6.64 µV next, ``DefaultMode.PCC``),
   and the signed-output/label metadata contract holds. The polarity check
   verifies the ``mean_flip`` **sign convention** (the signed-normal extraction
   returns the injected dipole orientation) — not anatomy;
3. **consumer** — the feature CSV carries the per-condition columns for the
   target ROI/component/metric and its N1 latency/amplitude agree with the npz
   the source stage wrote;
4. **amplitude retention (Opus B7 carry)** — the *projection gain* of the
   chain-output evoked onto ``w``, normalised by the **expected average-reference
   factor** (measured 0.949 after the §12.6 EEGLAB-standard ASR path; 0.966
   before it — see :class:`TestAmplitudeRetention`). The B7 e2e measured the
   same quantity on its anterior-injection fixture (0.516 measured / 0.541
   expected = 0.954) and the audit required the amplitude half of the
   ERP-retention statement, not only the waveform correlation.

Notes on the deliberate interface choices:

- the plain-MNE epoch call is used **without** ``event_id``, exactly as the plan
  writes it; MNE infers ``event_id={'1': 1}`` from the events array, so the
  condition is named ``'1'`` (pinned by a test; the P3 module would have used the
  annotation name ``stim``);
- one module-scoped fixture runs the chain, the source stage and the feature
  stage once for all test methods; the shared ``gsn129_dipole_setup`` conftest
  fixture (B8) builds the oct6 forward **through** ``SourceReconstructionModule``
  so the class-level forward cache is reused instead of rebuilt;
- ``compute_crosstalk``/``save_stc``/``generate_qc`` are off: they are covered by
  their own tests and would dominate the runtime here.
"""

from types import SimpleNamespace

import mne
import numpy as np
import pandas as pd
import pytest

from eegcpm.modules.features.source_erp_features import SourceERPFeatureModule
from eegcpm.modules.preprocessing import PreprocessingPipeline
from eegcpm.modules.source import SourceReconstructionModule
from tests.unit.test_hbn_chain_e2e import make_gsn129_recording

pytestmark = [pytest.mark.fsaverage, pytest.mark.chain_e2e, pytest.mark.slow]

TARGET_ROI = "SensoriMotor.Superior"
SFREQ = 250.0
SAMPLE_S = 1.0 / SFREQ
#: Injected N1 latency (the plan waveform; ``w`` peaks there on the sample grid).
N1_LATENCY_S = 0.100
N1_WINDOW = (0.05, 0.15)
P2_WINDOW = (0.15, 0.30)
#: Minimum Pearson correlation of the recovered ROI time course with ``w``.
MIN_TARGET_CORRELATION = 0.8
#: F7 (final-verification-audit fix round): the target ROI's N1 must be the
#: largest in the whole parcellation by at least this factor over the runner-up
#: (the audit measured 8.28 vs 6.64 µV next, ``DefaultMode.PCC``).
TARGET_N1_DOMINANCE_RATIO = 1.1
#: Control waveform: the injected ``w`` shifted by 0.15 s, so its N1/P2 do not
#: overlap the injected ERP (the time-*reversed* waveform is a weak control for
#: this waveform: its P2 lands on the injected P2, giving r = 0.42).
CONTROL_SHIFT_S = 0.15
#: Channels with ``|arf_c|`` below this fraction of the largest expected gain are
#: excluded from the retention median (the ratio is noise-dominated near the
#: dipole pattern's null).
ARF_FLOOR_FRACTION = 0.2
#: Amplitude-retention band; documented in :class:`TestAmplitudeRetention`.
RETENTION_LO = 0.7
RETENTION_HI = 1.3
#: Band for the same measure on the chain **input** (fixture-integrity control).
INPUT_RETENTION_LO = 0.9
INPUT_RETENTION_HI = 1.1


def _event_samples(raw) -> np.ndarray:
    events, _ = mne.events_from_annotations(raw, verbose=False)
    return events[:, 0].copy()


def _feature_config() -> dict:
    return {
        "stage": "features",
        "variant": "source_erp_b8",
        "components": {
            "N1": {"search_window": list(N1_WINDOW), "polarity": "negative"},
            "P2": {"search_window": list(P2_WINDOW), "polarity": "positive"},
        },
        "metrics": ["peak_amplitude", "peak_latency"],
        "rois": [TARGET_ROI],
        "generate_qc": False,
    }


def _control_waveform(waveform: np.ndarray) -> np.ndarray:
    """The injected waveform shifted by :data:`CONTROL_SHIFT_S` (non-vacuity control).

    ``np.roll`` moves the whole support; the wrapped tail is the waveform's zero
    padding, so the control is exactly "the same ERP at a latency where nothing
    was injected".
    """
    return np.roll(waveform, int(round(CONTROL_SHIFT_S * SFREQ)))


def _projection_gain(data: np.ndarray, waveform: np.ndarray) -> np.ndarray:
    """Per-channel least-squares projection gain of (n_ch, n_times) onto waveform."""
    return (data @ waveform) / (waveform @ waveform)


def _amplitude_retention(setup, evoked, raw_in, onsets_s) -> dict:
    """Projection gain and the expected average-reference factor (B7 carry).

    ``pattern`` is the injected per-channel sensor gain ``p`` (projection of the
    injected ERP onto ``w``, ``Cz`` zeroed as in the fixture). The chain's final
    ``ReferenceStep`` subtracts the mean over the 109 EEG channels, so the
    *expected* per-channel gain of the chain output is ``arf_c = p_c − mean(p)``
    (its scalar form is ``1 − mean(p)²/mean(p²)``). The measured per-channel gain
    ``g_c`` is the projection of the baseline-corrected chain-output evoked onto
    ``w``; the retention is the median of ``g_c / arf_c`` over the channels where
    ``|arf_c|`` is at least :data:`ARF_FLOOR_FRACTION` of the maximum (near the
    pattern null the ratio is dominated by noise).

    The chain **input** is measured the same way against ``p`` itself (the input
    is not average-referenced) over the same ``onsets_s`` (the ``stim`` event
    positions, taken from the events array rather than re-derived): it must come
    out ≈ 1, which proves the injection is intact and that the output number
    reflects the chain, not the measure.

    ``evoked`` is the plain-MNE ``epochs.average()``: ``epochs.info`` keeps the
    raw's full channel set while ``epochs.picks`` selects the data channels, so
    the channel list must come from the evoked, not from ``epochs.ch_names``.
    """
    eeg_idx = mne.pick_types(evoked.info, eeg=True)
    names = [evoked.ch_names[index] for index in eeg_idx]
    pattern = np.array(
        [setup["pattern"][setup["info"].ch_names.index(name)] for name in names]
    )
    mean_pattern = float(pattern.mean())
    arf = pattern - mean_pattern
    gain = _projection_gain(evoked.data, setup["w"])[eeg_idx]
    selected = np.abs(arf) >= ARF_FLOOR_FRACTION * np.abs(arf).max()

    # Fixture-integrity control: the same measure on the raw chain input.
    w_post = setup["w_post"]
    segments = []
    for onset in onsets_s:
        start = int(round(float(onset) * SFREQ))
        segments.append(
            raw_in.get_data(picks=names, start=start, stop=start + w_post.size)
        )
    input_evoked = np.stack(segments).mean(axis=0)
    input_gain = _projection_gain(input_evoked, w_post)
    input_selected = np.abs(pattern) >= ARF_FLOOR_FRACTION * np.abs(pattern).max()

    return {
        "names": names,
        "pattern": pattern,
        "mean_pattern": mean_pattern,
        "arf": arf,
        "gain": gain,
        "selected": selected,
        # Analytic expected factor of the average-reference step for this pattern.
        "arf_scalar": float((arf @ pattern) / (pattern @ pattern)),
        "retention": float(np.median(gain[selected] / arf[selected])),
        # Un-normalised median gain: ~arf when the chain preserves amplitude.
        "unnormalised": float(np.median(gain[selected] / pattern[selected])),
        "aggregate": float((gain @ arf) / (arf @ arf)),
        "input_retention": float(
            np.median(input_gain[input_selected] / pattern[input_selected])
        ),
        "input_correlation": float(np.corrcoef(input_gain, pattern)[0, 1]),
    }


@pytest.fixture(scope="module")
def source_erp_run(gsn129_dipole_setup, tmp_path_factory):
    """One chain run, one source run and one feature run (module-scoped).

    The recording is the B7 synthetic fixture with the B2 dipole projection
    injected at the ``stim`` onsets; the shared ``gsn129_dipole_setup`` fixture
    supplies the forward/labels/waveform and the sensor-space injection.
    """
    setup = gsn129_dipole_setup
    raw = make_gsn129_recording(erp_sensor_data=setup["injection"])
    events_in = _event_samples(raw)
    # Stim onsets relative to the first data sample (used by the input control).
    onsets_s = (events_in - raw.first_samp) / SFREQ

    chain_dir = tmp_path_factory.mktemp("source_erp_chain")
    pipeline = PreprocessingPipeline("hbn_langer", output_dir=chain_dir)
    chain = pipeline.process(raw, subject_id="b8", session_id="01", run_id="1")
    assert chain.success, f"chain errors={chain.errors} warnings={chain.warnings}"
    out = chain.outputs["data"]

    # Plain MNE epoching, exactly the plan's call (no P3 epochs module).
    events, _event_id = mne.events_from_annotations(out, verbose=False)
    epochs = mne.Epochs(
        out,
        events,
        tmin=-0.2,
        tmax=0.6,
        baseline=(None, 0),
        reject_by_annotation=True,
        preload=True,
        verbose=False,
    )

    source_dir = tmp_path_factory.mktemp("source_erp_source")
    module = SourceReconstructionModule(
        {
            "method": "dSPM",
            "min_trials": 10,
            "subjects_dir": str(setup["subjects_dir"]),
            "output": {
                "generate_qc": False,
                "compute_crosstalk": False,
                "save_stc": False,
            },
        },
        source_dir,
    )
    # The shared fixture must be the forward the module uses (no second build).
    # ``process()`` picks the EEG channels first, so the identity check must run
    # on the same picked info the module will hand to ``_get_template_forward``
    # (the plain ``epochs.info`` still carries the EOG channels).
    eeg_info = epochs.copy().pick("eeg").info
    assert module._get_template_forward(eeg_info)[0] is setup["fwd"], (
        "the conftest dipole setup's forward is not the module's template forward"
    )
    source = module.process(epochs, subject=SimpleNamespace(id="b8", session="01"))
    assert source.success, f"source errors={source.errors}"

    condition = source.metadata["conditions_produced"][0]
    with np.load(source_dir / "b8_roi_tc.npz") as npz:
        roi_names = [str(name) for name in npz["roi_names"]]
        target_tc = npz[condition][roi_names.index(TARGET_ROI)].copy()
        all_tc = npz[condition].copy()
        target_times = npz[f"{condition}_times"].copy()
        nave = int(np.asarray(npz[f"{condition}_nave"]).reshape(-1)[0])
        excluded_rois = [str(name) for name in npz["excluded_rois"]]
        vertex_counts = npz["roi_vertex_counts"].copy()

    feature_dir = tmp_path_factory.mktemp("source_erp_features")
    features = SourceERPFeatureModule(_feature_config(), feature_dir)
    feature_result = features.process(
        source_dir / "b8_roi_tc.npz", subject=SimpleNamespace(id="b8", session="01")
    )
    assert feature_result.success, f"feature errors={feature_result.errors}"
    table = pd.read_csv(feature_dir / "features.csv")

    evoked = epochs.average()
    retention = _amplitude_retention(setup, evoked, raw, onsets_s)
    return {
        "setup": setup,
        "raw": raw,
        "events_in": events_in,
        "chain": chain,
        "chain_raw": out,
        "events": events,
        "epochs": epochs,
        "evoked": evoked,
        "source": source,
        "source_dir": source_dir,
        "condition": condition,
        "roi_names": roi_names,
        "target_tc": target_tc,
        "all_tc": all_tc,
        "target_times": target_times,
        "nave": nave,
        "excluded_rois": excluded_rois,
        "vertex_counts": vertex_counts,
        "table": table,
        "retention": retention,
    }


class TestChainAndEpoching:
    """The chain runs clean and the plan's plain-MNE epoching sees every stimulus."""

    def test_chain_all_steps_ok(self, source_erp_run):
        result = source_erp_run["chain"]
        step_status = result.metadata["step_status"]
        assert set(step_status) == {
            "channel_roles",
            "zapline",
            "filter",
            "bad_channel_detection",
            "interpolate",
            "eog_regression",
            "asr",
            "block_rejection",
            "reference",
        }
        bad = {name: status for name, status in step_status.items()
               if status not in ("ok", "noop")}
        assert not bad, f"non-ok/noop step statuses: {bad}"
        assert not result.metadata.get("excluded")
        assert not result.metadata.get("degraded")

    def test_input_recording_contract(self, source_erp_run):
        """The B7 fixture contract this test relies on (plan §B8 task 1).

        250 Hz, ``first_samp != 0`` (a non-zero offset exercises the annotation
        frame recipe), a real ``meas_date``, ``stim`` annotations only, and a
        flat ``Cz`` before the chain (the average reference must recover it).
        """
        raw = source_erp_run["raw"]
        assert raw.info["sfreq"] == SFREQ
        assert raw.first_samp != 0
        assert raw.info["meas_date"] is not None
        assert set(raw.annotations.description) == {"stim"}
        assert np.all(raw.get_data(picks=["Cz"]) == 0.0)

    def test_stim_events_unchanged(self, source_erp_run):
        events_out = _event_samples(source_erp_run["chain_raw"])
        assert len(source_erp_run["events_in"]) == 74
        assert np.array_equal(events_out, source_erp_run["events_in"])

    def test_every_stimulus_survives_annotation_rejection(self, source_erp_run):
        """``reject_by_annotation=True`` must not drop epochs on a clean fixture."""
        assert len(source_erp_run["events"]) == 74
        assert len(source_erp_run["epochs"]) == 74
        assert len(source_erp_run["epochs"]) == len(source_erp_run["events"])

    def test_condition_name_is_the_plain_mne_inference(self, source_erp_run):
        """Pin the condition name: plain ``mne.Epochs`` without ``event_id``.

        MNE infers ``event_id={'1': 1}`` from the events array, so the source
        stage names the condition ``'1'`` (the P3 epochs module — deliberately
        not used here — would have used the ``stim`` annotation name). The name
        prefixes every feature column, so it is pinned rather than assumed.
        """
        assert source_erp_run["condition"] == "1"


class TestSourceERPRecovery:
    """The injected dipole ERP survives the chain and comes back signed and timed."""

    def test_target_n1_negative_within_one_sample(self, source_erp_run):
        """Polarity + latency + spatial dominance of the target N1.

        The polarity assertion verifies the **``mean_flip`` sign convention**
        (the signed-normal extraction returns the injected dipole orientation),
        not anatomy: with ``mode='mean_flip'`` the label vertices' normals are
        sign-flipped to agree with the dominant orientation, so the injected
        ``w`` (N1 −1.0) comes back negative. Spatial specificity is pinned by
        the dominance block below (F7, final-verification-audit fix round): the
        target ROI must carry the **largest N1 of the whole parcellation** —
        the audit measured −8.28 µV at the target vs −6.64 µV next
        (``DefaultMode.PCC``).
        """
        times = source_erp_run["target_times"]
        tc = source_erp_run["target_tc"]
        window = (times >= N1_WINDOW[0]) & (times <= N1_WINDOW[1])
        assert window.any()
        extremum = np.argmax(np.abs(tc[window]))
        latency = float(times[window][extremum])
        value = float(tc[window][extremum])
        assert value < 0, f"target N1 extremum is not negative ({value:.3e})"
        assert abs(latency - N1_LATENCY_S) <= SAMPLE_S + 1e-9, (
            f"target N1 latency {latency:.4f} s is more than one sample from the "
            f"injected {N1_LATENCY_S:.3f} s"
        )

        # F7: the target ROI's N1 dominates every other ROI's — the injected
        # dipole is the only ERP in the recording, so this is a source-ERP-
        # specific assertion, not just "some negative deflection exists".
        # Compare over all included ROIs (cerebellar ROIs are excluded from the
        # parcellation before roi_names/all_tc, so they cannot out-N1 the target).
        # TARGET_N1_DOMINANCE_RATIO = 1.1 pins the audit's 8.28 vs 6.64 µV (ratio 1.25).
        n1 = source_erp_run["all_tc"][:, window].min(axis=1)  # per-ROI N1
        names = source_erp_run["roi_names"]
        order = np.argsort(n1)  # most negative first
        target_index = names.index(TARGET_ROI)
        assert order[0] == target_index, (
            f"largest N1 belongs to {names[order[0]]} "
            f"({n1[order[0]]:.3e}), not {TARGET_ROI} ({n1[target_index]:.3e}); "
            f"top-3: {[(names[i], float(n1[i])) for i in order[:3]]}"
        )
        runner_up = float(n1[order[1]])
        assert n1[target_index] <= TARGET_N1_DOMINANCE_RATIO * runner_up, (
            f"target N1 {n1[target_index]:.3e} is not at least "
            f"{TARGET_N1_DOMINANCE_RATIO}x the runner-up {names[order[1]]} "
            f"({runner_up:.3e})"
        )

    def test_target_waveform_correlation_and_non_vacuity(self, source_erp_run):
        w = source_erp_run["setup"]["w"]
        tc = source_erp_run["target_tc"]
        correlation = float(np.corrcoef(tc, w)[0, 1])
        assert correlation > MIN_TARGET_CORRELATION, (
            f"corr(target ROI tc, injected w) = {correlation:.4f}"
        )
        # Non-vacuity: a shifted template (same shape, no temporal overlap with
        # the injected ERP) must not correlate with the recovered waveform.
        control_correlation = float(np.corrcoef(tc, _control_waveform(w))[0, 1])
        assert abs(control_correlation) < 0.5 * abs(correlation), (
            f"shifted-template control {control_correlation:.4f} is not far "
            f"below the real correlation {correlation:.4f}"
        )

    def test_signed_output_and_label_metadata(self, source_erp_run):
        tc = source_erp_run["target_tc"]
        assert tc.min() < 0, "ROI time course has no negative values (unsigned output)"
        assert source_erp_run["nave"] == 74
        assert "Cerebellar.Anterior" in source_erp_run["excluded_rois"]
        assert "Cerebellar.Posterior" in source_erp_run["excluded_rois"]
        assert int(source_erp_run["vertex_counts"].min()) >= 3

    def test_source_run_metadata_is_loud(self, source_erp_run):
        """The source run reports one produced condition and no missing/skipped ones."""
        metadata = source_erp_run["source"].metadata
        assert metadata["conditions_requested"] == ["1"]
        assert metadata["conditions_missing"] == []
        assert metadata["conditions_skipped"] == {}
        assert metadata["nave"] == {"1": 74}


class TestFeatureConsumer:
    """``SourceERPFeatureModule`` consumes the producer's archive end to end."""

    def test_csv_has_the_per_condition_columns(self, source_erp_run):
        condition = source_erp_run["condition"]
        expected = {
            f"{condition}_{TARGET_ROI}_N1_peak_latency",
            f"{condition}_{TARGET_ROI}_N1_peak_amplitude",
            f"{condition}_{TARGET_ROI}_P2_peak_latency",
            f"{condition}_{TARGET_ROI}_P2_peak_amplitude",
            f"{condition}_nave",
            "subject_id",
            "session_id",
        }
        columns = set(source_erp_run["table"].columns)
        assert expected <= columns, f"missing feature columns: {sorted(expected - columns)}"
        assert len(source_erp_run["table"]) == 1

    def test_csv_n1_agrees_with_the_roi_time_course(self, source_erp_run):
        condition = source_erp_run["condition"]
        row = source_erp_run["table"].iloc[0]
        latency = float(row[f"{condition}_{TARGET_ROI}_N1_peak_latency"])
        amplitude = float(row[f"{condition}_{TARGET_ROI}_N1_peak_amplitude"])
        assert abs(latency - N1_LATENCY_S) <= SAMPLE_S + 1e-9
        assert amplitude < 0
        # The CSV value is the same sample the npz holds (the consumer does not
        # re-window or re-scale).
        times = source_erp_run["target_times"]
        window = (times >= N1_WINDOW[0]) & (times <= N1_WINDOW[1])
        npz_amplitude = float(
            source_erp_run["target_tc"][window][
                np.argmin(source_erp_run["target_tc"][window])
            ]
        )
        assert amplitude == pytest.approx(npz_amplitude, rel=1e-9)

    def test_csv_nave_matches_the_source_metadata(self, source_erp_run):
        condition = source_erp_run["condition"]
        assert int(source_erp_run["table"].iloc[0][f"{condition}_nave"]) == 74


class TestAmplitudeRetention:
    """The Opus B7 carry: amplitude retention, not just waveform correlation.

    The B7 final audit measured the chain's ERP *amplitude* with a per-channel
    projection gain ``<evoked_c, w> / <w, w>`` and required the B8 synthetic
    source-ERP test to carry the assertion, normalised by the **expected
    average-reference factor** of the injected pattern. The B7 fixture's binary
    anterior mask gives the factor ``1 − n_injected/n_eeg = 0.541`` (measured
    retention 0.516/0.541 = 0.954); here the dipole pattern gives the
    per-channel factor ``arf_c = p_c − mean(p)``, whose scalar form is
    ``1 − mean(p)²/mean(p²)``.

    Documented values (this fixture, measured 2026-10-16; probe
    ``/tmp/opencode/b8_probe1.py``; re-measured after the plan §12.6
    EEGLAB-standard ASR path landed):

    - expected average-reference factor ``arf = 0.9952`` — the midline dipole
      pattern is spatially balanced (``mean(p) = −0.32 µV`` vs ``rms(p) =
      4.62 µV``), so the reference correction is small *for this pattern* (it
      moves the number by ~1 %: 0.960 un-normalised → 0.949 normalised); the
      per-channel factor is still used, and it matters most near the pattern
      null, where the floor excludes the noisiest channels. The B7 mask fixture
      is where the factor is material (0.541);
    - normalised retention **0.949** (median; 0.966 before §12.6), aggregate
      0.951, un-normalised 0.960, chain-input control **0.987** (the injection
      is intact, so the ~5 % loss is the chain's);
    - band ``[RETENTION_LO, RETENTION_HI] = [0.7, 1.3]``: fails if the chain loses
      more than 30 % of the injected amplitude or *amplifies* it by more than
      30 %. A chain that ASR-destroys the ERP — the audited failure mode — falls
      far below; the shifted-template control stays below 0.2 × retention.
    """

    def test_injection_is_intact_in_the_recording(self, source_erp_run):
        """Fixture integrity: the raw input really carries the projected dipole.

        The same projection measure on the chain *input* (not average-referenced,
        so its expected gain is the injected pattern itself) must come out ≈ 1.
        This is the non-vacuity control of the whole retention measure and a
        guard against fixture bugs that shift or scale the injection.
        """
        retention = source_erp_run["retention"]
        assert INPUT_RETENTION_LO <= retention["input_retention"] <= INPUT_RETENTION_HI, (
            f"chain-input retention {retention['input_retention']:.3f} outside "
            f"[{INPUT_RETENTION_LO}, {INPUT_RETENTION_HI}]: the injection is not "
            "in the recording as expected"
        )
        assert retention["input_correlation"] > 0.9

    def test_projection_gain_normalised_by_average_reference(self, source_erp_run):
        retention = source_erp_run["retention"]
        assert retention["selected"].sum() >= 20, (
            f"only {int(retention['selected'].sum())} channels carry a usable "
            "expected gain; the retention median would be meaningless"
        )
        assert RETENTION_LO <= retention["retention"] <= RETENTION_HI, (
            f"amplitude retention {retention['retention']:.3f} outside "
            f"[{RETENTION_LO}, {RETENTION_HI}] (expected average-reference factor "
            f"{retention['arf_scalar']:.3f}, un-normalised gain "
            f"{retention['unnormalised']:.3f})"
        )
        # The aggregate form (least-squares gain onto the expected pattern) must
        # agree with the median: a median-only pass could hide a few channels.
        assert RETENTION_LO <= retention["aggregate"] <= RETENTION_HI

    def test_average_reference_factor_is_the_analytic_one(self, source_erp_run):
        """The documented factor is the average-reference factor of the pattern.

        ``arf_c`` is derived from the injected pattern (``p_c − mean(p)``), not
        from the chain output, and the scalar identity
        ``1 − mean(p)²/mean(p²)`` is asserted explicitly: the number the
        retention is normalised by is the physics of the reference step, not a
        fitted constant.
        """
        retention = source_erp_run["retention"]
        assert retention["arf_scalar"] == pytest.approx(
            1.0 - retention["mean_pattern"] ** 2 / (retention["pattern"] ** 2).mean()
        )
        assert 0.0 < retention["arf_scalar"] <= 1.0
        # Documented for this fixture (spatially balanced midline dipole): the
        # correction is small; the un-normalised and normalised values differ by
        # only ~1.5 %.
        assert 0.9 < retention["arf_scalar"] <= 1.0
        assert retention["unnormalised"] == pytest.approx(
            retention["arf_scalar"] * retention["retention"], rel=0.1
        )

    def test_retention_measure_is_non_vacuous(self, source_erp_run):
        """A shifted template must score far below the real retention.

        The same per-channel gain, projected onto the injected waveform shifted
        by :data:`CONTROL_SHIFT_S` (no temporal overlap with the ERP), measures
        the part of the evoked that is *not* the injected ERP; if the measure
        passed on that control it would not be measuring the ERP.
        """
        setup = source_erp_run["setup"]
        retention = source_erp_run["retention"]
        evoked = source_erp_run["evoked"]
        control = _control_waveform(setup["w"])
        control_gain = _projection_gain(evoked.data, control)
        eeg_idx = mne.pick_types(evoked.info, eeg=True)
        control_retention = float(
            np.median(control_gain[eeg_idx][retention["selected"]]
                      / retention["arf"][retention["selected"]])
        )
        assert abs(control_retention) < 0.2 * retention["retention"], (
            f"shifted-template control {control_retention:.3f} is not far below "
            f"the real retention {retention['retention']:.3f}"
        )
