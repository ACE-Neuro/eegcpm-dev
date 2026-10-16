"""End-to-end synthetic run of the canonical ``hbn_langer`` chain (P1 C2, plan §B7 task 6).

One synthetic EGI-129 recording (129 names incl. ``Cz``, 250 Hz, 150 s,
``first_samp != 0``, ``meas_date`` set, 1/f + 60 Hz line + blinks on the EGI EOG
channels + an N1/P2 ERP at every 2-s ``stim`` annotation, ``Cz`` all zeros)
goes through the **real** chain (zapline → 0.1-45 Hz filter → bad-channel
detector → interpolate → EOG regression → ASR → block rejection → average
reference) and must come out with:

- identical ``stim`` event samples and every input annotation preserved;
- a FIF that reloads with the same annotations and ``meas_date``;
- a sidecar whose ``step_status`` is all ``ok``/``noop`` (no step silently
  skipped, no ``degraded``);
- the pinned ``config_hash == GOLDEN_RESOLVED_HASHES['hbn_langer']``;
- ``var(Cz) > 0`` after the average reference (the flat pre-reference channel
  must have been re-referenced, not silently dropped).

The second variant contaminates 30 % of the 2-s blocks with an in-band
high-amplitude oscillation: the block-rejection gate must **exclude** the
recording (sidecar ``*_preprocessing_EXCLUDED.json``, no FIF).

**Fidelity assertions (B7 rev2 fix 1b).** The clean fixture's ASR burden is
bounded (``repaired_fraction < 0.85``; measured 0.762 with this fixture — the
audit measured 0.76) and a second run of the same fixture with the ERP injected
on the **anterior EEG subset** (``erp_mode='anterior'``) asserts the injected
N1/P2 template is still measurable in the chain output (median per-channel
evoked↔template correlation ≥ 0.5; measured 0.88). The subset injection exists
because the base fixture's ERP is *common* to all channels and the chain's
average-reference step cancels a common-mode ERP exactly (measured r = −0.98 on
``Cz``): with the common injection the retention measure would be vacuous
(0.07), not a statement about ASR. Both bounds are recorded in
``VERIFICATION_TODO.md``; they tighten after the ASR high-pass design decision
(plan §12.6), which is recorded but deliberately not implemented in B7.

Marked ``chain_e2e`` + ``slow``; the CI workflow runs ``pytest -m chain_e2e``.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np
import pytest

from eegcpm.core.config_hash import GOLDEN_RESOLVED_HASHES
from eegcpm.modules.preprocessing import PreprocessingPipeline
from eegcpm.modules.qc.hbn_recording_qc import EOG_HYDROCEL_NAMES

pytestmark = [pytest.mark.chain_e2e, pytest.mark.slow]

SFREQ = 250.0
DURATION_S = 150.0
FIRST_SAMP = 1000
#: Block length of the chain's block-rejection gate (2-s windows).
BLOCK_S = 2.0
#: Per-channel amplitude heterogeneity (log-normal sigma of the 1/f noise gain).
CHANNEL_GAIN_SIGMA = 0.35
#: Amplitude of the in-band contamination in the "excluded" variant. ASR repairs
#: only a few percent of these samples, so the block gate still sees > 100 µV.
CONTAMINATION_PTP_V = 400e-6
#: Fraction of blocks to contaminate (> 0.20 max_rejected_fraction).
CONTAMINATED_FRACTION = 0.30


def _one_over_f(rng, n_channels, n_times, sfreq):
    """1/f-ish coloured noise (unit SD per channel)."""
    white = rng.standard_normal((n_channels, n_times))
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    shape = 1.0 / np.maximum(freqs, 1.0)
    shape[0] = 0.0
    coloured = np.fft.irfft(np.fft.rfft(white, axis=1) * shape, n=n_times, axis=1)
    return coloured / coloured.std(axis=1, keepdims=True)


def make_gsn129_recording(
    *,
    contaminated_fraction: float = 0.0,
    contamination_ptp_v: float = CONTAMINATION_PTP_V,
    contaminated_from_start: bool = False,
    erp_mode: str = "common",
    seed: int = 7,
) -> mne.io.BaseRaw:
    """Synthetic GSN-HydroCel-129 recording used by the chain e2e tests.

    Fixture design notes (probed 2026-10-16 while writing this test):

    - the 1/f noise is scaled by a per-channel log-normal gain (sigma
      :data:`CHANNEL_GAIN_SIGMA`), which keeps the bad-channel detector's
      relative-variance/kurtosis statistics homogeneous — a fixture with
      identical channel variances makes the detector's MAD collapse and it
      flags channels on noise alone;
    - the contamination is scaled by the *same* per-channel gain, tapered at the
      block edges and applied to the **EEG channels only** (contaminating the
      EOG channels lets the chain's EOG regression predict and remove it), so
      it survives the filter and the regression;
    - ``contaminated_from_start=True`` puts the contaminated blocks in the
      **ASR calibration window** (the chain's fixed first 60 s). ASR repairs
      bursts that lie *outside* its calibration window (measured: 22/75
      contaminated blocks drop to 0/75 after ASR), so a recording is only
      excluded by the block-rejection gate when the calibration itself is
      contaminated — which is the realistic "30 % contaminated blocks" case.
      **An interpolated channel makes the ASR calibration covariance exactly
      singular** (spherical-spline interpolation is an exact linear combination
      of the remaining channels) and ``eegprep.clean_asr`` then returns
      ~1e15 µV; ``ASRStep`` excludes the channels a previous ``interpolate``
      step reconstructed (B7-N1), which this fixture exercises.
    - ``erp_mode='anterior'`` injects the same N1/P2 template on the anterior
      EEG subset only (montage y ≥ median, EOG excluded). The default
      ``'common'`` injection puts the identical template on every channel; the
      chain's average-reference step cancels a common-mode ERP exactly
      (measured r = −0.98 on Cz), so a retention assertion on the common
      fixture would be vacuous — it would measure the reference step, not the
      chain's fidelity (B7 rev2 fix 1b).
    """
    if erp_mode not in ("common", "anterior"):
        raise ValueError(f"erp_mode must be 'common' or 'anterior', got {erp_mode!r}")
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    ch_names = list(montage.ch_names)
    assert len(ch_names) == 129 and "Cz" in ch_names
    n_channels = len(ch_names)
    n_times = int(DURATION_S * SFREQ)
    rng = np.random.default_rng(seed)
    t = np.arange(n_times) / SFREQ

    channel_gain = np.exp(rng.normal(0.0, CHANNEL_GAIN_SIGMA, n_channels))
    data = _one_over_f(rng, n_channels, n_times, SFREQ) * (15e-6 * channel_gain)[:, None]

    # 60 Hz line noise (zapline removes it; 60 Hz is outside the detector band)
    data += 5e-6 * np.sin(2 * np.pi * 60.0 * t)[None, :]

    # Blinks on the EGI EOG channels only (the detector runs on EEG picks)
    eog_idx = [ch_names.index(ch) for ch in EOG_HYDROCEL_NAMES if ch in ch_names]
    assert len(eog_idx) == 9
    blink_wave = np.exp(-0.5 * ((np.arange(int(0.3 * SFREQ)) / SFREQ - 0.15) / 0.06) ** 2)
    for onset in np.arange(5.0, DURATION_S - 5.0, 4.0):
        start = int(onset * SFREQ)
        stop = start + blink_wave.size
        data[eog_idx, start:stop] += (250e-6 * blink_wave)[None, :]

    # N1/P2 ERP at every 2-s stim annotation
    erp_t = np.arange(int(0.5 * SFREQ)) / SFREQ
    erp = (
        -8e-6 * np.exp(-0.5 * ((erp_t - 0.100) / 0.020) ** 2)
        + 6e-6 * np.exp(-0.5 * ((erp_t - 0.200) / 0.040) ** 2)
    )
    stim_onsets = np.arange(2.0, DURATION_S - 1.0, 2.0)
    if erp_mode == "common":
        erp_rows = np.ones(n_channels, dtype=bool)
    else:
        # Anterior EEG subset: the average-reference step cancels a common-mode
        # ERP exactly, so the retention assertion needs a spatially structured
        # injection (B7 rev2 fix 1b; see the docstring).
        erp_rows = _anterior_eeg_mask(ch_names)
    for onset in stim_onsets:
        start = int(onset * SFREQ)
        data[erp_rows, start:start + erp.size] += erp[None, :]

    # Contaminated blocks (the "excluded" variant): sustained in-band oscillation
    # on every **EEG** channel (not on the EOG channels: the chain's EOG
    # regression would otherwise predict and remove the contamination, because
    # it is identical across channels), scaled by the channel gain and tapered
    # at the block edges.
    if contaminated_fraction > 0:
        window = int(BLOCK_S * SFREQ)
        n_blocks = n_times // window
        n_bad = int(round(contaminated_fraction * n_blocks))
        n_taper = int(0.1 * SFREQ)
        envelope = np.ones(window)
        envelope[:n_taper] = 0.5 * (1 - np.cos(np.pi * np.arange(n_taper) / n_taper))
        envelope[-n_taper:] = envelope[:n_taper][::-1]
        eeg_rows = np.ones(n_channels, dtype=bool)
        eeg_rows[eog_idx] = False
        if contaminated_from_start:
            # Cover the ASR calibration window (the chain's fixed first 60 s).
            bad_blocks = np.arange(min(n_bad, 30))
        else:
            bad_blocks = np.sort(rng.choice(n_blocks, size=n_bad, replace=False))
        for block in bad_blocks:
            start = int(block) * window
            local_t = np.arange(window) / SFREQ
            contamination = (
                contamination_ptp_v
                * channel_gain[:, None]
                * envelope[None, :]
                * np.sin(2 * np.pi * 10.0 * local_t)[None, :]
            )
            data[eeg_rows, start:start + window] += contamination[eeg_rows]

    # Cz is the flat reference channel: all zeros BEFORE the chain (the average
    # reference must give it a non-zero variance afterwards).
    data[ch_names.index("Cz")] = 0.0

    info = mne.create_info(ch_names, SFREQ, ["eeg"] * n_channels)
    info.set_montage(montage, verbose=False)
    raw = mne.io.RawArray(data, info, first_samp=FIRST_SAMP, verbose=False)
    raw.set_meas_date(datetime(2020, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(
        mne.Annotations(
            stim_onsets,
            np.zeros_like(stim_onsets),
            ["stim"] * len(stim_onsets),
        )
    )
    return raw


def _event_samples(raw) -> np.ndarray:
    events, _ = mne.events_from_annotations(raw, verbose=False)
    return events[:, 0].copy()


def _erp_template() -> np.ndarray:
    """The N1/P2 template injected by :func:`make_gsn129_recording`."""
    erp_t = np.arange(int(0.5 * SFREQ)) / SFREQ
    return (
        -8e-6 * np.exp(-0.5 * ((erp_t - 0.100) / 0.020) ** 2)
        + 6e-6 * np.exp(-0.5 * ((erp_t - 0.200) / 0.040) ** 2)
    )


def _anterior_eeg_mask(ch_names) -> np.ndarray:
    """Boolean mask of the ``erp_mode='anterior'`` injection rows.

    Computed over the **full 129-channel montage** (exactly as
    :func:`make_gsn129_recording` does — a mask recomputed over a reduced
    channel list would shift the median y and pick a different subset), then
    mapped onto ``ch_names``. Anterior EEG channels (montage y >= median), with
    the EOG channels and the flat ``Cz`` reference excluded (Cz is zeroed after
    the injection).
    """
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    all_names = list(montage.ch_names)
    pos = montage.get_positions()["ch_pos"]
    y = np.array([pos[ch][1] for ch in all_names])
    mask_all = y >= np.median(y)
    for name in list(EOG_HYDROCEL_NAMES) + ["Cz"]:
        if name in all_names:
            mask_all[all_names.index(name)] = False
    return np.array([bool(mask_all[all_names.index(ch)]) for ch in ch_names])


def _template_retention(raw, channels, template=None) -> float:
    """Median per-channel correlation of the evoked with the injected template.

    The template-based fidelity measure (B7 rev2 fix 1b): stim-locked segments
    are averaged (74 trials on this fixture), the 0.5-s post-stim window is
    baseline-corrected, and each injected channel contributes the Pearson
    correlation between its evoked waveform and the template. The median over
    channels is the reported retention. ``template`` overrides the injected
    template (used by the non-vacuity control).
    """
    if template is None:
        template = _erp_template()
    half = int(round(0.05 * SFREQ))
    stim_onsets = np.arange(2.0, DURATION_S - 1.0, 2.0)
    segments = []
    for onset in stim_onsets:
        start = int(round(onset * SFREQ)) - half
        stop = start + template.size + 2 * half
        if start < 0 or stop > raw.n_times:
            continue
        segments.append(raw.get_data(picks=channels, start=start, stop=stop))
    assert segments, "no usable stim-locked segments"
    evoked = np.stack(segments).mean(axis=0)
    crop = evoked[:, half:half + template.size]
    crop = crop - crop[:, :half].mean(axis=1, keepdims=True)
    corrs = [
        float(np.corrcoef(seg, template)[0, 1])
        for seg in crop
        if float(np.std(seg)) > 0
    ]
    assert corrs, "all evoked channels are flat"
    return float(np.median(corrs))


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    """Run the real chain once on the clean synthetic recording."""
    raw = make_gsn129_recording()
    events_in = _event_samples(raw)
    annotations_in = list(raw.annotations.description)
    meas_date_in = raw.info["meas_date"]
    output_dir = tmp_path_factory.mktemp("chain_e2e_clean")
    module = PreprocessingPipeline("hbn_langer", output_dir=output_dir)
    result = module.process(raw, subject_id="e2e01", session_id="01", run_id="1")
    return {
        "raw_in": raw,
        "events_in": events_in,
        "annotations_in": annotations_in,
        "meas_date_in": meas_date_in,
        "result": result,
        "output_dir": output_dir,
    }


class TestHbnChainE2E:
    def test_run_succeeds(self, base_run):
        result = base_run["result"]
        assert result.success, f"errors={result.errors} warnings={result.warnings}"
        assert result.outputs["data"] is not None

    def test_step_status_all_ok_or_noop(self, base_run):
        result = base_run["result"]
        step_status = result.metadata["step_status"]
        # Every chain step must be recorded; a silently skipped step is a failure.
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
        assert not result.metadata.get("degraded")
        assert not result.metadata.get("excluded")

    def test_stim_events_identical(self, base_run):
        events_out = _event_samples(base_run["result"].outputs["data"])
        assert np.array_equal(events_out, base_run["events_in"])

    def test_all_input_annotations_present(self, base_run):
        out = base_run["result"].outputs["data"]
        descriptions_out = list(out.annotations.description)
        for desc in base_run["annotations_in"]:
            assert desc in descriptions_out
        # The chain may only ADD BAD_* annotations.
        added = set(descriptions_out) - set(base_run["annotations_in"])
        assert all(desc.startswith("BAD_") for desc in added), added

    def test_fif_reloads_with_annotations_and_meas_date(self, base_run):
        result = base_run["result"]
        fif_files = [Path(p) for p in result.output_files
                     if str(p).endswith("_preprocessed_raw.fif")]
        assert len(fif_files) == 1
        reloaded = mne.io.read_raw_fif(fif_files[0], preload=False, verbose=False)
        assert reloaded.info["meas_date"] == base_run["meas_date_in"]
        assert list(reloaded.annotations.description) == list(
            result.outputs["data"].annotations.description
        )
        assert np.array_equal(
            _event_samples(reloaded), base_run["events_in"]
        )

    def test_sidecar_records_golden_hash_and_provenance(self, base_run):
        output_dir = base_run["output_dir"]
        sidecars = sorted(output_dir.glob("*_preprocessing.json"))
        assert len(sidecars) == 1
        payload = json.loads(sidecars[0].read_text())
        assert payload["config_hash"] == GOLDEN_RESOLVED_HASHES["hbn_langer"]
        # `resolved_config` is the executor's `get_config()` payload: the
        # resolved step list (with every constructor parameter) + spec deviations.
        step_names = [step["name"] for step in payload["resolved_config"]["steps"]]
        assert step_names == [
            "channel_roles", "zapline", "filter", "bad_channel_detection",
            "interpolate", "eog_regression", "asr", "block_rejection", "reference",
        ]
        assert payload["resolved_config"]["spec_deviations"]
        assert "library_versions" in payload
        assert payload["step_status"]["asr"] == "ok"
        # The resolved ASR config must describe the fixed-first-60 s path.
        asr_cfg = next(
            step for step in payload["resolved_config"]["steps"] if step["name"] == "asr"
        )
        assert asr_cfg["params"]["calibration_window"] == "fixed_first_60s"
        assert asr_cfg["params"]["train_duration"] == 60.0

    def test_cz_variance_non_zero_after_reference(self, base_run):
        out = base_run["result"].outputs["data"]
        cz = out.get_data(picks=["Cz"])[0]
        assert np.var(cz) > 0, "Cz stayed flat after the average reference"

    def test_no_bad_blocks_in_clean_recording(self, base_run):
        block_meta = base_run["result"].metadata["block_rejection"]
        assert block_meta["n_blocks_rejected"] == 0
        assert block_meta["rejected_fraction"] == 0.0

    def test_clean_fixture_asr_burden_bound(self, base_run):
        """B7 rev2 fix 1b: bound on the clean fixture's ASR repair burden.

        The audit measured ``repaired_fraction = 0.76`` on this fixture and
        probed 94 %/<0.5 Hz + 52 %/0.5-2 Hz power removed — ``eegprep.clean_asr``
        at cutoff 20 on 0.1-Hz-high-passed data rewrites most of a *clean*
        recording. Measured here: **0.762**. The bound is deliberately loose
        (>0 is already true) but non-vacuous: it fails if the distortion grows
        past 85 % of the recording. It tightens after the recorded ASR
        high-pass design decision (plan §12.6) — do not raise it without that
        decision.
        """
        repaired = base_run["result"].metadata["asr"]["repaired_fraction"]
        assert repaired > 0.0, "ASR repaired nothing at all — fixture or step changed"
        assert repaired < 0.85, (
            f"ASR repaired {repaired:.3f} of a clean recording "
            "(measured 0.762; the audit measured 0.76)"
        )

    def test_qc_row_reads_chain_metadata(self, base_run):
        """The recording QC row must consume the chain metadata truthfully (P1-F10)."""
        from eegcpm.modules.qc.hbn_recording_qc import OCULAR_GATE, recording_qc_metrics

        result = base_run["result"]
        row = recording_qc_metrics(
            result.outputs["data"],
            task="RestingState",
            subject_id="e2e01",
            chain_metadata=result.metadata,
            raw_pre=base_run["raw_in"],
        )
        detector = result.metadata["bad_channel_detection"]
        assert row["bad_channels_n"] == len(detector["detected"])
        assert row["bad_channels_detected"] == list(detector["detected"])
        assert row["block_n_rejected"] == 0
        assert row["block_rejected_fraction"] == 0.0
        assert row["asr_burden"] is not None
        assert row["asr_burden"] == result.metadata["asr"]["asr_burden"]
        # The ocular gate ran on the pre-chain recording: the blinks in this
        # fixture live on the EOG channels only (no frontal leakage), so the
        # frontal blink-locked amplitude is *not* reduced — the gate reports
        # exactly that instead of a vacuous pass. Its semantics (pass/fail) are
        # covered by the dedicated unit tests in test_qc.py.
        assert row["blink_channel"] == "E8"
        assert row["blink_n"] > 0
        assert row["blink_residual_ratio"] is not None
        assert row["blink_residual_pass"] == (
            row["blink_residual_ratio"] < OCULAR_GATE
        )
        assert "residual_ocular_maxcorr_insample" in row

    def test_asr_excludes_interpolated_channels(self, base_run):
        """B7-N1: channels a previous interpolate step reconstructed stay out of ASR."""
        detector = base_run["result"].metadata["bad_channel_detection"]
        interpolate_meta = base_run["result"].metadata["interpolate"]
        asr_meta = base_run["result"].metadata["asr"]
        interpolated = list(interpolate_meta.get("interpolated_channels") or [])
        # The fixture must actually exercise the exclusion (non-vacuous).
        assert interpolated, "fixture detected no bad channels to interpolate"
        assert interpolated == list(detector["detected"])
        assert asr_meta["interpolated_excluded_from_asr"] == interpolated
        for name in interpolated:
            assert name in asr_meta["excluded_from_asr"]
        assert asr_meta["n_channels_asr"] == (
            109 - 1 - len(interpolated)  # 109 EEG incl. Cz; minus Cz and the interpolated
        )
        # B7 rev2 fix 7: the excluded channels are re-interpolated from the
        # ASR-cleaned data (recorded), so they stay consistent with the
        # repaired neighbours.
        assert asr_meta["reinterpolated_after_asr"] == interpolated
        assert asr_meta["reinterpolation_mode"] == interpolate_meta["mode"]
        for name in interpolated:
            assert asr_meta["reinterpolation_relative_norm"][name] > 0

    def test_interpolated_channel_matches_fresh_reinterpolation(self, base_run):
        """B7 rev2 fix 7 end-to-end: the saved interpolated channel is the spline
        reconstruction of the data it sits in.

        Before the fix the channel held its pre-ASR spline values; a fresh
        re-interpolation from the chain output diverged by 1.02/1.09 relative
        norm on this fixture (probed 2026-10-16, matching the audit's ~98 %).
        The chain's later steps (OLS EOG regression, average reference) are
        linear and preserve an interpolation relation whose coefficients sum to
        1, so the invariant survives to the final output.
        """
        result = base_run["result"]
        out = result.outputs["data"]
        interpolated = list(result.metadata["interpolate"]["interpolated_channels"])
        assert interpolated
        for name in interpolated:
            fresh = out.copy()
            fresh.info["bads"] = [name]
            fresh.interpolate_bads(reset_bads=False, mode="accurate", verbose=False)
            stored = out.get_data(picks=[name])[0]
            fresh_data = fresh.get_data(picks=[name])[0]
            rel = float(
                np.linalg.norm(stored - fresh_data) / np.linalg.norm(fresh_data)
            )
            assert rel < 0.05, (
                f"{name}: the saved interpolated channel diverges from a fresh "
                f"re-interpolation of the chain output by {rel:.3f} relative "
                "norm (1.02/1.09 before B7 rev2 fix 7)"
            )


@pytest.fixture(scope="module")
def contaminated_run(tmp_path_factory):
    """Chain run on the 30 %-contaminated recording (blocks in the calibration window)."""
    raw = make_gsn129_recording(
        contaminated_fraction=CONTAMINATED_FRACTION,
        contaminated_from_start=True,
        seed=11,
    )
    # Captured BEFORE the chain runs: the exclusion must not move the caller's
    # timeline, and the comparison must be against this capture — not against a
    # re-read of the same object (the old test compared raw_in with itself).
    events_in = _event_samples(raw)
    annotations_in = list(raw.annotations.description)
    output_dir = tmp_path_factory.mktemp("chain_e2e_contaminated")
    module = PreprocessingPipeline("hbn_langer", output_dir=output_dir)
    result = module.process(raw, subject_id="e2e02", session_id="01", run_id="1")
    return {
        "result": result,
        "output_dir": output_dir,
        "raw_in": raw,
        "events_in": events_in,
        "annotations_in": annotations_in,
    }


class TestHbnChainE2EContaminated:
    """30 % contaminated blocks → the run must be EXCLUDED, with no FIF."""

    def test_run_excluded(self, contaminated_run):
        result = contaminated_run["result"]
        assert not result.success
        excluded = result.metadata.get("excluded")
        assert excluded is not None, result.errors
        assert excluded["step"] == "block_rejection"
        assert excluded["reason"] == "block_rejection_fraction_exceeded"
        assert excluded["n_blocks_rejected"] / excluded["n_blocks_total"] > 0.20
        assert excluded["rejected_fraction"] > 0.20

    def test_excluded_sidecar_and_no_fif(self, contaminated_run):
        output_dir = contaminated_run["output_dir"]
        assert list(output_dir.glob("*_preprocessed_raw.fif")) == []
        assert list(output_dir.glob("*_preprocessing.json")) == []
        excluded_sidecars = list(output_dir.glob("*_preprocessing_EXCLUDED.json"))
        assert len(excluded_sidecars) == 1
        payload = json.loads(excluded_sidecars[0].read_text())
        assert payload["excluded"]["reason"] == "block_rejection_fraction_exceeded"
        assert payload["step_status"]["block_rejection"] == "excluded"
        # Steps after the exclusion never ran.
        assert payload["step_status"]["reference"] == "not_run"
        assert payload["config_hash"] == GOLDEN_RESOLVED_HASHES["hbn_langer"]
        # The exclusion record carries the block marks (the executor works on a
        # copy of the caller's raw, so the marks live here, not on the input).
        n_rejected = payload["excluded"]["n_blocks_rejected"]
        n_total = payload["excluded"]["n_blocks_total"]
        assert n_rejected / n_total > 0.20
        assert payload["excluded"]["rejected_fraction"] > 0.20
        assert len(payload["excluded"]["bad_block_onsets_s"]) == n_rejected
        assert payload["excluded"]["bad_block_onsets_s"][0] == 0.0

    def test_excluded_recording_keeps_stim_annotations(self, contaminated_run):
        raw = contaminated_run["raw_in"]
        descriptions = list(raw.annotations.description)
        assert descriptions.count("stim") > 0
        assert descriptions == contaminated_run["annotations_in"]
        events, _ = mne.events_from_annotations(raw, verbose=False)
        assert len(events) > 0
        # Every stim annotation still sits on its original sample (the chain
        # works on a copy; the exclusion must not move the caller's timeline).
        # Compared against the capture taken BEFORE the chain ran — not against
        # a re-read of the same object (B7 rev2 fix 8: the old assertion was a
        # tautology).
        assert np.array_equal(events[:, 0], contaminated_run["events_in"])
        assert len(contaminated_run["events_in"]) > 0


@pytest.fixture(scope="module")
def erp_run(tmp_path_factory):
    """Chain run with the ERP injected on the anterior subset (B7 rev2 fix 1b).

    ``erp_mode='anterior'`` keeps the injected template measurable after the
    chain's average-reference step; the default common-mode injection is
    cancelled by that step exactly (see the fixture docstring).
    """
    raw = make_gsn129_recording(erp_mode="anterior")
    output_dir = tmp_path_factory.mktemp("chain_e2e_erp")
    module = PreprocessingPipeline("hbn_langer", output_dir=output_dir)
    result = module.process(raw, subject_id="e2e03", session_id="01", run_id="1")
    return {"result": result, "raw_in": raw, "output_dir": output_dir}


class TestHbnChainE2EFidelity:
    """B7 rev2 fix 1b: the chain must not destroy the injected ERP."""

    def test_run_succeeds(self, erp_run):
        result = erp_run["result"]
        assert result.success, f"errors={result.errors} warnings={result.warnings}"

    def test_erp_retention_against_injected_template(self, erp_run):
        """Median per-channel evoked↔template correlation must stay ≥ 0.5.

        Measured 2026-10-16: **0.88** on the chain output (0.62 on the raw
        input — the chain's 0.1-45 Hz filter removes the low-frequency drift
        that limits the input correlation, so the retention improves rather
        than degrades). Threshold 0.5 is the audit's suggested bound; a chain
        that ASR-destroys the ERP (or a fixture regression that makes the ERP
        unmeasurable) fails loudly here.
        """
        result = erp_run["result"]
        out = result.outputs["data"]
        injected = _anterior_eeg_mask(out.ch_names)
        channels = [name for name, keep in zip(out.ch_names, injected) if keep]
        assert len(channels) >= 20, (
            f"only {len(channels)} injected channels survived into the output"
        )
        retention = _template_retention(out, channels)
        assert retention >= 0.5, (
            f"ERP retention {retention:.3f} < 0.5: the chain destroyed the "
            "injected N1/P2 template"
        )

    def test_retention_measure_is_non_vacuous(self, erp_run):
        """The measure must distinguish the ERP from noise (non-vacuity).

        A reversed-template control must score far below the real template on
        the same evoked data.
        """
        out = erp_run["result"].outputs["data"]
        injected = _anterior_eeg_mask(out.ch_names)
        channels = [name for name, keep in zip(out.ch_names, injected) if keep]
        real = _template_retention(out, channels)
        control = _template_retention(out, channels, template=_erp_template()[::-1])
        assert abs(control) < 0.5 * real, (
            f"reversed-template control {control:.3f} is not far below the "
            f"real retention {real:.3f}"
        )
