# EEGCPM Preprocessing Exclusion Criteria

**Date**: 2026-10-16 (regenerated from the code in B6 task 16)
**Purpose**: the exclusion criteria the *current* preprocessing code actually applies, with
the exact reason strings and metrics persisted in the run sidecar.
**Source of truth**: the code. If this document and the code disagree, the code wins — fix
the document.

This document replaces the 2025-11-28 version, which described a pre-B1 pipeline (8 stages,
channel dropping, ASR disabled, configs that no longer exist) and no longer matched the code.

---

## 1. How exclusion works (fail-loud semantics)

A recording is **excluded** when a step raises `RecordingExcluded`
(`eegcpm/modules/preprocessing/steps/base.py`) — a hard inclusion gate that no later step can
repair. The executor (`PreprocessingPipeline.process`,
`eegcpm/modules/preprocessing/pipeline_builder.py`) then:

- records `metadata['step_status'][step] = 'excluded'`, `metadata[step] = {applied: False,
  excluded: True, reason}`, `metadata['excluded'] = {step, reason, **metrics}`;
- aborts the remaining steps (`not_run`);
- sets `success = False`, clears `outputs['data']`/`outputs['ica']`;
- **never writes `*_preprocessed_raw.fif`**; writes `{stem}_preprocessing_EXCLUDED.json`;
- the CLI / `run_processor` persist `ProcessingStatus.EXCLUDED` (never `COMPLETED`).

A **required step** (every step of the `hbn_langer` chain) that raises, fails
`validate_inputs`, reports `skipped: True` / `applied: False` (outside its allow-listed no-op
reasons), or returns a non-`Raw` fails the run the same way: status `failed`, abort, no FIF,
`{stem}_preprocessing_FAILED.json`, `ProcessingStatus.FAILED`. Optional steps (`required:
False` in an explicit step list) are recorded `failed_tolerated` + `metadata['degraded']` and
the run continues.

Status vocabulary in the sidecar: `ok` | `noop` (allow-listed) | `disabled` | `failed` |
`failed_tolerated` | `excluded` | `not_run` (after an abort).

Allow-listed no-ops (not failures):

| Step | Reason | Meaning |
|---|---|---|
| `interpolate` | `no_bad_channels` | nothing to interpolate |
| `montage` | `montage_already_set` | the recording already carries a montage |

---

## 2. Recording-level exclusion gates

### 2.1 `too_many_bad_channels` — bad-channel budget

**Where**: `detect_bad_channels` (`eegcpm/modules/preprocessing/steps/bad_channels_pinned.py`),
used by the chain's `_BadChannelDetectorWrapper` (step 4).

**Criterion**: more than `max_bad` detections, `MAX_BAD_CHANNELS = 11` (≤10 % of the 109
scalp channels; Cz exempt). The detection itself never truncates: the full list is reported
and the recording is excluded instead of processing a silently altered analysis set.

**Metrics**: `n_detected`, `max_bad_channels`, `detected`.

What counts as a bad channel (mark-only; interpolation is the separate next step):

| Criterion | Rule |
|---|---|
| `flat` | `ptp == 0` or `variance <= 1e-20 V²` (absolute floor) |
| `flat` (near-flat floor, B4-D2) | `variance < 1e-3 × median` of the eligible channels — closes the blind spot above the absolute floor (e.g. a 1 nV dead electrode) |
| `variance` | log-variance robust z (`median/MAD`, std fallback) `> 3.0` (upper tail) |
| `hf_power` | 25–40 Hz Welch power, log-scale robust z `> 3.0` (upper tail; documented deviation from Langer's band because the detector runs after the 0.1–45 Hz low-pass) |
| `kurtosis` | excess kurtosis robust z `> 3.0` (upper tail) computed on **blink-free samples** (±200 ms around `find_eog_events` blinks; EOG-typed channel or most anterior non-flat EEG proxy; fallback to the full signal is recorded in `blink_mask.reason`) |
| `non_finite_<criterion>` | a non-finite z on an eligible channel |

Exempt: the reference channel from `metadata['channel_roles']['reference_channel']` (Cz),
passed explicitly as `exempt=(...)` — it is never flagged and never counted.

### 2.2 `too_many_bad_channels_for_interpolation` — interpolation budget

**Where**: `InterpolateBadChannelsStep.process`
(`eegcpm/modules/preprocessing/steps/interpolate.py`), chain step 5.

**Criterion**: `pct_bad > max_bad_percent` (chain: `11/109 × 100 ≈ 10.09 %`).
**Channels are never dropped** (the pre-B1 behaviour silently changed the analysis set).

**Metrics**: `n_bad`, `pct_bad`, `max_bad_percent`.

### 2.3 `block_rejection_fraction_exceeded` — contaminated recording

**Where**: `_BlockRejectionStepWrapper.process`
(`eegcpm/modules/preprocessing/hbn_langer_chain.py`), chain step 8.

**Criterion**: 2-s blocks whose **median** channel peak-to-peak exceeds 100 µV
(whole-head artifacts; single-channel spikes are handled by the detector + ASR).
`fraction > max_rejected_fraction` (0.20) → the recording is excluded.
`BAD_BLOCK` annotations are **appended before the raise**, so an excluded recording still
carries its rejection marks (P1-F4); `n_blocks == 0` (recording shorter than one block)
returns `applied: False`, which fails the required step.

**Metrics**: `n_blocks_rejected`, `n_blocks_total`, `rejected_fraction`,
`max_rejected_fraction`, `bad_block_onsets_s`.

---

## 3. Step-level hard failures (required steps)

These raise ordinary exceptions (not `RecordingExcluded`); on a required chain step they fail
the run (no FIF, `FAILED` sidecar). Listed with their message anchors:

| Step | Failure | Message anchor |
|---|---|---|
| `channel_roles` | missing `channel_roles` metadata downstream | `KeyError` |
| `zapline` | fewer than 2 EEG channels / `fline ≥ Nyquist` | `at least 2 EEG channels`, `Nyquist` |
| `zapline` | backend output missing a probed key | `KeyError` (`noisefreqs`, `noise_freq_<f>`) |
| `filter` / `interpolate` | MNE errors (e.g. interpolation without digitization) | MNE message |
| `bad_channel_detection` | failing RANSAC/PREP backend (never "0 bad channels") | `RANSAC bad-channel detection failed`, `PREP bad-channel detection failed` |
| `eog_regression` | no EOG channels present | `applied: False`, reason `no_eog_channels_present` (required step → run fails) |
| `asr` | recording shorter than the 60-s calibration | `Data too short for ASR calibration` |
| `asr` | <2 EEG channels after exclusions | `ASR needs at least 2 EEG channels` |
| `asr` | non-finite input/output or a shape change | `non-finite`, `ASR changed the data shape` |
| `block_rejection` | recording shorter than one 2-s block | `applied: False`, reason `recording shorter than one block` |
| `reference` | post-reference `Cz` variance not finite / `<= 0` | `post-reference variance of 'Cz'` |
| `reference` | `type='rest'` (not implemented) | `NotImplementedError` |
| `montage` (standalone) | EEG channel with NaN/zero position after `set_montage` | `without a valid position` |
| `ica` (standalone) | `n_components='rank-1'` with rank < 2 | `needs a data rank >= 2` |
| executor | required step failed/skipped/no-op/validation-failed | step status `failed` |

---

## 4. Marked but NOT excluded (annotations only)

These never remove samples or channels; they mark the timeline (append-only) and are consumed
downstream (`reject_by_annotation`, QC, covariates):

| Mark | Source | Meaning |
|---|---|---|
| `BAD_asr_window` | `ASRStep` from eegprep `clean_windows` | windows ASR flagged (data are repaired at full length; `repaired_fraction`/`asr_burden` in metadata) |
| `BAD_BLOCK` | block rejection | 2-s block above the ptp threshold (exclusion only above the fraction gate) |
| `BAD_amplitude` / `BAD_gradient` / `BAD_flatline` / `BAD_muscle` | `ArtifactAnnotationStep` (standalone) | artifact segments |

ASR **repairs** rather than excludes: the recording keeps its sample count, event samples and
`info`; `n_samples_before == n_samples_after` is reported.

---

## 5. Provenance of exclusions

Every run with an output directory writes a JSON sidecar
(`{stem}_preprocessing.json` / `_FAILED.json` / `_EXCLUDED.json`) containing
`step_status`, per-step metadata, `errors`, `warnings`, `excluded`, `degraded`,
`resolved_config`, `config_hash` (the golden resolved-config hash) and `library_versions`.
The workflow DB row records the same `config_hash` and the final status
(`COMPLETED` / `EXCLUDED` / `FAILED`).
