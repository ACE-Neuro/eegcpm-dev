# EEGCPM Development Changelog

This document chronicles the major development phases and architectural decisions in EEGCPM's evolution.

---

## Phase 2: Data Quality Detection System (Nov 2025)

### Motivation

50% of preprocessing failures were caused by LAPACK errors from rank-deficient data. Needed proactive quality detection before analysis.

### Implementation

**Created:** `eegcpm/modules/preprocessing/data_quality.py`

**Four Detection Methods:**

1. **Flatline Channels**: Variance < 1e-12 V² (disconnected electrodes)
2. **Bridged Channels**: Correlation > 0.98 (gel creates conductive paths)
3. **Data Rank**: SVD-based dimensionality estimation
4. **Comprehensive Assessment**: Combined detection with ICA recommendations

**Preprocessing Integration:**
- Added Step 1.5: Data quality assessment (after montage, before bad channels)
- Modified `_run_ica()` to use rank-based component selection
- Graceful ICA skip if rank too low

**Test Results (sub-NDARZN501NFF):**
- 3 flatline channels detected and removed
- 37 bridged channels detected and removed
- Data rank: 127/129 channels
- ICA: 20 components (rank-based)
- Zero LAPACK errors (previously crashed)

### Impact

- **Before**: 30% success rate (3/10 subjects)
- **After**: 100% success rate on available subjects (8/8)
- **Trade-off**: +3s processing time per subject to eliminate 50% failures

### Key Learnings

1. **Data quality trumps algorithms** - Best methods fail on poor data
2. **Proactive > Reactive** - Check quality BEFORE analysis
3. **Real-world data is messy** - 60% flatlines, 80% bridging in clinical EEG
4. **Rank deficiency is universal** - Fixed ICA components inappropriate
5. **Clear warnings matter** - "3 flatline channels" > "LAPACK error -5"

---

## Stage-First Architecture Refactor (Nov 2025)

### Motivation

Original pipeline-centric architecture caused data duplication when testing analysis variants. Needed flexible mix-and-match of processing stages.

### Architecture Change

**Before (Pipeline-Centric):**
```
eegcpm/pipelines/standard/{subject}/...
derivatives/pipeline-standard/
```

**After (Stage-First):**
```
derivatives/
├── preprocessing/{pipeline}/{subject}/ses-{session}/task-{task}/run-{run}/
├── epochs/{preprocessing}/{task}/{subject}/ses-{session}/
├── source/{preprocessing}/{task}/variant-{method}-{template}/{subject}/...
├── features/{preprocessing}/{task}/{source-variant}/{feature-type}/
└── prediction/{model-name}/
```

### Benefits

1. **Prevents Waste**: Each preprocessing variant reusable across analyses
2. **Flexible Combinations**: Mix preprocessing → epochs → source
3. **Cleaner Organization**: Stage-based directories more intuitive
4. **Single Source of Truth**: EEGCPMPaths eliminates path bugs

### Implementation

**Created:**
- `eegcpm/core/paths.py` (453 lines) - Centralized path management
- `eegcpm/core/dependencies.py` (354 lines) - Dependency resolver
- `tools/migrate_to_stage_first.py` (383 lines) - Migration script

**Modified:**
- `eegcpm/workflow/state.py` - Added stage tracking (current_stage, metadata)
- `eegcpm/core/config.py` - Added BaseStageConfig with dependencies
- `eegcpm/cli/preprocess.py` - Single `--project` argument
- `eegcpm/ui/pages/2_run_selection.py` - Simplified to single project root

### Breaking Changes

1. **Path Arguments**: `--bids-root`, `--derivatives` → `--project`
2. **State DB Location**: `derivatives/.eegcpm/` → `eegcpm/.eegcpm/`
3. **Output Structure**: `eegcpm/pipelines/` → `derivatives/preprocessing/`
4. **QC Reports**: Separate `derivatives/qc/` → Co-located with data

### Migration

```bash
python tools/migrate_to_stage_first.py --project /path --execute --symlinks
```

---

## Naming Convention Standardization (Nov 2025)

### Problem

Inconsistent naming across outputs:
- Mixed case formats (kebab vs snake)
- Redundant information in filenames
- Subject ID inconsistency (with/without `sub-` prefix)
- No version control for pipeline outputs

### Proposed Solution

**Principles:**
1. **Strict BIDS** for data files (`.fif`, `.epo.fif`)
2. **BIDS-inspired** for reports (`.html`, `.json`)
3. **Hierarchical context** - Less repetition
4. **Pipeline versioning** - Track parameters

**Examples:**

SAIIT Task:
```
OLD: task-saiit2afcblock1, task-saiit2afcblock2
NEW: task-saiit_run-01, task-saiit_run-02
```

QC Reports:
```
OLD: NDARZN501NFF_task-saiit2afcblock1_raw-qc.html
NEW: sub-NDARZN501NFF_ses-01_task-saiit_run-01_stage-raw_qc.html
```

### Implementation Priority

**High Priority (Must Have):**
1. Consistent BIDS naming for all data files
2. `task-{name}_run-{N}` instead of custom block names
3. Uniform QC naming (`stage-{name}_qc.html`)
4. Always include `sub-` prefix

**Medium Priority (Should Have):**
5. Pipeline versioning (`eegcpm-v0.1.0/`)
6. `config.yaml` alongside outputs
7. Subject-centric folder organization

**Low Priority (Nice to Have):**
8. BIDS validator compliance
9. Migration of existing data

**Status**: Proposal stage - not yet implemented

---

## QC System Enhancement (Dec 2025)

### Critical Fixes

#### 1. ASR Unit Conversion Bug ⚠️

**Problem**: eegprep stores data in µV, MNE expects Volts. Missing conversion caused 449x scaling error.

**Fix** (`eegcpm/modules/preprocessing/steps/asr.py:177`):
```python
# IMPORTANT: eegprep stores in µV, MNE expects V
data = eeg_cleaned['data'] * 1e-6  # µV → V
```

**Impact**: All preprocessed data amplitudes were wrong by 449x

#### 2. Event Preservation

**Problem**: ASR creates new Raw object without preserving annotations, losing all events.

**Fix** (`eegcpm/modules/preprocessing/steps/asr.py:226-228`):
```python
# Preserve annotations - CRITICAL for epoching
if raw.annotations is not None:
    raw_cleaned.set_annotations(raw.annotations)
```

### QC Report Enhancements

1. **Bad Channels Topography**: Color-coded by reason (dropped=red, interpolated=blue)
2. **PSD Before/After**: Show all channels including interpolated
3. **ICA Component Plots**: Red highlighting for rejected, ICLabel classifications
4. **Show ALL ICA Components**: Dynamic grid, no 20-component limit
5. **Before/After Metrics**: Added Min/Max amplitude rows
6. **Complete ICA Table**: All components with variance explained
7. **ERP Waveforms**: Auto-generated if events available

### ICLabel Migration

**Changed from**: mne-icalabel
**Changed to**: eegprep ICLabel (EEGLAB port)

**Why**: More reliable, better EEGLAB compatibility

**Requires**: PyTorch (CPU version installed)

**Workflow**:
1. Convert MNE Raw → EEGLAB via `eegprep.eeg_mne2eeg()`
2. Add ICA matrices
3. Run `eegprep.iclabel()`
4. Extract 7-class probabilities

### Other Fixes

- **Workflow State JSON**: Fixed serialization error from ICA objects
- **UI Default**: QC reports shown inline by default
- **Montage Restoration**: After ASR to prevent loss

---

## Current Development (Dec 2025)

### Documentation Consolidation

**Goal**: Reduce 18 markdown files → 10 focused documents

**New Structure:**
```
/Users/clive/eegcpm/
├── docs/                  # User/developer documentation
│   └── WORKFLOWS.md       # Multi-run workflow guide
├── planning/              # Architecture documentation
│   ├── QC_SYSTEM.md       # Quality control system
│   └── BAD_CHANNEL_DETECTION.md  # Clustering analysis
└── implementation/        # Historical records
    └── CHANGELOG.md       # This file
```

**Merges Completed:**
- ✅ QC docs (3 → 1): QUALITY_CONTROL_SYSTEM + comparison + multi-pipeline
- ✅ Clustering docs (3 → 1): bad_channel_clustering + implementations + batch
- ✅ Workflow docs (2 → 1): MULTIRUN_WORKFLOW + QUICK_REFERENCE
- ✅ Historical docs (4 → 1): This changelog

**Pending:**
- Move existing docs to new structure
- Create new planning docs (ARCHITECTURE, TASK_SYSTEM, UI_ARCHITECTURE)
- Trim CLAUDE.md (32K → 15-20K)
- Delete obsolete files

---

## References

### PREP Pipeline
Bigdely-Shamlo, N., et al. (2015). The PREP pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers in Neuroinformatics*, 9, 16.

### BIDS Specification
https://bids-specification.readthedocs.io/

### MNE-Python
https://mne.tools/stable/

---

## Remediation batch B1 — executor fail-fast, exclusion, one config schema (2026-10-14)

- B1 (plan `plan/261013-2240-p1p2-remediation.md`; P1-F1/F4/F6/F9-blanket-except/F17-ImportError): the preprocessing executor now fails loudly — required steps that raise, skip, fail validation or report `applied: False` abort the run (`success=False`, `outputs['data']=None`, no `*_preprocessed_raw.fif`), optional steps degrade it (`failed_tolerated` + `degraded`) unless the reason is allow-listed (`noop`); new `RecordingExcluded` (`steps/base.py`) yields an `excluded` run; per-step status is persisted in a `{stem}_preprocessing[_FAILED|_EXCLUDED].json` sidecar and `{stem}_preprocessed_raw.fif` is written only on success; dependency preflight (`check_dependencies`: pyzaplineplus, eegprep+eeglabio, mne-icalabel, autoreject) fails at construction; `PreprocessingPipeline([])` raises; the block-rejection gate appends `BAD_BLOCK` annotations (offset recipe, append-only) **before** raising `RecordingExcluded`; zapline's blanket `except` is gone and `validate`→`validate_inputs`; one pydantic schema (`StepSpec`/`PreprocessingSpec`, `extra='forbid'`, lazy `STEP_REGISTRY` validation, `load_preprocessing_spec`) replaces the flat/keyed dialects in the CLI and `run_processor`; `ProcessingStatus.EXCLUDED` is persisted on both entry points with a per-session/run CLI skip-check; 3 stale keyed-dialect YAMLs under `config/` deleted (default action) and example configs converted to nested `params:`.

## Remediation batch B1 rev1 — Opus audit fixes N1–N4, K1–K8 (2026-10-14)

- B1 rev1 (Opus 5.5 final-audit conditions, `planning/VERIFICATION_TODO.md` rev1 row): N1 — `run_processor` step-log loop iterates `metadata['step_status']` and skips non-dict entries (the top-level `degraded` flag previously crashed it into a false FAILED while a FIF + success sidecar existed); N2 — new `find_preprocessed_raw(run_dir)` helper used by `run_processor.load_processed_runs`, `epoch_combiner`, `aggregate_reports` and `cli/epochs.py` (all previously hardcoded the old `{subject}_preprocessed_raw.fif` stem); N3 — new `clear_run_outputs(run_dir)` called by both entry points before reprocessing so a failed forced re-run leaves no FIF and exactly one `_FAILED.json`; N4 — CLI QC failures are recorded as `metadata['qc_error']` and keep COMPLETED, each run gets its own try/except (an exception marks only that run FAILED, deletes its outputs and rewrites the sidecar) and later runs still proceed; K1 — `montage_already_set` / `already_at_target_sfreq` are allow-listed no-ops (not `required: false`); K2 — interpolate over-budget comment corrected (drop branch is B4/P1-F5 debt); K3/K4 — executor enforces a `(raw, dict)` return and normalizes truthiness (`np.bool_`, non-bool flags) for `skipped`/`applied`; K5 — `variant: hbn_langer` + explicit step list now raises; K6 — `preprocessing` envelope + top-level `steps` ambiguity raises (CLI catches → exit 1); K7 — `update_status` sets `completed_at`/`error` for completed/failed/excluded; K8 — save/sidecar ordering hardened (save failure cleans FIFs and writes `_FAILED.json`; sidecar failure deletes outputs and re-raises); plus `_subject_status([])` raises, a required-`skip_step`→`conditional_skip` test, window-sample-based `BAD_BLOCK` onsets, and exit code 2 when any run failed. Validated (90 focused tests; full suite 12 failed = documented baseline / 505 passed / 2 skipped).

## Remediation batch B1 rev2 — scoped re-audit fixes F1/F2 (2026-10-14)

- F1: the CLI now clears stale run outputs *before* loading the input (`read_raw_eeg`), so a forced re-run that fails at (or before) load cannot leave the previous FIF + success sidecar on disk; F2: `RunProcessor` reuses the shared `core.paths.fail_run_outputs` when a run succeeds and then crashes after `process()` (`preproc_result = None` before the try, cleanup in the `except`), so the UI entry point no longer keeps success-looking outputs beside a FAILED state. Tests added: forced re-run load-failure cleanup (F1), multi-run load failure still processes later runs with exit code 2, post-success crash cleanup + `load_processed_runs` reports no successful run (F2). Suite: focused 93 passed; full 12 failed (documented baseline) / 508 passed / 2 skipped.

## Remediation batch B2 — source-ERP core: anatomical labels, unique names, signed output (2026-10-14)

- B2 (plan `plan/261013-2240-p1p2-remediation.md`; P2-F1/F2/F3/F4, F11 estimator+guards and F13 `validate_input` moved forward, F16 dipole test): CONN ROI names are now unique (`{network}.{name}_{L|R}`, module-level uniqueness assertion) with `MIDLINE_X_MM=5.0`/`ROI.is_midline`/`SOURCE_EXCLUDED_NETWORKS` metadata; new `eegcpm/modules/source/labels.py::build_conn_labels` maps CONN MNI centres onto the inverse source space via `mne.vertex_to_mni` (10 mm lateral single-hemi `mne.Label`, 15 mm midline `BiHemiLabel` with both hemis required, per-ROI `radius_overrides`, `min_vertices=3` else `ValueError`, Cerebellar excluded with reasons) and returns `roi_names/roi_coords_mni/roi_radius_mm/roi_vertex_counts/roi_hemi/excluded_rois/excluded_reasons`; `SourceConfig` gains `roi_radius_midline/roi_radius_overrides/roi_min_vertices/subjects_dir`; `SourceReconstructionModule` v0.2.0 now picks EEG-only epochs (`copy().pick('eeg', exclude='bads')`), estimates the noise covariance with the approved estimator (`shrunk`, `tmin=None`, `tmax=0.0`, `rank=None`, `keep_sample_mean=True`) plus baseline/`tmin<0`/finite guards and a sample-count warning (after the forward, so forward errors stay `errors[0]`), builds labels once per instance, extracts **signed** ROI time courses (`pick_ori='normal'` + `mode='mean_flip'`, evoked `apply_inverse`, trials `apply_inverse_epochs(..., return_generator=True)` + `np.stack` → `(n_trials, n_rois, n_times)`) and always writes the label metadata + `sfreq` into `*_roi_tc.npz`; the modulo placeholder extraction is deleted; `EpochsModule.validate_input` uses `mne.BaseEpochs` and `SourceReconstructionModule.validate_input` returns `False` (documented contract deviation, re-checked loudly in `process()`) for non-epochs or missing/zero EEG montage; `source_qc` skips the new metadata keys/`_nave` and no longer guesses L/R suffixes; pytest markers (`slow`/`fsaverage`/`chain_e2e`) + `fsaverage_subjects_dir` fixture (cached-only, never downloads); tests: `test_conn_rois` uniqueness/midline, `test_source_labels` (30 labels, cerebellar exclusion, 12-vertex oct6 minimum, BiHemi geometry, overrides, `min_vertices` raise), `test_source_dipole_groundtruth` (assertions (1)–(6): dominance 178.0 vs 114.9 next-ROI dSPM, corr(target,w)=1.0000, N1 latency 0.1000 s negative for seeds 0/1, signed min<0, counts ≥3/exclusion/uniqueness) + covariance guards, `test_source` executor validation path. Validation: full standard suite **3 failed / 540 passed / 2 skipped** (only the 2 connectivity P4 + 1 robust_pca B6 baseline failures; the 9 source failures are fixed).

## Remediation batch B2 rev1 — audit fixes F2/N1/N2/N3 (2026-10-14)

- B2 rev1 (Opus 5.5 final-audit conditions; approved deviation from plan B2 task 4): **F2** — `SourceReconstructionModule.__init__` no longer calls `mne.datasets.fetch_fsaverage` (it stored the subjects dir and triggered a ~500 MB download for ~25 unmarked construction tests); the config value is stored as-is and a new `subjects_dir` property resolves `fetch_fsaverage(verbose=False)`'s parent on first use and caches it, and both `_get_conn_labels` and `_get_template_forward` (`fs_dir = self.subjects_dir / 'fsaverage'`) read the property so labels and forward share one fsaverage copy; **N1** — the label cache is keyed on the source space vertno arrays (both hemispheres) and **rebuilds** when they change (rebuild, not raise: reusing one module instance across subjects with individual head models is legitimate, and the N2 guard still rejects individual-subject spaces), with a malformed src forcing the `build_conn_labels` validation path; **N2** — `build_conn_labels` raises `ValueError` when `getattr(src, '_subject', None) != subject` so an individual-subject source space is never silently mapped onto the fsaverage surface (MNE only warns); **N3** — unknown `radius_overrides` keys raise `ValueError` (stale pre-B2 names such as `DefaultMode.LP` no longer silently do nothing; an override for an excluded ROI stays a valid key); plus the EEG pick in `process()` moved inside the `try` so an all-bad-channels case returns `success=False` instead of raising. Tests added: 3 × lazy `subjects_dir` (no fetch during `__init__` / resolved once then cached / configured dir never fetches), label-cache vertno keying (same vertno cached incl. a distinct object, different vertno rebuilt), individual-subject source space raise, unknown override raise, all-bad-channels `success=False`. Validation: targeted B2 command **2 failed / 80 passed / 2 skipped** (only the 2 connectivity P4 baseline failures); full standard suite **3 failed / 547 passed / 2 skipped** (2 connectivity P4 + 1 robust_pca B6); dipole ground truth unchanged (seed 0: target 178.0 vs next 114.9, corr 1.0000, N1 −178.0 at 0.1000 s, min vertex count 12).

## Remediation batch B3 — ASR timeline integrity: full-length repair, no `info['temp']` (2026-10-14)

- B3 (plan `plan/261013-2240-p1p2-remediation.md` §B3; P1-F3/F4/F8/F15): `ASRStep` v2.0 now **repairs at full length** — it runs `eegprep.clean_asr` on the EEG picks (`pick_types(eeg=True, exclude='bads')` minus explicit `exclude_channels` minus the `channel_roles` reference channel, `< 2` picks → `ValueError`) with either data-driven calibration (`ref_maxbadchannels=0.075`, `ref_tolerances=(-inf, 5.5)`) or `calibration='fixed_first_60s'` (`ref_maxbadchannels=<ndarray (n_channels, n_samples) in µV>`, probed: a transposed array raises), then checks the returned shape against the input and raises `RuntimeError` on any non-finite output or input (the `np.nan_to_num` zero-fill and the whole `BAD_asr_naninf` branch are deleted), writes back with `raw._data[picks] = repaired * 1e-6` (the `RawArray` rebuild and its montage-restore code are deleted, so `meas_date`/`highpass`/`lowpass`/`bads`/montage/annotations survive), converts the runs of `~eegprep.clean_windows(EEG_rep, max_bad_channels=window_criterion, zthresholds=window_tolerances)` into `BAD_asr_window` annotations via the §0.3 offset recipe and reports repaired (burst-corrected) intervals in metadata only; `append_bad_annotations` (`steps/asr.py`, shared with the block-rejection gate) implements the recipe and additionally corrects mne 1.13.2's `meas_date=None` frame behaviour (probed: `set_annotations` re-adds `raw.first_time` to *every* onset when `orig_time is None`, so a literal append shifted pre-existing annotations by 4 s at `first_samp=1000`); metadata is the full provenance list (`applied/implementation/calibration/n_calibration_samples/cutoff/window_criterion/window_tolerances/window_length/n_channels_asr/excluded_from_asr/n_samples_before==n_samples_after/window_rejected_fraction/n_bad_windows/bad_window_intervals_s/repaired_fraction/repaired_intervals_s/asr_burden`, `asr_burden` = fraction of samples rejected **or** repaired), all `info['temp']` writes are deleted and `validate_inputs` enforces `train_duration` only for the fixed-calibration branch; `_ASRStepWrapper` v2.0 delegates to `ASRStep(calibration='fixed_first_60s')` and returns its metadata unchanged plus `retain_burst_masks` (phantom `calibration_window` metadata gone, unsupported value raises, `check_dependencies` delegates), `_derive_asr_burden` is deleted, `_BlockRejectionStepWrapper` uses the shared append helper, and `hbn_recording_qc` reads `asr_burden`/`window_rejected_fraction`/`repaired_fraction` from `chain_metadata['asr']` instead of `info['temp']`; tests: new `tests/unit/test_asr_timeline.py` (14 test functions, 16 cases after rev1 — event-sample/sample-count/`info` integrity and annotation survival for both calibrations and both `meas_date` branches, `BAD_asr_window` placement, NaN input/output raises, reference-channel exclusion, wrapper metadata truthfulness, JSON-native metadata, executor sidecar provenance, QC chain-metadata mapping, executor-level `train_duration` gate, `BAD_BLOCK` offsets in both frames) and the reachability burden test rewritten against `ASRStep` metadata. Validation (pre-rev1): B3 command `test_asr_timeline + test_preprocessing_reachability + test_preprocessing_chain + test_executor_failure_semantics` → **62 passed** (64 after rev1); full standard suite → **3 failed / 562 passed / 2 skipped** (failures exactly the documented baseline: 2 `mne_connectivity` P4 + 1 `robust_pca` B6); greps clean (`info['temp']` absent from `eegcpm/`, no `RawArray`/`nan_to_num` in `asr.py`, `except` only re-raises `ImportError`).

### B3 rev1 — K3 pre-audit fixes (2026-10-14)

- **B3-K1**: `_ASRStepWrapper` now overrides `validate_inputs` (delegating to the inner `ASRStep` fixed-first-60 s gate) — the canonical chain fails loudly on recordings shorter than `train_duration` instead of silently capping the calibration reference; the false "executor rejects it first" comment is corrected; new test covers the direct and executor paths. **Encoder**: `workflow/state.py::_make_json_serializable` is now numpy-aware (`np.integer`/`np.floating`/`np.bool_`/`ndarray`), keeps `None`, and renders non-finite floats as `"inf"`/`"-inf"`/`"nan"` strings so sidecars stay strict JSON (`n_calibration_samples: null` now survives the round-trip); unit + data-driven sidecar tests added. Suites: B3 scope 16 passed; full 3 failed (baseline) / 566 passed / 2 skipped.

### B3 rev2 — Qwen3.8-Max final-audit conditions (2026-10-14)

- Audit performed by **Qwen3.8-Max** as a user-approved stand-in for Opus 5.5 (provider endpoint outage). Verdict: **approve**; conditions on record/follow-up only. Applied in this rev2: **B3-K6** — `append_bad_annotations` now self-checks the append-only invariant (captures pre-existing annotation sample positions and raises `RuntimeError` if any moved or disappeared; covers the private-state `orig_time` desync MNE's public API never produces) with a test; **B3-K7** — the `clean_windows`-on-repaired-data semantics (residual post-repair badness; fully repaired windows deliberately unmarked) documented at the mask computation, and a spy test now pins that `clean_windows` receives `EEG_rep` (a regression to the pre-repair dict would previously have gone unnoticed); **B3-K8** — `montage_restored` → `montage_present` (+ `run_processor` readers; the phantom-action key name is gone); **B3-K10** — `n_samples_before` captured before the write-back (no longer tautological); **B3-K11** — stale counts annotated; **B3-K3** — dead `has_nan_inf` log block and spurious asrpy "lost montage" warning removed; **data_roundtrip** provenance key added (eegprep round-trips through a float32 µV EEGLAB `.set`). Tracking obligations assigned: **B3-K9** (four uncorrected `artifacts.py` append sites → plan B6 task 17) and **B3-K5** (workflow-level `metadata_json` sanitizer bypass → plan B6 task 18); B8 checklist gains the eeg_mne2eeg temp-file memory/IO note. Suites: `test_asr_timeline.py` 18 passed; full 3 failed (baseline) / 568 passed / 2 skipped.

## Remediation batch B4 — source usability (config, modes, consumer, nave, loud failures) + P1 bad-channel detector (2026-10-15)

- B4 (plan `plan/261013-2240-p1p2-remediation.md` §B4; P2-F5/F6/F7/F8/F9 + P1-F5): the source stage now has one `extra='forbid'` schema (`ForwardConfig`/`InverseConfig`/`SourceOutputConfig`/`SourceConfig`; `inverse.snr_trial`/`method_params`, `output.compute_crosstalk`, `mode: evoked|trials|both`, `min_trials`, `equalize_trials`, `random_seed`, `variant` defaulting to `{method}-CONN32` with mismatch → `ValidationError`); `SourceReconstructionModule` validates through `SourceConfig.model_validate` (flat `snr`/`spacing`/`trial_level` now raise) while keeping the test-facing attributes; the CLI exits 1 on `ValidationError`; the dSPM example config is rewritten to the nested dialect (flat keys/`name`/`use_conn_rois`/`conductivity` dropped); outputs are `{subject}_roi_tc.npz` (`{cond}: (n_rois, n_times)`) for `mode='evoked'`, `{subject}_roi_tc_trials.npz` (`{cond}: (n_trials, n_rois, n_times)`) for `mode='trials'`, both files for `both`, each carrying `{cond}_times`, `{cond}_nave`, `sfreq` + the label metadata, with `output.save_stc` (evoked only)/`save_roi_tc`/`generate_qc` honoured; condition handling is one `_select_condition_epochs` helper (requested = task-config names else `event_id` keys; no matching events → `conditions_missing`; `n < min_trials` → `conditions_skipped` `below_min_trials`; seeded per-condition `equalize_trials` subset in sorted order with short conditions → `below_equalize_trials`; zero produced → `success=False` "no conditions produced; requested=…, missing=…, skipped=…" and nothing written), metadata gains `conditions_requested/produced/missing/skipped`, `nave`, `nave_confound_warning`, and selection runs after the inverse operator so the forward error stays `errors[0]` and the covariance guards keep precedence (trial `snr_trial` lambda2 stays B5 task 2); the feature consumer accepts `Mapping`/`NpzFile`/path, finds conditions via the `{k}_times` partner keys (no deny-list), keeps negative tmin from `{cond}_times`, derives `sfreq` from the axis (500.0 default deleted), emits per-condition `{cond}_{roi}_{comp}_{metric}` columns + `{cond}_nave`, uses `np.trapezoid`, and raises `ValueError` for 3-D arrays / zero matches before writing the CSV once at the end, with the CLI gaining `--feature-type source_erp` and `--source/--preprocessing/--task/--subjects/--sessions` discovery (`paths.get_source_dir(...) / f"{subject}_roi_tc.npz"`); P1-F5: `channel_roles` drops the `raw._eegcpm_reference_channels` hack (metadata is the only source of truth), `detect_bad_channels(raw, threshold_sd, exempt, max_bad) -> (bads, report)` uses NaN-safe median/MAD robust z-scores (`mad==0 → std → degenerate`), flat (`ptp==0 or var<=1e-20`) non-exempt channels bad outright and excluded from the stats, a documented **25–40 Hz** HF band (post-45 Hz-lowpass; `hf_band_hz` in the report), non-finite z → bad, `> max_bad` → `RecordingExcluded('too_many_bad_channels', …)` (no silent truncation) with report keys `detected/reasons/z_scores/n_detected_precap/flat/exempt/hf_band_hz/degenerate_metrics`, `apply_bad_channels(..., exempt=)` never adds exempt names, `_BadChannelDetectorWrapper` takes `exempt` from the channel_roles metadata (missing → `KeyError` fails the run) and returns the report + `n_bads`, and `interpolate` raises `RecordingExcluded('too_many_bad_channels_for_interpolation', …)` instead of dropping channels. Validation: B4 command (source/config/consumer/dipole/detector/chain/reachability) **116 passed / 2 skipped**; full standard suite **3 failed / 616 passed / 2 skipped** (failures exactly the baseline: 2 `mne_connectivity` P4 + 1 `robust_pca` B6); greps clean (`trapz|500.0` absent from `source_erp_features.py`; no `except` in `bad_channels_pinned.py`). Known breaks recorded as deferred: the documented P4 connectivity break (unchanged) and the Streamlit source page's `aparc`/`schaefer100` + free-text variant options no longer validate (UI out of B4 scope).

### B4 rev1 — audit fixes (2026-10-15)

- (Opus 5.5 final-audit `revise` conditions; user decision: log-scale robust z): `detect_bad_channels` now z-scores variance and 25–40 Hz power on the **`log`** scale with the **upper tail only** (`z > threshold_sd`), keeps kurtosis two-sided on the raw scale, reports **signed** z-scores plus a new `tails` key, and treats a robust scale as degenerate via a relative-epsilon test (`scale <= 1e-12 * max(|median|, 1.0)`, MAD → std fallback kept; `z = 0` + `degenerate_metrics` entry) instead of `mad == 0`; the log is computed only on stats-eligible channels (flat channels excluded from the stats). New regression fixture (GSN-HydroCel-129-like, spatially correlated 3 cm, Cz-referenced, one injected heavy-tailed channel) → 2 detections ≤ `MAX_BAD_CHANNELS` (pre-rev1 raw two-sided rule: 23 → would exclude), lower tail never flagged, flat Cz exempt, plus a relative-epsilon degeneracy unit test. Real-data validation (read-only, chain order channel roles → 0.1–45 Hz filter → detector) on the 5 HBN pilot5 recordings: 3/5 clean recordings ≤ 11 (8/9/9; the pre-rev1 24- and 11-count recordings drop to 9), `sub-NDARAC853CR6` stays over budget at 19 (16 kurtosis detections — rule kept as pinned) and `sub-NDARAM487XU3` is corrupt (94 of 109 EEG channels identically zero → correct exclusion); table in `planning/VERIFICATION_TODO.md`. `SourceConfig` gains numeric bounds (`min_trials ge=1`, `equalize_trials ge=1`, `random_seed ge=0`); `_select_condition_epochs` raises `ValueError` for duplicate/reserved (`roi_*`, `sfreq`, `excluded_rois`, `excluded_reasons`)/companion-suffix (`_times`/`_nave`) condition names on both the task-config and `event_id` paths; the seeded-equalize test now also pins seed sensitivity and non-`arange` subsetting; accepted items assigned to plan B5 tasks 7–9 + λ² spy note and B7 task 11. Validation: B4 command **131 passed / 2 skipped**; full standard suite **3 failed / 631 passed / 2 skipped** (failures exactly the baseline: 2 `mne_connectivity` P4 + 1 `robust_pca` B6).

## Remediation batch B5 — source hardening + provenance (P2-F10/F11/F12/F13, B4-K5/K1/N4/N5) (2026-10-15)

- B5 (plan `plan/261013-2240-p1p2-remediation.md` §B5 tasks 1–9): `SourceReconstructionModule` now logs `mne.compute_rank` for the picked epochs and the covariance (`data_rank`/`cov_rank`, surfaced in metadata + provenance), inverts single trials with `lambda2 = 1/snr_trial**2` (a missing kwarg would silently fall back to the evoked λ², pinned by a spy test with a **non-default** `snr_trial`, B4-N5) via `apply_inverse_epochs(..., return_generator=True)`, and passes `method_params` to both inverse paths through one helper (`_resolve_method_params`: eLORETA with `0 < loose < 1` → `{'force_equal': True}` merged under any user params, mne 1.13.2 defaults it to False); the template forward keeps the **3-layer BEM only** (`_load_bem` raises `FileNotFoundError`; the single-shell `make_bem_model(conductivity=(0.3,))` branch and the dead `bem_path` reassignment are gone), uses the head→MRI transform read from the *same* `subjects_dir` the pre-flight validated (`trans='fsaverage'` resolves to mne's bundled template transform in 1.13.2 — probed identical here, but not guaranteed to be the validated file) and is cached class-wide by `sha256(ch_names + rounded loc[:3]) | spacing | BEM | mne version | subjects_dir`; a new montage pre-flight (`_preflight_montage`, run before the forward) fails on missing/zero EEG positions ("montage"), missing LPA/Nasion/RPA cardinal fiducials or non-head dig frames ("fiducials"/"head frame"), and on electrode-to-fsaverage-scalp distances (`cKDTree` over `fsaverage-head.fif` after `read_trans`; warn median > 10 mm, fail max > 25 mm — probed GSN-HydroCel-129 7.25/18.74 mm, +40 mm shift 13.4/31.0 mm) returning `{'median_mm','max_mm','n_electrodes','status','warnings'}`; every run that produces data artifacts also writes `{subject}_source_config.yaml`, `{subject}-cov.fif`, `{subject}-inv.fif` and `{subject}_provenance.json` (eegcpm/mne/numpy/scipy versions, subjects_dir/BEM/trans, spacing, `n_sources`, cov method + ranks, `n_channels`/`n_epochs`/sfreq, `lambda2_evoked`/`lambda2_trial`, `method_params`, `pick_ori='normal'`/`mean_flip`, `nave` + condition bookkeeping + `nave_confound_warning`, the full label block incl. `excluded_reasons`, the per-ROI flip summary `mean_dot` (+ `lh_mean_dot`/`rh_mean_dot` for BiHemiLabel ROIs; Opus B2 cond. 5), montage distances, `epochs_path` + streamed 1 MiB-chunk sha256, `config_hash` = sha256 of `json.dumps(cfg.model_dump(mode='json'), sort_keys=True)`, the resolved config and an `artifacts` map of filename→sha256), all via new `eegcpm/core/provenance.py` helpers that write to a sibling temp file + `os.replace` (mne suffix constraints respected for `-cov.fif`/`-inv.fif`; `np.savez` temp keeps `.npz`); writes are **stale-safe** (this subject's own patterns are deleted before writing, so a re-run never mixes old and new) and a failure anywhere in the write phase removes everything this run wrote (B4-K5: no partial artifact set behind `success=False`; proven by monkeypatching the provenance writer); the source CLI sorts every glob, selects the session's epoch file through `_select_epoch_file` (0 or >1 `*_epo.fif` counts the session failed **with the reason** instead of silently taking the first/skipping) and passes `epochs_path` into the provenance, and the unreadable-task-config `except: continue` now reports the file and exits 1; condition selection matches by **event-code value** (`np.isin(epochs.events[:, 2], values)`) instead of `epochs[[names]]`, which fixes the hierarchical `/`-tag over-selection (probed: `epochs[['a']]` with `event_id={'a':1,'a/b':2}` selects 3 epochs, the value match selects 1) and adds the numeric-code fallback (`7`/`"7"` → `event_id` value; mirrors `EventSorter.sort_by_condition`) so codes that are numbers no longer produce a loud "no matching events" for epochs not produced by this repo's stage; the source-ERP consumer now requires the run's provenance for every **on-disk** archive (`verify_archive_provenance`: missing/unreadable/`config_hash`-less provenance rejected, recorded `artifacts` sha256 cross-checked; in-memory `Mapping` input stays sidecar-free) and reports `source_config_hash`/`source_epochs_sha256`/`source_epochs_path` in its metadata; the features CLI reports every expected-but-absent subject/session on stderr and returns `n_missing`/`missing` in its summary (raises when nothing is found) instead of silently producing an empty table. Tests: new `tests/unit/test_source_hardening.py` (30 tests: method-param resolution, BEM/pre-flight cases incl. GSN pass + shifted-positions fail + fiducial/frame strips, the λ²/pick_ori spy, eLORETA smoke, provenance key/value + cov/inv reload + `config_hash` sensitivity, atomic-write failure/stale-cleanup cases, `_select_epoch_file`, event-value matching, features-CLI missing reporting) and `test_source_erp_features.py` extended with the provenance requirement (reject path/no-config-hash/hash-mismatch/unreadable, mapping exemption, metadata identity, CLI rejection). Validation: B5 command `pytest tests/unit/test_source*.py` → **143 passed / 2 skipped** (79 s); full standard suite → **3 failed / 667 passed / 2 skipped** (failures exactly the baseline: 2 `mne_connectivity` P4 + 1 `robust_pca` B6); plan greps clean (`conductivity=(0.3,)`/`make_bem_model` absent, every `pick_ori='normal'`, no direct `np.savez` in the source module); manual CLI smoke (synthetic GSN-129 project → `source-reconstruct` → `features --feature-type source_erp`) wrote the full 8-file artifact set with no temp leftovers and reported the missing `sub-99` while extracting 63 feature columns.

### B5 rev1 — Opus final-audit fixes (2026-10-16)

- **N1 (major)**: the features CLI now enforces a single source `config_hash` across the archives it combines — a table mixing different source configs raises by default (new `--allow-mixed-source-configs` override warns and reports the hashes); the hash(es) are printed and returned in the summary. **N2**: `verify_archive_provenance` requires the archive to be listed in the provenance `artifacts` map with a matching sha256 (omitting the map no longer skips the check). **A1**: `provenance_path_for_archive` maps `*_roi_tc_trials.npz` to `{subject}_provenance.json` (the file actually written). **A2**: provenance records `forward_source` (`template`/`caller`) and sets `bem`/`trans`/`spacing` to `None` for caller-supplied forwards. **A3**: `_preflight_montage` docstring documents that the fiducial check only catches dig lists without/stripped fiducials (mne `set_montage` fabricates them) and that the scalp-distance check is the real guard. **A7**: cache-key test now varies electrode positions (montage half was code-inspected only). **N6**: `compute_covariance` uses the `COV_METHOD` constant. **A6 sign-off**: stale-output deletion stays in the write phase (previous good artifacts survive pre-write failures) with the new N1 consumer check as the guard. Tests added: mixed-config refusal/override, mandatory-artifacts rejection (missing + unlisted), trials-provenance pairing, caller-forward provenance, montage cache key.

## Remediation batch B6 — Checkpoint A (RPCA removed), golden resolved-config hash, P1 tier-2 debt (2026-10-16)

- **Checkpoint A (P1-F2, decision (a), user-resolved 2026-10-15)**: `RobustPCAStep` is **removed from `build_hbn_langer_chain()`** — the canonical chain is now **9 steps** (`channel_roles → zapline → filter → bad_channel_detection → interpolate → eog_regression → asr → block_rejection → reference`). The step class and its `robust_pca.py` constants stay for standalone use (it was never in `STEP_REGISTRY`). The removal is an explicit, machine-readable spec deviation: `HBN_LANGER_SPEC_DEVIATIONS = ["P1-F2: robust_pca removed (destroys ERP; synthetic retention 2-28%)"]` is returned by `PreprocessingPipeline.get_config()['spec_deviations']` and therefore part of the golden resolved-config hash. Evidence: `ialm_decompose(..., return_info=True)` now also returns `{'n_iter', 'converged'}` and `RobustPCAStep` metadata gains `nnz_S_frac`/`energy_S_frac`/`n_iter`/`converged` (P1-F2 instrumentation); new `tests/unit/test_robust_pca_erp.py` injects a fixed-topography N1/P2 waveform at 25 onsets into rank-10 mixed AR(1) noise (109 ch, 250 Hz) and measures the epoch-average **retention** — the plan fixture (60 s) retains ~0.67 and the audit's destruction regime (30 s) ~0.19 with `nnz_S_frac ≈ 1.0`, both asserted `xfail(strict=True)` on the rejected acceptance criterion `retention >= 0.95`, plus a normal test pinning that `robust_pca` is absent from the chain and the spec deviation is recorded. The published-regime fixture `tests/fixtures/robust_pca/lin2011_exact_recovery.npz` is restored (sha256 `2ec82181…`, added to `tests/fixtures/INDEX.md`, `.gitignore` now negates `!tests/fixtures/**/*.npz`) and loaded via `Path(__file__).resolve().parents[1]` — the `test_robust_pca.py` baseline failure is fixed.
- **Golden config (P1-F7)**: every `hbn_langer` chain step/wrapper now implements `get_config()` with **all** constructor parameters plus `name`/`version`/`enabled`/`required` (`ChannelRolesStep`, `ZaplineStep` (+`nkeep`/`picks`/`verbose`), `_FilterStepWrapper`, `_BadChannelDetectorWrapper`, `InterpolateBadChannelsStep` (+`reset_bads`), `EOGRegressionStep`, `_ASRStepWrapper`, `_BlockRejectionStepWrapper`, `ReferenceStep`, `RobustPCAStep`). `core/config_hash.py` gains `library_versions()` (eegcpm/mne/numpy/scipy/eegprep/pyzaplineplus/eeglabio/autoreject/mne-icalabel, `'not-installed'` fallback), `resolved_config_hash(pipeline)` = sha256 of `{pipeline: get_config(), libraries: library_versions()}`, and the literal `GOLDEN_RESOLVED_HASHES = {'hbn_langer': 'f3a40cd6ab825513b840ffed333f1de0d425c51f584023caa8b685e28613f378'}` computed in the pinned venv **after** the RPCA removal; `load_golden_hashes()` keeps returning all 5 `GOLDEN_CONFIG_NAMES` keys (parity-harness compatibility) by mapping the `'preprocessing'` stage to the resolved hash and leaving the other 4 stages on file-content hashes. The executor sidecar now records `resolved_config`, `config_hash` and `library_versions` (success, FAILED and EXCLUDED runs alike), and the CLI + `run_processor` set `workflow.config_hash` on the DB row before `save_state`. `ReferenceStep` implements the post-reference check the YAML used to only claim: `check_reference_channel_variance='Cz'` (the chain passes it) asserts the post-reference column is finite and `> 0` else `RuntimeError`, reporting `reference_channel_post_ref_variance`. New tests: `test_config_hash.py` (+7: literal golden hash, per-step param sensitivity across all 9 steps, library-version sensitivity, `'not-installed'` fallback, 5-key mapping + parity-harness golden path, `test_hbn_chain_all_required`, sidecar provenance), `test_preprocessing_chain`/`test_preprocessing_schema` updated to the 9-step order.
- **Tier-2 debt (P1-F9/F13/F16/F18/F20/F21/F22 + B3-K9/B3-K5/B5-N3/B5-A4)**: zapline passes `noisefreqs=[fline]` when a line frequency is set (else its own list, `[]` = auto), unpacks the real 4-tuple and takes removed-component counts from `analytics['noise_freq_<f>']['n_remove_final']` (per-chunk array → `n_components_removed`/`_by_chunk`/`_total`; a missing key raises `KeyError` instead of the old invented default 1) with detected frequencies from `zap_config['noisefreqs']`; EOG regression is a single multi-target `np.linalg.lstsq(X.T, Y.T)` (verified equal to the old per-channel normal-equations solve within 1e-10) whose condition number is computed once on the **z-scored** design rows (+intercept, invariant to EOG scaling) and whose metadata records the frontal-ERP spec deviation (sensitivity check at B8); `ReferenceStep(type='rest')` raises `NotImplementedError` (no silent average fallback) and the dead `exclude_bads` knob is gone; `BadChannelDetector._detect_by_correlation` compares against **montage neighbours** within 5 cm (closest-channels/all-channels fallbacks) with `np.nanmedian`, and flat channels are bad outright (previously one flat channel NaN-ed the criterion for every channel) while the RANSAC/PREP `except Exception: print(...); return set(), {}` handlers now raise `RuntimeError` (B3-K9 — a failing path can no longer masquerade as a clean recording); `ICAStep` `rank-1` uses `min(max(5, r-1), r-1)` and raises `ValueError` when `r-1 < 1`; the `pipeline` optional extra pins the verified versions exactly (`mne==1.13.2`, `mne-icalabel==0.10.0`, `eegprep==0.4.0`, `pyzaplineplus==0.1.0`, `eeglabio==0.1.4`, `autoreject==0.5.1`); `MontageStep` raises `ValueError` listing EEG channels left with a NaN/zero position after `set_montage` (it used to drop them silently); `planning/EXCLUSION_CRITERIA.md` is regenerated from the code (`too_many_bad_channels`, `too_many_bad_channels_for_interpolation`, `block_rejection_fraction_exceeded`, required-step failure + the mark-only annotations); the four `modules/preprocessing/artifacts.py` append sites route through `append_bad_annotations` (B3-K9 — the uncorrected recipe shifted every existing annotation by `first_time` at `meas_date=None` + `first_samp≠0`, now covered by a `test_asr_timeline.py` case); `workflow/state.py::save_state` runs `state.metadata` through the numpy-aware encoder (B3-K5 — a numpy scalar used to raise `TypeError` and a non-finite float wrote `-Infinity`; numpy-scalar round-trip test added); the source module records the resolved `{condition: codes, matched_event_values}` mapping plus a `run_identity_hash` (config hash + mapping) in both the module metadata and the provenance (B5-N3), the features CLI writes `{feature_type}_{condition}.provenance.json` next to the parquet (source config hash(es), run identity hash(es), parquet sha256, library versions) and the template provenance test pins `forward_source == 'template'`; the forward cache key now includes the BEM/trans `mtime_ns:size` so replacing a template file in place invalidates the cached forward (B5-A4). `source/__init__.py` documents the `mean_flip` decision (plan §12 item 4): uniform-patch attenuation 11/30 dominant ROIs vs 22/30 under `pca_flip` and the midline polarity convention (near-tied flip axes, `MPFC` lh −0.70 / rh +0.71, one inverted ROI) are recorded in the module docstring.
- **B4-D2 blink-aware bad-channel detector** (user decision 2026-10-15, plan §12.5; the last open B4 condition): kurtosis is now computed on **blink-free samples** — blinks are located with `mne.preprocessing.find_eog_events` on the EOG-typed channel with the highest **1-10 Hz band-passed** variance (B6 rev2 NEW-2: raw-variance ranking is dominated by electrode drift and under-detects blinks; the earlier highest-raw-variance rule was replaced), else on a synthetic frontal proxy (most anterior non-flat EEG channel in the head frame), masking ±200 ms around each event, with an explicit recorded fallback to the full signal when blinks are too frequent (`> 50 %` masked or `< 50 %` usable samples) or no proxy exists (the ICLabel cross-check is documented as inapplicable: the detector is chain step 4, before any ICA); kurtosis is flagged on the **upper tail** on the **log scale** — `log(Pearson kurtosis) = log(excess + 3)`, always defined because sample Pearson kurtosis `m4/m2² >= 1` (plan §12.5 item 2; corrected 2026-10-16, B6 rev2 NEW-3: the earlier "log undefined for signed excess kurtosis" text applied the log to the *signed excess*) — instead of two-sided; a **relative near-flat floor** (`variance < 1e-3 × median` of the eligible channels → reason `flat`, reported under `near_flat`) closes the blind spot above the absolute `1e-20 V²` floor (a 1 nV dead electrode) — the lower-tail z variant of the decision is deliberately not used because the log-variance spread of a healthy recording makes z a non-flatness measure (measured: the lowest channel of the healthy fixture at 29 % of the median variance has z = −54); the report gains `near_flat` + `blink_mask` (source/channel/n_blinks/blinks_per_minute/masked_fraction/half_width_s/plausibility_reason/reason, plus the band-passed `selection` ranking) and the `tails` entry for kurtosis flips to `upper`; budget and mark-only semantics unchanged. Real-data validation (read-only, chain order channel roles → 0.1–45 Hz filter → detector, `exempt=('Cz',)`, `max_bad=1e9` to count) on pilot5 × 3 tasks: see the table in `planning/VERIFICATION_TODO.md` — **6/12 non-corrupt recordings stay within the 11-channel budget** (from 5/12 at B4 rev1; RestingState 4/4, DespicableMe 1/4, FunwithFractals 1/4 — corrected 2026-10-16, B6-K1: an earlier "12/12" claim was a counting error, the table is authoritative), with the corrupt `sub-NDARAM487XU3` still excluded (94 of 109 EEG channels identically zero). **Superseded 2026-10-16 (B6 rev2 NEW-6):** this rev0 table ran *without* `ZaplineStep` and before the NEW-1/NEW-2/NEW-3 fixes; the authoritative pilot5 numbers are the true-chain-order table in the B6 rev2 section below.
- **Validation**: full standard suite → **2 failed / 713 passed / 2 skipped / 2 xfailed** (only the 2 `mne_connectivity` P4 baseline failures; the `robust_pca` fixture test now passes); the eLORETA/dipole/source tests stay green; the parity-harness golden path (5 stage keys) is pinned by a test and does not raise.

### B6 rev1 — K3 pre-audit fixes (2026-10-16)

- **B6-K1 (major, reporting integrity)**: the pilot5 counts are corrected to **6/12 non-corrupt recordings within the 11-channel budget** (RestingState 4/4, DespicableMe 1/4, FunwithFractals 1/4). The recorded probe table and its raw JSON (`/tmp/opencode/b6_pilot_full.json`) were re-checked row by row (15 rows; the 3 corrupt `sub-NDARAM487XU3` recordings excluded); the "12/12" text in this changelog and the "7/11, FF 2/4" text in `planning/VERIFICATION_TODO.md` were counting errors — the table itself was correct and is unchanged. All three documents now state 6/12 (plus a marked correction note in `planning/audits/2026-10-16-B6-prior-state-briefing.md`, kept as part of the audit record).
- **B6-K2**: `bad_channels_pinned.py` module docstring: the near-flat floor is the `variance < 1e-3 × median` ratio disjunct only; the never-implemented "log-variance z < -threshold_sd" clause is replaced by the rejection rationale (z = −54 at 29 % of the median variance flags healthy channels).
- **B6-K3**: `detect_bad_channels` docstring: "raw kurtosis (two-sided)" → "raw excess kurtosis (upper tail, B4-D2)".
- **B6-K4**: `hbn_langer_chain.py` docstring: "The step class stays importable for standalone use (it was never registered)" — there was no `STEP_REGISTRY` entry to keep.
- **B6-K5**: `_find_blink_mask` now builds an ordered candidate list (EOG-typed channel, then synthetic frontal proxy) and retries the proxy when `find_eog_events` raises on a dead EOG electrode, instead of falling back to the full signal; every attempt is recorded under the new `blink_mask['attempts']` key and `source`/`channel` name the path used. Test: flat E8 + blinks on the proxy → `source == 'frontal_proxy'`, mask applied and covering the injected blinks, detector report records the same path.
- **B6-K6**: `modules/qc/preprocessed_qc.py` reads the scalar `n_components_removed_total` (legacy scalar fallback) for the zapline note and the "Zapline Components" metric — the B6 per-frequency dict is no longer embedded; test pins value 4 and the note text with dict-form metadata.
- **B6-K7**: `steps/zapline.py` writes the cleaned data back only after every metadata read that can raise (`zap_config['noisefreqs']`, `analytics['noise_freq_<f>']`), so a `KeyError` leaves the raw unmutated; the two KeyError tests now assert the raw data is unchanged (the fake backend returns deliberately modified data).
- **B6-K8**: `RobustPCAStep` reports `applied: False` + `reason: 'ialm_not_converged'` when IALM hits `max_iter` without reaching tolerance (a required run then fails in the executor instead of silently saving a non-converged decomposition). Chosen option: the low-rank write-back is kept (the step's only output; standalone/ERP-retention inspection) — the ERP xfail fixtures converge (`n_iter` 31/27, `converged` True), so the documented xfail regime is untouched. New test forces `max_iter=1` (not applied + reason) and pins the converged path (`applied: True`).
- **B6-K9 (record only, owner B7)**: `modules/preprocessing/bad_channels.py:633,645` legacy `except Exception: print(...)` + silent `drop_channels` is recorded as plan B7 task 15 (no code change in B6).
- **Validation**: full standard suite (with the two intentionally-uncollectable `test_specparam_*` files ignored per plan §0.1) → **2 failed / 717 passed / 2 skipped / 2 xfailed** (only the 2 `mne_connectivity` P4 baseline failures); targeted B6 command (`test_bad_channel_detector`, `test_preprocessing_steps_tier2`, `test_robust_pca`, `test_robust_pca_erp`, `test_config_hash`) → **58 passed / 2 xfailed**; `test_preprocessed_qc.py` (K6) → 21 passed. The re-check of the 60-s RPCA ERP fixture gives `n_iter` 31 / `converged` True (the briefing's "n_iter 200 / converged False" was stale; retention 0.675 / 0.188 reproduce).

### B6 rev2 — Opus final-audit fixes (NEW-1 … NEW-10, 2026-10-16)

- **NEW-1 (critical) — EOG regression re-injected unfiltered-EOG noise.** The chain's filter step now band-passes the **EEG + EOG** picks (`_FilterStepWrapper(picks=('eeg','eog'))`, version 2.0; the picks are recorded in the step metadata and `get_config()`). `raw.filter()`'s default picks resolve to the *data* channels, which **exclude EOG** (probed 2026-10-16, mne 1.13.2: a 60 Hz sine on an EOG channel passes untouched) — so the EOG regressors kept their full DC drift and 60 Hz line noise and `eog_regression` added both back into the already-filtered EEG (audit real-data probe: >50 Hz power ×33, 59–61 Hz ×~14,000, frontal std 51.6 µV vs 10.9 µV with filtered EOG). Zapline stays EEG-only (the 0.1–45 Hz filter removes the EOG line noise). `EOGRegressionStep` version 4.0 documents the band-passed-regressor contract. Tests: `test_filter_step_bandpasses_eog_channels` (EOG line power ratio < 1e-4 after filtering, sub-0.1 Hz ratio <= 1, EEG unchanged) and `test_eog_regression_after_filter_does_not_inject_eog_noise` (chain order: <0.1 Hz ratio <= 1, >45 Hz ratio <= 1.05, blink component <= 0.5 of its pre-regression amplitude on a shared ocular/environmental fixture; pre-fix contrast: 59–61 Hz power ×>100, blink not removed). Real-data probe after the fix (read-only, `sub-NDARAC853CR6` RestingState + DespicableMe, chain order up to the regression): see below.
- **NEW-2 (major) — blink channel chosen on unfiltered variance under-detected.** `_find_blink_mask` now ranks the EOG-typed channels on **1–10 Hz band-passed variance** (`BLINK_BANDPASS_HZ`, the conventional blink band; raw-variance ranking picks a drift-dominated infraorbital channel — E126/E127 on the pilot data — and `find_eog_events` then finds only 2–6 blinks vs 21–84 with a band-passed reference, masking <= 1.3 % of samples in 9/12 recordings). The frontal-proxy fallback + `attempts` are kept, and a detected rate below `MIN_BLINK_RATE_PER_MINUTE` (1/min) is recorded as `blink_mask['plausibility_reason']` (the mask is still applied; provenance, not a fallback). The report also gains `blinks_per_minute` and the band-passed `selection` ranking. Tests: drifting-vs-blink EOG fixture (raw-variance argmax is the drift channel, band-passed argmax is the blink channel, >= 15/20 blinks found and masked; the drift channel yields a flood of spurious events) and the implausible-rate reason test.
- **NEW-3 — log-scale kurtosis adopted (the rejection premise was false).** Kurtosis is now `log(Pearson kurtosis) = log(excess + 3)`, upper tail (the user's recorded §12.5 decision). The earlier "log-scale is undefined" text was wrong: it applied the log to the *signed excess*; sample Pearson kurtosis is `m4/m2² >= 1` by Cauchy-Schwarz, so the log is always defined. Corrected in the detector module docstring/`detect_bad_channels` docstring/`TAILS` comment, this changelog (B4-D2 paragraph) and `planning/VERIFICATION_TODO.md`; the audit briefing carries a rev2 correction note. The auditor's probe measured 8/12 vs 6/12 pilot recordings within budget with log-kurtosis before the NEW-1/NEW-2 fixes.
- **NEW-4 (major, provenance) — golden hash covered only constructor parameters.** `_BadChannelDetectorWrapper.get_config()` now returns the detector's module constants under `algorithm` (`max_bad_channels`, `tails`, `hf_band_hz`, `flat_variance_floor`, `near_flat_variance_ratio`, `degeneracy_rel_eps`, blink half-width/max-masked/min-blink-free/band/rate) read from `steps/bad_channels_pinned.py` at call time; `_ASRStepWrapper.get_config()` returns the delegated `ASRStep` constants (`method`, `train_duration`, and the new named `DATA_DRIVEN_REF_MAXBADCHANNELS`/`DATA_DRIVEN_REF_TOLERANCES` from `steps/asr.py`). Version bumps: detector **3.0**, filter **2.0**, EOG regression **4.0**, ASR wrapper **2.1**. New sensitivity test monkeypatches `TAILS`/`NEAR_FLAT_VARIANCE_RATIO`/`MAX_BAD_CHANNELS` and asserts the resolved hash changes; a second test pins the exposed values. `GOLDEN_RESOLVED_HASHES['hbn_langer']` **re-pinned to `731756264f5c1c79783acba79dee1913b491bd5e02801c3b54105f254edef975`** (was `f3a40cd6…`); `load_golden_hashes()` still returns the 5 `GOLDEN_CONFIG_NAMES` keys and the parity path does not raise.
- **NEW-5 — numpy/scipy pinned + diagnosable hash failures.** The `pipeline` extra pins `numpy==2.5.4` and `scipy==1.18.1` (mne's filter/PSD/SVD numerics are what the detector and the resolved hash depend on). The re-pin one-liner is documented at `GOLDEN_RESOLVED_HASHES`; `GOLDEN_RESOLVED_LIBRARY_VERSIONS` records the pinned versions and the literal-hash test prints the exact `library_versions()` diff (pinned vs installed) when it fails.
- **NEW-6 — pilot5 re-run in the true chain order (incl. zapline).** Read-only, `channel_roles → ZaplineStep(fline=60) → filter 0.1–45 Hz (EEG+EOG) → detector`, `exempt=('Cz',)`, `max_bad=1e9` to count, after the NEW-1/NEW-2/NEW-3 fixes. Table in `planning/VERIFICATION_TODO.md` (the rev0 table omitted zapline and ran before these fixes; its "6/12" summary is superseded). Summary: **8/12 non-corrupt recordings within the 11-channel budget** (RestingState 4/4, DespicableMe 2/4, FunwithFractals 2/4; up from 6/12 at rev1 — CD661GL0/DM 12→7 and AM873GAC/FF 24→10 moved inside, AC853CR6/FF 26→20 and AC904DMU/DM 24→18 stayed outside; the auditor's log-kurtosis-only probe also measured 8/12 with the same per-task split). The blink channel is now chosen on band-passed variance (E126/E125/E127/E25 vs the drift-dominated raw-variance winners), and AC853CR6/DM is the only row where the low-rate plausibility reason fires (0.70/min).
- **NEW-7 — stale RPCA numbers fixed.** `planning/VERIFICATION_TODO.md` said "`n_iter` 200, not converged"; the frozen fixtures converge (`n_iter` 31 for the 60-s plan fixture, 27 for the 30-s audit regime, retention 0.675 / 0.188). Corrected in the TODO; the non-converged path is exercised by the forced-`max_iter=1` test.
- **NEW-8 — xfail honesty.** Both ERP-retention `xfail(strict=True)` markers now carry `raises=AssertionError` (a fixture/step error can no longer masquerade as the expected failure), and the instrumentation/metadata facts moved into normal tests (`test_rpca_erp_destruction_measured_plan_fixture` / `..._audit_regime`): measured retention below the rejected `>= 0.95` criterion (0.675 / 0.188), topography corr, `nnz_S_frac > 0.9`, `energy_S_frac`, `n_iter`, `converged is True`. The two IALM runs are module-scoped fixtures (computed once).
- **NEW-9 — behavioural detector tests.** (a) a 1 nV channel is flagged `flat` and listed in `near_flat` (above the absolute floor, `z_scores['var']` still computed); (b) blink-dominated frontal channels are flagged by kurtosis on the full signal and not flagged with the mask applied (fixture tuned so the variance criterion stays below threshold; the no-mask run uses the documented full-signal fallback report); (c) a large sub-Gaussian sinusoid (Pearson kurtosis ~1.5) has `z_kurtosis < -3` and is never flagged by the kurtosis upper tail (it is flagged by variance, a different criterion).
- **NEW-10 — RPCA write-back only on convergence.** `RobustPCAStep.process` writes the low-rank component back only when IALM converged; a non-converged solve reports `applied: False` + `reason: 'ialm_not_converged'` **and leaves the input data untouched** (the K8 residual "a caller ignoring `applied` still receives the non-converged L" is gone). The ERP fixtures converge, so the documented xfail regime is unaffected; the forced-`max_iter=1` test now asserts the data are unchanged and pins the converged write-back.
- **Validation (rev2)**: full standard suite (with the two intentionally-uncollectable `test_specparam_*` files ignored per plan §0.1) → **2 failed / 728 passed / 2 skipped / 2 xfailed** (only the 2 `mne_connectivity` P4 baseline failures; +11 tests vs rev1). Targeted B6 files (`test_bad_channel_detector`, `test_preprocessing_steps_tier2`, `test_robust_pca`, `test_robust_pca_erp`, `test_config_hash`) → **69 passed / 2 xfailed** (rev1: 58 passed / 2 xfailed). Real-data NEW-1 probe (read-only, chain order up to `EOGRegressionStep`, `sub-NDARAC853CR6` RestingState + DespicableMe): chain order (channel_roles → zapline(60) → filter EEG+EOG → EOG regression) vs the pre-fix EEG-only filter, measured on the EEG picks before vs after the regression: `sub-NDARAC853CR6`/RestingState — chain order >45 Hz ×1.15, 59–61 Hz ×47.7 (absolute 4.5e-19 → 2.1e-17 V²/Hz, i.e. at the post-filter nV²/Hz floor), <0.1 Hz ×0.72, frontal std 43.2 → 12.4 µV; pre-fix 59–61 Hz ×250,954, frontal std 43.2 → 36.2 µV. `sub-NDARAC853CR6`/DespicableMe — chain order >45 Hz ×1.005, 59–61 Hz ×1.12, <0.1 Hz ×0.975, frontal std 64.5 → 20.9 µV; pre-fix 59–61 Hz ×4,456, frontal std 64.5 → 40.5 µV. So the dominant failure mode is reduced by 3.6–4.7 orders of magnitude and the frontal std drops to ~1/3 of the pre-fix value (matching the audit's 51.6 µV vs 10.9 µV); the two marginal band-ratio increases (>45 Hz on the Rest recording, 59–61 Hz on both) are the unavoidable single-coefficient regression residual of the EOG's *filter stopband* content (≈ nV/√Hz, ~6–7 orders below the in-band signal) — the EEG's own post-filter line floor is essentially zero, so any residual shows up as a large ratio on a tiny absolute level. The synthetic unit test asserts the non-increase on a fixture with a realistic post-filter EEG line floor (60 µV pre-filter). Raw probe JSON: `/tmp/opencode/b6r2_new1_realdata.json`.

### B6 rev2a — scoped-re-audit fix (2026-10-16)

- **ASR wrapper `get_config` truthfulness (re-audit condition 1).** `_ASRStepWrapper.get_config()` reported the *data-driven* `ref_maxbadchannels=0.075` / `ref_tolerances=[-inf, 5.5]`, but the wrapper runs `calibration='fixed_first_60s'` (calibration reference = the first 60 s passed as an ndarray; `ref_tolerances` never passed). It now reports `calibration_reference: "first_train_duration_s_of_data"`, `ref_maxbadchannels: "calibration_array"`, `ref_tolerances: "eegprep_default"` (the data-driven constants are no longer claimed). New spy test `test_asr_wrapper_config_matches_fixed_path_kwargs` asserts the `clean_asr` kwargs (ndarray shape `(n_channels, 60*sfreq)`, no `ref_tolerances`, `cutoff=20`) and the wrapper config. **`GOLDEN_RESOLVED_HASHES['hbn_langer']` re-pinned to `72c394e7801d894edf7b39dc26e543e401b2d023f67b445b83d8a5460719b3dd`** (was `73175626…`). Suite: 2 failed (connectivity P4) / 729 passed / 2 skipped / 2 xfailed.

## Remediation batch B7 — QC truthfulness, test-suite honesty, CI job (2026-10-16)

Plan `plan/261013-2240-p1p2-remediation.md` §B7 tasks 1–15 + the B6-carried EOG-regressor guard. **P1-F10/F11/F12, P2-F14/F15/F16.**

### QC truthfulness (tasks 1–3, P1-F10)

- **`modules/qc/preprocessed_qc.py`** (tasks 1–2): new metric **"Pipeline step status"** built from `metadata['step_status']` — `bad` when any step is `failed`/`excluded` (the metric value is the count and the unit names the steps), `warning` when `metadata['degraded']` or a step is `not_run` (only reachable after an abort), `ok` otherwise. The bad-channel accounting now reads the **detector's report** (`metadata['bad_channel_detection']['detected']`/`n_detected_precap`) — authoritative because the chain's interpolate step runs `reset_bads=True`, so `info['bads']` is empty afterwards — and counts `channel_roles` exclusions (`neck_face_dropped` + EEG→EOG retyped) in a separate informational metric **"Channel-role exclusions"**, subtracting them from `n_dropped` (the safety-net note no longer labels them "Bad/removed channel(s)"). "Bad Channels" therefore counts detected/interpolated channels only; without chain metadata the old marked/dropped fallback is kept.
- **`cli/preprocess.py`** (task 2): the `removed_channels` map is built from `metadata['bad_channel_detection']` (per-channel `reasons`, legacy `bad_channels` dialect kept as fallback); the success path stores the per-recording chain QC row under `recording_qc` in the run's `*_qc_metrics.json` (`recording_qc_metrics(post_raw, task, subject_id, chain_metadata=result.metadata, raw_pre=raw)`); **FAILED and EXCLUDED runs now write the same `*_qc_metrics.json`** with `status`/`reason` (+ the exclusion metrics / error list, no HTML), including the crashed-run path — a downstream QC table can distinguish "excluded for contamination" from "not processed".
- **`modules/qc/hbn_recording_qc.py`** (task 3): block-rejection fields read the names the step emits (`rejected_fraction`, `n_blocks_rejected`); `bad_channels_n` comes from the detector report (`bad_channels_n_precap`, `bad_channels_detected` added; `info['bads']` fallback); the **vacuous in-sample `residual_ocular_maxcorr` gate is replaced** by `blink_residual_ratio` — blinks are found on the *pre-chain* recording via the first available EGI EOG name (or a synthetic frontal proxy), and the ratio of blink-locked frontal scalp ptp post/pre is gated at `OCULAR_GATE` (0.30); the old correlation survives only as the informational `residual_ocular_maxcorr_insample` (after the OLS EOG regression it is ~0 *by construction*). `blink_residual_ratio`/`_pass`/`_reason`, `blink_n` and `blink_channel` are `None`/recorded when `raw_pre` is missing — a gate that cannot run never reports a pass. The frontal set excludes EOG-typed channels (an EOG channel keeps its blink by definition, which would pin the ratio near 1).

### Test-suite honesty (tasks 4–5, P1-F11/F12)

- **`tests/unit/test_no_silent_continue.py`**: the gate resolved the repo root one level too high (`parents[1]` = `tests/`) and scanned **zero** files; it now uses `parents[2]` (the package root), asserts every `CORE_PATH` resolves and was scanned (>= 1 file each, **88 files** in total), adds `eegcpm/modules/source` and `eegcpm/modules/qc`, skips `_legacy` trees, and gains a companion test that no live module imports from `_legacy`. The AST rule was extended: a handler whose body is only `pass`, `continue` **or `print(...)`** is a silent swallow (a bare print is not a record — that is exactly why the B6-K9 legacy handlers were invisible). **All 18 offenders found by the widened gate were fixed with no allow-list** (14 in the original scope + 4 print-only handlers in `modules/qc`): `FileNotFoundError`-swallowing `unlink` calls became `unlink(missing_ok=True)`; `dua_paths` no longer swallows a missing `yaml` (hard dependency); the ICA auto-detection (`find_bads_eog`/`find_bads_ecg`) narrows to `(ValueError, RuntimeError)` and records `auto_detected['eog_error']`/`['ecg_error']`; the networkx graph metrics record why the columns are absent (`logger.warning`) and path length narrows to `NetworkXError`/`NetworkXPointlessConcept`; DTF counts and logs singular frequency bins; `roc_auc_score` failures record `auc=None` + `auc_error`; option discovery logs unreadable connectivity archives; `aggregate_reports`/`run_processor`/`epoch_combiner` log the state/JSON/combined-epoch/stdout failures; `preprocessed_qc` records the ERP-plot skip as a QC note and the ICA-topography mismatch in the log; `preprocessing_comparison_qc` records a failed montage; `metrics_io`/`raw_qc` log skipped files/recordings.
- **Vacuous checks (task 5)**: `np.trapz` is gone from `modules/features/erp_features.py` (→ `np.trapezoid`), and the two B3-era QC tests in `test_preprocessing_chain.py` that asserted the *vacuous* in-sample ocular gate were rewritten to the new contract (schema keys, `raw_pre`-missing behaviour, and a non-vacuous blink-gate pass/fail pair).

### End-to-end chain test + CI (tasks 6–7, C2/C6)

- **`tests/unit/test_hbn_chain_e2e.py`** (`chain_e2e`, `slow`): synthetic GSN-HydroCel-129 recording (129 names incl. `Cz`, 250 Hz, 150 s, `first_samp=1000`, `meas_date` set, per-channel-scaled 1/f + 5 µV 60 Hz + blinks on the 9 EGI EOG channels + an N1/P2 ERP at every 2-s `stim`, `Cz` identically zero) through the **real** 9-step chain. Asserts: success, `step_status` all `ok`/`noop` (no `degraded`/`excluded`), `stim` event samples identical input↔output, every input annotation preserved (only `BAD_*` added), the FIF reloads with the same annotations + `meas_date`, the sidecar's `config_hash == GOLDEN_RESOLVED_HASHES['hbn_langer']` with the resolved 9-step list and the fixed-first-60 s ASR config, `var(Cz) > 0` after the average reference, zero bad blocks, the QC row reading the chain metadata, and ASR excluding the interpolated channels. Second variant (30 % contaminated blocks **inside the ASR calibration window**): `RecordingExcluded('block_rejection_fraction_exceeded')`, `*_preprocessing_EXCLUDED.json` (with `n_blocks_rejected`/`rejected_fraction`/`bad_block_onsets_s`, `reference: not_run`) and **no FIF**. The fixture is documented in-file (why the contamination is per-channel-scaled, EEG-only and calibration-window-placed: ASR repairs bursts outside its calibration window — measured 22/75 → 0/75 — so the gate only fires when the calibration itself is contaminated).
- **`.github/workflows/pipeline-verification.yml`** (repo root, `working-directory: eegcpm-0.1`): Python 3.13, `pip install -e ".[pipeline]" pytest pytest-cov`, an explicit `import eegprep, pyzaplineplus, eeglabio, autoreject, mne_icalabel` step (a missing backend fails the job instead of silently degrading a step), a cached `~/mne_data` + `fetch_fsaverage()` prefetch (the only allowed download), the §0.1 standard command, `pytest -m chain_e2e`, and the no-silent-except gate with its scan report.
- **`tests/unit/conftest.py`**: conditional `collect_ignore` for the two specparam test files when the optional backend is absent (they raise at import) — without it every pytest invocation except the one with two explicit `--ignore` flags died at collection, including the plan's `pytest -m chain_e2e`.

### Source crosstalk + source QC (tasks 8–10, P2-F14/F15/F16)

- **`modules/qc/source_qc.py`** (task 8): the invalid "simulate a delta at the label's middle vertex, correlate with a dSPM magnitude, `fill_diagonal(1.0)`" crosstalk code (and the `vertices[len//2]` "centre") is **deleted**. The new `compute_crosstalk_matrix(fwd, info, noise_cov, labels, method, lambda2, depth)` builds a QC-only fixed-orientation pair, `make_inverse_resolution_matrix` and `get_cross_talk(..., mode='mean', norm=False)` (one whole-source-space STC per label) and forms `C[i, j] = mean(|ctf_i in label j|) / mean(|ctf_i in label i|)` per pair. **Two documented deviations from the plan's literal recipe**, both probed (`/tmp/opencode/b7_probe_crosstalk2.py`, mne 1.13.2): (i) `make_inverse_operator(info, fwd_fixed, …, fixed=True, depth=0.8)` raises — mne's compatibility table requires a **surface-oriented, free-orientation** forward for fixed orientation + depth weighting, so the inverse is built from `convert_forward_solution(fwd, surf_ori=True, force_fixed=False)` (the operator is still fixed-orientation) while the `force_fixed` forward is used for the resolution matrix; (ii) `get_cross_talk`'s `idx` handling tests `type(idx[0]) is Label`, so a **`BiHemiLabel` (every midline CONN ROI) raises `IndexError`** — the per-label call uses explicit integer vertex lists (`label_vertex_indices`), verified numerically identical to the single-`Label` call (max |Δ| 0.0). `compute_crosstalk_matrix_numpy(..., align_signs=False)` is an independent numpy mirror (`apply_inverse` on the identity → `R = J @ G`, direct label-row means) used by the tests.
- **`modules/source/__init__.py`**: when `cfg.output.compute_crosstalk` the matrix is computed once per (forward, noise-covariance digest, method, lambda2) — the cache key includes the **covariance digest** beyond the plan's "montage hash" because the inverse operator is whitened by it (two subjects must never share a matrix) — written as `{subject}_crosstalk.npz` (matrix + `roi_names` + method + definition string) with a `crosstalk` block in the provenance (definition, off-diagonal max/mean, pairs > 0.3, or `computed: False`), and passed into `qc_data` together with `labels` and `roi_min_vertices`. A crosstalk failure (e.g. eLORETA, which `make_inverse_resolution_matrix` does not support) is recorded as a warning and never fails an otherwise successful run.
- **Note (caught by narrowing the crosstalk failure handler)**: the first version of the wiring caught `Exception` and recorded it as a warning; that silently swallowed a `TypeError` (`_forward_cache_key` needs `spacing` + `subjects_dir`), so the matrix was never computed while every test still passed. The handler is now `(ValueError, RuntimeError, np.linalg.LinAlgError)` (MemoryError deliberately not caught), the call is fixed, and `test_process_with_template_forward` asserts the written `{subject}_crosstalk.npz` (shape, unit diagonal, `roi_names`) plus the provenance summary.
- **QC metrics (task 9)**: `SourceQC.compute` now marks the result `bad` when any `roi_vertex_counts < roi_min_vertices`, any ROI value is non-finite, or a condition has no `{cond}_times` axis (metrics "ROI Vertex Counts", "ROI Values Finite", "Condition Time Axes"), the **fabricated time axis and the `sfreq=500.0` default are gone** (the ERP plot uses only conditions that carry their own axis and is skipped otherwise), and labels without a matrix produce an explicit note instead of silently recomputing the deleted (invalid) matrix.
- **`tests/unit/test_source.py`** (task 10): both `@pytest.mark.skip` tests are replaced by real fsaverage tests (`oct6`, nested config): the module's own template forward (signed outputs, 30 unique ROI names, cerebellar exclusion, provenance files + config hash) and a caller-forward sLORETA run; new `TestSourceCrosstalk` asserts (a) the dipole-fixture target ROI `SensoriMotor.Superior`'s strongest off-diagonal entry is finite and **< 1** (measured 0.597, strongest partner `DorsalAttention.IPS_R`), (b) permuting two label indices permutes C's rows/columns (and the permutation moves entries), and (c) the numpy mirror matches the library matrix exactly while the `label_sign_flip`-aligned variant does **not** (critic N-3, quantified: 34 % of vertices flipped on average, max |ΔC| 2.08 — the library's signed mean cancels across sulcal walls; the saved matrix keeps the library definition and the discrepancy is pinned by a test rather than silently "fixed"). New `tests/unit/test_source_qc_metrics.py` covers the task-9 metrics.

### CLI, provenance and legacy fixes (tasks 11–15)

- **Features CLI argparse validation (B4-K6, task 11)**: `validate_features_args(parser, args)` runs right after parsing in `cli/main.py` — `--input-dir` is **required** for every non-`source_erp` type and, for `source_erp`, either `--input-dir` or all of `--preprocessing`/`--task`/`--source` must be resolvable; violations exit **2** with a usage message instead of a runtime traceback.
- **B5-A5 (task 12)**: `core/provenance.stale_output_patterns` now covers this subject's hidden atomic-writer leftovers (`.{subject}_*`, `.{subject}-*` — the SIGKILL case) plus `{subject}_crosstalk.npz`, and the module docstring records the single-writer assumption (two concurrent writers for the same subject are unsupported).
- **B5-N4 (task 13)**: condition names containing `/` or `\` are rejected by `_validate_condition_names` with a clear `ValueError` before any file is written (they become `{subject}_{condition}_stc` names and used to fail the whole run deep inside `stc.save`).
- **B5-N5 (task 14)**: the `source-reconstruct` CLI raises `SystemExit(1)` when an explicitly requested task config is missing or unreadable (no more warning + per-event fallback that silently changes the condition structure) and exits non-zero when any session failed.
- **B6-K9 (task 15)**: the legacy `detect_and_interpolate_bad_channels` `except Exception: print(...)` + silent `drop_channels` handlers now re-raise (`RuntimeError`, "refusing to drop channels silently"); the deliberate zero-variance drop is logged and returned in `bad_channels`.

### Carried B6 item — EOG regressor guard

- **`steps/eog_regression.py` v5.0**: `_check_band_limited` enforces the NEW-1 contract instead of only documenting it — (1) `raw.info['lowpass']` must be finite and `<= 50 Hz` (the chain's filter sets 45; an unfiltered `RawArray` declares Nyquist and is refused outright), (2) the fraction of EOG power above 50 Hz (Welch; capped at 45 % of Nyquist) must be `<= 5 %` (catches a recording whose metadata claims a 45 Hz lowpass while the EOG data stayed unfiltered — the exact pre-NEW-1 chain behaviour). Both measurements and the check outcome (`eog_hf_power_fraction`, `eog_lowpass_hz`, `eog_band_limited_check`) are recorded; `allow_unfiltered_regressors=True` is the explicit, recorded opt-out. The guard constants are in `get_config()` (B6-NEW-4 principle). Tests: refusal on an unfiltered recording, refusal on EEG-only-filtered data (metadata lies), acceptance + recorded fraction for the chain contract, the explicit bypass, and the Nyquist-capped cutoff. **`GOLDEN_RESOLVED_HASHES['hbn_langer']` re-pinned to `ca91b7f280d62804a63d5bc18aa4acfb18a1e80a193a0cedb64f343b3d75cbb7`** (was `72c394e7…`; superseded by the B7 rev2 re-pin to `1ef8780d…` — itself re-pinned to `b47de382…` by the F1–F7 round; see the B7 rev2 section and the final-verification-audit section at the end of this changelog).

### New critical finding (B7-N1) — ASR degenerates on interpolated channels

Found while building the task-6 e2e fixture and **fixed in the minimal, non-architectural way** (flag for approval; the alternative is a chain reorder):

- **Evidence (probed 2026-10-16, `/tmp/opencode/b7_e2e_probe*.py`, mne 1.13.2 + eegprep 0.4.0).** Spherical-spline interpolation reconstructs a bad channel as an **exact** linear combination of the remaining channels, so including it in the ASR input makes the calibration covariance exactly singular: on the EGI-129 fixture with a **single** interpolated channel the condition number goes from 35 (0 interpolated) to **1.2e15** with 1 exact zero eigenvalue; `eegprep.clean_asr` then takes `sqrtm` of that matrix (`LinAlgWarning: Matrix is singular`) and returns **~1e15 µV** instead of repaired data (control without the interpolated channel: sane ~12 µV output). Every 2-s block then exceeds the 100 µV block-rejection threshold, so **any recording with >= 1 interpolated channel was excluded by the chain** — the detector flags 1–11 channels on real HBN pilot recordings (B6: 8/12 recordings within the 11-channel budget, i.e. *with* interpolations), so this was a real-data blocker for Checkpoint B. The B3 ASR tests never combined `interpolate` with ASR (they mark channels bad instead), which is why the audits did not see it.
- **Fix.** `InterpolateBadChannelsStep` records `interpolated_channels`; `ASRStep._channel_exclusions` adds them to the exclusion set (they are synthetic, carry no independent information, stay in the recording and are re-mixed by the average reference — only the ASR *input* excludes them), reporting `interpolated_excluded_from_asr`. Verified: 0 zero eigenvalues, condition number 34.9, sane output; the e2e chain now succeeds on a recording with 2 interpolated channels. Tests: `test_asr_timeline.py::test_interpolated_channels_excluded_from_asr` + `::test_interpolated_channel_does_not_break_asr` (with the rank-deficiency control assertion). **Not applied** (recorded for approval): moving interpolation after ASR (the EEGLAB ordering) — it changes the chain composition and the golden resolved-config hash.

### Validation (B7)

- Focused suites: `test_hbn_chain_e2e.py` → **13 passed**; `pytest -m chain_e2e` → **13 passed**; `test_no_silent_continue.py -rA` → **16 passed** (88 files scanned, 0 offenders); `test_source.py` → **55 passed** (both former skips now real tests); `test_asr_timeline.py` → 23 passed; `test_qc.py` → 49 passed; `test_preprocessed_qc.py` → 31 passed; `test_source_hardening.py` → 33 passed; `test_preprocessing_steps_tier2.py` → 35 passed; `test_cli_arg_validation.py` → 12 passed (new); `test_source_qc_metrics.py` → 6 passed (new); `test_preprocess_status_persistence.py` → 18 passed.
- **Full standard suite → 2 failed / 811 passed / 2 xfailed** (the two failures are exactly the documented `mne_connectivity` P4 baseline; **zero skips** — both `test_source.py` skips became real tests; the two `test_specparam_*` collection errors are now conditionally ignored by `conftest.collect_ignore` and are never counted as passes).
- Greps: `info['temp']` → 0 hits; `np.trapz(` → 0 hits (comment only); `set_annotations(annots)` → 0; `mne.make_inverse_operator|mne.apply_inverse` → 0; the no-silent-except gate scans **88** files with **0** offenders.

### B7 rev1 — K3 fixes (2026-10-16)

Pre-audit findings B7-A1…A5 (Kimi K3), all applied on the uncommitted B7 working tree.

- **B7-A1 (major) — no-silent-except gate widened and the 9 offenders fixed.** `CORE_PATHS` now also
  covers `eegcpm/cli`, `eegcpm/workflow` and `eegcpm/data` (the `_legacy` exclusion and the
  import-guard test are unchanged) — the gate scans **108** files (was 88) with **0 offenders and no
  allow-list**. Fixes, all narrowed + recorded:
  `cli/connectivity.py` — the auto-detect loop records unreadable / non-mapping task configs (log +
  a "Task config problems" summary) instead of `except Exception: continue`, and the load failure is
  narrowed to `(OSError, yaml.YAMLError, pydantic.ValidationError, TypeError)` + logged (this CLI
  only prints the config, so it is recoverable); `cli/epochs.py` — the QC-report handler is narrowed
  to `(OSError, ValueError, RuntimeError, KeyError, TypeError)`, logged and listed in the run
  summary ("QC reports not generated for N run(s)"), and `discover-events`' `find_events` handler is
  narrowed to `(RuntimeError, ValueError, TypeError)` + logged; `workflow/state.py` — the three
  `except sqlite3.OperationalError: pass` migration handlers are replaced by one
  `_add_column_if_missing()` helper that tolerates only a message-checked duplicate-column error
  (anything else re-raises) and logs the no-op at debug; `data/hbn.py` — non-numeric phenotype
  values are logged (debug per value, warning summary) instead of a bare `pass`; `data/conn_rois.py`
  — `load_nifti_template` logs the missing-nibabel / missing-template no-op and only tolerates
  `ImportError` (a corrupt template propagates).
- **B7-A2 (minor) — no false "all steps ok".** `preprocessed_qc.py`: when `metadata['step_status']`
  is absent (legacy dialect or standalone QC) the "Pipeline step status" metric is now
  `warning` / unit `no chain metadata` (value 0) instead of `ok` / `(all steps ok)`; an explicitly
  empty `step_status` is a warning too.
- **B7-A3 (minor) — crosstalk pair count is directed.** `C[i, j]` is the leakage of source ROI i into
  target ROI j, so the above-threshold count is now over directed off-diagonal entries
  (`(C > 0.3) & ~eye`, reciprocal pairs count twice). The provenance key is renamed
  `n_pairs_above_0.3` → **`n_directed_above_0.3`** and the definition string documents the
  directionality; QC metadata `crosstalk_high_pairs` → **`n_directed_above_0.3`**, and the QC
  figure statistics, the HTML caption/summary and the `scripts/test_new_qc_figures.py` consumer were
  updated to the same definition (the old `triu` count silently halved reciprocal pairs).
- **B7-A4 (evidence note, no code change).** The e2e EXCLUDED variant places the 30 % contamination
  **inside the ASR calibration window by design**: ASR repairs post-calibration contamination
  (measured 22/75 → 0/75 contaminated blocks), so the block-rejection gate can only fire when the
  calibration itself is contaminated; at that amplitude the gate's *post-calibration* sensitivity is
  untestable (there is no contamination left to detect). Recorded in `VERIFICATION_TODO.md` for the
  final auditor to adjudicate the coverage caveat.
- **B7-A5 (no code change).** Added to the B8 checklist: validate the EOG-regression HF guard
  (`MAX_HF_POWER_FRACTION=0.05`) on the real Checkpoint B subject; if it false-fires, measure the HF
  fraction on blink-free segments or raise the threshold — never silently bypass.
- **Tests**: gate **19 passed** (16 before; 3 new scope cases incl. the two new dirs' parametrized
  existence checks); `test_preprocessed_qc.py` **33** (rewrote `test_without_metadata_is_ok` →
  `test_without_metadata_is_warning_not_ok` + legacy-metadata and empty-`step_status` cases);
  `test_source_qc_metrics.py` **8** (directed-vs-`triu` count, symmetric-matrix double count);
  `test_source.py` provenance assertion pins `n_directed_above_0.3` against the saved matrix;
  `test_cli_arg_validation.py` **14** (unreadable auto-detect config reported, unloadable explicit
  config reported); `test_conn_rois.py` **11** (template loads; missing template returns None and is
  logged); new `tests/unit/test_hbn_data_utils.py` **8** (numeric/skip/requested-but-absent/
  `score_columns`/NaN boundary); `test_preprocess_status_persistence.py` **21** (idempotent
  migration, duplicate column tolerated + debug-logged, non-duplicate `OperationalError` propagates).
- **Validation (rev1)**: full standard suite → **2 failed / 833 passed / 2 xfailed / 0 skipped**
  (failures exactly the two documented `mne_connectivity` P4 baseline tests); `pytest -m chain_e2e`
  → **13 passed**; focused B7 set (gate + preprocessed_qc + source_qc_metrics + cli_arg_validation +
  conn_rois + hbn_data_utils + status persistence) → **113 passed**; the gate reports **108 files
  scanned / 0 offenders**.

### B7 rev2 — Opus final-audit fixes (2026-10-16)

All nine conditions of the Opus B7 final audit applied on the uncommitted B7 working tree; B7
stays `[~]` (awaiting re-audit). **Golden resolved hash re-pinned `ca91b7f2…` →
`1ef8780d87f228609e8c55c233426e51778b5e2a0a8648ba42318e3433b8d421`** (`ASRStep` 2.0 → 2.1,
`_ASRStepWrapper` 2.1 → 2.2; **later re-pinned to `b47de382…` by the F1–F7 round**, see the
final-verification-audit section at the end of this changelog).

- **B7-N2 (critical) — ASR output blow-up guard.** `eegprep.clean_asr` at cutoff 20 on
  0.1-Hz-high-passed data can return garbage at ~1e15–1e16 µV (probed: 4.6e15 µV from a 13 µV
  input at calibration-covariance condition number 2.2e8). `ASRStep` now fails loudly with reason
  **`asr_output_magnitude_blowup`** when `max|post| > 1e3 × max|pre|` — before the write-back, so
  the caller's data are never corrupted and the block-rejection gate can never report a false
  "block contamination" exclusion. The fixed-calibration covariance condition number / min
  eigenvalue are recorded (`calibration_cov_cond`, `calibration_cov_min_eig`; exactly-singular →
  `asr_calibration_covariance_degenerate`). The guard immediately caught a pre-existing test that
  asserted the garbage path (flat Cz in the ASR covariance → 2.06e16 µV) as a success.
- **e2e fidelity assertions (fix 1b).** `test_hbn_chain_e2e.py` bounds the clean fixture's ASR
  burden (`repaired_fraction < 0.85`, **measured 0.762**; audit 0.76) and asserts ERP retention
  (median evoked↔template correlation ≥ 0.5, **measured 0.88**, reversed-template non-vacuity
  control). The retention fixture injects the N1/P2 template on the anterior EEG subset
  (`erp_mode='anterior'`) because the average-reference step cancels a common-mode ERP exactly
  (measured r = −0.98 on Cz).
- **Fix 7 — interpolated channels after ASR.** The channels a previous interpolate step
  reconstructed (excluded from the ASR filter, B7-N1) are now **re-interpolated from the
  ASR-cleaned data** with the same mode, so the saved channel is the spline reconstruction of the
  data it sits in (1.02/1.09 relative-norm divergence before; the e2e asserts < 0.05 now).
  Recorded as `reinterpolated_after_asr` / `reinterpolation_mode` /
  `reinterpolation_relative_norm`; `bads` restored in a `finally`.
- **CI can go green (fix 2).** The two P4 `mne_connectivity` tests are deselected in
  `pipeline-verification.yml` (`mne_connectivity` is not in `.[pipeline]`; documented P4
  baseline) and a new step runs `pytest -m fsaverage` and **fails on any skip** (≥1 pass
  required). Local dry run of the exact CI commands (final code): §0.1 (with the two deselects) →
  853 passed / 2 deselected / 2 xfailed / 0 failed / 0 skipped; `-m fsaverage` → 43 passed /
  0 skipped; `-m chain_e2e` → 18 passed; gate → 19 passed with the scan count printed. A real
  green GitHub run requires pushing the branch.
- **Blink gate (fixes 3–4).** `blink_residual_ratio` refuses to run on a mismatched timeline
  (`sfreq`/`n_times`/`first_samp`; `reason='timeline_mismatch'` — a resampled post used to get a
  bogus pass/fail verdict) and measures the pre/post peak-to-peak on **identically 1–15 Hz
  band-passed copies**, so line noise in the pre recording can no longer bias the ratio toward
  pass (100 µV 60 Hz fixture: 0.20 false pass before, 1.0 correct fail after).
- **Crosstalk (fixes 5–6).** Direction convention corrected everywhere (verified against mne
  1.13.2: rows are CTFs → `C[i, j]` is leakage **from ROI j into ROI i**; row = target, column =
  source; axis labels, captions, provenance/npz definitions). The saved **primary** matrix is now
  the **flip-aligned** estimator (matches `mean_flip`; rank agreement 0.76 vs 0.61 for the
  library signed mean); the library matrix is kept as `crosstalk_unflipped` and the exact
  numpy-vs-library cross-check still pins the unflipped path. Provenance records
  `primary_matrix`, `n_directed_above_0.3` (on the primary), the unflipped summary and
  `max_abs_delta_primary_vs_unflipped`; QC records `crosstalk_matrix_kind` and states it in the
  figure/caption. `{"computed": false}` carries a `reason`.
- **Test honesty + smaller gaps (fixes 8–9).** The contaminated e2e variant compares against the
  events captured **before** the chain (the old check compared `raw_in` with itself);
  `assert interpolated` added. The no-silent-except gate prints the scan count; line-noise
  failures return **NaN + note** instead of 0.0 (RawQC and QualityControlModule);
  `run_processor.load_processed_runs` records an unreadable preprocessed FIF (logged reason,
  `unknown`/`review`, warning) instead of a silent 128-channel default and reads the canonical
  `bad_channel_detection` key first; `preprocessed_qc` subtracts only role channels that were
  **EEG in the before set** (a native EOG electrode no longer hides a genuine EEG drop).
- **⛔ Recorded, not implemented — open user decision blocking B8:** ASR high-pass design
  (plan §12 item 6, decisions log): (i) compute the ASR correction on a ≥0.5–1 Hz high-passed
  copy and add it to the 0.1 Hz data, (ii) 1 Hz chain variant, (iii) keep as-is with the guard +
  documented distortion. Evidence: 76 % (audit) / 0.63 repaired fraction, 94 %/<0.5 Hz + 52 %/
  0.5–2 Hz power removed, the blow-up regime.
- **Validation (rev2)**: full standard suite → **2 failed / 853 passed / 2 xfailed / 0 skipped**
  (failures exactly the two `mne_connectivity` P4 baseline tests); `pytest -m chain_e2e` → 18
  passed; `pytest -m fsaverage` → 43 passed / 0 skipped; gate → 19 passed / 108 files / 0
  offenders. Probes: `/tmp/opencode/b7rev2_probe{1..8}.py`.

---

## Remediation batch B8 — synthetic dipole source-ERP end-to-end test (task 1) (2026-10-16)

Plan `plan/261013-2240-p1p2-remediation.md` §B8 task 1 + the Opus B7 amplitude carry. **Checkpoint B**
(real EGI-129 data) remains ⛔ blocked by the ASR high-pass decision (plan §12.6) and the second
dataset path; it is deliberately not attempted here.

### New test — `tests/unit/test_e2e_synthetic_source_erp.py` (`fsaverage` + `chain_e2e` + `slow`, 16 tests)

- **Fixture** (`source_erp_run`, module-scoped: one chain run, one source run, one feature run): the
  B7 synthetic GSN-HydroCel-129 recording (`make_gsn129_recording`) carrying an ERP generated from
  the **B2 dipole** — the plan waveform (N1 −1.0 at 0.100 s, P2 +0.6 at 0.200 s) on every
  `SensoriMotor.Superior` label vertex × `mne.label_sign_flip`, projected with `mne.apply_forward`
  (10 µV sensor peak) and injected at every 2-s `stim` onset — then the full `hbn_langer` chain,
  plain-MNE epoching (`mne.Epochs(raw, events, tmin=-0.2, tmax=0.6, baseline=(None, 0),
  reject_by_annotation=True)`; the P3 epochs module is deliberately not used),
  `SourceReconstructionModule` (evoked dSPM, oct6 fsaverage, `compute_crosstalk=False`) and
  `SourceERPFeatureModule` (reads the archive **from disk**, so the provenance sidecar check runs).
- **Measured**: chain all steps `ok` (`asr.repaired_fraction` 0.719, detector flags/interpolates
  `E104`), `stim` samples unchanged, 74/74 epochs; target ROI N1 **−8.28 at 0.100 s** (within one
  sample), `corr(tc, w) = 0.964` (shifted-template control far below), signed output, cerebellar
  exclusion, ≥3 vertices/ROI; feature CSV columns
  `1_SensoriMotor.Superior_{N1,P2}_peak_{latency,amplitude}` + `1_nave == 74`, values equal to the
  npz; condition name pinned `'1'` (plain MNE infers `event_id={'1': 1}` from the events array; the
  P3 module would have used the `stim` annotation name).
- **Amplitude retention (Opus B7 carry)**: per-channel projection gain `<evoked_c, w>/<w, w>`
  normalised by the expected average-reference factor `arf_c = p_c − mean(p)` (scalar
  `1 − mean(p)²/mean(p²) = 0.9952`; the balanced midline dipole pattern makes the reference
  correction small here — 0.970 un-normalised → 0.966 normalised — while the B7 binary-mask fixture
  is where the factor is material at 0.541). Measured **retention 0.966** (aggregate 0.994), band
  **[0.7, 1.3]**; the chain-input control returns **0.987** (the injection is intact, so the 3.4 %
  loss is the chain's) and the shifted-template control < 0.2 × retention — the measure is
  non-vacuous on both ends.
- **Fixture sharing**: new `gsn129_dipole_setup` conftest fixture builds the oct6 forward **through**
  `SourceReconstructionModule._get_template_forward` on the 109-channel EEG set the chain leaves
  behind, so the class-level `_FORWARD_CACHE` serves the module under test (the test asserts the same
  forward object; one ~10 s build per session). `make_gsn129_recording` gained the additive,
  backwards-compatible `erp_sensor_data` parameter (verbatim sensor-space injection at every `stim`
  onset; only the B8 test uses it).
- **Probe findings while writing the test** (both caught before landing):
  (1) injecting the *epoch-window* array (−0.2…0.6 s) at the stim onset carries the 0.2-s pre-stim
  padding and shifts the injected ERP by 0.2 s — the first probe measured retention 0.015 and
  `corr(tc, w) = −0.05`; the conftest fixture injects the post-stim window (126 samples) and the
  input-gain control guards the regression;
  (2) the two-step `apply_forward` scaling must derive the amplitude from the probe's peak of the
  *probe-scaled* waveform: the first version computed the probe at 1 nA but scaled the unscaled
  `stc`, producing a 1e4 V injection, which the chain's ASR covariance guard correctly rejected as
  degenerate (the fixture now asserts the projected peak == 10 µV).
- **Validation**: `pytest tests/unit/test_e2e_synthetic_source_erp.py -q --no-cov -p no:cacheprovider`
  → **16 passed** (28 s); `pytest -m chain_e2e` → **34 passed** (18 B7 + 16 B8); full standard suite
  → **2 failed / 870 passed / 2 xfailed / 0 skipped** (failures exactly the two `mne_connectivity`
  P4 baseline tests). Probes: `/tmp/opencode/b8_probe1.py`, `/tmp/opencode/b8_probe2.py`.

---

## Final-verification-audit fixes (F1–F7) (2026-10-16)

Fix round applied after the final deep audit of the P1/P2 acceptance conditions (briefing:
`planning/audits/2026-10-16-final-verification-briefing.md`). **Golden resolved hash re-pinned
`1ef8780d…` → `b47de382047676631d2d096b8553ba70b79734aaaebde6774ca00a08e83f1a52`** (F2/F3).

- **F1 (major) — CI file never committed.** `.github/workflows/pipeline-verification.yml` was
  **untracked** (the B7 commit message claimed it but it was never added); the file is complete
  and is committed with this round. A **green remote CI run requires pushing** — recorded as a
  user-gated open item under B8/Checkpoint B in `VERIFICATION_TODO.md` (local dry run ≠ remote
  green run).
- **F2 (major) — block-rejection threshold not hashed.** The 100 µV median-ptp threshold was a
  local in `_BlockRejectionStepWrapper.process`; it is now the module constant
  `BLOCK_REJECTION_PTP_THRESHOLD` in `hbn_langer_chain.py`, read at call time by `process` and
  exposed via `get_config()` (`ptp_threshold_v`, wrapper version 1.0 → 1.1) and recorded in the
  step metadata. Sensitivity test: changing the constant changes `resolved_config_hash`.
- **F3 (major) — ASR window tolerances + blow-up factor not hashed.**
  `ASR_OUTPUT_BLOWUP_FACTOR` and the canonical `window_tolerances=(-inf, 7.0)` (the
  `clean_windows` z-thresholds that decide the `BAD_asr_window` marks) were missing from
  `_ASRStepWrapper.get_config()`. `ASR_WINDOW_TOLERANCES` is now a named constant in
  `steps/asr.py` (also the `ASRStep` default), the wrapper passes it explicitly to `ASRStep` and
  reports it plus `output_blowup_factor` in `get_config()` — read at call time (wrapper version
  2.2 → 2.3). Sensitivity tests for both constants, plus a test that the wrapper really passes
  the constant through to `ASRStep` (the config is not cosmetic).
- **F4 (minor) — `forward.template` silently ignored.** `ForwardConfig.template` is now
  `Literal["fsaverage"]` (`core/config.py`): any other template is rejected at validation
  instead of being accepted and silently ignored (the source module's forward/BEM/trans/labels
  are fsaverage-specific). `spacing` deliberately left unrestricted (an existing test exercises
  `oct4`; restriction is not trivial). Rejection tests in `test_source_config.py`.
- **F5 (minor) — briefing count.** `planning/audits/2026-10-16-final-verification-briefing.md`
  said the gate gives "20 passed"; the actual result is **19 passed** (108 files / 0 offenders)
  — corrected, and the now-stale "guard factor not in `get_config`" residual marked fixed.
- **F6 (minor) — QC copy replaced annotations.** `preprocessing_comparison_qc.py` called
  `set_annotations(annot)` on the display copy, wiping its stim/event annotations. It now calls
  the new `append_eegprep_removed_mark` → shared append-only `append_bad_annotations` helper
  (offset recipe included); display-only, never processed or saved. New tests:
  `tests/unit/test_preprocessing_comparison_qc.py` (3).
- **F7 (minor) — B8 target dominance.** `test_e2e_synthetic_source_erp.py` now asserts the target
  ROI carries the **largest N1 of the whole parcellation** (measured −8.277 vs −6.636 µV next,
  `DefaultMode.PCC`; ≥ 1.1× bound, ratio 1.247), folded into the existing N1 polarity/latency test
  so the suite counts stay pinned; the polarity docstring states it verifies the `mean_flip`
  **sign convention**, not anatomy.
- **Validation**: full standard suite → **2 failed / 876 passed / 2 xfailed / 0 skipped**
  (failures exactly the two `mne_connectivity` P4 baseline tests); `pytest -m chain_e2e` →
  **34 passed** (18 B7 + 16 B8); `pytest -m fsaverage` → **59 passed / 0 skipped**; gate →
  **19 passed** / 108 files / 0 offenders. Per-file: `test_config_hash` 19 (+2),
  `test_source_config` 22 (+1), `test_e2e_synthetic_source_erp` 16 (unchanged),
  `test_preprocessing_comparison_qc` 3 (new).

## ASR EEGLAB-standard path + per-dataset configurability (plan §12.6 decision) (2026-10-16)

The B7-N2 open decision (plan §12 item 6) is resolved as the **EEGLAB-standard path** and
implemented: `eegprep.clean_asr` requires zero-mean / high-passed input (its docstring:
*"assumed to be zero mean, e.g., appropriately high-passed (e.g. >0.5Hz or with a 0.5Hz - 1.0Hz
transition band)"*) and EEGLAB's `clean_artifacts` applies `clean_drifts(EEG, (0.25, 0.75))`
internally before ASR and keeps the high-passed output (verified in the installed eegprep
source). Running ASR on the chain's 0.1-Hz data rewrote most of a *clean* recording
(`repaired_fraction` 0.762) and blew up on the real pilot recording.

- **`steps/asr.py` (`ASRStep` 2.1 → 2.2).** New constructor params
  `drift_transition_hz=(0.25, 0.75)` (EEGLAB's own transition band; `None`/`'off'` skips the
  filter with a `RuntimeWarning` + `drift_filter_applied: False` in the metadata) and
  `max_dims=0.66` (forwarded to `clean_asr`). The ASR **input** is high-passed with
  `eegprep.clean_drifts` after the picks/exclusions and before the calibration reference is
  extracted; **the write-back is the high-passed repaired data — no add-back**, exactly EEGLAB's
  behaviour. Metadata adds `drift_transition_hz`, `drift_filter_applied`,
  `drift_filter_modified_fraction` and `max_dims`; `pre_max_uv` and `repaired_fraction` are
  measured against the post-drift ASR input (the high-pass effect is recorded separately, so
  `repaired_fraction` keeps meaning "samples ASR rewrote"). Docstrings fixed: **cutoff 20 is
  EEGLAB's conservative setting** (aggressive 3, default 5, conservative 20; GUI default 20;
  Chang et al. 2019 optimal 20–30) — the old "aggressive 20" wording was wrong. The B7-N2 guard
  and the B7-N1 exclusion/re-interpolation stay (re-probed 2026-10-16: the exactly-singular
  interpolated-channel case is still caught with the filter on and off). **Correction
  (2026-10-16, ASR scoped audit R2):** the earlier claim that "the near-singular blow-up regime
  is tamed by the high-pass" was wrong for real data — the drift filter's justification is
  EEGLAB-standard input + lower repair burden, not blow-up prevention; the fix is the
  MATLAB-equivalent pinv tolerance (see the ASR scoped-audit R1–R6 entry below).
- **`hbn_langer_chain.py` (wrapper 2.3 → 2.4).** `build_hbn_langer_chain(asr_cutoff=20.0,
  asr_drift_transition_hz=(0.25, 0.75), asr_max_dims=0.66, asr_calibration='fixed_first_60s')`
  forwards the ASR parameters and returns a `ResolvedChainSteps` (prebuilt steps +
  `HBN_LANGER_SPEC_DEVIATIONS`, so the chain-name path, `PreprocessingSpec.pipeline_steps()` and
  a direct call resolve to the same config/hash). `_ASRStepWrapper` gains `drift_transition_hz` /
  `max_dims` (validated through `ASRStep`, exposed in `get_config`) and accepts
  `calibration_window='data_driven'` — EEGLAB's automatic clean-window calibration, the built-in
  per-recording adaptivity; `get_config` reports the branch it actually runs (the fixed-path
  keys are unchanged; the data-driven branch reports the `DATA_DRIVEN_REF_*` constants read at
  call time).
- **Per-dataset configurability (`core/config.py`, `pipeline_builder.py`).** New pydantic schema
  `HbnLangerParams` (`extra='forbid'`: `asr_cutoff`, `asr_drift_transition_hz`, `asr_max_dims`,
  `asr_calibration`), exposed as `PreprocessingSpec.chain_params` and forwarded by
  `pipeline_steps()` via `build_hbn_langer_chain(**params)`. `chain_params` next to an explicit
  step list is rejected (it would be silently ignored), as are unknown keys, wrong types and
  non-increasing transition bands. Example per-dataset YAML:
  `chain_params: {asr_cutoff: 30, asr_calibration: data_driven}`; the applied values land in the
  resolved config, the sidecar and the ASR step metadata.
- **Real-data probe** (`/tmp/opencode/p1_chain_asr_realdata_probe.py`, chain steps 1–6 then the
  real `ASRStep`, `sub-NDARAC853CR6` RestingState, 368 s): the new path is **finite with
  `max_ratio` 0.312** (target < 10; the `clean_artifacts` probe variant measured 0.35),
  `repaired_fraction` 0.048, `drift_filter_modified_fraction` 0.955; the pre-§12.6
  configuration fails loudly with `asr_output_magnitude_blowup` at **ratio 8.85e13**.
- **Bounds re-measured:** clean-fixture `repaired_fraction` 0.481 (bound `< 0.65`; was
  0.762/`< 0.85`), ERP retention 0.866 (anterior fixture), B8 amplitude retention 0.949 (band
  [0.7, 1.3]; `repaired_fraction` 0.404).
- **`GOLDEN_RESOLVED_HASHES['hbn_langer']` re-pinned
  `b47de382…` → `b07bfe39930a4525ef48d0d2ca053df2144bbd396e95388adcaf963edf82fb8f`.**
- **Tests:** `test_asr_timeline.py` 28 → 33 (drift-before-ASR identity spy, sub-transition
  band-power removal, skip/warning/metadata, `max_dims` plumbing, blow-up guard re-scoped +
  non-vacuity control); `test_config_hash.py` 19 → 21 (chain-params sensitivity, data-driven
  truthfulness); `test_preprocessing_schema.py` 25 → 29 (chain_params validation + forwarding);
  `test_hbn_chain_e2e.py` / `test_e2e_synthetic_source_erp.py` re-measured bounds and provenance
  assertions.

## ASR scoped-audit fixes R1–R6 (2026-10-16)

The §12.6 scoped audit found the drift filter correct but **not** the real-data fix: running the
full chain on pilot5, **4 of 6 recordings that reach ASR failed** with
`asr_output_magnitude_blowup` *under the EEGLAB-standard path*. Root cause (reproduced):
eegprep 0.4.0 calls `np.linalg.pinv` with numpy's default `rcond=1e-15` at
`plugins/clean_rawdata/asr_process.py:214` (via `finite_pinv(Q)`), while MATLAB's `pinv` uses
the absolute tolerance `max(size(A)) * eps(norm(A))` (≈2.2e-14 for ~100 channels).

- **R1 — pinv tolerance, scoped + provenance (critical).** `ASRStep` runs `clean_asr` under a
  temporary, source-verified patch of that call site with the MATLAB tolerance (module-global
  `finite_pinv` in `eegprep.plugins.clean_rawdata.asr_process`, restored in `finally`).
  `pinv_tolerance: 'matlab_max_size_eps(norm)'` is recorded in the step metadata and in
  `ASRStep.get_config` + `_ASRStepWrapper.get_config` (golden hash covers it); a changed eegprep
  call site raises `eegprep_pinv_assumption_broken` instead of silently reverting. Deterministic
  32-channel reproducer (calibration covariance condition number ~1.2e4 + 10 Hz common
  sinusoid): default pinv → `asr_output_magnitude_blowup`; MATLAB tolerance → clean (ratio ~1).
  **Pilot5 re-run: all 6 recordings that reach ASR process cleanly (ratios 0.28–1.27); before,
  4/6 failed at 4.3e12–6.9e13.** Upstream bug-report draft (not filed, user-gated):
  `planning/audits/eegprep-pinv-bug-report-draft.md`.
- **R2 — overclaim corrected.** The drift filter's justification is EEGLAB-standard input +
  lower repair burden (`repaired_fraction` 0.762 → 0.481), **not** blow-up prevention; the
  "near-singular blow-up regime is tamed" claims are removed from the plan §12.6, the code
  docstrings, this changelog and `VERIFICATION_TODO.md`.
- **R3 — resume check compares config hashes.** `cli/preprocess.py` skips a
  COMPLETED/EXCLUDED run only when the stored `config_hash` equals
  `resolved_config_hash(current pipeline)`; a mismatch **reprocesses** (loud message) instead of
  silently reusing a stale result from a different `chain_params`/step configuration.
  `run_processor` has no COMPLETED/EXCLUDED skip path (it always clears outputs and
  reprocesses); its hash semantics are documented there. Tests: matching-hash skip, stale-hash
  reprocess, two-config reprocess.
- **R4 — truthful `info['highpass']`.** When the drift filter runs, `info['highpass']` is set to
  the passband edge (0.75 Hz default) inside `info._unlock()`, matching MNE's own `filter`
  convention; the full band stays in the metadata (`drift_transition_hz`, `info_highpass_hz`).
  Measured attenuation (design probe): N1 0.995, P2 0.978, P3 0.889, LPP 0.573; an 0.8-s ramp
  is attenuated to 0.49 with its peak shifted 196 ms — an add-back option for slow-ERP datasets
  is a recorded follow-up, not implemented.
- **R5 — bound honesty.** The chain-e2e `repaired_fraction < 0.65` bound is documented as a
  regression tripwire, not a fidelity measure (the fixture spectrum is too red; the
  ERP-retention assertion remains the fidelity measure). Docstring only.
- **R6 — bounds.** `HbnLangerParams.asr_cutoff` bounded to [1, 200];
  `asr_drift_transition_hz` requires `0 < low < high <= 5` Hz at schema level and
  `high < Nyquist` at runtime in `ASRStep`; `asr_max_dims` was already bounded.
- **`GOLDEN_RESOLVED_HASHES['hbn_langer']` re-pinned
  `b07bfe39…` → `a216dc003f898808c7e52e0c964bf752cacf15322b13f4cc320f8f7934658422`**
  (R1 `pinv_tolerance` in both `get_config`s + `ASRStep` 2.2 → 2.3 / wrapper 2.4 → 2.5).
