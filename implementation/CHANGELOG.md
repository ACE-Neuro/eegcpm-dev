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
