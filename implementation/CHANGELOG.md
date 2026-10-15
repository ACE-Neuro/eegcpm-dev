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
