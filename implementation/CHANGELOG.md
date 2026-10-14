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
