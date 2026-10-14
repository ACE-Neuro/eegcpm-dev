# P1 + P2 Deep-Audit Required Fixes (implementation brief)

Source: **Pass B deep audits, Opus 5.5, 2026-10-13** — both verdicts **BLOCK**.
This file distills the *required fixes* (the "what to implement" spec). Status tracking:
`planning/VERIFICATION_TODO.md`. If this file and the audits ever disagree, the audits are
the source of truth; re-audit after fixes.

Environment: work in `eegcpm-0.1/.venv` (mne **1.13.2**, eegprep 0.4.0, pyzaplineplus 0.1.0,
eeglabio 0.1.4, autoreject 0.5.1). NOTE: audits probed mne 1.12.1; in 1.13.2
`make_inverse_operator`/`apply_inverse`/`apply_inverse_epochs` are only under
`mne.minimum_norm`. Verified signatures/recipes: skill `eeg-pipeline-apis`
(`.opencode/skills/eeg-pipeline-apis/`).

---

# P1 — Preprocessing

## P1-F1 (blocker) — Executor failure semantics
`pipeline_builder.py:187-231`, `cli/preprocess.py:244,368`, `pipeline/run_processor.py:375,385,505`,
`hbn_langer_chain.py:291-298`

Problems: every step exception is caught and logged; `success` is always True
(`raw_processed` never None); block-rejection gate raises **before** writing any annotations, so
excluded recordings are saved with zero `BAD_BLOCK` marks; `fatal_on_required_step_error` has no
consumer; CLI skip-check compares `load_state(subject, task, pipeline)` while rows store
`session='01', run='N'` (so the CLI never matches — but `run_processor` does persist COMPLETED).

Required fix:
1. `steps/base.py`: add `required: bool = True` (YAML-overridable via `required: false`).
2. New `RecordingExcluded(Exception)` carrying `reason` + `metrics`.
3. Executor: required step raises / returns `skipped` / `applied is False` / fails
   `validate_inputs` → stop, `success=False`. `RecordingExcluded` → `success=False`,
   `metadata['excluded']={reason,...}`, no FIF written. Optional-step failure → record
   `status:'failed_tolerated'`, `metadata['degraded']=True`, continue. Legitimate no-ops must be
   allow-listed explicitly (e.g. interpolate "no_bad_channels").
4. `success = not errors`; save only on success. Never write `*_preprocessed_raw.fif` for
   failed/excluded runs (quarantine JSON allowed).
5. `__init__`: raise on empty step list; per-step `check_dependencies()` at construction
   (zapline→pyzaplineplus, ASR→eegprep+eeglabio, RANSAC→autoreject).
6. Remove zapline's blanket `except` (`zapline.py:266-273`) or re-raise.
7. CLI: COMPLETED only if all runs succeeded; new EXCLUDED status; `manager.save_state(workflow)`
   after final status; pass session/run to `load_state`; `run_processor.py:505` conditional.
8. Required steps for `hbn_langer`: all 10.

## P1-F2 (blocker, scientific) — Robust PCA destroys the ERP
`steps/robust_pca.py:33,74,142-147`, `hbn_langer_chain.py:66`

Synthetic probe: at pinned λ=0.003162, S is dense (97–99.9% entries) and the injected ERP is
**2–28% retained** (topography corr 0.44); λ=0.01 leaves 101%, λ≥0.03 ≈ no-op. Decision gate
before any real ERP run:
- (a) remove `RobustPCAStep` from the ERP chain (record spec deviation), **or**
- (b) validate on ≥5 real HBN pilot recordings: grand-average ERP with vs without RPCA;
  accept only retention ≥0.95 and topography corr ≥0.95.
Record `nnz_S_frac`, `energy_S_frac`, `n_iter`, `converged` in step metadata; unit test injecting
a known ERP with retention assertion. (F14: μ-cap deviation immaterial; λ pinned to 100k samples
is minor next to this.)

## P1-F3 (blocker, timing) — ASR window rejection corrupts the timeline
`steps/asr.py:167-177,186-191,247-252,327-331`

Problems: `clean_artifacts(WindowCriterion=…)` cuts windows; the step throws away eegprep's
remapped events/mask, rebuilds a `RawArray` and copies **original** annotations → events shift;
annotations past the new end are dropped silently; `meas_date` mismatch raises (ASR then skipped
silently per F1); `info` metadata (`highpass`/`lowpass`, `meas_date`, `bads`, `subject_info`)
lost; no boundary annotations.

Required fix (approved recipe):
- `clean_artifacts(..., WindowCriterion='off')` to get ASR-repaired data at full length;
  then `eegprep.clean_windows(EEG_repaired, max_bad_channels=0.25, zthresholds=(-inf,7))`
  to obtain `sample_mask`;
- convert `~sample_mask` runs into `BAD_asr_window` annotations;
- write data back into a copy of the original Raw (`raw_out._data[picks] = data*1e-6`), never a
  fresh `RawArray`, so `info`/annotations/`meas_date` survive;
- if cutting must stay for non-ERP use: convert eegprep events back to annotations, add
  `BAD_boundary` at each splice, save mask to sidecar `.npy`/JSON;
- NaN/Inf in output → raise (no zero-fill, `asr.py:238-239`);
- unit test: inject artifact window + events; each event must land on the same data sample
  before/after ASR.

## P1-F4 (blocker, data) — Block rejection wipes annotations
`hbn_langer_chain.py:306-309`

`raw.set_annotations(annots)` replaces everything (also wipes `BAD_asr_naninf`). Append using
the offset recipe (see skill / audit F4); test that pre-existing stim annotations survive.

## P1-F5 (blocker, data) — Bad-channel detector
`steps/channel_roles.py:92-100` ↔ `steps/bad_channels_pinned.py:31-33,58-59,76-85,104-122`,
`steps/interpolate.py:93-102`

- Cz exemption wire broken (custom attr written, MNE info read). Pass `exempt=('Cz',)`
  explicitly (from `metadata['channel_roles']['reference_channel']`); drop the info-flag hack.
- Flat Cz (all zeros) → `kurtosis` NaN → kurtosis test silently disabled for **all** channels.
  Use NaN-safe robust stats (median/MAD); zero-variance non-exempt channels = bad outright.
- >11 bads silently truncated (rest stay in data). Instead raise `RecordingExcluded` with
  `n_detected`; remove the channel-drop branch of interpolate (or `on_exceed='raise'`).
- Record detected list, per-criterion z-scores, `n_detected_precap`.
- Remove `except: pass` blocks (lines 84, 101, 115). High-frequency metric runs after the
  45 Hz low-pass — compute before, or document the band.

## P1-F6 — One config schema
`pipeline_builder.py:133`, example YAMLs, `cli/preprocess.py:238`, `run_processor.py:257`

Three dialects exist: nested `params:` (code), flat (examples), keyed-sections (legacy).
Flat/keyed silently no-op or drop params. Fix: pydantic `StepSpec{name: Literal[...], params: dict,
required: bool=True}` + `PreprocessingSpec{steps: list[StepSpec] | Literal['hbn_langer']}` with
`extra='forbid'`, `min_length=1`; validate in CLI and `run_processor`; convert/delete stale configs.

## P1-F7 — Pin the golden config
`config_hash.py:27,41-50`, `examples/.../hbn_langer.yaml`

Hash is self-referential (pins current bytes, nothing enforced); YAML keys disagree with code
(window_criterion 0.2 vs 0.25; max_iter 1000 vs 200; EOG channel ints vs "E8"; `seed`,
`fatal_on_required_step_error`, `fixed_first_60s`, `post_ref_cz_variance_check` unconsumed).
Fix: build the chain *from* YAML (reject unknown keys) or strip YAML to provenance; hash
`get_config()` of resolved params + step versions + library versions; store in state DB + JSON
sidecar; literal golden hashes in a test; implement the post-reference Cz variance check.

## P1-F8 — ASR wrapper metadata
`hbn_langer_chain.py:213-237`, `asr.py:147-177`

Wrapper discards ASR `step_meta` (skipped/window stats/NaN counts) and reports `applied: True`
+ phantom `calibration_window: 'fixed_first_60s'`. Merge metadata, propagate `skipped`, record the
**true** calibration (`data_driven_clean_windows(ref_tolerances=(-inf,5.5))`), or implement fixed
calibration via `clean_asr(ref_maxbadchannels=<ndarray of first 60 s>;)` — ndarray = reference
data (verified). ASR must run on EEG picks only (not EOG/zero Cz).

## P1-F9 — Zapline parameters + metadata
`steps/zapline.py:196-205,214,266-273,275`

Pass `noisefreqs=[fline]` when set; read removed-component counts from `analytics_results`
(NOT `result[1]['n_removed_components']` — fabricated, always 1); rename `validate` →
`validate_inputs`; remove blanket except.

## P1-F10 — QC surfaces failures + channel accounting
`preprocessed_qc.py:289,333-348,671-679`, `hbn_recording_qc.py:70-83,92-105`

Add "Pipeline step status" metric (bad if required step failed/skipped; warning if degraded);
read chain metadata keys (`bad_channel_detection`, not `bad_channels`); count `channel_roles`
exclusions separately from bads (currently ~18% "warning" every recording); wire
`recording_qc_metrics` into the CLI; fix block-rejection keys (`rejected_fraction`,
`n_blocks_rejected`); read ASR burden from sidecar (not `info['temp']`); replace vacuous
`residual_ocular_maxcorr` gate.

## P1-F11/F12 — Test-suite vacuity
`test_no_silent_continue.py:21-22` (`parents[2]`, assert ≥1 file scanned, fix 19 offenders,
add executor failure test); `test_preprocessing_chain.py:192-204` (real Cz test with flat Cz +
heavy-tailed channel); `test_preprocessing_reachability.py` (cap test must now assert
`RecordingExcluded`); `test_robust_pca.py:10` (restore fixture `lin2011_exact_recovery.npz`,
resolve path from `__file__`); CI job with eegprep/pyzaplineplus/eeglabio running the full
chain on a synthetic recording with events (event alignment + annotation survival + metadata).

## P1-F13…F22 — Tier-2 debt (order after the above)
EOG regression: single multi-target `lstsq`; standardize EOG rows before condition number; record
spec deviation; frontal-ERP sensitivity check (`eog_regression.py:59-60,68-102`). ASR default
`window_criterion=0.25` never fall back to `window_length` (`asr.py:170-172`). Reference: raise on
`rest`, remove dead `exclude_bads` (`reference.py:61,105-113`). Bad-channels: `nanmedian` over
montage neighbours; flat = bad (`bad_channels.py:197-206`). ICLabel owns `ica.apply` (double apply
is harmless — idempotent projector). `max(5, rank-1) → min(max(5,r-1), r-1)` (`ica.py:372`).
Pin dependency versions; record library versions. EGI montage guard (`steps/montage.py`).
Regenerate stale `planning/EXCLUSION_CRITERIA.md`. Fix latent QC metrics (`hbn_recording_qc.py:98`).

## P1 conditions for approval
C1 fail-fast semantics persisted (FAILED/EXCLUDED, no FIF) tested on CLI + run_processor paths;
empty step list / missing dep fails at construction. C2 end-to-end synthetic event-timing test
(events stable within 1 sample; annotations survive; `BAD_*` marks; `meas_date` inputs work).
C3 detector test (flat Cz not flagged; heavy-tailed channel detected; >11 bads → `RecordingExcluded`).
C4 RPCA decision recorded (removed, or real-data retention ≥0.95). C5 single config schema +
resolved-config hash + literal golden hashes. C6 QC step status + working no-silent-except gate +
CI job. Re-audit F1–F5 + C1–C4 after fixes.

---

# P2 — Source reconstruction

## P2-F1 (blocker) — Real ROI extraction
`modules/source/__init__.py:332,412` (current: `stc.data[i % n_sources]` — arbitrary vertices)

Required fix:
- Build labels from `inv['src']`: `mne.vertex_to_mni(src[h]['vertno'], h, 'fsaverage', subjects_dir)`;
  select vertices within `roi_radius` mm (default 10); L/R → single-hemi `mne.Label`;
  midline → `mne.BiHemiLabel`; error if label < `min_vertices` (default 3).
- Extract: `mne.extract_label_time_course(stc, labels, src, mode='mean_flip', allow_empty=False)`.
- Save `roi_names`, `roi_coords_mni`, `roi_radius_mm`, `roi_vertex_counts`, `roi_hemi`,
  `excluded_rois` (QC already expects these keys, `source_qc.py:61`). Delete modulo code.

## P2-F2 — Unique ROI names
`data/conn_rois.py:23-25,97-99` — 32 names → 20 unique; 12 L/R collisions overwrite each other.
Fix: key `f"{network}.{name}" + (f"_{hemisphere}" if hemisphere else "")`; QC reads `roi_hemi`
instead of guessing order (`source_qc.py:524-540`).

## P2-F3 — Cerebellar / midline ROI geometry
`data/conn_rois.py:31,34,39,42,43,48,75,76` — Cerebellar ROIs: 0 vertices within 10 mm; midline
ROIs (MPFC 2/4, ACC 3/2) sparse. Fix: exclude cerebellar ROIs from extraction (log
`excluded_rois`); bilateral `BiHemiLabel` for |x| ≤ 5 mm; per-ROI radius override (e.g. 15 mm
midline); log vertex counts.

## P2-F4 (blocker) — Signed orientation (ERP polarity)
`source/__init__.py:85-92,119-125,131-137,401-407` — with `loose=0.2`, `pick_ori=None` yields the
unsigned norm (no negative N1/P2 components; baseline noise floor positive; trial averages
noise-biased). Fix: `pick_ori='normal'` everywhere + `mode='mean_flip'` extraction. With the
template geometry, flip vectors are identical across subjects → polarity comparable. Keep
`loose=0.2, depth=0.8` defaults. [Connectivity computed from magnitude data is invalid — regen later.]

## P2-F5 — Config wiring + evoked path
`source/__init__.py:31-38`, `cli/source_reconstruct.py:249`, `core/config.py:240-266` — nested
`forward`/`inverse` never reach the module; `trial_level` missing from `SourceConfig` (evoked/STC
path unreachable); dSPM example writes to sLORETA-labelled variant. Fix: read
`config['forward']['spacing']`, `config['inverse'][...]`, `config['roi_radius']`,
`config['output'][...]`; extend `SourceConfig` with `mode: Literal['evoked','trials','both']='evoked'`,
`inverse.snr_trial=1.0`, `inverse.method_params`, `min_trials`, `equalize_trials: Optional[int]`,
`roi_min_vertices`, `subjects_dir`; `extra='forbid'` on all config models; derive/enforce `variant`;
fix/delete `examples/configs/source/dspm-conn32.yaml`.

## P2-F6 — Feature consumer contract
`source_erp_features.py:52,68,101,111,117-124,179` — expects ROI-keyed data but producer is
condition-keyed; times start at 0 (ignore tmin); sfreq hardcoded 500; averages across conditions;
writes partial `features.csv` then crashes on unbound `times`. Fix: accept
`{cond: (n_rois, n_times)}` + `{cond}_times` + `roi_names`; convert `NpzFile` → dict; per-condition
feature columns `{cond}_{roi}_{comp}_{metric}`; raise before writing if nothing matched;
`np.trapezoid`; add CLI entry.

## P2-F7 — √nave amplitude confound
`source/__init__.py:95,118-125` — noise-normalised amplitudes scale with `sqrt(nave)` (probe
ratio 4.4721 = √20). Fix: `equalize_trials=N` (seeded `np.random.default_rng(seed)`) or use
MNE-method for amplitude features; always save `{cond}_nave`; flag as covariate.

## P2-F8/F9 — Loud failures
Missing conditions silently skipped; empty output "succeeds" (`source/__init__.py:113-117,155-165,
375-378`). Zero/few-trial conditions: evoked `.average()` raises → kills subject; trial mode writes
`(0,32,T)` arrays. Fix: warn per missing condition; `success=False` when zero conditions produced;
metadata requested-vs-produced; skip conditions < `min_trials`; never write zero-trial arrays.

## P2-F10 — Provenance
`cli/source_reconstruct.py:56,64,141`, `source/__init__.py:145-165` — write `source_config.yaml` +
`provenance.json` (mne/numpy versions, fsaverage path, BEM filename, spacing, n_sources, cov rank,
n_channels, per-condition nave, label vertex counts, epochs path + sha256); save `-cov.fif` +
`-inv.fif`; sort globs; fail on >1 `*_epo.fif`.

## P2-F11 — Covariance hardening
`source/__init__.py:76-82` — H4 **refuted**: baseline-corrected epochs are correct. Real fix:
`method='shrunk'`, `rank=None`, `keep_sample_mean=True`; guards: baseline present, finite data,
warn if `n_epochs*n_prestim < 10*n_eeg`; log `compute_rank`.

## P2-F12 — Trial inversion + eLORETA
`source/__init__.py:95,396-407` — per-trial `apply_inverse` re-prepares operator (~10× slower);
single trials use evoked λ²=1/9. Fix: `apply_inverse_epochs(..., lambda2=1/snr_trial**2,
pick_ori='normal', return_generator=True)`; `method_params={'force_equal': True}` for eLORETA with
0<loose<1 (verified default False in 1.13.2).

## P2-F13 — Forward/montage preflight
`source/__init__.py:254-290`, `pipeline/base.py:148-151` — H6 refuted (1-layer BEM raises for EEG:
delete the fallback + dead `bem_path`, raise FileNotFoundError if BEM missing). H7 confirmed:
add pre-flight (EEG `loc[:3]` finite/non-zero; dig cardinal points; frame=head; transform with
`trans='fsaverage'` and warn >10 mm / fail >25 mm distance to `fsaverage-head.fif`). Use
`mne.BaseEpochs` for isinstance (EpochsFIF). Cache src/forward by hash; recompute per subject is waste.

## P2-F14/F15 — QC truthfulness
`source_qc.py:203-207,935,989-1001` — crosstalk QC never runs and is invalid as written (center
vertex = middle index; delta-vs-reconstructed ≈ 1 for any leakage). Fix: compute once per
template/montage via `make_inverse_resolution_matrix` + `get_cross_talk`/`get_point_spread`
aggregated over label vertices; pass labels/fwd/inv into `qc_data`. QC pass/fail on label vertex
counts + finite data; ERP display from evoked output; fail if `{cond}_times` missing.

## P2-F16 — Tests
`tests/unit/test_source.py` — 9 failed / 17 passed / 2 skipped reproduced. Un-skip e2e behind
presence check (fsaverage is cached); add simulated-dipole ground-truth test: signed dipole
waveform at a CONN ROI vertex via `mne.apply_forward` + noise → assert ROI peak dominance,
polarity consistency across two runs, latency ≤1 sample, min vertex counts, cerebellar exclusion,
unique names, nested config reaches module.

## P2 conditions for approval
1) F1–F6 implemented as specified; 2) F7–F10 (nave saved/equalised, loud failures, provenance);
3) F11–F13; 4) tests green incl. e2e + simulated dipole; 5) one real EGI-129 subject end-to-end
with `plot_alignment` check + crosstalk matrix in QC; 6) re-audit changed areas.

---

# Cross-cutting notes for the plan

- Implementation batches should follow the audits' fix order (P1-F1 → P2-F1–F4 → P1-F3/F4 →
  P2-F5/F6 → P1-F2 decision → rest), with validation + re-audit after each batch.
- Every batch must include the tests named above; the no-silent-except gate must pass honestly.
- The RPCA gate (P1-F2) and the real-subject EGI-129 run (P2 condition 5) need a user decision /
  real data — mark them as explicit checkpoints in the plan.
- Do not run the canonical chain on real data until C1–C4 (P1) and P2 conditions 1–4 hold.
