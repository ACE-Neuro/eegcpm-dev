# B3 audit briefing (prior passes + state since)

Companion to `plan/261013-2240-p1p2-remediation.md` §0/§B3. B3 = timeline integrity:
ASR mark-only (repair at full length; `BAD_asr_window` from the `clean_windows` mask;
repaired intervals in metadata, never annotated — user decision). HEAD `4afe675` (B1+B2
committed); B3 uncommitted in the working tree.

## Prior passes (untrusted — verify)

**builder-validator → VALIDATED WITH CONCERNS.** Core correct: full-length repair, write-back
into the original Raw, no `RawArray`, no `info['temp']`, non-finite input/output → `RuntimeError`,
reference channel excluded, truthful calibration metadata, wrapper delegation, QC reads
`chain_metadata['asr']`, F15 default. The one open item was the §0.4 numpy-aware/None-keeping
sidecar encoder (a stated B1-rev1 prerequisite) — **now landed** (see below). Verified
independently: offset-recipe correction is real (MNE 1.13.2 `set_annotations` adds `first_time`
to every onset when `orig_time is None`; identity call moved a stored onset 5.0→9.0 s at
`first_samp=1000`); `append_bad_annotations` pre-shift keeps every annotation on its data sample
in both `meas_date` branches; events `np.array_equal` before/after; burden union confirmed
(0.0321 = rejected 0.0244 ∪ repaired 0.0263, not sum).

**K3 pre-audit → revise.** Findings:
- **B3-K1 (major) — FIXED since:** `_ASRStepWrapper` did not override `validate_inputs` (base
  returns True), so the golden chain silently capped the fixed-first-60 s calibration on short
  recordings; comment falsely claimed the executor rejected it. Fix: wrapper `validate_inputs`
  delegates to the inner `ASRStep` fixed gate; comment corrected; test
  `test_golden_chain_wrapper_enforces_minimum_length` (direct + executor path).
- **B3-K2 (minor) — FIXED since:** stale docs (encoder "not implemented") updated in
  `planning/VERIFICATION_TODO.md` + `implementation/CHANGELOG.md`.
- **B3-K3 (minor, deferred):** `run_processor` dead `has_nan_inf` block (key never emitted) +
  spurious "ASR lost montage" warning for the asrpy branch.
- **B3-K4 (minor, pre-existing, deferred):** `hbn_recording_qc` block-rejection fallback reads
  `fraction`/`n_bad` but the step emits `rejected_fraction`/`n_blocks_rejected`.
- **B3-K5 (minor, deferred):** workflow-level `metadata_json` (`state.py:317`) bypasses the
  sanitizer (step metadata + sidecar use it).
- K3 also confirmed: eegprep µV/`(C, n_samples)` calibration semantics; executor runs
  `validate_inputs` before `process`; `clean_windows` on repaired data measures **residual**
  badness post-repair; no test is vacuous; greps clean.

## State since K3 (already in the working tree)

- **Encoder landed** (`workflow/state.py::_make_json_serializable`): numpy-aware
  (`np.integer→int`, `np.floating→float`, `np.bool_→bool`, `ndarray→list`), keeps `None` in
  dicts/lists, non-finite floats → `"inf"`/`"-inf"`/`"nan"` strings (strict JSON). Tests:
  `test_state_schema.py` (2) + `test_data_driven_sidecar_keeps_none_and_is_strict_json`
  (sidecar keeps `n_calibration_samples: null`; no `Infinity` tokens; `window_tolerances ==
  ["-inf", 7.0]`).
- Suites: B3 scope 16 passed; full **3 failed (baseline: 2 `mne_connectivity` + 1 robust_pca) /
  566 passed / 2 skipped**; plan greps clean (`info['temp']` none in `eegcpm/`; no
  `RawArray`/`nan_to_num` in `asr.py`; `except` only `ImportError` re-raises).

## Highest-cost-to-be-wrong items for this audit

1. `append_bad_annotations` frame correction — probe MNE semantics yourself; confirm annotations
   keep their samples in both branches and B1's committed block-rejection did not regress.
2. Encoder behavior change (`None` kept, non-finite → strings) vs all consumers.
3. `clean_windows`-on-repaired-data: what `window_rejected_fraction` means scientifically now.
4. Adjudicate B3-K3/K4/K5 (accept with batch owners or require now).
5. Is the K1 fix complete (any other path silently caps the fixed calibration — direct `ASRStep`
   vs wrapper, asrpy)?

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, k_id or new}],
risk_assessment,
reproducibility_data_integrity_notes,
residual_risks,
conditions_for_approval
```
