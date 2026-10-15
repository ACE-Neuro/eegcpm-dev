# B4 audit briefing (prior passes + state)

Companion to `plan/261013-2240-p1p2-remediation.md` §0/§B4. B4 = source usability + P1 detector:
nested `extra='forbid'` SourceConfig; module wiring; condition selection + nave/equalize + loud
failures; evoked/trials output split; feature consumer rewrite; CLI; bad-channel detector rewrite;
interpolate raise. HEAD `e7398fc` (B1–B3 committed); B4 uncommitted.

## Prior passes (untrusted — verify)

**builder-validator → VALIDATED.** B4 command 116 passed / 2 skipped; full suite 3 failed (exact
baseline: 2 `mne_connectivity` + 1 robust_pca) / 616 passed / 2 skipped; greps clean
(`trapz|500.0` absent from `source_erp_features.py`; no `except` in `bad_channels_pinned.py`;
`_eegcpm_reference_channels` hack gone). Independent probes reproduced: flat-key rejection; variant
derivation/mismatch; nested values reach module attrs; condition selection (0-trial→missing,
3-trial→below_min, all-missing→empty, equalize seeded, `nave_confound_warning`); output shapes/flags;
consumer (N1 latency 0.100 s, −1.01 amplitude, negative tmin kept, 3-D→ValueError no CSV, NpzFile);
CLI exit 1; detector (flat Cz exempt + heavy-tail detected, flat non-exempt, 16 bads→
`RecordingExcluded` n_detected=16/max=11, `hf_band_hz==[25,40]`); wrapper KeyError on missing
`channel_roles` metadata; interpolate raise with channel count unchanged. No vacuous tests found.

**K3 pre-audit → approve (all findings minor).**
- **B4-K1 (minor):** condition matching is exact `str(code) in epochs.event_id` key membership;
  `EventSorter.sort_by_condition` (`core/task_config.py:150-157`) has a code-value fallback the new
  helper lacks. Loud "no matching events" for epochs not produced by this repo's stage. Decide B4 vs follow-up.
- **B4-K2 (minor):** duplicate condition names in `task_config['conditions']` silently overwrite
  (probe: two `{"name":"dup"}` → requested `['dup']`, first spec lost).
- **B4-K3 (minor):** `degenerate_metrics` reporting is float-rounding-dependent (`np.nanstd` of
  identical ~1e-12 values can be ulp-nonzero); suggest relative-epsilon degeneracy.
- **B4-K4 (minor):** `test_equalize_trials_seeded_identical_outputs` proves determinism but not
  random subsetting (a regression to first-N would pass); also confirm the intended reading:
  `min_trials=10, equalize_trials=5` → nave 5.
- **B4-K5 (minor):** `mode='both'` writes stc/evoked before trials — IO failure mid-write leaves
  partial artifacts with `success=False` (compute failures are atomic).
- **B4-K6 (minor):** `--input-dir` optional now; non-`source_erp` types fail at runtime with a
  `FileNotFoundError` instead of an argparse error.
- K3 cleared: error precedence (forward/covariance before selection); empty-selection guard;
  consumer closed-handle safety; no deny-list; detector no-except; wrapper KeyError intended;
  `cli/main.py` already wired; deferred connectivity/UI breaks documented.

## Known/deferred (do not treat as B4 regressions)
- Evoked npz schema change breaks `modules/connectivity` + `ui/pages/11_connectivity.py` until P4.
- `ui/pages/10_source_reconstruction.py` can still offer old parcellations (CLI rejects loudly).
- Single-shell BEM fallback → B5 task 3; QC truthfulness → B7; `np.trapz` in sensor-level
  `erp_features.py` → B7 acceptance grep.

## Highest-cost-to-be-wrong items for this audit
1. `_select_condition_epochs` semantics (B4-K1/K4): matching fallback, equalize randomness,
   min_trials vs equalize ordering.
2. Detector degeneracy threshold (B4-K3) and `n_detected == max_bad` allowed vs the ≤10%-budget intent.
3. Deferred-breakage list completeness (e.g. `modules/qc/source_qc.py` 3-D handling under
   `mode='trials'` QC path).
4. `mode='both'` partial-write tail (B4-K5).
5. Any failure mode both prior passes missed.

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, k_id or new}],
risk_assessment,
reproducibility_data_integrity_notes,
residual_risks,
conditions_for_approval
```
