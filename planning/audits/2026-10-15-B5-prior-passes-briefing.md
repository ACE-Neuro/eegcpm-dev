# B5 audit briefing (prior passes + state)

Companion to `plan/261013-2240-p1p2-remediation.md` §0/§B5. B5 = source hardening + provenance:
rank logging; trial λ² (`snr_trial`) + `method_params`; 3-layer-BEM-only forward + cache; montage
pre-flight; full provenance JSON + atomic/stale-safe writes; provenance-gated feature consumer;
event-code-value condition matching; CLI glob/selection/reporting fixes. HEAD `cb49fed` (B1–B4
committed); B5 uncommitted.

## Prior passes (untrusted — verify)

**builder-validator → VALIDATED.** All nine §B5 tasks reproduce independently: `tests/unit/test_source*.py`
→ 143 passed / 2 skipped; full standard suite → 3 failed (exact baseline: 2 `mne_connectivity` + 1
robust_pca) / 667 passed / 2 skipped; plan greps clean. Probes confirmed: `_resolve_method_params`
(eLORETA `0<loose<1` → `{'force_equal': True}`; user params win); trial λ² spy with **non-default**
`snr_trial=2.0 → 0.25` + `pick_ori='normal'`; `_load_bem` → `FileNotFoundError`; pre-flight
(montage-less/fiducials/+40 mm fail; GSN-HydroCel-129 median 7.25 / max 18.74 mm); provenance keys +
`mne.read_cov`/`read_inverse_operator` reload + `config_hash` sensitivity; atomic failure → empty dir,
no temp leftovers; condition matching by value (`{'a':1,'a/b':2}` + code `['a']` → 3 trials, no
over-selection) + numeric fallback; consumer gating (missing/empty/mismatched provenance rejected;
Mapping exempt); CLI reporting. No vacuous tests. Residual notes: (1) `cli/source_reconstruct.py:172`
retains a pre-existing `except Exception` around **explicit** task-config loading (warning + fallback;
out of B5 scope); (2) forward cache key includes `subjects_dir` (safe superset); (3) the montage half
of the cache-key test is only code-inspected.

**K3 pre-audit → approve (all minor).**
- **B5-A1 (minor):** `provenance_path_for_archive("…_roi_tc_trials.npz")` → `…_trials_provenance.json`
  which is never written, while the trials archive's sha256 **is** recorded in `{subject}_provenance.json`;
  `verify_archive_provenance(trials_path)` raises "missing provenance". Fix: map the trials suffix to
  the subject provenance, or drop the trials entry from `artifacts`.
- **B5-A2 (minor):** `_build_provenance` records `bem`/`trans` as fsaverage template files even when
  the caller supplied their own `forward` (`montage_check.status` honestly says "not_run").
  Fix: record `None`/"caller-supplied" in that branch.
- **B5-A3 (minor, docs):** the fiducial pre-flight check (b) is near-vacuous on the montage path —
  mne 1.13.2 `set_montage` fabricates LPA/Nasion/RPA. The distance check (c) is the real guard.
  Document the limitation (plan-mandated check).
- **B5-A4 (minor):** cache key covers BEM **filename**, not contents/mtime; benign GIL double-build.
- **B5-A5 (minor):** dot-temp leftovers (`.*.tmp*`) not in stale patterns; deterministic temp names
  share paths under concurrency (single-writer assumed).
- **B5-A6 (minor, conscious sign-off):** delete-at-start means a failed re-run destroys the previous
  run's good artifacts (plan-sanctioned; consumer requires provenance, also deleted).
- **B5-A7 (minor, test):** `test_forward_cache_key_is_montage_and_spacing_specific` only varies
  spacing; add shifted `loc[:3]` → different key.
- **B5-A8 (minor, docs):** B4 row still says "⛔ B4-D2 open" while the decisions log records it
  resolved; changelog says "29 tests" (actual 30).

## Highest-cost-to-be-wrong items for this audit
1. Trials-archive provenance pairing (A1) — fix or explicitly scope out.
2. Caller-supplied-forward provenance honesty (A2).
3. Atomic-write semantics under crash/concurrency + delete-at-start trade-off (A5/A6) — conscious sign-off.
4. Fabricated-fiducial limitation (A3) — confirm the distance check is the accepted guard.
5. Anything both prior passes missed.

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, k_id or new}],
risk_assessment,
reproducibility_data_integrity_notes,
residual_risks,
conditions_for_approval
```
