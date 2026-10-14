# Pipeline Verification TODO

**Goal:** verify the EEGCPM pipeline end-to-end, prioritized for **source-based ERP**:
raw → preprocessing → epochs (bad segment/trial rejection) → source reconstruction → ERP.

**Definition of done (per phase):** traceable data flow documented, critical findings fixed or
explicitly accepted, tests green, risks recorded in this file.

## Working agreement

- **Two-pass review per phase:**
  - *Pass A — Triage (Kimi K3-256k, `kimi-code-plan-global/k3-256k`):* map module, data flow,
    candidate defects with `file:line`, open questions. Cheap/free via subscription.
  - *Pass B — Deep audit (Opus 5.5, `opencode/claude-opus-5-5`, builder-deep-auditor):* verifies
    triage candidates independently, deep failure-mode analysis, verdict
    (approve | revise | block) with critical findings and conditions for approval.
- Review agents are **read-only**. Fixes go through implementer + validator agents separately,
  then only changed areas are re-audited.
- Findings are logged below with `file:line`; accepted risks are explicitly recorded.

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Batch 0 — Environment & tooling (done 2026-10-13)

- [x] Repo venv created: `eegcpm-0.1/.venv` — mne **1.13.2**, mne-icalabel 0.10.0, eegprep 0.4.0,
      pyzaplineplus 0.1.0, eeglabio 0.1.4, autoreject 0.5.1, streamlit, pytest.
      torch intentionally not installed (neural extras out of scope).
- [x] Verified all formerly-missing deps import (audit F1 consequence fixed: chain no longer
      silently skips ASR/zapline in this environment).
- [x] Skill `.opencode/skills/eeg-pipeline-apis/` created — verify-don't-guess protocol,
      verified signature references, signature dump script. Caveat: audits probed mne 1.12.1;
      in 1.13.2 `make_inverse_operator`/`apply_inverse(_epochs)` live under `mne.minimum_norm`
      (verified; repo code does not use the old top-level aliases).
- [x] `AGENTS.md` created at repo root — non-negotiable invariants + environment + workflow.
- [x] Full audit required-fixes brief saved: `planning/audits/2026-10-13-P1-P2-required-fixes.md`.

## P0 — Scope & definition of done

- [x] Phase ordering and acceptance criteria defined (below, per phase)
- [ ] Identify a representative subject/session for end-to-end verification

## P1 — Preprocessing *(priority — feeds source reconstruction)*

**Targets:** `eegcpm-0.1/eegcpm/modules/preprocessing/` (steps/, `pipeline_builder.py`,
`hbn_langer_chain.py`), `modules/qc/preprocessed_qc.py`, `modules/qc/hbn_recording_qc.py`,
config wiring (`core/config.py`, `config/preprocessing/*.yaml`).

**Tests:** `test_preprocessing_chain.py`, `test_preprocessing_reachability.py`,
`test_robust_pca.py`, `test_qc.py`, `test_preprocessed_qc.py`, `test_no_silent_continue.py`.

- [x] Pass A (Kimi triage) — done 2026-10-13 · 24 findings (5 high)
- [x] Pass B (Opus 5.5 audit) — done 2026-10-13 · **verdict: BLOCK** · 22 findings (F1–F22)

### P1 audit outcome (blocks real-data runs)

| # | Finding | file:line | Pass A match |
|---|---------|-----------|--------------|
| F1 | Failed/skipped step never fails a run; block-rejection gate raises **before** writing annotations → worst recordings saved with **zero** BAD_BLOCK marks; CLI skip-check mismatches session/run (H19 refuted — CLI doesn't persist COMPLETED) | `pipeline_builder.py:187-231`, `cli/preprocess.py:244,368`, `hbn_langer_chain.py:291-298` | confirmed + worse |
| F2 | **RPCA (IALM, λ=0.003162) removes most of the ERP on synthetic data (2–28% retention)**; λ pinned to 100k samples; must validate on ≥5 real recordings (retention ≥0.95) or remove from ERP chain | `steps/robust_pca.py:33,74,142-147`, `hbn_langer_chain.py:66` | new |
| F3 | ASR window rejection shortens data but keeps original annotation times; annotations dropped silently; `meas_date` mismatch raises (ASR silently skipped); `info` metadata lost; enhancements: prefer `WindowCriterion='off'` + annotate from `clean_windows` mask, write back into `raw.copy()` | `steps/asr.py:167-177,186-191,247-252,327-331` | confirmed + worse |
| F4 | Block rejection wipes all annotations (also `BAD_asr_naninf`); append instead of replace | `hbn_langer_chain.py:306-309` | confirmed + worse |
| F5 | Bad-channel detector: Cz exemption wire break; **flat Cz makes kurtosis NaN → kurtosis test disabled for all channels**; >11 bads silently truncated (remain in data) | `steps/channel_roles.py:92-100` ↔ `steps/bad_channels_pinned.py:31-33,58-59,76-85,104-122`, `steps/interpolate.py:93-102` | H2 confirmed (consequence re-scoped), H12 partial |
| F6 | Three config formats; flat/`keyed` forms silently no-op or lose params | `pipeline_builder.py:133`, example YAMLs, `cli/preprocess.py:238` | confirmed |
| F7 | Golden config not pinned (self-referential hash; unconsumed keys; post-ref Cz check unimplemented) | `config_hash.py:27,41-50`, `hbn_langer.yaml` | confirmed + more |
| F8 | ASR wrapper discards metadata, reports `applied: True` + phantom `fixed_first_60s` calibration | `hbn_langer_chain.py:213-237`, `asr.py:147-177` | new |
| F9 | Zapline params never passed; metadata invented | `steps/zapline.py:196-205,214` | confirmed |
| F10 | QC blind to step failures; miscounts `channel_roles` exclusions as bad channels; `hbn_recording_qc` has no caller | `preprocessed_qc.py:289,333-348,671-679` | confirmed + latent |
| F11 | No-silent-except gate scans 0 files; 19 real offenders found when fixed | `test_no_silent_continue.py:21-22` | confirmed |
| F12 | Vacuous tests (Cz test never calls detector; cap test encodes truncation; missing RPCA fixture; trivially-true copy check) | `test_preprocessing_chain.py:192-204`, `test_preprocessing_reachability.py`, `test_robust_pca.py:10` | confirmed + new |
| F13–F22 | Tier-2 debt: EOG regression frontal-ERP concern, μ cap immaterial (H16 μ refuted), ASR window default 0.5 (H11), reference knobs (H14), neighbor correlation (H18), double ICA harmless (H13 refuted), `max(5, rank-1)` (H22), unpinned deps, EGI montage NaN risk, stale docs, latent metrics | see audit report | mixed |

**Ordered fix plan:** 1) executor/status semantics + dependency preflight (F1, F6 empty-pipeline
guard) → 2) timeline integrity (F3, F4, F8) → 3) bad-channel detector (F5) → 4) RPCA decision
gate (F2) → 5) config schema + resolved-config hash (F6, F7) → 6) zapline + ASR window default
(F9, F15) → 7) QC (F10) → 8) tests (F11, F12) → 9) debt (F13–F22).

**Conditions for approval:** C1 fail-fast semantics; C2 end-to-end synthetic event-timing test;
C3 detector test (flat Cz not flagged, >11 bads excludes); C4 RPCA gate decided/validated;
C5 single validated config schema + provenance hash; C6 QC status + working no-silent-except
gate + CI job with eegprep/pyzaplineplus installed. Re-audit F1–F5 + C1–C4 after fixes.

## P2 — Source reconstruction *(priority — source-based ERP target)*

**Targets:** `eegcpm-0.1/eegcpm/modules/source/__init__.py`, `cli/source_reconstruct.py`,
`data/conn_rois.py`, `data/templates/`, `modules/qc/source_qc.py`, SourceConfig in
`core/config.py`, `config/source/*.yaml`, consumer `modules/features/source_erp_features.py`.

**Tests:** `tests/unit/test_source.py` (9 failed / 17 passed / 2 skipped).

- [x] Pass A (Kimi triage) — done 2026-10-13 · 14 findings (4 high, 1 blocker)
- [x] Pass B (Opus 5.5 audit) — done 2026-10-13 · **verdict: BLOCK** · 16 findings (F1–F16)

### P2 audit outcome (blocks the source-ERP goal)

| # | Finding | file:line | Pass A match |
|---|---------|-----------|--------------|
| F1 | ROI extraction is a placeholder (`stc.data[i % n_sources]`); MNI coords/`roi_radius`/atlas unused. Fix: sphere labels from CONN MNI coords via `vertex_to_mni`, BiHemi for midline, `extract_label_time_course(mode='mean_flip')` | `source/__init__.py:332,412` | confirmed |
| F2 | ROI names not unique (32 names → 20 unique; 12 L/R collisions) → silent overwrites in features/QC; fix naming + use `roi_hemi` | `data/conn_rois.py:23-25,97-99` | new |
| F3 | Cerebellar ROIs have **0 vertices** within 10 mm; midline ROIs sparse (2–4); fix: exclude cerebellum (log `excluded_rois`), bilateral labels for midline, per-ROI radius | `data/conn_rois.py:31,34,39,42,43,48,75,76` | new |
| F4 | **Output is unsigned magnitude → ERP polarity destroyed** (no N1/P2 sign); `pick_ori='normal'` + `mean_flip` required; also invalidates phase-based connectivity | `source/__init__.py:85-92,119-125,131-137,401-407` | new (refutes H14) |
| F5 | Nested config never reaches module; no evoked path (trial mode hard-coded); dSPM config mislabeled sLORETA; `extra='forbid'` + `mode: evoked/trials/both` needed | `source/__init__.py:31-38`, `cli/source_reconstruct.py:249`, `config.py:240-266` | confirmed |
| F6 | Feature consumer can't read producer output (writes partial CSV then fails; `np.trapz` deprecation) | `source_erp_features.py:52,68,101,111,117-124,179` | partial |
| F7 | dSPM/sLORETA amplitude scales with √nave (trial-count confound for CPM): equalize trials or use MNE-method for amplitude features; save `{cond}_nave` | `source/__init__.py:95,118-125` | new |
| F8 | Missing conditions dropped silently; empty output counts as success | `source/__init__.py:113-117,375-378,155-165` | confirmed + escalated |
| F9 | Zero/few-trial conditions: evoked mode crashes whole subject; trial mode writes `(0,32,T)` arrays | `source/__init__.py:117-118,386-416` | new |
| F10 | No provenance (no config/cov/inv/MNE versions/hashes saved; unsorted globs) | `cli/source_reconstruct.py:56,64,141`, `source/__init__.py:145-165` | new |
| F11 | Noise covariance: **H4 refuted** (baseline-corrected epochs are correct per MNE); real weakness is `empirical` w/o shrinkage → use `'shrunk'` + guards + rank logging | `source/__init__.py:76-82` | refuted (core) |
| F12 | Single trials use evoked λ²=1/9; per-trial `apply_inverse` slow → `apply_inverse_epochs`; eLORETA needs `force_equal=True` (H14 claim refuted) | `source/__init__.py:95,396-407` | new |
| F13 | BEM single-shell fallback is dead code (MNE raises; H6 refuted); no montage/dig pre-flight (H7 confirmed); `validate_input` misses `EpochsFIF`; forward recomputed per subject | `source/__init__.py:254-290`, `pipeline/base.py:148-151` | H6 refuted, H7 confirmed |
| F14–F16 | Crosstalk QC never runs and is invalid as written; QC false confidence; tests (9 failing; un-skip e2e + add simulated-dipole ground-truth test) | `source_qc.py:77,145-150,203-207,553-555,935,989-1001` | confirmed + new |

**Ordered fix plan:** F1–F3 → F4 → F5 → F6 → F7–F10 → F11–F16.

**Conditions for approval:** 1) F1–F6 implemented per design (unique names, cerebellum excluded,
signed `mean_flip`, reachable evoked mode, `extra='forbid'`, working consumer); 2) F7–F10 (nave
saved/equalized, loud failures, provenance files); 3) F11–F13 (shrunk covariance, per-trial λ²,
`apply_inverse_epochs`, eLORETA `force_equal`, BEM fallback removed, montage pre-flight);
4) tests green incl. e2e + simulated-dipole ground truth; 5) one real EGI-129 subject end-to-end
with `plot_alignment` check + crosstalk matrix in QC; 6) re-audit changed areas.

## P3 — Epoching & trial rejection

**Targets:** `modules/epochs/` (`trial_filter.py`, `binning.py`, `__init__.py`), `cli/epochs.py`,
`modules/qc/epochs_qc.py`, event mapping (`data/event_mapping.py`).

**Tests:** `tests/unit/test_epochs.py`.

- [ ] Pass A (Kimi triage) — note: P1/P2 audits found epoch-corrupting issues (annotation
      timeline, BAD_asr_naninf, montage) that must land first
- [ ] Pass B (Opus 5.5 audit) — verdict + critical findings
- [ ] Fix confirmed findings; validate with tests
- [ ] Re-audit changed areas; mark P3 verified

## P4 — Features & prediction *(later)*

**Targets:** `modules/features/` (erp, source_erp, connectivity, specparam, isc, drowsiness),
`evaluation/prediction/` (`ridge_cpm.py`, `workflow.py`), `core/leakage.py`,
`core/parity_harness.py`.

- [ ] Pass A (Kimi triage) — focus on CV leakage and feature math
- [ ] Pass B (Opus 5.5 audit) — verdict + critical findings
- [ ] Fix confirmed findings; validate with tests
- [ ] Re-audit changed areas; mark P4 verified
- [ ] Regenerate connectivity from the B2/B4 source format: `*_roi_tc.npz` is now evoked
      2-D with 30 ROIs + label metadata keys (`roi_coords_mni/roi_radius_mm/roi_vertex_counts/
      roi_hemi/excluded_rois`) and trials live in `*_roi_tc_trials.npz` (B4); the old
      magnitude-based connectivity outputs are invalid (P2-F4) and
      `modules/connectivity/__init__.py` still treats every non-`_times` npz key as a condition.

## P5 — End-to-end verification

- [ ] Real run through CLI/`run.sh` on a representative subject (EGI-129)
- [ ] Verify executor/checkpoint/resume behavior (`pipeline/`)
- [ ] Confirm raw → preprocessed → epochs → source → ERP traceability

## Backlog *(after pipeline verification)*

- [ ] REST API frontend
- [ ] User management
- [ ] Regenerate connectivity outputs (P2 audit: existing magnitude-based PLV/wPLI invalid).
      **B2+ live breakage (Opus B2 audit, confirmed 2026-10-14):** `ConnectivityModule`
      (`modules/connectivity/__init__.py:115-116`) and UI page 11 (`ui/pages/11_connectivity.py:573`)
      treat the new npz metadata keys as conditions and fail with `IndexError` on every B2+
      `*_roi_tc.npz`. Loud failure (counted failed), deferred to P4 by deep-auditor decision —
      fix together with the schema migration (optionally raise an explicit
      "npz schema v2 unsupported" error instead of the IndexError). Do not half-enable
      connectivity on evoked 2-D data without the P4 audit.

---

## Findings & decisions log

| Date | Phase | Source | Finding | Status |
|------|-------|--------|---------|--------|
| 2026-10-13 | P1 | Kimi triage (Pass A) | H1 executor swallows failures; H2 Cz wire break; H3 annotation wipe; H4 flat-YAML loss; H5 vacuous gate; +19 more | → verified by Pass B (F1/F5/F4/F6/F11 + more) |
| 2026-10-13 | P2 | Kimi triage (Pass A) | H1 ROI placeholder; H2 nested config ignored; H3 9/26 tests fail; H4 covariance bias; H5 consumer mismatch; +9 more | → verified by Pass B (F1/F5/F16; H4 refuted) |
| 2026-10-13 | P1 | Opus 5.5 audit (Pass B) | **BLOCK** — F1 executor semantics, F2 RPCA ERP destruction (2–28% retention, synthetic), F3 ASR timeline corruption, F4 annotation wipe, F5 detector (Cz + kurtosis NaN + truncation) | confirmed, fix pending |
| 2026-10-13 | P1 | Opus 5.5 audit (Pass B) | Refuted/de-risked: H13 double ICA harmless (idempotent projector); H16 μ cap immaterial; H19 CLI does not persist COMPLETED | accepted |
| 2026-10-13 | P2 | Opus 5.5 audit (Pass B) | **BLOCK** — F1 placeholder ROI (confirmed), F2 non-unique ROI names, F3 cerebellar/midline labels, F4 unsigned magnitude destroys ERP polarity, F5 no evoked path/config wiring, F6 consumer broken, F7 √nave amplitude confound | confirmed, fix pending |
| 2026-10-13 | P2 | Opus 5.5 audit (Pass B) | Refuted: H4 noise covariance (baseline-corrected epochs are correct; switch to `shrunk`); H6 single-shell fallback dead (MNE errors); H14 partially refuted (`force_equal` actually False) | accepted |
| 2026-10-14 | P2 | Opus 5.5 B2 audit (Pass B) | **revise** — F2 download-on-construction (`__init__` calls `fetch_fsaverage`; ~25 unmarked tests; forward ignores `self.subjects_dir`); new N2/N1 individual-subject source space silently mapped onto fsaverage + label cache not keyed by source space; N3 unknown `radius_overrides` keys silently ignored; F1 connectivity break deferred to P4 by decision (loud IndexError; UI page 11 also affected) | **fixed in rev1 2026-10-14**, awaiting confirmation |
| 2026-10-14 | P2 | Opus 5.5 B2 audit — **science escalation** | `mean_flip` vs `pca_flip` under uniform-patch injection: 11/30 vs 22/30 dominant; midline flip axes near-tied (0.008–0.21); one ROI polarity-inverted. **Blocks P2 condition 4; user decision required** (plan §12 item 4) | open |
| 2026-10-14 | P2 | Opus 5.5 B2 re-audit + K3 confirm | **CF-1 fixed**: label cache key now includes the source-space subject (a cache hit previously skipped the N2 subject guard — a scaled template with equal `vertno` silently reused fsaverage labels); regression test added and empirically probed non-vacuous. B2 verified; accepted residuals: trial path untested until B4/B5, connectivity IndexError until P4, single-shell BEM until B5, `fetch_fsaverage` config side-effect recorded for B5 provenance | closed |
| 2026-10-14 | — | Validator wording note | The 2 "connectivity P4 failures" are `mne_connectivity`-missing import failures; the deferred npz-schema `IndexError` breakage (P4 backlog) is not exercised by the suite | noted |
| 2026-10-15 | P2 | Qwen3.8-Max B3 final audit (user-approved Opus stand-in) | `approve` — new findings: **K6** `append_bad_annotations` lacked an append-only self-check; **K7** mask-on-repaired-data semantics undocumented + suite could not detect a regression to pre-repair `clean_windows`; **K8** `montage_restored` phantom key; **K10** tautological `n_samples_before`; **K11** stale counts; **K9** (pre-existing, default-reachable) four uncorrected `artifacts.py` append sites corrupt the timeline when `meas_date=None` + `first_samp≠0`; K3–K5 arbitrated | **rev2 applied + confirmed; B3 verified 2026-10-15** |

## Remediation batches — `plan/261013-2240-p1p2-remediation.md`

| Batch | Scope | Status |
|---|---|---|
| **B1** | Executor fail-fast + exclusion + dependency preflight + one preprocessing schema + status persistence (P1-F1, F4, F6, F9 blanket-except only, F17 ImportError only) | `[x]` **verified 2026-10-14** — Opus rev2 approval, no conditions left (residuals → B3/B6) |
| B2 | Source-ERP core: anatomical labels, unique names, geometry, signed output + dipole ground truth | `[x]` **verified 2026-10-14** — Opus conditions met (F2/N1/N2/N3 + CF-1 fixed, K3-confirmed) |
| B3 | Timeline integrity: ASR without cutting, sidecar instead of `info['temp']`, ASR wrapper metadata | `[x]` **verified 2026-10-15** — final audit by Qwen3.8-Max (user-approved Opus 5.5 stand-in): `approve`; rev2 confirmed (`confirm-with-notes`) |
| B4 | Source usability + config wiring + P1 bad-channel detector | `[ ]` not started |
| B5 | Source hardening + provenance (covariance, trial inversion, eLORETA, BEM, montage) | `[ ]` not started |
| B6 | ⛔ Checkpoint A (RPCA) + instrumentation + golden resolved-config hash + P1 tier-2 debt | `[ ]` not started |
| B7 | QC truthfulness, test-suite honesty, CI job | `[ ]` not started |
| B8 | End-to-end verification prep; ⛔ Checkpoint B (real EGI-129 subject) | `[ ]` not started |

B1 evidence (2026-10-14, after rev1 + rev2): focused suite green (six files: executor semantics,
schema, status persistence [15 tests], reachability, chain, output discovery → **93 passed**);
full suite **12 failed / 508 passed / 2 skipped**, failures exactly the documented baseline
(2 connectivity P4, 1 robust_pca B6, 9 source B2/B4; plus 2 specparam collection errors from the
intentionally missing optional dependency — recorded, never counted as passes);
`grep -rn "info\[.temp.\]" eegcpm --include=*.py` unchanged (11 hits) as required in B1
(removed in B3). Audit trail: Opus rev1 → `revise` (N1–N4/K1–K8) → fixed + validated → scoped
re-audit → `revise` (F1/F2 edge cases) → **rev2 fixed 2026-10-14** (F1: the CLI clears stale
outputs *before* input load; F2: `run_processor` cleanup via shared `core.paths.fail_run_outputs`;
plus a multi-run CLI test and a post-success-crash test) → awaiting confirmation. B1 is **not**
self-approved: re-audit scope `pipeline_builder.py`, `steps/base.py`, `_BlockRejectionStepWrapper`,
`zapline.py`, `core/config.py` (preprocessing schema), `cli/preprocess.py`, `run_processor.py`
status logic, `core/paths.py` helpers, new tests.
**B1 verified 2026-10-14** (Opus quick review after rev2: `approve`, no conditions left). Accepted
residuals (follow-ups): (1) `fail_run_outputs` builds the sidecar stem with `_ses-{session}` while
the pipeline omits the session segment when empty — both callers always pass a session today;
promote one shared stem helper in B3/B6. (2) An excluded run that later crashes shows
`processing_state failed` beside an `_EXCLUDED.json` — labelling only, never a false success.
(3) `aggregate_reports`/`cli/epochs.py` call-site swaps lack dedicated tests (simple swaps, read-verified).

B2 evidence (2026-10-14, working tree on `remediation/p1p2`, not committed): targeted validation
`pytest tests/unit/test_conn_rois.py tests/unit/test_source.py tests/unit/test_source_labels.py
tests/unit/test_source_dipole_groundtruth.py tests/unit/test_connectivity.py` → **73 passed /
2 skipped / 2 failed** (only the 2 connectivity P4 baseline failures); full standard suite →
**3 failed / 540 passed / 2 skipped** (2 connectivity P4 + 1 robust_pca B6; all 9 previously
failing `test_source.py` tests now pass, 2 specparam collection errors ignored as documented).
Greps: `grep -n "i % \|% stc.data\|roi_idx %" eegcpm/modules/source/__init__.py` → empty;
every `pick_ori` call is `'normal'`. Dipole ground truth (fsaverage cached, oct6, dSPM, seeds
0/1): target `SensoriMotor.Superior` has the largest max|tc| in 0.05–0.25 s (178.0 vs next
114.9 = `DefaultMode.PCC`), `corr(tc_target, w) = 1.0000`, N1 extremum negative at 0.1000 s
(±0 samples), same sign for both seeds, `tc.min() < 0`, min vertex count 12, Cerebellar
excluded, names unique — assertion (1) **passed, no triage needed** (FEF leakage baseline
34.96 mm recorded in the test). Probes: `build_conn_labels` at oct6 = 30 labels / 12–104
vertices; most isolated ROI = `DefaultMode.MPFC` (44.07 mm). Re-audit scope: `conn_rois.py`,
`modules/source/labels.py`, `modules/source/__init__.py` extraction/inversion,
`pipeline/base.py` EpochsModule, `source_qc.py` parsing, the two new test files + `conftest.py`.
Follow-up recorded for **P4**: `*_roi_tc.npz` now always contains
`roi_coords_mni/roi_radius_mm/roi_vertex_counts/roi_hemi/excluded_rois`, and the evoked file is
2-D with 30 ROIs while trials move to a separate file in B4 — `cli/connectivity.py` /
`modules/connectivity/__init__.py` still glob `*_roi_tc.npz` and treat every non-`_times` key as
a condition, so connectivity outputs must be regenerated (backlog, out of B2 scope).

B2 rev1 evidence (2026-10-14; **B2 verified 2026-10-14**): F2 — construction no longer calls
`fetch_fsaverage`; the new `subjects_dir` property resolves it on first use and caches it, and
`_get_conn_labels` + `_get_template_forward` both use it (one fsaverage copy); N1 — the label cache
is keyed on the source-space subject + vertno arrays and rebuilds on change (reused module across
subjects stays legal; N2 still rejects individual-subject spaces, now also on cache hits — **CF-1**);
N2 — `build_conn_labels` raises `ValueError` when `src._subject != subject`; N3 — unknown
`radius_overrides` keys raise `ValueError` (excluded-ROI overrides stay valid); the EEG pick in
`process()` moved inside the `try` (all-bad channels → `success=False`). Targeted B2 command →
**81 passed / 2 skipped / 2 failed** (only the 2 `mne_connectivity`-missing baseline failures);
full standard suite → **3 failed / 548 passed / 2 skipped** (2 connectivity + 1 robust_pca B6);
dipole ground truth unchanged (seed 0: 178.0 vs 114.9 next-ROI, `corr = 1.0000`, N1 −178.0 at
0.1000 s, min vertex count 12, Cerebellar excluded). Opus scoped re-audit found **CF-1** (cache hit
skipped the subject guard) → fixed with a regression test (empirically probed non-vacuous) →
K3 confirmation pass: `confirm-with-notes` (cosmetic annotation fixed).

B3 evidence (2026-10-14, working tree on `remediation/p1p2`, not committed): probe (task 1,
`/tmp/opencode/`) — `eeg_mne2eeg` returns `(n_channels, n_times)` float32 in **µV** (×1e6);
`clean_asr` returns a **new** dict at the same shape (float64, µV, never cut) and
`ref_maxbadchannels=<ndarray>` requires `(n_channels, n_samples)` (transposed → `ValueError`);
`clean_windows` returns `(EEG, sample_mask)` with `mask.shape == (n_times,)`, `True` = kept, and
`max_bad_channels` in (0,1) is a **fraction of channels per window**. Implementation: `ASRStep`
v2.0 full-length repair (write-back into the original Raw; `BAD_asr_window` from the mask;
repaired intervals in metadata only; NaN input/output → `RuntimeError`; all `info['temp']`
writes deleted; `validate_inputs` enforces `train_duration` for the fixed branch only);
`_ASRStepWrapper` v2.0 delegates and forwards metadata unchanged (+`retain_burst_masks`);
`_derive_asr_burden` deleted; `hbn_recording_qc` reads `chain_metadata['asr']`. **Offset-recipe
finding (needs auditor attention):** the §0.3 recipe as written moves pre-existing annotations by
`raw.first_time` whenever `meas_date is None` and `first_samp != 0` (mne 1.13.2 `set_annotations`
re-adds `first_time` to *every* onset when `orig_time is None` — probed with an identity
`set_annotations(raw.annotations)`: stored onset 5.0 s → 9.0 s at `first_samp=1000`, 250 Hz);
`append_bad_annotations` pre-shifts the existing onsets back in that branch only, so both frames
keep every annotation on its sample (used by ASR **and** the B1 block-rejection gate).
B3 command (`test_asr_timeline + test_preprocessing_reachability + test_preprocessing_chain +
test_executor_failure_semantics`) → **62 passed**; full standard suite → **3 failed / 562 passed /
2 skipped** (exactly the baseline: 2 `mne_connectivity` P4 + 1 `robust_pca` B6). Greps:
`info['temp']` absent from `eegcpm/`; no `RawArray`/`nan_to_num` in `asr.py`; `except` only
re-raises `ImportError`. New `tests/unit/test_asr_timeline.py` (14 tests, incl. event-sample
integrity for both calibrations × both `meas_date` branches, sidecar provenance, QC
chain-metadata mapping, executor-level `train_duration` gate, and `BAD_BLOCK` offsets in both
frames). **B3 rev1 (orchestrator, 2026-10-14, after K3 pre-audit):** (1) **B3-K1 fixed** — the
`_ASRStepWrapper` now overrides `validate_inputs` (delegates to the inner `ASRStep` fixed-first-60 s
gate), so the golden chain fails loudly on recordings shorter than 60 s instead of silently capping
the calibration; false comment corrected; test `test_golden_chain_wrapper_enforces_minimum_length`
added (executor path included). (2) **Encoder landed** — `workflow/state.py::_make_json_serializable`
is now numpy-aware (`np.integer→int`, `np.floating→float`, `np.bool_→bool`, `ndarray→list`), keeps
`None` in dicts/lists, and converts non-finite floats to `"inf"`/`"-inf"`/`"nan"` strings (strict
JSON); tests added (`test_state_schema.py` encoder tests, data-driven sidecar test) — the earlier
residual (1) is resolved. (3) `asr_burden` is the **union** of rejected+repaired samples (plan
wording "rejected or repaired"), not the old sum. Remaining minors: `hbn_recording_qc` block-rejection
fallback key names still `fraction`/`n_bad` vs emitted `rejected_fraction`/`n_blocks_rejected`
(B3-K4, owned by plan B7 task 3); workflow-level `metadata_json` sanitizer bypass (B3-K5 → plan B6
task 18); `artifacts.py` four uncorrected append sites (B3-K9 → plan B6 task 17). `run_processor`'s
dead `has_nan_inf` block + spurious asrpy warning were removed in rev2 (B3-K3, closed).
**B3 verified 2026-10-15** — final audit by **Qwen3.8-Max** (user-approved stand-in; Opus 5.5
endpoint outage): `approve`; rev2 applied (K6 self-check + test, K7 mask-semantics doc + spy test,
K8 rename, K10 pre-write-back capture, K11 counts annotated, K3 removals, `data_roundtrip` key) and
confirmed (`confirm-with-notes`): both new tests proven non-vacuous (K6 exercises a real
92-annotation drop; K7 pins `clean_windows(EEG_rep)`), suites reproduce (18 passed; 3F/568P/2S).
The audit independently verified the `eeg_mne2eeg` channel-order identity (non-contiguous picks),
the frame correction across FIF round-trip / `set_meas_date` / `anonymize`, the encoder (37 cases)
and the repaired-detector behaviour. Residuals accepted: K6 raises after `set_annotations` mutated
the in-memory raw (fail-loud, nothing persisted); asrpy caps calibration with no gate (non-golden,
truthful metadata); plain `pytest tests/unit` interrupts on the two specparam collection errors
(the documented command ignores those files).
Re-audit scope: `steps/asr.py` (incl. `append_bad_annotations`), `_ASRStepWrapper`,
block-rejection append, `hbn_recording_qc` ASR fields, `workflow/state.py` encoder, the test files.

## Next actions (queue)

- [x] Batch 0 — environment, skill, AGENTS.md, audit brief (done 2026-10-13)
- [x] Remediation plan (Opus 5.5): `plan/261013-2240-p1p2-remediation.md` — 8 batches
- [x] Plan critique (Sonnet 5): verdict `revise` → all 14 findings patched into the plan
      (parity-harness compatibility, non-vacuous crosstalk recipe/test, covariance estimator moved
      to B2, dipole leakage distance corrected to 34.96 mm, avg-ref projector preservation note,
      `modules/qc` added to the no-silent-except gate, plus minor clarifications); delta
      re-critique (Sonnet 5.5) `accept-with-fixes` → 4 residual items patched (B2 covariance
      ordering, dipole triage protocol, crosstalk cross-check, precedence rule)
- [ ] **User checkpoints before implementation:** (A) RPCA gate decision (blocks B6 only);
      (B) real EGI-129 subject/session + data path (blocks B8); (C) confirm deletion of 3 stale
      keyed-dialect YAMLs (blocks B1 task 9; default = delete) — **resolved 2026-10-14: the
      documented default was applied; the three files are deleted and their deletion is asserted
      in `tests/unit/test_preprocessing_schema.py`**
- [ ] Implementation batches B1…B8 (DeepSeek V4.1 Flash `#max`): per batch → builder-validator →
      K3 pre-audit → Opus 5.5 final audit on milestone batches (B1, B2, B4, B6, B7) and whenever
      K3 flags a blocker. → **B1 verified 2026-10-14; B2 verified 2026-10-14; B3 verified 2026-10-15
      (Qwen3.8-Max stand-in audit + rev2 confirmation). Next: B4** (source config wiring + modes +
      consumer + P1 bad-channel detector) — ready to start.
- [ ] **Final-audit model note:** Opus 5.5 was intermittently unavailable for B3; per user decision
      the B3 final audit ran on Qwen3.8-Max. B4/B6/B7 milestone audits: try Opus 5.5 first, fall back
      to Qwen3.8-Max (or Kimi K3) only with a recorded note.
- [ ] **Science decision before P2 condition 4:** `mean_flip` vs `pca_flip` (plan §12 item 4;
      11/30 vs 22/30 dominance under uniform-patch injection, midline polarity near-ties) — does
      not block B3–B8.
- [ ] Real-subject end-to-end verification (Checkpoint B artifacts → `planning/audits/`)
