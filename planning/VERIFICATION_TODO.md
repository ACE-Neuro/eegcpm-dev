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

## P5 — End-to-end verification

- [ ] Real run through CLI/`run.sh` on a representative subject (EGI-129)
- [ ] Verify executor/checkpoint/resume behavior (`pipeline/`)
- [ ] Confirm raw → preprocessed → epochs → source → ERP traceability

## Backlog *(after pipeline verification)*

- [ ] REST API frontend
- [ ] User management
- [ ] Regenerate connectivity outputs (P2 audit: existing magnitude-based PLV/wPLI invalid)

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

## Remediation batches — `plan/261013-2240-p1p2-remediation.md`

| Batch | Scope | Status |
|---|---|---|
| **B1** | Executor fail-fast + exclusion + dependency preflight + one preprocessing schema + status persistence (P1-F1, F4, F6, F9 blanket-except only, F17 ImportError only) | `[x]` **verified 2026-10-14** — Opus rev2 approval, no conditions left (residuals → B3/B6) |
| B2 | Source-ERP core: anatomical labels, unique names, geometry, signed output + dipole ground truth | `[ ]` not started |
| B3 | Timeline integrity: ASR without cutting, sidecar instead of `info['temp']`, ASR wrapper metadata | `[ ]` not started |
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

## Next actions (queue)

- [x] Batch 0 — environment, skill, AGENTS.md, audit brief (done 2026-10-13)
- [x] Remediation plan (Opus 5.5): `plan/261013-2240-p1p2-remediation.md` — 8 batches
- [x] Plan critique (Sonnet 4.6): verdict `revise` → all 14 findings patched into the plan
      (parity-harness compatibility, non-vacuous crosstalk recipe/test, covariance estimator moved
      to B2, dipole leakage distance corrected to 34.96 mm, avg-ref projector preservation note,
      `modules/qc` added to the no-silent-except gate, plus minor clarifications)
- [ ] **User checkpoints before implementation:** (A) RPCA gate decision (blocks B6 only);
      (B) real EGI-129 subject/session + data path (blocks B8); (C) confirm deletion of 3 stale
      keyed-dialect YAMLs (blocks B1 task 9; default = delete) — **resolved 2026-10-14: the
      documented default was applied; the three files are deleted and their deletion is asserted
      in `tests/unit/test_preprocessing_schema.py`**
- [ ] Implementation batches B1…B8 (DeepSeek V4.1 Flash `#max`): per batch → builder-validator →
      K3 pre-audit → Opus 5.5 final audit on milestone batches (B1, B2, B4, B6, B7) and whenever
      K3 flags a blocker. **Start with B1.** → B1 implemented 2026-10-14, awaiting audit
- [ ] Real-subject end-to-end verification (Checkpoint B artifacts → `planning/audits/`)
