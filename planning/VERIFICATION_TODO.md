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
      `cli/connectivity.py:177` also globs `*_roi_tc.npz` (now the **evoked-only** file) and feeds it to
      that module; it must switch to `*_roi_tc_trials.npz` when the connectivity stage moves to trial-level
      data.

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
      connectivity on evoked 2-D data without the P4 audit. `cli/connectivity.py:177` globs
      `*_roi_tc.npz`, which after B4 is the **evoked-only** archive (trials are in
      `*_roi_tc_trials.npz`). UI page 10 (`ui/pages/10_source_reconstruction.py:91`) still hardcodes the
      default variant `eLORETA-CONN32` next to a free method selectbox (changing the method without renaming
      the variant now raises `ValidationError`) and still offers `aparc`/`schaefer100` parcellations +
      `depends_on.epochs: null`, all of which the new `extra='forbid'`/`Literal` schema rejects — UI is out of
      B4 scope.

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
| 2026-10-15 | P1/P2 | Opus 5.5 B4 final audit (Pass B) | `revise` — decisive: the new detector's raw-scale two-sided variance/HF rule flags 11–97 channels on real HBN pilot recordings (8/10 would be excluded); minors K1–K6 (condition matching, duplicates, degeneracy float-rounding, equalize test honesty, `mode='both'` partial writes, `--input-dir`) | rev1 fixes applied 2026-10-15 — awaiting re-audit |
| 2026-10-15 | P1 | User decision (B4 rev1) | Bad-channel detector variance/HF statistics move to **log-scale robust z with upper-tail detection** (option a); kurtosis stays two-sided on the raw scale; degeneracy becomes a relative-epsilon test | applied in B4 rev1 (pilot5 validation recorded) |
| 2026-10-15 | P1 | Opus B4 re-audit — **B4-D2 (resolved)** | Detector calibration on real HBN remains over-budget: **5/12 non-corrupt pilot recordings within budget**; over-flagging driven by kurtosis (frontal blink clusters — detector runs before EOG regression) + variance in movie tasks; plus a near-flat lower-tail blind spot. **Resolved 2026-10-15 (user): blink-aware design at B6 tier-2** — kurtosis on blink-free samples (EOG when present, else synthetic frontal proxy via `find_eog_events`; optional ICLabel cross-check), log-scale/upper-tail kurtosis, relative near-flat floor; no chain reordering; validate on pilot5 × 3 tasks + the other dataset | resolved (implement in B6) |
| 2026-10-15 | P1 | **RPCA decision (resolved)** | Option (a): **remove `RobustPCAStep` from the ERP chain** (9 steps), recorded as a spec deviation; golden hash pinned after removal in B6 | resolved (implement in B6) |
| 2026-10-15 | P2 | **`mean_flip` decision (resolved)** | Option (a): keep `mean_flip`; document the uniform-patch attenuation (11/30 vs 22/30) and the midline polarity convention; P2 condition 4 sign-off after the docs land | resolved (docs at B6 tier-2) |
| 2026-10-15 | P0/B8 | **Real data (resolved)** | Checkpoint B will use a pilot5 subject (`~/proj/adhd-hbn/data_raw/hbn-eeg/pilot5`) **and** the user's other dataset ("one EP one attention"; path TBD) | resolved |
| 2026-10-16 | P1 | B6 implementation notes (for the auditor) | **RPCA removed** from `build_hbn_langer_chain()` (9 steps) with the deviation in `get_config`/`spec_deviations`; the golden resolved hash is pinned (`f3a40cd6…` rev1 → `73175626…` rev2 → **re-pinned `72c394e7…` after the scoped-re-audit ASR fix**, which now reports the fixed-first-60 s path truthfully) and covers the 9 chain steps + spec deviations + 9 library versions + the detector/ASR algorithm constants; **`robust_pca` was never in `STEP_REGISTRY`**, so "keep the step class + registry" was implemented as "keep the class importable" (adding a registry entry would newly enable it in YAML step lists); **B4-D2 detector**: kurtosis upper-tail on blink-free samples + relative near-flat floor implemented as the ratio rule only — the decision's `log-variance z < −K` alternative is documented as unusable (the lowest channel of the healthy fixture, at 29 % of the median variance, has z = −54, i.e. it flags healthy channels); pilot5 × 3 at rev1: **6/12** (Rest 4/4, DM 1/4, FF 1/4) — **superseded by rev2: 8/12 (Rest 4/4, DM 2/4, FF 2/4)** after log-Pearson kurtosis + band-passed blink ranking + EEG+EOG filtering; the movie-task over-budget (4/8 movie recordings) is genuine artifact burden, not blinks (a vertical-EOG reference mask finds 10–40× more blinks yet changes pass/fail on zero recordings); **blink under-detection persists** (band-passed-variance ranking still picks infraorbital channels in 9/12; 2–6 blinks vs 17–88 for the vertical-EOG reference; plausibility floor 1/min fires once) — carried to B8 (vertical-EOG derivation, floor ≈5/min); follow-ups: pilot-50 budget recalibration. **B6 rev2 (2026-10-16, Opus final-audit conditions NEW-1…NEW-10): EOG band-passed by the chain filter (NEW-1), band-passed blink-channel ranking + rate plausibility (NEW-2), log-Pearson upper-tail kurtosis (NEW-3), algorithm constants in `get_config` (NEW-4), numpy/scipy pinned (NEW-5), pilot table re-run in true chain order incl. zapline (NEW-6), stale RPCA numbers fixed (NEW-7), xfail `raises=AssertionError` + normal instrumentation tests (NEW-8), behavioural detector tests (NEW-9), RPCA write-back only on convergence (NEW-10); scoped-re-audit fix: ASR wrapper `get_config` fixed-path truthfulness + hash re-pin** | implemented 2026-10-16 + rev1 K3 fixes + rev2 final-audit fixes + scoped-re-audit fix; awaiting desk-check/commit |

## Remediation batches — `plan/261013-2240-p1p2-remediation.md`

| Batch | Scope | Status |
|---|---|---|
| **B1** | Executor fail-fast + exclusion + dependency preflight + one preprocessing schema + status persistence (P1-F1, F4, F6, F9 blanket-except only, F17 ImportError only) | `[x]` **verified 2026-10-14** — Opus rev2 approval, no conditions left (residuals → B3/B6) |
| B2 | Source-ERP core: anatomical labels, unique names, geometry, signed output + dipole ground truth | `[x]` **verified 2026-10-14** — Opus conditions met (F2/N1/N2/N3 + CF-1 fixed, K3-confirmed) |
| B3 | Timeline integrity: ASR without cutting, sidecar instead of `info['temp']`, ASR wrapper metadata | `[x]` **verified 2026-10-15** — final audit by Qwen3.8-Max (user-approved Opus 5.5 stand-in): `approve`; rev2 confirmed (`confirm-with-notes`) |
| B4 | Source usability + config wiring + P1 bad-channel detector | `[x]` **verified 2026-10-15 (code scope)** — Opus final audit `revise` → rev1 applied (log-scale detector + config bounds + condition guards + test honesty) → scoped re-audit `approve`; B4-D2 **resolved 2026-10-15**: blink-aware detector design at B6 tier-2 (decisions log / plan §12.5) |
| B5 | Source hardening + provenance (covariance, trial inversion, eLORETA, BEM, montage) | `[x]` **verified 2026-10-16** — Opus final audit `revise` → rev1 applied (N1/N2/A1/A2/A3/A6/A7/N6) → scoped re-audit `approve`, no conditions left (residuals → B6/B7/B8) |
| B6 | ⛔ Checkpoint A (RPCA) + instrumentation + golden resolved-config hash + P1 tier-2 debt | `[~]` implemented 2026-10-16 + rev1 K3 fixes + **rev2 Opus final-audit fixes (NEW-1…NEW-10) 2026-10-16** — awaiting re-audit |
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

B4 evidence (2026-10-15, working tree on `remediation/p1p2`, not committed) — **P2-F5/F6/F7/F8/F9
and P1-F5**:
- **P2-F5 schema**: `ForwardConfig`/`InverseConfig`/`SourceOutputConfig`/`SourceConfig` are
  `extra='forbid'`; `InverseConfig` gains `snr_trial`/`method_params`, `SourceOutputConfig` gains
  `compute_crosstalk`, `SourceConfig` gains `mode`/`min_trials`/`equalize_trials`/`random_seed` +
  `variant` defaulting to `{method}-CONN32` (mismatch → `ValidationError`); `SourceReconstructionModule`
  now does `SourceConfig.model_validate(config)` (flat `snr`/`spacing`/`trial_level` raise) while
  keeping the test-facing attributes (`method, parcellation, snr, snr_trial, loose, depth, spacing,
  roi_radius, mode`); the CLI exits 1 on `ValidationError`; `examples/configs/source/dspm-conn32.yaml`
  rewritten to the nested dialect (flat keys/`name`/`use_conn_rois`/`conductivity` dropped) and both
  `eegcpm/config/source/*.yaml` validate.
- **P2-F7/F8/F9 outputs**: `mode` evoked/trials/both → `{subject}_roi_tc.npz` (`{cond}: (n_rois,
  n_times)`) / `{subject}_roi_tc_trials.npz` (`{cond}: (n_trials, n_rois, n_times)`) / both; every
  file carries `{cond}_times`, `{cond}_nave`, `sfreq` + label metadata; `output.save_stc` (evoked
  only) / `save_roi_tc` / `generate_qc` are honoured; one `_select_condition_epochs` helper:
  requested = task-config names else `event_id` keys, no matching events → `conditions_missing`,
  `n < min_trials` → `conditions_skipped` (`below_min_trials`), `equalize_trials` draws a seeded
  (`random_seed`) per-condition subset in sorted order (short conditions → `below_equalize_trials`),
  zero produced → `ModuleResult(success=False, "no conditions produced; requested=…, missing=…,
  skipped=…")` with nothing written; metadata gains `conditions_requested/produced/missing/skipped`,
  `nave`, `nave_confound_warning` (dSPM/sLORETA/eLORETA without equalize). Selection runs **after**
  the inverse operator so the forward error stays `errors[0]` and the covariance guards keep
  precedence. **B5 note:** the trial path still uses the evoked lambda2; `snr_trial` →
  `lambda2_trial` + `method_params` remain B5 task 2 as planned.
- **P2-F6 consumer**: `SourceERPFeatureModule` accepts `Mapping`/`NpzFile`/path, derives conditions
  from `{k}_times` keys (no deny-list), keeps negative tmin from `{cond}_times`, derives `sfreq`
  from the axis (500.0 default deleted), writes per-condition columns `{cond}_{roi}_{comp}_{metric}`
  (metric = the real metric key, e.g. `N1_peak_latency`; mean windows `{cond}_{roi}_mean_amp_{w}`)
  + `{cond}_nave`, uses `np.trapezoid`, and raises `ValueError` for 3-D arrays / zero matched
  conditions/ROIs **before** writing the CSV (written once at the end); CLI gains
  `--feature-type source_erp` + `--source/--preprocessing/--task/--subjects/--sessions` (input-dir
  mode or `paths.get_source_dir(...) / f"{subject}_roi_tc.npz"`).
- **P1-F5 detector**: `channel_roles` drops the `raw._eegcpm_reference_channels` hack
  (`metadata['channel_roles']['reference_channel']` is the only source of truth);
  `detect_bad_channels(raw, threshold_sd, exempt, max_bad) -> (bads, report)` uses NaN-safe
  median/MAD robust z (`mad==0 → std → degenerate`), flat (`ptp==0 or var<=1e-20`) non-exempt
  channels bad outright and excluded from the stats, HF band **25–40 Hz** documented
  (`hf_band_hz`, post-45 Hz-lowpass deviation), non-finite z → bad, `> max_bad` →
  `RecordingExcluded('too_many_bad_channels', {n_detected, max_bad_channels, detected})` (no
  truncation), report keys `detected/reasons/z_scores/n_detected_precap/flat/exempt/hf_band_hz/
  degenerate_metrics`; `apply_bad_channels(..., exempt=)` never adds exempt names;
  `_BadChannelDetectorWrapper` passes `exempt` from the channel_roles metadata (missing metadata →
  `KeyError` fails the run) and returns the report + `n_bads`; `interpolate` raises
  `RecordingExcluded('too_many_bad_channels_for_interpolation', {n_bad, pct_bad, max_bad_percent})`
  instead of dropping channels.
- **Validation**: B4 command (`test_source + test_source_config + test_source_erp_features +
  test_source_dipole_groundtruth + test_bad_channel_detector + test_preprocessing_chain +
  test_preprocessing_reachability`) → **116 passed / 2 skipped**; full standard suite → **3 failed /
  616 passed / 2 skipped**, failures exactly the documented baseline (2 `mne_connectivity` P4 + 1
  `robust_pca` B6); greps clean (`trapz|500.0` absent from `source_erp_features.py`, no `except` in
  `bad_channels_pinned.py`); dipole ground truth unchanged; new tests: `test_source_config.py` (17),
  `test_source_erp_features.py` (12), `test_bad_channel_detector.py` (9), 9 `fsaverage`-marked
  condition-selection tests in `test_source.py` (missing/below-min trials, all-missing loud failure,
  seeded equalize identity, below-equalize skip, mode both/trials file shapes, save flags, QC HTML).

**B4 rev1 — Opus 5.5 final-audit conditions applied (2026-10-15)** — verdict `revise`; the decisive finding
was the detector's raw-scale two-sided variance/HF rule flagging 11–97 channels on real HBN pilot recordings
(8/10 excluded). User decision recorded: **log-scale robust z** (option a).
- **B4-D1 detector recalibration** (`steps/bad_channels_pinned.py`; structure kept: flat bad outright and
  excluded from the stats, budget `> max_bad` → `RecordingExcluded`, no `except`, report keys kept): variance
  and 25–40 Hz power are z-scored on the **`log`** scale with the **upper tail only** (`z > threshold_sd`);
  kurtosis stays **two-sided on the raw scale**; the reported z-scores are **signed**; the report gains
  `tails = {'variance': 'upper', 'hf_power': 'upper', 'kurtosis': 'two-sided'}`. Degeneracy is now a
  **relative-epsilon** test (B4-K3): scale `<= 1e-12 * max(|median|, 1.0)` → degenerate (MAD → std fallback
  kept), `z = 0`, metric recorded in `degenerate_metrics`; the log is computed only on stats-eligible
  channels (flat channels would be `log(0)`).
- **Regression fixture** (`test_bad_channel_detector.py::test_healthy_cz_referenced_recording_stays_within_budget`):
  GSN-HydroCel-129-like, spatially correlated (3 cm), Cz-referenced healthy synthetic recording + one injected
  heavy-tailed channel → 2 detections (injected `E70` + one chance kurtosis outlier) `<= MAX_BAD_CHANNELS`,
  lower tail never flagged, flat Cz exempt; the pre-rev1 raw-scale two-sided rule flags 23 channels on the same
  fixture (would raise `RecordingExcluded`). New unit test for the relative-epsilon degeneracy (ulp-level
  spread → degenerate, `z = 0`; `MAD == 0` with a real outlier still falls back to std).
- **Real-data validation (read-only; chain order channel roles → 0.1–45 Hz filter → detector)** on the 5 pilot
  recordings `~/proj/adhd-hbn/data_raw/hbn-eeg/pilot5/*/eeg/*_task-RestingState_eeg.set` (`exempt=('Cz',)`,
  `max_bad=1e9` to count instead of raise; nothing written into the data directory). The canonical chain also
  runs `ZaplineStep(fline=60)` between channel roles and the filter; it is immaterial for the 25–40 Hz band
  and was omitted to match the audit's decisive probe exactly:

| recording | pre-rev1 (raw two-sided) | rev1 (log upper-tail) | rev1 reasons |
|---|---|---|---|
| sub-NDARAC853CR6/RestingState | 28 → EXCLUDED | 19 → EXCLUDED | flat 1, variance 3, kurtosis 16, hf_power 1 |
| sub-NDARAC904DMU/RestingState | 9 | 8 | variance 8, hf_power 3, kurtosis 1 |
| sub-NDARAM487XU3/RestingState | 97 → EXCLUDED | 97 → EXCLUDED | **flat 94**, kurtosis 2, hf_power 1 |
| sub-NDARAM873GAC/RestingState | 24 → EXCLUDED | 9 | flat 1, variance 1, hf_power 4, kurtosis 4 |
| sub-NDARCD661GL0/RestingState | 11 | 9 | variance 7, hf_power 7, kurtosis 6 |

| recording | rev1 — DespicableMe | rev1 — FunwithFractals |
|---|---|---|
| sub-NDARAC853CR6 | 7 | 27 → EXCLUDED |
| sub-NDARAC904DMU | 25 → EXCLUDED | 17 → EXCLUDED |
| sub-NDARAM487XU3 | 97 → EXCLUDED (corrupt) | 95 → EXCLUDED (corrupt) |
| sub-NDARAM873GAC | 36 → EXCLUDED | 21 → EXCLUDED |
| sub-NDARCD661GL0 | 12 → EXCLUDED | 11 |

  **Summary across the 12 non-corrupt recordings: only 5 stay within budget** (3/4 RestingState,
  1/4 DespicableMe, 1/4 FunwithFractals). Over-budget recordings are flagged by both kurtosis (up to
  23 channels, often a frontal cluster) and variance (up to 19). Honest residuals:
  (a) **sub-NDARAC853CR6** stays over budget on RestingState (19) and FunwithFractals (27), dominated
  by kurtosis detections (frontal cluster E4–E19 + E54/E78/E113/E119, excess-kurtosis z 3.2–9.1) —
  the kurtosis rule was kept two-sided/raw as pinned by the user decision, so this is a real
  over-budget recording under the current rule (not a log-scale artifact); (b) **sub-NDARAM487XU3 is
  a corrupt recording**: 114 of 129 raw channels are identically zero
  (94 of the 109 EEG channels flat), so exclusion is correct data-quality behaviour, not detector
  over-flagging.
- **B4-N1 config bounds** (`core/config.py::SourceConfig`): `min_trials: int = Field(10, ge=1)`,
  `equalize_trials: Optional[int] = Field(None, ge=1)`, `random_seed: int = Field(42, ge=0)`; tests:
  `equalize_trials` 0/−1 → `ValidationError`, `min_trials=0` → `ValidationError`, `random_seed=-1` →
  `ValidationError`, bounds reach module construction.
- **B4-K2/N2 condition-name guards** (`modules/source/__init__.py::_select_condition_epochs`): `ValueError`
  for (i) duplicate condition names, (ii) reserved names (`roi_names`/`sfreq`/`excluded_rois`/
  `excluded_reasons`, any `roi_*` prefix), (iii) names ending in `_times`/`_nave`; the guard runs on
  task-config names **and** on the `event_id`-derived path. Tests: duplicate, 5 reserved names, 2 companion
  suffixes, event_id-derived reserved name.
- **B4-K4 test honesty**: `test_equalize_trials_seeded_identical_outputs` now also pins that a different seed
  yields different output arrays (`{cond}_nave == 5` kept) and that the selected trial indices are not
  `arange(5)` (checks `Epochs.selection`).
- **Accepted items assigned**: plan B5 tasks 7–9 (event-code-value condition matching + numeric fallback
  (K1/N3); atomic/stale-output-safe source writes + provenance-requiring consumer (K5 widened); features CLI
  missing subjects/sessions (N4)) + B5 task 2 λ² spy note (N5); plan B7 task 11 (`--input-dir`
  per-feature-type argparse validation, K6); `examples/configs/source/dspm-conn32.yaml` `snr_trial` comment
  annotated "(wired in B5)".
- **Known breaks recorded (deferred, do not fix here)**: the evoked/trials schema change keeps the
  documented P4 connectivity break (unchanged; `cli/connectivity.py:177` globs `*_roi_tc.npz`, now the
  **evoked-only** archive — trials live in `*_roi_tc_trials.npz`); the Streamlit source page
  (`ui/pages/10_source_reconstruction.py`) still offers `aparc`/`schaefer100` parcellations, a free-text
  variant (default `eLORETA-CONN32`, line 91 — changing the method without renaming the variant now raises
  `ValidationError`) and `depends_on.epochs: null`, all rejected by the new schema — UI is out of B4 scope.
- Re-audit scope: `core/config.py` source models, `modules/source/__init__.py` (config/selection/
  output paths), `cli/source_reconstruct.py`, `modules/features/source_erp_features.py`,
  `cli/features.py` + parser, `steps/channel_roles.py`, `steps/bad_channels_pinned.py`,
  `steps/interpolate.py`, `_BadChannelDetectorWrapper`, the new/updated tests.

B5 evidence (2026-10-15, working tree on `remediation/p1p2`, not committed) — **P2-F10/F11/F12/F13 +
the B4-accepted items (K1/K5 widened/N4/N5)**, plan §B5 tasks 1–9:
- **Task 1/5 ranks + provenance**: `process()` computes `data_rank = mne.compute_rank(epochs, rank=None)`
  and `cov_rank = mne.compute_rank(noise_cov, info=epochs.info)` (both in `ModuleResult.metadata` and the
  provenance JSON). Every run with `output.save_stc or output.save_roi_tc` writes
  `{subject}_source_config.yaml` (`cfg.model_dump(mode='json')`), `{subject}-cov.fif`,
  `{subject}-inv.fif` and `{subject}_provenance.json` next to the data artifacts (with both flags off,
  nothing is written — the B4 contract is kept). Provenance keys: `stage/module/module_version/
  created_utc/subject_id/eegcpm/mne/numpy/scipy/method/parcellation/mode/spacing/subjects_dir/bem/trans/
  n_sources/n_channels/n_epochs/sfreq/cov_method/cov_rank/data_rank/loose/depth/pick_ori/
  label_extraction_mode/lambda2_evoked/lambda2_trial/method_params/nave/nave_confound_warning/
  conditions_requested/conditions_produced/conditions_missing/conditions_skipped/labels{roi_names,
  roi_coords_mni,roi_radius_mm,roi_vertex_counts,roi_hemi,excluded_rois,excluded_reasons}/
  roi_flip_summary{mean_dot,lh_mean_dot,rh_mean_dot,n_vertices}/montage_check{median_mm,max_mm,
  n_electrodes,status,warnings}/epochs_path/epochs_sha256/config_hash/config/artifacts{filename→sha256}`;
  `config_hash = sha256(json.dumps(cfg.model_dump(mode='json'), sort_keys=True))` (tested sensitive to
  `inverse.snr`); `epochs_path` is passed by the CLI and hashed in streamed 1 MiB chunks. Test
  `test_provenance_and_reloadable_artifacts` asserts every key, `mne.read_cov` /
  `mne.minimum_norm.read_inverse_operator` reload both FIFs, the per-ROI flip summary matches the vertex
  counts, and `verify_archive_provenance` accepts the archive it produced.
- **Task 2 inversion**: `lambda2_trial = 1/snr_trial**2` passed positionally to `apply_inverse_epochs`
  (spy test wraps the original with **`snr_trial=2.0`** → captures 0.25, `pick_ori='normal'`,
  `return_generator=True`, `method_params=None`; asserting the default 1.0 would not catch the missing
  kwarg — B4-N5); `_resolve_method_params(method, loose, user)` returns `{'force_equal': True, **user}`
  for eLORETA with `0 < loose < 1` (unit-tested incl. user override + not-mutated + loose 0/1 → no flag)
  and `method_params` reaches both `apply_inverse` and `apply_inverse_epochs`; eLORETA smoke run
  (fsaverage, slow) succeeds and records `method_params == {'force_equal': True}`.
- **Task 3 forward**: single-shell `make_bem_model(conductivity=(0.3,))` branch + dead `bem_path`
  reassignment deleted; `_load_bem(subjects_dir)` → `FileNotFoundError` on a missing
  `fsaverage-5120-5120-5120-bem-sol.fif` (tested); class-level `_FORWARD_CACHE` keyed by
  `sha256(ch_names + rounded loc[:3] bytes)|spacing|BEM|mne.__version__|subjects_dir` (the resolved
  subjects_dir is an added component beyond the plan's list: two fsaverage copies must not share a
  forward) — cache hit/`oct5` miss tested; the forward is built with `trans=<subjects_dir>/fsaverage/bem/
  fsaverage-trans.fif` instead of the literal `trans='fsaverage'`, because mne 1.13.2 resolves the string
  to its **bundled** `mne/data/fsaverage/fsaverage-trans.fif` (probed numerically identical to the cached
  copy here, but not guaranteed to be the file the pre-flight validated) — documented deviation.
- **Task 4 pre-flight**: `_preflight_montage(info, subjects_dir)` checks (a) finite non-zero EEG
  `loc[:3]` ("montage"), (b) LPA/Nasion/RPA `FIFFV_POINT_CARDINAL` + every dig point in
  `FIFFV_COORD_HEAD` ("fiducials"/"head frame"), (c) `read_trans` head→MRI + `cKDTree` over
  `fsaverage-head.fif`: warn median > 10 mm, fail max > 25 mm (probed GSN-129 7.25/18.74 mm; +40 mm shift
  13.4/31.0 mm); runs inside the template-forward path, i.e. before the forward is computed (kept there so
  the unmarked `test_process_with_mock_forward` and the covariance-guard tests never need fsaverage);
  with a caller-supplied forward the alignment check is recorded as `status: not_run` (an individual head
  model has no fsaverage trans) — documented deviation. mne 1.13.2 `set_montage` **fabricates** cardinal
  points from `ch_pos` (probed), so the no-fiducial test strips them via `info._unlock()`.
- **Task 6 CLI**: every glob sorted (subjects/sessions/task configs/epoch files); `_select_epoch_file`
  raises on 0 (`FileNotFoundError`) or >1 (`ValueError`) `*_epo.fif` and the session is counted failed
  **with the reason**; `module.process(..., epochs_path=epoch_file)`; the task-config scan's
  `except Exception: continue` now prints the file and `SystemExit(1)`.
- **Task 7 condition matching**: `_match_event_codes` (name keys → their values; `int`/numeric-string codes
  → matching `event_id` values, mirroring `EventSorter.sort_by_condition`) + `np.isin(epochs.events[:, 2],
  values)`; tests pin the hierarchical over-selection fix (probed: `epochs[['a']]` with
  `event_id={'a':1,'a/b':2}` selects 3 epochs, the value match 1), both numeric dialects, the unchanged
  name dialect and the still-loud "no matching events" (now including the codes).
- **Task 8 atomic/stale-safe writes**: new `eegcpm/core/provenance.py` (temp + `os.replace`; mne suffix
  constraints for `-cov.fif`/`-inv.fif`, `.npz` kept on the npz temp name, temp leftovers removed on
  failure; `sha256_file`/`sha256_json`/`to_jsonable`/`package_version`; `provenance_path_for_archive` =
  the single `*_roi_tc.npz` ↔ `*_provenance.json` naming rule; `load/verify_archive_provenance`).
  `_write_source_artifacts` deletes this subject's own previous outputs first (patterns from
  `stale_output_patterns`), writes every artifact atomically and the provenance last, and removes
  everything this run wrote if any write fails (tested by monkeypatching the provenance writer →
  `success=False`, output dir empty). The consumer requires provenance for on-disk archives (path or
  `NpzFile` via `fid.name` — numpy 2.5 dropped `NpzFile.filename`), rejects missing/unreadable/
  `config_hash`-less provenance and a recorded sha256 mismatch, keeps in-memory `Mapping` input
  sidecar-free, and reports `source_config_hash`/`source_epochs_sha256`/`source_epochs_path` in metadata.
- **Task 9 features CLI**: discovery builds the expected subject × session set, prints one stderr line per
  expected-but-absent input, raises `FileNotFoundError` when nothing is found, and returns
  `n_missing`/`missing` in the summary (tested: one present + two absent sessions → `n_rows=1`,
  `n_missing=2`; a requested absent subject → raise, no parquet).
- **Validation**: B5 command `pytest tests/unit/test_source*.py` → **143 passed / 2 skipped** (79 s);
  full standard suite → **3 failed / 667 passed / 2 skipped** (failures exactly the baseline: 2
  `mne_connectivity` P4 + 1 `robust_pca` B6); plan greps clean
  (`grep -n "conductivity=(0.3,)\|make_bem_model" eegcpm/modules/source/__init__.py` → nothing; every
  `pick_ori` is `'normal'`; no direct `np.savez`); manual CLI smoke on a synthetic GSN-129 project
  (`source-reconstruct` then `features --feature-type source_erp`) → 8 artifacts with no temp leftovers,
  provenance with `epochs_path`/sha256 + `montage_check: ok` + `artifacts`, features table 1 row × 63 cols
  and `sub-99` reported missing.
- **Observations for the auditor**: (i) the output stem uses the *stripped* subject id
  (`01_roi_tc.npz` inside `sub-01/ses-01/`), which is what the source CLI has always written (it passes
  `subject_id` without the `sub-` prefix) and what the features CLI + `ui/pages/11_connectivity.py`
  look for — renaming it would touch the documented P4/UI debt; (ii) `montage_check` is `null`-valued
  (`status: not_run`) only when the caller supplies a forward; (iii) provenance is written whenever
  `save_stc`/`save_roi_tc` is on, so `test_save_roi_tc_and_stc_disabled_writes_nothing` (nothing at all
  written) still holds.
- Re-audit scope: `core/provenance.py`, `modules/source/__init__.py` (ranks/inverse/forward/pre-flight/
  provenance/atomic writes/condition matching), `cli/source_reconstruct.py`,
  `modules/features/source_erp_features.py`, `cli/features.py`, `tests/unit/test_source_hardening.py`,
  `tests/unit/test_source_erp_features.py`.

**B5 rev1 (2026-10-16, Opus final-audit conditions):** **N1** — the features CLI enforces one source
`config_hash` per table (mixed configs raise; `--allow-mixed-source-configs` warns + reports the
hashes; hash(es) printed and returned). **N2** — `verify_archive_provenance` requires the archive in
the `artifacts` map with a matching sha256 (missing map/unlisted archive rejected). **A1** — trials
archives pair with `{subject}_provenance.json`. **A2** — provenance records `forward_source`
(`template`/`caller`); `bem`/`trans`/`spacing` are `None` for caller-supplied forwards. **A3** —
pre-flight docstring documents the fabricated-fiducial limitation (distance check is the real guard).
**A7** — cache-key test now varies electrode positions. **N6** — `compute_covariance` uses the
`COV_METHOD` constant. **A6 sign-off (user-approved default):** stale-output deletion stays in the
write phase (pre-write failures keep the previous good set) with the N1 consumer check as the guard;
moving deletion to `process()` start remains available if full "latest-config-only" is wanted.
Assignments: N3 + A4 → plan B6 tasks 19–20; A5 + N4 + N5 → plan B7 tasks 12–14. Tests added:
mixed-config refusal/override, mandatory-artifacts rejection (missing + unlisted), trials-provenance
pairing, caller-forward provenance, montage cache key. Scoped re-audit pending.

B6 evidence (2026-10-16, working tree on `remediation/p1p2`, not committed) — **Checkpoint A
decision (a), P1-F2/F7/F9/F13–F22 + B3-K9/B3-K5/B5-N3/B5-A4 + B4-D2**, plan §B6 tasks 1–20:

- **Checkpoint A (RPCA removed)**: `build_hbn_langer_chain()` is now **9 steps**
  (`channel_roles → zapline → filter → bad_channel_detection → interpolate → eog_regression → asr →
  block_rejection → reference`); the step class stays for standalone use (it was never in
  `STEP_REGISTRY` — there was no registry entry to "keep"). The deviation is machine-readable:
  `HBN_LANGER_SPEC_DEVIATIONS = ["P1-F2: robust_pca removed (destroys ERP; synthetic retention
  2-28%)"]` is part of `PreprocessingPipeline.get_config()` and therefore of the golden resolved
  hash; the chain docstring records it. Instrumentation: `ialm_decompose(..., return_info=True)` →
  `{'n_iter', 'converged'}` and `RobustPCAStep` metadata `nnz_S_frac`/`energy_S_frac`/`n_iter`/
  `converged`. ERP-injection test (`tests/unit/test_robust_pca_erp.py`): 109 ch × 60 s / 250 Hz,
  rank-10 mixed AR(1) + fixed-topography N1/P2 at 25 onsets → **retention 0.675** (topography corr
  0.979, `nnz_S_frac` 0.999, `energy_S_frac` 0.131, `n_iter` **31, converged**) for the plan
  fixture and **0.188** (corr 0.834, `n_iter` 27, converged) for the audit's 30-s destruction
  regime [corrected 2026-10-16 (B6 rev2 NEW-7): the earlier "`n_iter` 200, not converged" was
  stale — the frozen fixture converges; the forced-`max_iter=1` unit test exercises the
  non-converged path]; both assertions on the
  rejected acceptance criterion `retention >= 0.95` are `xfail(strict=True, raises=AssertionError)`
  with the instrumentation/metadata facts in normal tests (B6 rev2 NEW-8), plus a normal test
  asserting `robust_pca` is absent from the chain and the spec deviation is recorded. Fixture
  restored (`tests/fixtures/robust_pca/lin2011_exact_recovery.npz`, sha256
  `2ec821810142bcfae3dba35c0f85fcbac3cb8fae1c016caa27dff12ed8210e50`, in `INDEX.md`, `.gitignore`
  negation `!tests/fixtures/**/*.npz`), loaded via `Path(__file__).resolve().parents[1]` — the
  `test_robust_pca.py` baseline failure is fixed.
- **Golden config (P1-F7)**: every chain step/wrapper `get_config()` now returns all constructor
  params + name/version/enabled/required (verified mechanically for all 9 steps).
  `core/config_hash.py`: `library_versions()` (9 packages, `'not-installed'` fallback),
  `resolved_config_hash(pipeline)`, and the literal
  **`GOLDEN_RESOLVED_HASHES['hbn_langer'] = 731756264f5c1c79783acba79dee1913b491bd5e02801c3b54105f254edef975`**
  (rev2 value, re-pinned after NEW-1…NEW-4; the rev1 value was
  `f3a40cd6ab825513b840ffed333f1de0d425c51f584023caa8b685e28613f378`; computed in the pinned venv;
  `library_versions` = eegcpm 0.1.0, mne 1.13.2,
  numpy 2.5.4, scipy 1.18.1, eegprep 0.4.0, pyzaplineplus 0.1.0, eeglabio 0.1.4, autoreject 0.5.1,
  mne-icalabel 0.10.0). `load_golden_hashes()` keeps all 5 `GOLDEN_CONFIG_NAMES` keys (parity-harness
  compatibility): `'preprocessing'` → the resolved hash, the other 4 stages stay file-content hashes.
  The executor sidecar records `resolved_config`/`config_hash`/`library_versions` for success,
  FAILED and EXCLUDED runs; the CLI and `run_processor` set `workflow.config_hash` before
  `save_state`. `ReferenceStep(check_reference_channel_variance='Cz')` (chain) asserts the
  post-reference Cz column is finite and `> 0` else `RuntimeError` and reports
  `reference_channel_post_ref_variance`. Tests: `test_config_hash.py` (+7 incl. the literal hash,
  per-step param sensitivity over all 9 steps, library-version sensitivity, `'not-installed'`
  fallback, the 5-key parity-harness golden path, `test_hbn_chain_all_required`, sidecar provenance);
  chain-order tests updated to 9 steps.
- **Tier-2**: zapline (`noisefreqs=[fline]` when set, real 4-tuple unpack, counts from
  `analytics['noise_freq_<f>']['n_remove_final']` per chunk → `n_components_removed` dict +
  `_by_chunk`/`_total`, `KeyError` on a missing key, detected freqs from `zap_config['noisefreqs']`);
  EOG regression (single multi-target `lstsq`, equals the per-channel normal-equations solve within
  1e-10, z-scored design condition number invariant to EOG scaling, `spec_deviation` metadata);
  `ReferenceStep(type='rest')` → `NotImplementedError`, dead `exclude_bads` removed; correlation
  detector over 5-cm **montage neighbours** with `np.nanmedian` (flat channel = bad; one flat channel
  no longer NaNs everyone) and the RANSAC/PREP silent handlers now raise `RuntimeError`; ICA
  `rank-1` = `min(max(5, r-1), r-1)` with `ValueError` for `r-1 < 1`; `pipeline` extra pins the six
  exact versions; `MontageStep` raises `ValueError` listing NaN/zero-position EEG channels instead of
  dropping them; `planning/EXCLUSION_CRITERIA.md` regenerated from the code; the four
  `modules/preprocessing/artifacts.py` sites use `append_bad_annotations` (test: `meas_date=None`,
  `first_samp=1000` → event samples unchanged in both frames); `workflow/state.py::save_state` routes
  metadata through the numpy-aware encoder (numpy-scalar round-trip test); the source module records
  `resolved_event_codes` + `run_identity_hash` (config hash + condition mapping) in metadata and
  provenance, the features CLI writes `{feature_type}_{condition}.provenance.json` (source config
  hashes, run identity hashes, parquet sha256, library versions) next to the parquet, the template
  provenance test pins `forward_source == 'template'`, and the forward cache key includes the
  BEM/trans `mtime_ns:size`. The `mean_flip` decision (plan §12 item 4) is documented in
  `modules/source/__init__.py` (uniform-patch 11/30 vs 22/30 dominance; midline polarity convention
  with `MPFC` lh −0.70 / rh +0.71 and one inverted ROI) and in plan §0.5.
- **B4-D2 blink-aware detector** (user decision 2026-10-15, plan §12.5; rev2 fixes 2026-10-16:
  NEW-1/NEW-2/NEW-3): kurtosis on **blink-free samples** (EOG-typed channel, else the most anterior
  non-flat EEG proxy; `find_eog_events` ±200 ms; explicit recorded fallback to the full signal when
  `>50%` masked or `<50%` usable; the ICLabel cross-check is documented as inapplicable — the
  detector is chain step 4, before any ICA); the blink channel is ranked on **1–10 Hz band-passed
  variance** (NEW-2: raw-variance ranking picks a drift-dominated infraorbital channel and
  under-detects blinks on real data; a blink rate below `MIN_BLINK_RATE_PER_MINUTE` is recorded as
  `blink_mask['plausibility_reason']`); kurtosis flagged on the **upper tail** on the **log scale**
  (`log(Pearson kurtosis) = log(excess + 3)`, always defined since sample Pearson kurtosis
  `m4/m2² ≥ 1` — corrected 2026-10-16, B6 rev2 NEW-3: the earlier "log undefined for signed excess
  kurtosis" text applied the log to the signed excess; the auditor's probe measured 8/12 vs 6/12
  within budget with log-kurtosis before the NEW-1/NEW-2 fixes) — this supersedes the B4-D1
  "two-sided" note; **relative near-flat floor** `variance < 1e-3 × median` → reason `flat` (new
  `near_flat` report key) closing the 1 nV blind spot above the absolute `1e-20 V²` floor — the
  decision's lower-tail z alternative is deliberately **not** used because the log-variance spread of
  a healthy recording makes z a non-flatness measure (measured: the lowest channel of the healthy
  regression fixture at 29 % of the median variance has z = −54, i.e. the z rule flags healthy
  channels exactly like the lower-tail rule B4-D1 rejected); report gains `blink_mask`
  (source/channel/n_blinks/blinks_per_minute/masked_fraction/half_width_s/plausibility_reason/reason,
  + `selection` ranking) and `tails['kurtosis'] == 'upper'`; budget and mark-only semantics
  unchanged. The detector's module constants are part of the resolved config (`get_config()
  ['algorithm']`, B6 rev2 NEW-4) and the chain's filter step band-passes the **EEG + EOG** picks
  (NEW-1; `get_config()['picks']`). **Pilot5 × 3 tasks** (read-only, TRUE chain order channel roles →
  `ZaplineStep(fline=60)` → 0.1–45 Hz filter (EEG+EOG) → detector, `exempt=('Cz',)`, `max_bad=1e9`
  to count — re-run after the NEW-1/NEW-2/NEW-3 fixes, B6 rev2 NEW-6):

| recording | task | n_bads | ≤11 | flat | near-flat | blinks (masked) | criteria |
|---|---|---|---|---|---|---|---|
| sub-NDARAC853CR6 | RestingState | 5 | yes | 1 | 0 | 36 (3.9 %) | flat 1, hf 1, kurt 2, var 3 |
| sub-NDARAC853CR6 | DespicableMe | 6 | yes | 1 | 0 | 2 (0.5 %) | flat 1, hf 1, kurt 3, var 2 |
| sub-NDARAC853CR6 | FunwithFractals | 20 | **NO** | 1 | 0 | 5 (1.2 %) | flat 1, hf 1, kurt 14, var 6 |
| sub-NDARAC904DMU | RestingState | 8 | yes | 0 | 0 | 17 (1.7 %) | hf 3, var 8 |
| sub-NDARAC904DMU | DespicableMe | 18 | **NO** | 0 | 0 | 4 (0.9 %) | hf 7, kurt 4, var 12 |
| sub-NDARAC904DMU | FunwithFractals | 17 | **NO** | 0 | 0 | 5 (1.2 %) | hf 13, kurt 3, var 11 |
| sub-NDARAM487XU3 | RestingState | 96 | **NO** (corrupt) | 94 | 1 | 420 (38.5 %) | flat 95, hf 1, kurt 1 |
| sub-NDARAM487XU3 | DespicableMe | 95 | **NO** (corrupt) | 94 | 0 | 272 (40.8 %) | flat 94, kurt 1 |
| sub-NDARAM487XU3 | FunwithFractals | 96 | **NO** (corrupt) | 94 | 0 | 253 (41.3 %) | flat 94, hf 1, kurt 1 |
| sub-NDARAM873GAC | RestingState | 6 | yes | 1 | 0 | 23 (2.3 %) | flat 1, hf 4, var 4 |
| sub-NDARAM873GAC | DespicableMe | 31 | **NO** | 3 | 0 | 37 (8.5 %) | flat 3, hf 3, kurt 12, var 19 |
| sub-NDARAM873GAC | FunwithFractals | 10 | yes | 1 | 0 | 24 (5.8 %) | flat 1, hf 7, kurt 1, var 8 |
| sub-NDARCD661GL0 | RestingState | 8 | yes | 0 | 0 | 6 (0.6 %) | hf 7, kurt 5, var 7 |
| sub-NDARCD661GL0 | DespicableMe | 7 | yes | 1 | 0 | 3 (0.6 %) | flat 1, var 6 |
| sub-NDARCD661GL0 | FunwithFractals | 5 | yes | 1 | 0 | 5 (1.2 %) | flat 1, var 4 |

  **Summary — 12 non-corrupt recordings: 8 within budget** (RestingState **4/4**, DespicableMe
  **2/4**, FunwithFractals **2/4**; up from 6/12 at rev1 — CD661GL0/DM 12→7 and AM873GAC/FF 24→10
  moved inside the budget, AC853CR6/FF 26→20 and AC904DMU/DM 24→18 stayed outside). The auditor's
  log-kurtosis-only probe (without zapline/NEW-1/NEW-2) also measured 8/12 with the same per-task
  split. This table is the **true chain order** (`channel_roles → ZaplineStep(fline=60) → 0.1–45 Hz
  filter (EEG+EOG) → detector`) after the NEW-1/NEW-2/NEW-3 fixes; it supersedes the rev0/rev1 table,
  which omitted zapline and ran before those fixes (the B6-K1 arithmetic correction of that table
  stands; this table is authoritative now; raw probe JSON `/tmp/opencode/b6r2_pilot_full.json`).
  The blink channel is now chosen on **band-passed** variance (E126/E125/E127/E25 on the healthy
  recordings — the drift-dominated E126/E127/E125 were the raw-variance winners; the corrupt
  AM487XU3 picks E8 with 253–420 spurious events). `sub-NDARAC853CR6`/DespicableMe is the only row
  where the low-rate plausibility reason fires (0.70/min < 1/min). The corrupt
  `sub-NDARAM487XU3` (94 of 109 EEG channels identically zero) stays excluded on all three tasks.
  **Target ("most clean recordings within budget") is met for RestingState; the movie tasks remain
  partially over budget — reported honestly.** The remaining over-budget rows (AC853CR6/FF 20 with
  14 kurtosis flags; AC904DMU/DM 18 and FF 17 with variance/HF flags; AM873GAC/DM 31 with variance
  19 + kurtosis 12) are genuine movie-task artifact burden (movement/muscle) under the pinned
  `threshold_sd=3.0` and ≤11 budget — not a blink artifact: the mask applied on every row and a
  diagnostic that additionally masks blinks for the variance/HF criteria moves counts by ≤2 channels
  (the rev0 diagnostic; the as-implemented kurtosis already uses the mask). Recorded follow-ups:
  plan §12.5 option (iv) recalibrate the budget on pilot-50, and/or a user decision on extending the
  blink mask to the variance/HF criteria. The user's other dataset ("one EP one attention", path TBD)
  is not available yet — the second validation dataset remains open.
- **Validation**: full standard suite → **2 failed / 713 passed / 2 skipped / 2 xfailed** (only the
  2 `mne_connectivity` P4 baseline failures; the `robust_pca` fixture test now passes; the 2 xfails
  are the deliberate ERP-retention documentations). New/changed tests: `test_robust_pca_erp.py`,
  `test_robust_pca.py` (fixture path + `return_info`), `test_config_hash.py` (+7), `test_state_schema.py`
  (+numpy round-trip), `test_asr_timeline.py` (+2 artifact-annotation cases),
  `test_preprocessing_steps_tier2.py` (26), `test_bad_channel_detector.py` (tails/keys),
  `test_preprocessing_chain.py`/`test_preprocessing_schema.py` (9-step order),
  `test_source_hardening.py` (run-identity test, `forward_source` + new provenance keys),
  `test_source_erp_features.py` (features provenance sidecar + run identity).
  Re-audit scope: chain composition + RPCA metadata/tests, `config_hash.py`, every chain
  `get_config`, reference/zapline/EOG/ICA/montage/bad-channels edits, the detector (B4-D2),
  `artifacts.py`, `workflow/state.py`, `modules/source/__init__.py` (identity/provenance/docstring),
  `cli/features.py`, the new/updated tests.
- **B6 rev1 — K3 pre-audit fixes (2026-10-16)** (details in `implementation/CHANGELOG.md`):
  **B6-K1** pilot5 counts corrected to **6/12** (Rest 4/4, DM 1/4, FF 1/4; table + raw probe JSON
  re-checked; all three documents consistent now — the *arithmetic* was correct for that table; the
  table itself is superseded by the rev2 true-chain-order re-run below); **B6-K2** module docstring
  keeps only the `var < 1e-3 × median` near-flat disjunct + the z-rule rejection rationale;
  **B6-K3** detector
  docstring "raw excess kurtosis (upper tail, B4-D2)"; **B6-K4** chain docstring: the step class
  stays importable (never registered); **B6-K5** `_find_blink_mask` retries the synthetic frontal
  proxy when `find_eog_events` fails on a dead EOG electrode (new `blink_mask['attempts']`, test
  pins proxy path + applied mask); **B6-K6** `preprocessed_qc.py` reads
  `n_components_removed_total` (scalar) for the zapline note/metric (test with dict payload);
  **B6-K7** `zapline.py` writes the cleaned data back only after the metadata reads that can raise
  (`KeyError` leaves raw unmutated — both KeyError tests assert it); **B6-K8** `RobustPCAStep`
  returns `applied: False` + `reason: 'ialm_not_converged'` at `max_iter` exhaustion, keeping the
  low-rank write-back (the ERP xfail fixtures converge: `n_iter` 31/27, retention 0.675/0.188 — the
  documented xfail regime is untouched; a forced `max_iter=1` test pins the new metadata);
  **B6-K9 (record only)** legacy `bad_channels.py:633,645` silent drops → plan B7 task 15.
  Validation: full standard suite (specparam files ignored per §0.1) **2 failed / 717 passed /
  2 skipped / 2 xfailed** (only the 2 `mne_connectivity` P4 failures); targeted B6 command
  **58 passed / 2 xfailed**; `test_preprocessed_qc.py` **21 passed**.
- **B6 rev2 — Opus final-audit fixes NEW-1…NEW-10 (2026-10-16)** (details in
  `implementation/CHANGELOG.md`): **NEW-1** the chain filter band-passes the EEG **+ EOG** picks
  (`_FilterStepWrapper` v2.0, picks in metadata + `get_config`; `raw.filter()`'s default data picks
  exclude EOG — probed) so the EOG regressors carry no drift/line noise; `EOGRegressionStep` v4.0
  documents the contract; tests pin the mechanism and the regression non-injection (pre-fix contrast:
  59–61 Hz ×>100). **NEW-2** blink channel ranked on 1–10 Hz band-passed variance + low-rate
  `plausibility_reason`; **NEW-3** kurtosis on `log(Pearson)` (upper tail; the "log undefined"
  premise corrected in all docs); **NEW-4** detector/ASR constants in `get_config`
  (`algorithm` block; named `DATA_DRIVEN_REF_*` constants in `asr.py`), versions bumped (detector
  3.0, filter 2.0, EOG 4.0, ASR 2.1), sensitivity test, golden hash re-pinned
  `731756264f5c1c79783acba79dee1913b491bd5e02801c3b54105f254edef975`; **NEW-5** numpy/scipy pinned in
  the `pipeline` extra + re-pin one-liner + library-diff failure output; **NEW-6** pilot5 true
  chain order table above (8/12); **NEW-7** stale RPCA numbers fixed; **NEW-8** xfail
  `raises=AssertionError` + normal instrumentation tests; **NEW-9** behavioural detector tests
  (1 nV near-flat, blink-dominated masked vs full signal, sub-Gaussian sinusoid); **NEW-10** RPCA
  write-back only on convergence.
  Validation: full standard suite (specparam files ignored per §0.1) **2 failed / 728 passed /
  2 skipped / 2 xfailed** (only the 2 `mne_connectivity` P4 failures); targeted B6 files
  **69 passed / 2 xfailed**.

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
      (Qwen3.8-Max stand-in audit + rev2 confirmation). B4 verified 2026-10-15 (code scope). B5
      implemented 2026-10-15 + rev1 fixes 2026-10-16 (N1/N2/A1/A2/A3/A7/N6; A6 sign-off) — **verified
      2026-10-16 (scoped Opus re-audit `approve`)**. **Checkpoint A resolved (remove RPCA, plan §12.1).**
      B6 implemented 2026-10-16 (tasks 1–20 + B4-D2; full suite 2 failed / 713 passed / 2 skipped /
      2 xfailed = only the 2 `mne_connectivity` P4 failures) + **rev1 K3 pre-audit fixes 2026-10-16**
      (B6-K1…K9; suite 2 failed / 717 passed / 2 skipped / 2 xfailed; pilot5 corrected to 6/12) +
      **rev2 Opus final-audit fixes 2026-10-16 (NEW-1…NEW-10)**: EOG band-passed with the EEG
      (NEW-1), band-passed blink-channel ranking + rate plausibility (NEW-2), log-Pearson upper-tail
      kurtosis (NEW-3), detector/ASR constants in `get_config` + re-pinned golden hash `73175626…`
      (NEW-4), numpy/scipy pinned + library-diff failure output (NEW-5), pilot5 re-run in the true
      chain order incl. zapline (NEW-6), stale RPCA numbers fixed (NEW-7), xfail
      `raises=AssertionError` + normal instrumentation tests (NEW-8), behavioural detector tests
      (NEW-9), RPCA write-back only on convergence (NEW-10) — **awaiting re-audit (Opus 5.5,
      milestone batch)**. Next: **B7** (QC truthfulness, test honesty, CI)
      after the B6 audit; B6 residuals to watch: movie-task detector budget + the "other dataset" half
      of the B4-D2 validation.
- [ ] **Final-audit model note:** Opus 5.5 was intermittently unavailable for B3; per user decision
      the B3 final audit ran on Qwen3.8-Max. B4/B6/B7 milestone audits: try Opus 5.5 first, fall back
      to Qwen3.8-Max (or Kimi K3) only with a recorded note.
- [ ] **Science decision before P2 condition 4:** `mean_flip` vs `pca_flip` (plan §12 item 4;
      11/30 vs 22/30 dominance under uniform-patch injection, midline polarity near-ties) — does
      not block B3–B8.
- [ ] Real-subject end-to-end verification (Checkpoint B artifacts → `planning/audits/`)
