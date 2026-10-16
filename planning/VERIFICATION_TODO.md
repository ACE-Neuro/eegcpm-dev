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
| 2026-10-16 | P1 | **ASR high-pass decision (resolved, plan §12.6)** | Option (iv), the **EEGLAB-standard path**: `clean_drifts(0.25, 0.75)` on the ASR **input** before `clean_asr`, write-back = the high-passed repaired data (**no add-back**, EEGLAB's own `clean_artifacts` behaviour), cutoff 20 recorded as **conservative** (aggressive 3 / default 5 / GUI default 20 / Chang 2019 20–30), `max_dims` 0.66, all per-dataset configurable via `PreprocessingSpec.chain_params` (`HbnLangerParams`; `asr_calibration: data_driven` = EEGLAB's automatic clean-window calibration). Real pilot probe (`sub-NDARAC853CR6` RestingState): new path finite, max ratio **0.312** (pre-§12.6: `asr_output_magnitude_blowup` at ratio 8.85e13). Golden hash re-pinned `b47de382…` → `b07bfe39…` | **implemented 2026-10-16; awaiting audit** (see the §12.6 evidence section) |
| 2026-10-16 | P1/P2 | **B7 rev2 — Opus final audit (B7-N2 + majors)** | `revise` conditions applied on the uncommitted B7 tree: **(1) B7-N2 critical — ASR output blow-up guard** (`ASRStep` fails loudly with reason `asr_output_magnitude_blowup` when `max|post| > 1e3·max|pre|`; calibration-covariance condition number recorded; blow-ups can no longer be misattributed to block rejection) **+ e2e fidelity assertions** (`repaired_fraction < 0.85`, measured 0.762; ERP-retention median evoked↔template correlation ≥ 0.5, measured 0.88 on the anterior-injection fixture) **+ fix 7**: the ASR-excluded interpolated channels are **re-interpolated from the ASR-cleaned data** (they previously kept pre-ASR spline values; 1.02/1.09 relative-norm divergence) → `ASRStep` 2.0→2.1, wrapper 2.1→2.2, **golden hash re-pinned `ca91b7f2…` → `1ef8780d…`**; **(2) CI can go green**: the two P4 `mne_connectivity` tests deselected in the workflow + a step that fails on any fsaverage skip; **(3) blink gate timeline guard** (`timeline_mismatch` when sfreq/n_times/first_samp differ — a resampled post used to get a bogus pass/fail verdict); **(4) blink ratio bias**: pre and post ptp now measured on identically 1–15 Hz band-passed copies (100 µV 60 Hz in the pre recording was a false pass); **(5) crosstalk direction** corrected everywhere (rows are CTFs: `C[i, j]` is leakage **from ROI j into ROI i**; row = target, column = source); **(6) crosstalk artifact**: the saved primary `crosstalk` is now the **flip-aligned** numpy estimator (matches `mean_flip`), the library matrix is kept as `crosstalk_unflipped`, QC/provenance state which matrix the statistics use; **(8) e2e honesty** (contaminated-variant event check no longer compares a file with itself; `assert interpolated`); **(9) smaller gaps** (gate prints the scan count; line-noise failures are NaN + note instead of 0.0; `run_processor.load_processed_runs` records an unreadable FIF instead of defaulting to 128 channels and reads `bad_channel_detection`; crosstalk `{"computed": false}` carries a reason; role-exclusion accounting uses the before-EEG set) | **rev2 applied 2026-10-16; awaiting re-audit. ⛔ NEW OPEN USER DECISION (blocks B8): ASR high-pass design (plan §12.6). Hash later re-pinned `1ef8780d…` → `b47de382…` by the F1–F7 fix round (see the B8 evidence)** |
| 2026-10-16 | P1 | **ASR scoped audit → fixes R1–R6** | `revise` (scoped): the §12.6 drift filter is correct but **did not fix the real-data blow-ups** — full chain on pilot5, **4 of 6 recordings that reach ASR failed with `asr_output_magnitude_blowup` under the EEGLAB-standard path**. Root cause (reproduced): eegprep 0.4.0 calls `np.linalg.pinv` with numpy's default `rcond=1e-15` (`asr_process.py:214` via `finite_pinv(Q)`), MATLAB uses `max(size(A))·eps(norm(A))` (absolute). **R1** (critical): `clean_asr` runs under a temporary, source-verified patch of that call site with the MATLAB tolerance; `pinv_tolerance: 'matlab_max_size_eps(norm)'` recorded in metadata + both `get_config`s; changed call site → `eegprep_pinv_assumption_broken`; deterministic reproducer both directions; **pilot5: 6/6 reach-ASR recordings clean (ratios 0.28–1.27; before 4/6 at 4.3e12–6.9e13)**; upstream bug-report **draft** (not filed). **R2** overclaims corrected (justification = EEGLAB-standard input + lower burden, 0.762→0.481, not blow-up prevention). **R3** resume check compares `config_hash` (mismatch → reprocess; `run_processor` has no skip path). **R4** `info['highpass']` = 0.75 Hz when the drift filter runs (attenuation N1 0.995/P2 0.978/P3 0.889/LPP 0.573, 0.8-s ramp 0.49 @ +196 ms; add-back recorded as follow-up). **R5** e2e burden bound labelled a regression tripwire (docstring). **R6** `HbnLangerParams` bounds ([1,200] cutoff; `0<lo<hi<=5` drift; Nyquist at runtime). Hash re-pinned `b07bfe39…` → `a216dc00…` | **implemented 2026-10-16; awaiting re-audit** (see the §ASR R1–R6 evidence section) |

## Remediation batches — `plan/261013-2240-p1p2-remediation.md`

| Batch | Scope | Status |
|---|---|---|
| **B1** | Executor fail-fast + exclusion + dependency preflight + one preprocessing schema + status persistence (P1-F1, F4, F6, F9 blanket-except only, F17 ImportError only) | `[x]` **verified 2026-10-14** — Opus rev2 approval, no conditions left (residuals → B3/B6) |
| B2 | Source-ERP core: anatomical labels, unique names, geometry, signed output + dipole ground truth | `[x]` **verified 2026-10-14** — Opus conditions met (F2/N1/N2/N3 + CF-1 fixed, K3-confirmed) |
| B3 | Timeline integrity: ASR without cutting, sidecar instead of `info['temp']`, ASR wrapper metadata | `[x]` **verified 2026-10-15** — final audit by Qwen3.8-Max (user-approved Opus 5.5 stand-in): `approve`; rev2 confirmed (`confirm-with-notes`) |
| B4 | Source usability + config wiring + P1 bad-channel detector | `[x]` **verified 2026-10-15 (code scope)** — Opus final audit `revise` → rev1 applied (log-scale detector + config bounds + condition guards + test honesty) → scoped re-audit `approve`; B4-D2 **resolved 2026-10-15**: blink-aware detector design at B6 tier-2 (decisions log / plan §12.5) |
| B5 | Source hardening + provenance (covariance, trial inversion, eLORETA, BEM, montage) | `[x]` **verified 2026-10-16** — Opus final audit `revise` → rev1 applied (N1/N2/A1/A2/A3/A6/A7/N6) → scoped re-audit `approve`, no conditions left (residuals → B6/B7/B8) |
| B6 | ⛔ Checkpoint A (RPCA) + instrumentation + golden resolved-config hash + P1 tier-2 debt | `[x]` **verified 2026-10-16** — Opus final audit `revise` → rev2 (NEW-1…NEW-10) + scoped-re-audit ASR-fix applied; desk-check + suites (2F connectivity-only / 729P) per the re-audit; hash `72c394e7…` |
| B7 | QC truthfulness, test-suite honesty, CI job | `[x]` **verified 2026-10-16** — Opus final audit `revise` → rev2 (B7-N2 ASR guard + majors) + scoped re-audit: all conditions met (hash `1ef8780d…`; suite 2F connectivity-only / 853P / 0 skipped; chain e2e 18; fsaverage 43/0; gate 108 files / 0 offenders) — **hash later re-pinned to `b47de382…` by the final-verification-audit fix round (F1–F7), see the B8 evidence** |
| B8 | End-to-end verification prep; ⛔ Checkpoint B (real EGI-129 subject) | `[~]` **task 1 verified 2026-10-16 (code scope)** — final-verification audit `revise` → fixes F1–F7 applied + scoped re-check `confirm-with-notes`; **P1 C1–C6 and P2 conditions 1–4 verified for code scope**; Checkpoint B (real data) gates: second-dataset path **provided** (`~/proj/25SusAttn`, established 2026-10-16); **green remote CI run DONE** (run `38892244381`, success); ASR §12.6 resolved (EEGLAB-standard path + R1 pinv fix, audited/approved 2026-10-16). Remaining: choose 25SusAttn tasks + run the real-data checklists |

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
      (NEW-9), RPCA write-back only on convergence (NEW-10) — scoped-re-audit ASR-fix +
      hash re-pin `72c394e7…` — **B6 verified 2026-10-16** (desk-check + suites 2F connectivity-only /
      729P). **B7 verified 2026-10-16** (final-audit rev2 + scoped re-audit all met; hash `1ef8780d…`).
      Next: **B8** — task 1 (synthetic dipole source-ERP e2e + the Opus B7 amplitude-retention carry)
      **implemented 2026-10-16, awaiting audit**; **final-verification-audit fix round F1–F7 applied
      2026-10-16** (golden hash re-pinned `1ef8780d…` → `b47de382…`; green remote CI run requires
      pushing — user-gated open item under B8/Checkpoint B); **ASR high-pass decision (plan §12.6)
      resolved 2026-10-16 — EEGLAB-standard path implemented, golden hash re-pinned
      `b47de382…` → `b07bfe39…`** (see the §12.6 evidence section); ⛔ Checkpoint B (real data)
      blocked by the second dataset path **and the push-gated CI run**.
      B6/B7 residuals to watch: movie-task detector budget + the "other dataset" half of the B4-D2
      validation.
- [ ] **Final-audit model note:** Opus 5.5 was intermittently unavailable for B3; per user decision
      the B3 final audit ran on Qwen3.8-Max. B4/B6/B7 milestone audits: try Opus 5.5 first, fall back
      to Qwen3.8-Max (or Kimi K3) only with a recorded note.
- [ ] **Science decision before P2 condition 4:** `mean_flip` vs `pca_flip` (plan §12 item 4;
      11/30 vs 22/30 dominance under uniform-patch injection, midline polarity near-ties) — does
      not block B3–B8.
- [x] **ASR high-pass design decision (B7-N2, plan §12 item 6) — RESOLVED 2026-10-16:**
      **EEGLAB-standard path implemented** (option (iv)): `clean_drifts(0.25, 0.75)` on the ASR
      input before `clean_asr`, write-back = the high-passed repaired data (**no add-back**),
      `max_dims` configurable, cutoff 20 documented as **conservative**, per-dataset via
      `chain_params`. Evidence, real-data numbers and the new golden hash in the §12.6 evidence
      section below. ⛔ Checkpoint B no longer blocked by this item — **the unblocking fix is R1 (the MATLAB-tolerance pinv patch), not the drift filter**: with R1, **6/6 pilot5 recordings that reach ASR process cleanly** (was 2/6; see the R1–R6 table).
- [ ] Real-subject end-to-end verification (Checkpoint B artifacts → `planning/audits/`)
- [x] **✅ B8/Checkpoint B open item — green remote CI run (F1):** **DONE 2026-10-16** — run `38892244381` completed **success** (13m47s; first run failed on the missing `pyarrow` parquet engine, fixed in the `pipeline` extra). The previous line's description follows:
      `.github/workflows/pipeline-verification.yml` is tracked with the F1–F7 round (it was
      **untracked** through B7 — the B7 commit message claimed it but never added it), and the
      exact workflow commands are green in a local dry run. A **green remote GitHub run requires
      pushing** the branch (or opening a PR) — a user-gated action, not performed by the
      implementer and not performable from this workspace. Until the user pushes, the remote CI
      item stays open (local dry run ≠ remote green run).
- [ ] **B8 checklist — EOG-regression HF guard on real data (B7-A5):** validate
      `MAX_HF_POWER_FRACTION=0.05` (`steps/eog_regression.py::_check_band_limited`) on the real
      Checkpoint B subject. If the guard false-fires (e.g. blink transients pushing the > 50 Hz
      fraction over the limit), measure the HF fraction on **blink-free segments** or raise the
      threshold with the measured distribution as evidence — **never** silently bypass with
      `allow_unfiltered_regressors=True`.

### B7 evidence (2026-10-16, working tree on `remediation/p1p2`, not committed)

Plan §B7 tasks 1–15 + the B6-carried `EOGRegressionStep` guard. Files: `modules/qc/{preprocessed_qc,hbn_recording_qc,source_qc,metrics_io,raw_qc,preprocessing_comparison_qc}.py`,
`modules/source/__init__.py`, `modules/preprocessing/{bad_channels.py,steps/{asr,interpolate,ica,eog_regression}.py}`,
`core/{provenance,paths,dua_paths,config_hash}.py`, `evaluation/{metrics,prediction/workflow}.py`,
`pipeline/{aggregate_reports,epoch_combiner,run_processor}.py`, `cli/{preprocess,features,main,source_reconstruct}.py`,
`modules/{connectivity,features}/…`, `.github/workflows/pipeline-verification.yml`, tests.

- **QC truthfulness (tasks 1–3)**: "Pipeline step status" metric (`bad` on any `failed`/`excluded`,
  `warning` on `degraded`/`not_run`, else `ok`); "Bad Channels" now counts the detector's
  `detected`/`n_detected_precap` (authoritative after `reset_bads=True`) with a separate
  "Channel-role exclusions" metric (neck/face + EEG→EOG retyped, subtracted from `n_dropped`);
  FAILED/EXCLUDED runs write `*_qc_metrics.json` (`status`/`reason` + exclusion metrics/errors) and
  the success path stores the `recording_qc` row; `hbn_recording_qc` reads the emitted block keys
  (`rejected_fraction`/`n_blocks_rejected`), the detector's bad-channel count, and replaces the
  vacuous in-sample `residual_ocular_maxcorr` gate with `blink_residual_ratio` (blinks on `raw_pre`
  via the first EGI EOG name or a frontal proxy; blink-locked frontal scalp ptp post/pre;
  `None`/reason when `raw_pre` is missing — never a false pass); the old correlation survives as
  `residual_ocular_maxcorr_insample` only.
- **Gate + test honesty (tasks 4–5)**: `test_no_silent_continue.py` root fixed (`parents[2]`), scans
  **88** files (asserted per `CORE_PATH` + > 50 total), covers `modules/source` and `modules/qc`,
  excludes `_legacy` and asserts no live module imports it; the AST rule now also flags
  `print(...)`-only handlers and **all 18 offenders were fixed with no allow-list**
  (`unlink(missing_ok=True)`, narrowed ICA auto-detection with recorded `*_error`, logged networkx/
  DTF/AUC/option-discovery/aggregate/stdout/QC failures). `np.trapz` → `np.trapezoid` in
  `erp_features.py`; the two vacuous B3-era QC tests rewritten to the new contract.
- **e2e + CI (tasks 6–7)**: new `tests/unit/test_hbn_chain_e2e.py` (`chain_e2e`/`slow`, **13 passed**;
  `pytest -m chain_e2e` → 13 passed) — synthetic GSN-129 chain run asserting identical `stim` event
  samples, preserved annotations, FIF reload with `meas_date`, all-`ok`/`noop` `step_status`, the
  pinned `config_hash`, `var(Cz) > 0`, and the 30 %-contaminated variant excluded with an
  `*_EXCLUDED.json` and no FIF; `.github/workflows/pipeline-verification.yml` (py3.13,
  `.[pipeline]` + pytest, an explicit backend-import step, cached `~/mne_data` +
  `fetch_fsaverage()`, the §0.1 command, `pytest -m chain_e2e`, the gate with `-rA`); conditional
  `conftest.collect_ignore` for the two specparam files so `pytest -m chain_e2e` can collect.
- **Crosstalk + source QC (tasks 8–10)**: the invalid delta-vs-reconstructed crosstalk code is
  deleted; `compute_crosstalk_matrix` builds the QC-only fixed-orientation pair (probed deviations:
  the inverse needs a **free-orientation surface** forward for fixed+depth, and `get_cross_talk`
  rejects `BiHemiLabel` → per-label integer vertex lists, verified numerically identical),
  `compute_crosstalk_matrix_numpy` mirrors it in numpy; the module caches per
  (forward, **covariance digest**, method, lambda2), writes `{subject}_crosstalk.npz` and passes
  `labels`/`crosstalk_matrix`/`roi_min_vertices` into QC; QC marks `bad` on
  `roi_vertex_counts < roi_min_vertices`, non-finite ROI values or a missing `{cond}_times` (the
  fabricated `arange/500.0` axis is gone); `test_source.py`'s two skips are real fsaverage tests and
  the crosstalk test asserts the target ROI's strongest off-diagonal entry is finite and < 1
  (measured 0.597, `DorsalAttention.IPS_R`), permutation of rows/columns, and the numpy cross-check
  (exact match) vs the `label_sign_flip`-aligned variant (**does not** match: 34 % flipped vertices
  on average, max |ΔC| 2.08 — critic N-3 quantified; the saved matrix keeps the library definition,
  pinned by a test, reported rather than silently "fixed").
- **CLI/provenance/legacy (tasks 11–15)**: per-feature-type argparse validation (exit 2 + usage for a
  non-`source_erp` type without `--input-dir` and for `source_erp` without `--input-dir` or all of
  `--preprocessing/--task/--source`); hidden `.{subject}_*`/`.{subject}-*` atomic temps + the
  crosstalk npz added to the stale patterns (single-writer assumption documented); condition names
  with `/` or `\` rejected before any write; `source-reconstruct` exits 1 on a missing/unreadable
  requested task config and on any failed session; the legacy
  `detect_and_interpolate_bad_channels` handlers re-raise instead of dropping channels (the
  zero-variance drop is logged).
- **Carried B6 item**: `EOGRegressionStep` v5.0 refuses unfiltered regressors — `info['lowpass']`
  must be finite and ≤ 50 Hz **and** the EOG power fraction above 50 Hz (Welch, capped at
  45 % Nyquist) must be ≤ 5 %, both recorded (`eog_hf_power_fraction`, `eog_lowpass_hz`,
  `eog_band_limited_check`), with the explicit recorded `allow_unfiltered_regressors` opt-out; the
  guard constants are in `get_config()`; **golden hash re-pinned to `ca91b7f2…`** (was `72c394e7…`;
  superseded by the B7 rev2 re-pin to `1ef8780d…`, itself re-pinned to `b47de382…` by the
  F1–F7 round and to `b07bfe39…` by the §12.6 ASR round — see the B7 rev2 / B8 / §12.6 evidence
  sections).
- **⛔ NEW CRITICAL FINDING B7-N1 (found by the task-6 fixture; fixed minimally, needs approval)**:
  spherical-spline interpolation makes an interpolated channel an **exact** linear combination of the
  others, so the ASR calibration covariance is exactly singular (probed on the EGI-129 fixture:
  condition number 35 with 0 interpolated channels → **1.2e15** with 1, one exact zero eigenvalue);
  `eegprep.clean_asr` then takes `sqrtm` of that matrix and returns **~1e15 µV** instead of repaired
  data, every 2-s block exceeds the 100 µV gate and **any recording with ≥ 1 interpolated channel was
  excluded** (the detector flags 1–11 channels on real HBN recordings → a real-data blocker for
  Checkpoint B; B3's ASR tests mark channels bad instead of interpolating, which is why it was never
  seen). Fix applied: `InterpolateBadChannelsStep` records `interpolated_channels` and `ASRStep`
  excludes them from the ASR input (they are synthetic and stay in the recording; only the ASR
  covariance must not be degenerate), reported as `interpolated_excluded_from_asr`; verified 0 zero
  eigenvalues, condition number 34.9, sane output, and the e2e chain now succeeds with 2 interpolated
  channels. Regression tests in `test_asr_timeline.py`. **Not applied** (needs approval): moving
  interpolation after ASR (EEGLAB ordering) — it changes the chain composition and the golden hash.
  Probe scripts: `/tmp/opencode/b7_e2e_probe{1..8}.py`, `/tmp/opencode/b7_probe_crosstalk2.py`.
- **Validation**: full standard suite → **2 failed / 811 passed / 2 xfailed** (exactly the two
  `mne_connectivity` P4 baseline failures; **0 skipped** — both `test_source.py` skips are real tests
  now); focused: e2e 13, gate 16 (88 files, 0 offenders), `test_source.py` 55, `test_asr_timeline.py`
  23, `test_qc.py` 49, `test_preprocessed_qc.py` 31, `test_source_hardening.py` 33,
  `test_preprocessing_steps_tier2.py` 35, new `test_cli_arg_validation.py` 12, new
  `test_source_qc_metrics.py` 6, `test_preprocess_status_persistence.py` 18, `test_config_hash.py` 17.
  Greps: `info['temp']` 0, `np.trapz(` 0, `set_annotations(annots)` 0,
  `mne.make_inverse_operator|mne.apply_inverse` 0.
- **Re-audit scope**: QC modules (`preprocessed_qc`, `hbn_recording_qc`, `source_qc` + the new
  crosstalk functions), `cli/preprocess.py` (QC records + detector metadata), `cli/features.py` +
  `cli/main.py` (argparse validation), `cli/source_reconstruct.py` (exit codes), `modules/source`
  (crosstalk wiring/cache/npz + condition-name guard), the gate + all fixed offenders
  (**including the rev1-widened `cli`/`workflow`/`data` scope**), `steps/{asr,interpolate,eog_regression}.py`
  (B7-N1 + the regressor guard), `core/provenance.py` (stale temps), the new/updated tests, the CI
  workflow.

### B7 rev1 — K3 pre-audit fixes (2026-10-16, same uncommitted tree)

Kimi K3 pre-audit findings B7-A1…A5; B7 stays `[~]` (awaiting the Opus final audit).

- **B7-A1 (major) — gate scope + the 9 offenders.** `tests/unit/test_no_silent_continue.py` now
  scans `eegcpm/cli`, `eegcpm/workflow`, `eegcpm/data` as well (`_legacy` exclusion and the
  import-guard test unchanged): **108 files scanned, 0 offenders, no allow-list** (was 88 files).
  Fixes: `cli/connectivity.py:108` records unreadable / non-mapping task configs (log + "Task config
  problems" summary — the loop reports every file it dropped) and the load handler is narrowed to
  `(OSError, yaml.YAMLError, ValidationError, TypeError)` + logged; `cli/epochs.py:416` QC-report
  failures are narrowed to `(OSError, ValueError, RuntimeError, KeyError, TypeError)`, logged and
  listed in the run summary, `cli/epochs.py:546` `find_events` failures narrowed to
  `(RuntimeError, ValueError, TypeError)` + logged; `workflow/state.py:258,263,270` replaced by
  `_add_column_if_missing()` — only a message-checked duplicate-column `OperationalError` is
  tolerated (debug log), anything else re-raises; `data/hbn.py:170,177` non-numeric phenotype values
  are logged (debug + warning summary); `data/conn_rois.py:185` logs the missing-nibabel /
  missing-template no-op and tolerates only `ImportError`.
- **B7-A2 (minor).** `preprocessed_qc.py`: with no `step_status` (legacy dialect / standalone QC) the
  "Pipeline step status" metric is `warning` / unit `no chain metadata` (value 0) — the old
  `ok` / `(all steps ok)` was a false clean bill. Empty `step_status` is a warning too.
- **B7-A3 (minor).** Crosstalk pair count is now directed: `n_directed_above_0.3` (provenance key
  renamed from `n_pairs_above_0.3`; QC metadata key renamed from `crosstalk_high_pairs`) counts
  `(C > 0.3) & ~eye`, i.e. reciprocal pairs count twice, matching the directional
  `C[i, j] = leakage of i into j` definition (documented in the provenance `definition` string, the
  QC figure statistics, the HTML caption/summary and the `scripts/test_new_qc_figures.py` consumer).
- **B7-A4 (evidence, no code change).** The `chain_e2e` EXCLUDED variant places the 30 % block
  contamination **inside the ASR calibration window by design** — ASR repairs contamination outside
  its calibration window (measured 22/75 → 0/75 contaminated blocks), so the block-rejection gate
  only fires when the calibration itself is contaminated; at that amplitude the gate's
  *post-calibration* sensitivity is **untestable** (nothing is left to detect), which is a coverage
  caveat for the final auditor to adjudicate rather than a defect.
- **B7-A5 (no code change).** Added to the B8 checklist: validate the EOG-regression HF guard
  (`MAX_HF_POWER_FRACTION=0.05`) on the real Checkpoint B subject; if it false-fires, measure the HF
  fraction on blink-free segments or raise the threshold — never silently bypass.
- **Validation (rev1)**: full standard suite → **2 failed / 833 passed / 2 xfailed / 0 skipped**
  (exactly the two `mne_connectivity` P4 baseline failures); `pytest -m chain_e2e` → 13 passed; gate
  → 19 passed / 108 files / 0 offenders; focused B7 set → 113 passed. New/extended tests:
  `test_no_silent_continue.py` 19, `test_preprocessed_qc.py` 33, `test_source_qc_metrics.py` 8,
  `test_cli_arg_validation.py` 14, `test_conn_rois.py` 11, new `test_hbn_data_utils.py` 8,
  `test_preprocess_status_persistence.py` 21, plus the `test_source.py` provenance assertion.

### B7 rev2 — Opus final-audit fixes (2026-10-16, same uncommitted tree)

All 9 audit conditions applied. B7 stays `[~]` (awaiting the re-audit). **Golden resolved hash
re-pinned `ca91b7f2…` → `1ef8780d87f228609e8c55c233426e51778b5e2a0a8648ba42318e3433b8d421`** (fix 7:
`ASRStep` 2.0 → 2.1 + `_ASRStepWrapper` 2.1 → 2.2); **that value was later re-pinned to
`b47de382047676631d2d096b8553ba70b79734aaaebde6774ca00a08e83f1a52` by the final-verification-audit
fix round F1–F7 (F2/F3 — see the B8 evidence section).** Files: `steps/asr.py`,
`hbn_langer_chain.py`, `core/config_hash.py`, `modules/qc/{hbn_recording_qc,source_qc,raw_qc,
preprocessed_qc}.py`, `modules/qc/__init__.py`, `modules/source/__init__.py`,
`pipeline/run_processor.py`, `.github/workflows/pipeline-verification.yml`, tests
(`test_asr_timeline`, `test_hbn_chain_e2e`, `test_qc`, `test_source`, `test_source_qc_metrics`,
`test_preprocessed_qc`, `test_preprocess_status_persistence`, `test_no_silent_continue`).

- **1. B7-N2 (critical) — ASR guard + e2e fidelity.**
  - **(a) Guard.** `ASRStep` now fails loudly with reason `asr_output_magnitude_blowup` when the
    repaired output's max |amplitude| exceeds `ASR_OUTPUT_BLOWUP_FACTOR` (1e3) × the input max —
    *before* the write-back, so the caller's recording is never corrupted and the block-rejection
    gate can never report a false "block contamination" exclusion. The fixed-calibration
    covariance condition number + min eigenvalue are recorded in the step metadata
    (`calibration_cov_cond` / `calibration_cov_min_eig`; exactly-singular → separate
    `asr_calibration_covariance_degenerate` failure). Probed on a 32-channel 0.1-Hz fixture with
    three near-duplicate pairs (condition number 2.2e8): eegprep returns **4.6e15 µV** from a
    13 µV input → the guard fires (real-eegprep regression test
    `test_asr_blowup_guard_fires_on_real_eegprep_output`, plus a monkeypatched deterministic
    test). The guard also caught a pre-existing test that asserted the garbage path
    (`test_reference_channel_excluded`'s no-metadata control: flat Cz in the covariance →
    2.06e16 µV) as a success — that test now expects the loud failure.
  - **(b) e2e fidelity.** `test_hbn_chain_e2e.py` asserts `repaired_fraction < 0.85` on the clean
    fixture (**measured 0.762**; audit 0.76; the bound tightens after the §12.6 decision) and an
    **ERP-retention** measure: median per-channel correlation of the stim-locked evoked with the
    injected N1/P2 template **≥ 0.5** (measured **0.88**; non-vacuity control with the reversed
    template far below). The retention fixture injects the ERP on the anterior EEG subset
    (`erp_mode='anterior'`) because a common-mode ERP is cancelled **exactly** by the chain's
    average-reference step (measured r = −0.98 on Cz) — with the default common injection the
    retention measure is 0.07 and says nothing about ASR.
  - **(c) Recorded, not implemented (⛔ blocks B8):** ASR high-pass design — plan **§12 item 6**,
    decisions log row 2026-10-16, B8 checklist item. Evidence: 0.76/0.63 repaired fraction,
    94 %/<0.5 Hz + 52 %/0.5–2 Hz power removed, the blow-up regime (cond 9e4–1.4e6 → 2.6e16–5.7e16
    µV; our near-duplicate probe 2.2e8 → 4.6e15 µV).
- **2. CI can go green (major).** `pipeline-verification.yml`: the two P4 tests are deselected in
  the §0.1 step (`--deselect tests/unit/test_connectivity.py::test_wpli_close_to_mne_connectivity
  --deselect …::test_dwpli_close_to_mne_connectivity`; `mne_connectivity` is not in
  `.[pipeline]`, so they fail on import — documented P4 baseline), and a new step runs
  `pytest -m fsaverage` and **fails on any skip** (grep on the summary) with ≥1 pass required.
  Local dry run of the exact CI commands (final code): §0.1 (with the two deselects) →
  **853 passed / 2 deselected / 2 xfailed / 0 failed / 0 skipped**; `-m fsaverage` →
  **43 passed / 0 skipped** (cached fsaverage); `-m chain_e2e` → **18 passed**; gate with
  `-rA -s` → **19 passed**, prints `no-silent-except gate: scanned 108 files across 11 core
  paths, 0 offenders`. An actual green GitHub run requires pushing the branch (not done here).
- **3. Blink gate timeline guard (major).** `blink_residual_ratio` returns `ratio=None`,
  `pass=None`, `reason='timeline_mismatch'` (plus the pre/post `timeline` record) unless
  `sfreq`, `n_times` **and** `first_samp` match — blink indices from `raw_pre` are meaningless on
  a resampled/trimmed `raw_post` (probed: a 250→200 Hz resample produced a bogus ratio 0.50 with
  a pass/fail verdict; the audit measured 0.149/pass=True).
- **4. Blink ratio bias (major).** The pre and post peak-to-peak amplitudes are now measured on
  **identically 1–15 Hz band-passed copies** (`BLINK_BANDPASS_HZ`), so line noise / drift in the
  pre recording can no longer bias the ratio toward pass. Test fixture: 100 µV 60 Hz in the pre
  data with the blinks fully retained → pre-fix ratio 0.20 (false pass), post-fix ratio 1.0
  (correct fail); the unfiltered false pass is asserted as the non-vacuity control.
- **5. Crosstalk direction (major).** Verified against mne 1.13.2 source ("the columns of the
  resolution matrix are the PSFs and the rows are the CTFs"): `C[i, j]` is the leakage **from ROI
  j (source, column) into ROI i (target, row)** — row = target, column = source. Corrected in
  `source_qc.py` (`crosstalk_from_resmat` docstring, QC metadata comment, figure docstring, axis
  labels (x = Source ROI, y = Target ROI), statistics block, HTML caption) and the provenance/npz
  definition strings in `modules/source/__init__.py`; the tests' comments follow.
- **6. Crosstalk artifact (major).** The saved primary `crosstalk` matrix is now the
  **flip-aligned** estimator (`compute_crosstalk_matrix_numpy(..., align_signs=True)` — matches
  the pipeline's `mean_flip` ROI extraction; rank agreement 0.76 vs 0.61 for the library signed
  mean); the library `get_cross_talk(mode='mean')` matrix is kept as `crosstalk_unflipped` and
  the exact numpy-vs-library cross-check still pins the unflipped path. `_get_crosstalk_matrices`
  caches both; the npz gains `crosstalk_unflipped` + `primary` + a definition stating the
  convention and which matrix is which; provenance records the primary summary
  (`primary_matrix`, `n_directed_above_0.3` — **computed on the primary flip-aligned matrix**),
  the unflipped summary and `max_abs_delta_primary_vs_unflipped`; QC gets
  `crosstalk_matrix_kind` and states it in `result.metadata`, the figure statistics and a note.
  `{"computed": false}` now carries a `reason` (config-disabled or the exception).
- **7. Interpolated channels after ASR (major, B7-N1 residual).** After the eegprep repair,
  `ASRStep` re-interpolates the excluded interpolated channels from the ASR-cleaned data with the
  same mode (temporarily marking them bad; `bads` restored in a `finally`), records
  `reinterpolated_after_asr` / `reinterpolation_mode` / `reinterpolation_relative_norm`. The
  e2e now asserts the saved interpolated channel equals a fresh re-interpolation of the chain
  output (rel-norm < 0.05; **1.02/1.09 before the fix**, matching the audit's ~98 %). Version
  bumps + golden hash re-pin as above.
- **8. e2e assertion honesty (minor).** The contaminated fixture captures `events_in` **before**
  the chain and the test compares against that capture (the old assertion compared `raw_in` with
  itself); `test_asr_excludes_interpolated_channels` now `assert interpolated` (≥1) and checks
  the re-interpolation metadata.
- **9. Smaller gaps.** The no-silent-except gate prints the scan count
  (`no-silent-except gate: scanned 108 files …`) and embeds it in the failure message;
  `raw_qc._compute_line_noise_ratio` / `QualityControlModule._compute_line_noise` return **NaN
  with a note/log** on failure instead of a false 0.0 (the metric is `warning`, never `ok`);
  `run_processor.load_processed_runs` records an unreadable preprocessed FIF (logged reason,
  `quality_status='unknown'`, `recommended_action='review'`, warning in the metrics) instead of a
  silent `n_original=128` assumption, and `_extract_quality_metrics` reads the canonical
  `bad_channel_detection` key first (legacy `bad_channels` still honoured); crosstalk
  `{"computed": false}` gains a `reason` (see fix 6); `preprocessed_qc` subtracts only role
  channels that were **EEG in the before set** from `n_dropped` (a native EOG electrode no longer
  hides a genuine EEG drop).
- **Validation (rev2)**: full standard suite → **2 failed / 853 passed / 2 xfailed / 0 skipped**
  (failures exactly the two `mne_connectivity` P4 baseline tests); `pytest -m chain_e2e` →
  **18 passed**; `pytest -m fsaverage` → 43 passed / 0 skipped; gate → 19 passed / 108 files /
  0 offenders (scan count printed). Per-file: `test_asr_timeline` 28, `test_hbn_chain_e2e` 18,
  `test_qc` 54, `test_source` 60, `test_source_qc_metrics` 8, `test_preprocessed_qc` 34,
  `test_preprocess_status_persistence` 24, `test_no_silent_continue` 19. Local CI dry run (the
  exact workflow commands, final code, with the P4 deselects) → **853 passed / 2 deselected /
  2 xfailed / 0 failed / 0 skipped**. Probes: `/tmp/opencode/b7rev2_probe{1..8}.py`.
- **Re-audit scope (rev2)**: `steps/asr.py` (guard, re-interpolation, metadata, version),
  `hbn_langer_chain.py` (wrapper version), `core/config_hash.py` (re-pin),
  `modules/qc/hbn_recording_qc.py` (timeline guard, band-pass), `modules/qc/source_qc.py` +
  `modules/source/__init__.py` (direction, primary/unflipped matrices, provenance/npz/QC wiring),
  `modules/qc/{raw_qc.py,__init__.py}` + `pipeline/run_processor.py` + `preprocessed_qc.py`
  (smaller gaps), the updated tests, and the CI workflow (deselects + fsaverage-skip gate).

### B8 evidence (task 1) — synthetic dipole source-ERP e2e + the B7 amplitude carry (2026-10-16)

Plan §B8 task 1. **Checkpoint B** (real EGI-129 data, plan §B8 task 3) remains ⛔ blocked by the ASR
high-pass decision (plan §12.6) and the second-dataset path — not attempted.

- **New test** `tests/unit/test_e2e_synthetic_source_erp.py` (`fsaverage` + `chain_e2e` + `slow`,
  **16 tests**): the B7 synthetic GSN-HydroCel-129 recording (`make_gsn129_recording`) carrying an ERP
  generated from the **B2 dipole** — the plan waveform (N1 −1.0 at 0.100 s, P2 +0.6 at 0.200 s) on
  every `SensoriMotor.Superior` label vertex × `mne.label_sign_flip`, projected with
  `mne.apply_forward` (10 µV sensor peak) and injected at every 2-s `stim` onset — through the full
  `hbn_langer` chain, plain-MNE epoching (`mne.Epochs(raw, events, tmin=-0.2, tmax=0.6,
  baseline=(None, 0), reject_by_annotation=True)`; the P3 module is deliberately not used),
  `SourceReconstructionModule` (evoked dSPM, oct6 fsaverage) and `SourceERPFeatureModule` (archive
  read from disk → provenance sidecar check). One module-scoped fixture runs chain/source/features
  once (**27 s** for the whole file).
- **Measured**: chain all steps `ok` (`asr.repaired_fraction` 0.719; detector flags/interpolates
  `E104`), `stim` samples unchanged, 74/74 epochs; target ROI N1 **−8.28 at 0.100 s** (within one
  sample), `corr(tc, w) = 0.964` (shifted-template control far below), signed output, cerebellar
  exclusion, ≥3 vertices/ROI; feature CSV columns
  `1_SensoriMotor.Superior_{N1,P2}_peak_{latency,amplitude}` + `1_nave == 74` with values equal to
  the npz; condition name pinned `'1'` (plain MNE infers `event_id={'1': 1}`; the P3 module would
  have used `stim`).
- **Amplitude retention (Opus B7 carry)**: per-channel projection gain `<evoked_c, w>/<w, w>`
  normalised by the expected average-reference factor `arf_c = p_c − mean(p)` (scalar
  `1 − mean(p)²/mean(p²) = 0.9952`; the balanced midline dipole pattern makes the correction small
  here — 0.970 un-normalised → 0.966 normalised — vs 0.541 in the B7 binary-mask fixture). Measured
  **retention 0.966** (aggregate 0.994), band **[0.7, 1.3]**; chain-input control **0.987**
  (injection intact → the 3.4 % loss is the chain's) and shifted-template control < 0.2 × retention.
- **Fixture sharing**: new `gsn129_dipole_setup` conftest fixture builds the oct6 forward **through**
  `SourceReconstructionModule._get_template_forward` on the 109-channel EEG set the chain leaves
  behind (class-level forward cache reused — the test asserts the same forward object; one ~10 s
  build per session); `make_gsn129_recording` gained the additive, backwards-compatible
  `erp_sensor_data` parameter. Two probe bugs caught before landing (pre-stim padding shifting the
  injected ERP by 0.2 s; a mis-scaled `apply_forward` producing a 1e4 V injection that the ASR
  covariance guard correctly rejected) — both guarded by tests now (input-gain control + the
  `apply_forward` peak assertion).
- **Validation**: new file **16 passed** (28 s); `pytest -m chain_e2e` → **34 passed** (18 B7 + 16
  B8); full standard suite → **2 failed / 870 passed / 2 xfailed / 0 skipped** (failures exactly the
  two `mne_connectivity` P4 baseline tests). Probes: `/tmp/opencode/b8_probe{1,2}.py`.

### Final-verification-audit fixes F1–F7 (2026-10-16, same uncommitted tree)

Fix round applied after the final deep audit of the P1/P2 acceptance conditions (briefing:
`planning/audits/2026-10-16-final-verification-briefing.md`). **Golden resolved hash re-pinned
`1ef8780d…` → `b47de382047676631d2d096b8553ba70b79734aaaebde6774ca00a08e83f1a52`** (F2/F3).

- **F1 (major) — CI file never committed.** `.github/workflows/pipeline-verification.yml` was
  **untracked** (the B7 commit message claimed it, but it was never added); the file is present and
  complete and is added to version control with this round (the orchestrator commits). **A green
  remote CI run requires pushing** — recorded as a user-gated open item under B8/Checkpoint B
  (see the checklist above).
- **F2 (major) — block-rejection threshold not hashed.** The 100 µV median-ptp threshold was a
  local in `_BlockRejectionStepWrapper.process`. It is now the module constant
  `BLOCK_REJECTION_PTP_THRESHOLD` in `hbn_langer_chain.py`, read at call time by `process` and
  exposed via `get_config()` (`ptp_threshold_v`; wrapper version 1.1); recorded in the step
  metadata too. Sensitivity test: changing the constant changes `resolved_config_hash`.
- **F3 (major) — ASR window tolerances + blow-up factor not hashed.**
  `ASR_OUTPUT_BLOWUP_FACTOR` and the canonical `window_tolerances=(-inf, 7.0)` (the
  `clean_windows` z-thresholds that decide the `BAD_asr_window` marks) were missing from
  `_ASRStepWrapper.get_config()`. `ASR_WINDOW_TOLERANCES` is now a named constant in `steps/asr.py`
  (also the `ASRStep` default), the wrapper passes it explicitly and reports both it and the
  blow-up factor (`output_blowup_factor`) in `get_config()` — read at call time (wrapper version
  2.3). Sensitivity tests for both, plus a test that the wrapper really passes the constant through
  to `ASRStep`.
- **F4 (minor) — `forward.template` silently ignored.** `ForwardConfig.template` is now
  `Literal["fsaverage"]` (`core/config.py`); any other template is rejected at validation instead
  of being silently ignored. `spacing` deliberately left unrestricted (oct4 is exercised by an
  existing test; restriction is not trivial). Rejection tests in `test_source_config.py`.
- **F5 (minor) — briefing count.** `planning/audits/2026-10-16-final-verification-briefing.md`
  said the gate gives "20 passed"; the actual result is **19 passed** (108 files / 0 offenders) —
  corrected (and the stale "guard factor not in `get_config`" residual removed).
- **F6 (minor) — QC copy replaced annotations.** `preprocessing_comparison_qc.py` called
  `set_annotations(annot)` on the display copy, wiping its stim/event annotations. It now calls
  `append_eegprep_removed_mark` → the shared append-only `append_bad_annotations` helper; still
  display-only. New tests: `tests/unit/test_preprocessing_comparison_qc.py` (3).
- **F7 (minor) — B8 target dominance.** `test_e2e_synthetic_source_erp.py` asserts the target ROI
  carries the **largest N1 of the whole parcellation** (measured −8.277 vs −6.636 next,
  `DefaultMode.PCC`; ≥ 1.1× bound), and the polarity test docstring now states it verifies the
  `mean_flip` **sign convention**, not anatomy.
- **Validation**: full standard suite → **2 failed / 876 passed / 2 xfailed / 0 skipped** (failures
  exactly the two `mne_connectivity` P4 baseline tests); `pytest -m chain_e2e` → **34 passed**
  (18 B7 + 16 B8); `pytest -m fsaverage` → **59 passed / 0 skipped**; gate →
  **19 passed** / 108 files / 0 offenders. Per-file: `test_config_hash` 19 (+2 sensitivity/config
  tests), `test_source_config` 22 (+1 F4), `test_e2e_synthetic_source_erp` 16 (F7 folded into the
  existing N1 polarity/latency test — no new test method, counts unchanged),
  `test_preprocessing_comparison_qc` 3 (new). F7 measured top-3 N1 (fixture npz, N1 window):
  **−8.277** `SensoriMotor.Superior`, −6.636 `DefaultMode.PCC`, −2.943 `FrontoParietal.PPC_L`
  (target/runner-up ratio 1.247 ≥ the 1.1 bound).

### §12.6 ASR EEGLAB-standard path + per-dataset configurability (2026-10-16, same uncommitted tree)

Plan §12 item 6 (B7-N2 follow-up), resolved as option (iv) — the **EEGLAB-standard path** —
and implemented; awaiting audit. **Golden resolved hash re-pinned
`b47de382…` → `b07bfe39930a4525ef48d0d2ca053df2144bbd396e95388adcaf963edf82fb8f`.**

- **What changed (`steps/asr.py`, `ASRStep` 2.1 → 2.2).** The ASR **input** is high-passed with
  `eegprep.clean_drifts(eeg, drift_transition_hz)` (default `(0.25, 0.75)`, EEGLAB's own
  `clean_artifacts` band) after the picks/exclusions and before the calibration reference is
  extracted; `max_dims` (default 0.66) is forwarded to `clean_asr`; the **write-back is the
  high-passed repaired data — no add-back** (exactly EEGLAB's `clean_artifacts`). `None`/`'off'`
  skips the filter with a `RuntimeWarning` and records `drift_filter_applied: False` (the
  non-EEGLAB regime the B7-N2 guard exists for). Metadata gains `drift_transition_hz`,
  `drift_filter_applied`, `drift_filter_modified_fraction`, `max_dims`; `pre_max_uv` /
  `repaired_fraction` are measured against the **post-drift ASR input** (the high-pass effect is
  recorded separately, so `repaired_fraction` keeps meaning "samples ASR rewrote"). The B7-N2
  blow-up guard and the B7-N1 exclusion/re-interpolation are unchanged (re-probed: the
  exactly-singular interpolated-channel covariance is still caught — the EGI-129 fixture raises
  `asr_calibration_covariance_degenerate` with the drift filter on and off). **Correction
  (2026-10-16, ASR scoped audit R2):** the earlier claim that the near-singular 32-channel
  control "is tamed by the high-pass" was wrong for real data — 4 of 6 pilot5 recordings that
  reach ASR still failed under this path; the fix is the MATLAB-equivalent pinv tolerance (see
  the §ASR R1–R6 section below).
- **Cutoff semantics corrected.** 20 = EEGLAB **conservative** (aggressive 3, default 5,
  conservative 20; GUI default 20; Chang et al. 2019 optimal 20–30) — the old "aggressive 20"
  wording is fixed in the `ASRStep`/chain docstrings and the golden YAML.
- **Chain + configurability (`hbn_langer_chain.py`, wrapper 2.3 → 2.4; `core/config.py`;
  `pipeline_builder.py`).** `build_hbn_langer_chain(asr_cutoff=20, asr_drift_transition_hz=(0.25,
  0.75), asr_max_dims=0.66, asr_calibration='fixed_first_60s')` returns a `ResolvedChainSteps`
  (prebuilt steps + `HBN_LANGER_SPEC_DEVIATIONS`, so every construction path resolves to the same
  config/hash). `_ASRStepWrapper` accepts `calibration_window='data_driven'` (EEGLAB's automatic
  clean-window calibration — the built-in per-recording adaptivity; the spec default stays
  fixed-first-60 s) and reports the branch it actually runs in `get_config`
  (`calibration_reference`/`ref_maxbadchannels`/`ref_tolerances` switch with the branch;
  `DATA_DRIVEN_REF_*` read at call time). New pydantic schema `HbnLangerParams` (`extra='forbid'`)
  exposed as `PreprocessingSpec.chain_params`, forwarded by `pipeline_steps()` via
  `build_hbn_langer_chain(**params)`; `chain_params` next to an explicit step list is rejected
  (`model_fields_set`), unknown keys / wrong types / non-increasing transition bands are hard
  errors. Per-dataset YAML example: `chain_params: {asr_cutoff: 30, asr_calibration: data_driven}`.
- **Real-data probe (`/tmp/opencode/p1_chain_asr_realdata_probe.py`; chain steps 1–6 then the
  actual `ASRStep`, `sub-NDARAC853CR6` RestingState, 368 s, 109 EEG channels, 5 interpolated):**
  - new path: **finite**, `max_ratio = 0.312` (target < 10; the `clean_artifacts` variant C of
    `/tmp/opencode/asr_design_probe.py` measured 0.35), `repaired_fraction = 0.048`,
    `drift_filter_modified_fraction = 0.955`, band-power ratios vs the 0.1 Hz input: <0.5 Hz 0.078,
    0.5–2 Hz 0.49, 2–8 Hz 0.83, 8–45 Hz 0.81, window-rejected fraction 0.022;
  - pre-§12.6 configuration (`drift_transition_hz=None`): loud
    `asr_output_magnitude_blowup` at **ratio 8.85e13** (matches the recorded 9e13 evidence).
- **e2e re-measurements (bounds updated):** clean fixture `repaired_fraction` **0.481** (bound
  `< 0.65`; was 0.762/`< 0.85`; separate drift effect 0.355), ERP retention **0.866** on the
  anterior fixture (≥ 0.5 bound; was 0.88), B8 dipole fixture: `repaired_fraction` 0.404, amplitude
  retention **0.949** (band [0.7, 1.3]; was 0.966), aggregate 0.951, chain-input control 0.987.
- **Tests added/updated:** `test_asr_timeline.py` (+5: clean_drifts-before-clean_asr identity spy,
  sub-transition band-power removal, `None`/`'off'` skip + warning + metadata, `max_dims` reaches
  `clean_asr` through step and wrapper, blow-up guard re-scoped to the non-high-passed regime with
  the EEGLAB-path non-vacuity control; wrapper metadata/calibration tests extended);
  `test_config_hash.py` (+2: `chain_params` sensitivity across cutoff/drift/max_dims/calibration,
  data-driven calibration truthfulness + `DATA_DRIVEN_REF_*` sensitivity; existing constant/config
  tests extended); `test_preprocessing_schema.py` (+4: unknown key, wrong type, explicit-step-list
  rejection, accepted values reach the wrapper + default chain equality); `test_hbn_chain_e2e.py`
  (burden bound + drift/max_dims provenance assertions); `test_e2e_synthetic_source_erp.py`
  (docstring re-measurements).
- **Validation** (see the final report for the exact commands): full standard suite → failures
  exactly the two `mne_connectivity` P4 baseline tests; `pytest -m chain_e2e` → **34 passed**;
  `pytest -m fsaverage` → **59 passed / 0 skipped**; gate → **19 passed** / 108 files / 0 offenders.

### §ASR R1–R6 — ASR scoped-audit fixes (2026-10-16, same uncommitted tree)

Scoped audit of the §12.6 drift filter: the filter itself is correct, but it does **not** fix the
real-data blow-ups. Full chain on pilot5: **4 of 6 recordings that reach ASR failed with
`asr_output_magnitude_blowup` under the EEGLAB-standard path**. Root cause (auditor, reproduced):
eegprep 0.4.0 calls `np.linalg.pinv` with numpy's default `rcond=1e-15`
(`plugins/clean_rawdata/asr_process.py:214`, via the module-global `finite_pinv(Q)`), while
MATLAB's `pinv` uses the absolute tolerance `max(size(A)) * eps(norm(A))` (≈2.2e-14 for ~100
channels). Task list R1–R6; plan record: §12.6 ("ASR scoped-audit fixes R1–R6"). **Golden
resolved hash re-pinned `b07bfe39930a4525ef48d0d2ca053df2144bbd396e95388adcaf963edf82fb8f` →
`a216dc003f898808c7e52e0c964bf752cacf15322b13f4cc320f8f7934658422`** (R1 `pinv_tolerance` in
both `get_config`s + `ASRStep` 2.2 → 2.3 / `_ASRStepWrapper` 2.4 → 2.5).

- **R1 (critical) — pinv tolerance, scoped + provenance (`steps/asr.py`).** `clean_asr` runs
  under a temporary patch of exactly the eegprep call site: the module global `finite_pinv` in
  `eegprep.plugins.clean_rawdata.asr_process` is replaced with
  `_pinv_with_matlab_tolerance` (SVD, then MATLAB's absolute cutoff `max(size)·eps(norm)` — the
  exact semantics, not numpy's relative `rcond`) and restored in `finally`. Before every patched
  call the assumption is source-verified (the module exposes `finite_pinv` and `asr_process`
  contains `finite_pinv(Q)`); a mismatch raises `eegprep_pinv_assumption_broken` instead of
  silently reverting. Metadata records `pinv_tolerance: 'matlab_max_size_eps(norm)'`,
  `eegprep_version`, `eegprep_pinv_verified_version`; `get_config` (step + wrapper) carries
  `pinv_tolerance` so the golden hash covers it. **Reproducer (both directions):** deterministic
  32-channel fixture (calibration covariance condition number 1.2e4, 10 Hz common sinusoid;
  `test_pinv_tolerance_is_the_blowup_fix`) — with the real eegprep default pinv the step raises
  `asr_output_magnitude_blowup` (and writes nothing back); with the fix the same call is clean
  (ratio 1.0). Helper pinned against MATLAB semantics on a crafted 64×64 matrix
  (`test_matlab_pinv_tolerance_helper_matches_matlab_semantics`: numpy inverts a 3e-15 singular
  value, the MATLAB cutoff zeroes it). Patch scoping + guard pinned by
  `test_pinv_patch_is_scoped_and_guarded`.
- **Pilot5 re-run table** (read-only; chain steps 1–6 then the **real** `ASRStep` with the chain
  defaults — cutoff 20, fixed-first-60 s, drift (0.25, 0.75), max_dims 0.66; RestingState +
  DespicableMe = the 10 recordings; `/tmp/opencode/pilot5_step_probe.py`, 2026-10-16). "Before"
  = the pre-fix tree (`/tmp/opencode/pilot5_pinv_probe.py`; the same 4/6 failure the audit
  reproduced). 4 recordings are excluded by the bad-channel detector before ASR (unchanged by
  R1), 6 reach ASR:

| recording | task | reaches ASR | before (default pinv) | after (R1) | `repaired_fraction` | `window_rejected_fraction` | max ratio | `info['highpass']` |
|---|---|---|---|---|---|---|---|---|
| sub-NDARAC853CR6 | RestingState | yes | ok (0.379) | **ok** | 0.0483 | 0.0219 | 0.3785 | 0.75 |
| sub-NDARAC853CR6 | DespicableMe | yes | **blow-up 4.26e12** | **ok** | 0.2171 | 0.0563 | 0.2828 | 0.75 |
| sub-NDARAC904DMU | RestingState | yes | **blow-up 3.25e13** | **ok** | 0.1958 | 0.3206 | 0.4893 | 0.75 |
| sub-NDARAM873GAC | RestingState | yes | **blow-up 6.87e13** | **ok** | 0.0820 | 0.0116 | 0.5697 | 0.75 |
| sub-NDARCD661GL0 | RestingState | yes | ok (1.267) | **ok** | 0.0805 | 0.5833 | 1.2666 | 0.75 |
| sub-NDARCD661GL0 | DespicableMe | yes | **blow-up 4.23e13** | **ok** | 0.0393 | 0.0311 | 0.7434 | 0.75 |
| sub-NDARAC904DMU | DespicableMe | no | — | — | — | — | — | — |
| sub-NDARAM487XU3 | RestingState | no (corrupt: 94/109 flat) | — | — | — | — | — | — |
| sub-NDARAM487XU3 | DespicableMe | no (corrupt) | — | — | — | — | — | — |
| sub-NDARAM873GAC | DespicableMe | no | — | — | — | — | — | — |

  **ASR success rate: 6/6 recordings that reach ASR (100 %), target met**; before R1 it was 2/6.
  Every patched run reports `pinv_tolerance: matlab_max_size_eps(norm)`, `eegprep_version 0.4.0`
  and a finite output; `info['highpass']` is 0.75 Hz on all of them (R4). Upstream bug-report
  **draft** (title, call site, MATLAB semantics, reproducer, observed vs expected) at
  `planning/audits/eegprep-pinv-bug-report-draft.md` — **not filed** (user-gated).
- **R2 — overclaims corrected.** The drift filter's justification is EEGLAB-standard input +
  lower repair burden (`repaired_fraction` 0.762 → 0.481 on the e2e fixture), **not** blow-up
  prevention. Removed the "near-singular blow-up regime is tamed by the high-pass" claims from
  plan §12.6, `steps/asr.py` docstrings, `hbn_langer_chain.py`, `implementation/CHANGELOG.md`
  and this file's §12.6 evidence.
- **R3 — resume check compares config hashes (`cli/preprocess.py`).** The CLI computes
  `resolved_config_hash(PreprocessingPipeline(spec.pipeline_steps()))` once per batch (identical
  for every run; a construction failure is a global error → exit 1) and skips a
  COMPLETED/EXCLUDED run **only** when the stored `config_hash` equals it. A mismatch prints the
  stored/current hashes and **reprocesses**. `run_processor` has no COMPLETED/EXCLUDED skip path
  (verified: it clears the output dir and always reprocesses; `load_processed_runs` only reads
  results for QC) — its hash semantics are documented there. Tests: matching-hash skip
  (`test_cli_skips_completed_run_by_session_and_run`, updated), stale-hash reprocess
  (`test_cli_config_hash_mismatch_is_not_a_resume_hit`), two-config reprocess
  (`test_cli_reprocesses_when_config_changes`); the `chain_params` hash sensitivity itself is
  pinned by `test_config_hash.py::test_resolved_config_hash_sensitive_to_chain_params`.
- **R4 — truthful `info['highpass']` (`steps/asr.py`).** When the drift filter runs,
  `info['highpass']` is set inside `info._unlock()` to the transition band's passband edge
  (`drift_transition_hz[1]`, 0.75 Hz default) — MNE's own `filter` records the passband edge
  (probed on 1.13.2). Metadata gains `info_highpass_hz` (None when the filter is off, and the
  chain's value is then left untouched). Tests:
  `test_info_highpass_updated_when_drift_filter_runs`,
  `test_info_highpass_untouched_when_drift_filter_off`, and the timeline test now pins 0.75.
  **Measured attenuation (design probe, recorded in plan §12.6):** N1 0.995, P2 0.978, P3 0.889,
  LPP 0.573; an 0.8-s ramp is attenuated to 0.49 with its peak shifted 196 ms — an **add-back
  option for slow-ERP datasets is a recorded follow-up, not implemented.**
- **R5 — bound honesty (`test_hbn_chain_e2e.py`, docstring only).** The
  `repaired_fraction < 0.65` bound is labelled a **regression tripwire**, not a fidelity measure
  (the fixture's 1/f spectrum is too red); the ERP-retention assertion remains the fidelity
  measure. Measured after R1: still 0.481 (drift effect 0.355).
- **R6 — bounds (`core/config.py`, `steps/asr.py`).** `HbnLangerParams.asr_cutoff`:
  `ge=1, le=200`; `asr_drift_transition_hz`: `0 < low < high` **and** `high <= 5` Hz at schema
  level, plus `high < Nyquist` at runtime in the step (`_check_drift_nyquist`, called from
  `validate_inputs` and the eegprep path). Tests:
  `test_preprocessing_schema.py::test_chain_params_r6_bounds_are_hard_errors` (cutoff 0.5/500 and
  drift (0, 0.75)/(1, 0.5)/(0.25, 0.25)/(0.5, 5.5) rejected; (1, 200)/(0.5, 5) accepted),
  `test_asr_timeline.py::test_drift_transition_above_nyquist_refused` (sfreq 4 Hz, band
  (0.5, 2.5) → ValueError from both `validate_inputs` and `process`).
- **Validation commands (2026-10-16, working tree; exact commands in the final report):**
  `pytest tests/unit/test_asr_timeline.py` → **39 passed** (33 → 39: +6 R1/R4/R6 tests, 2 B7-N2
  tests re-scoped to the default-tolerance regime); `pytest tests/unit/test_preprocessing_schema.py
  tests/unit/test_preprocess_status_persistence.py` → **56 passed** (R6 bounds, R3
  skip/reprocess); `pytest tests/unit/test_config_hash.py` → **21 passed** (hash re-pinned,
  `PINV_TOLERANCE_SPEC` sensitivity); `pytest -m chain_e2e` → **18 passed**; `pytest -m
  fsaverage` → **59 passed / 0 skipped**; **full standard suite → 2 failed / 896 passed /
  2 xfailed**, failures exactly the two `mne_connectivity` P4 baseline tests
  (`test_connectivity.py::test_wpli_close_to_mne_connectivity`,
  `::test_dwpli_close_to_mne_connectivity`); `test_no_silent_continue` gate green (19 passed,
  108 files / 0 offenders).


### 2026-10-17 — Track B pilot: 25SusAttn (user's dataset), setup + first runs

**Scope chosen by the user**: their own dataset first (HBN is public), tasks **`cptvis`**
(visual GradCPT, attention) + **`stroopvis`** (spatial Stroop, ERP).

**New assets**

- Task configs `eegcpm/config/tasks/cptvis.yaml` (target/nontarget, occipital N1, N2, P3,
  `key_resp.*`) and `stroopvis.yaml` (congruent/incongruent, N2/N450/P3, `response.*`) —
  both validated by `TaskConfig.from_yaml`.
- Preprocessing config `~/proj/25SusAttn/configs/25susattn_optimized.yaml`: montage (sfp) →
  1 Hz HP → drop_flat → variance bad channels (7 SD) → **interpolate** (missing in the
  packaged optimized.yaml despite its comment) → zapline (adaptive 50 Hz) → ASR (20,
  EEGLAB-standard path + R1 pinv fix) → average reference → ICA (EOG-informed auto-detect) →
  ICLabel (0.5) → 40 Hz LP.
- Montage `~/proj/25SusAttn/25susattn_64.sfp` from the CapTrak `electrodes.tsv`
  (64 EEG channels; `coordsystem.json` has NAS/LPA/RPA for a later coregistration).

**Montage findings (all verified on real data)**

1. The TSV has `n/a` coordinates for VEOG/HEOG/EKG/EMG; one NaN row makes MNE's `.sfp`
   reader scale **every** position to NaN (median-norm divide) → exclude those rows.
2. `set_montage(match_alias=True)` resolves raw `M1/M2/Cb1/Cb2` to the canonical
   `TP9/TP10/POO7/POO8`; the .sfp must therefore store those canonical names.
3. BrainVision records `Trigger` as an EEG channel; the loader now retypes
   trigger/trig/status → `stim` (`eegcpm/data/bids_utils.py`), regression test
   `test_preprocess.py::TestReadRawEeg::test_brainvision_trigger_channel_typed_as_stim`.
4. The dataset **does** have VEOG/HEOG (typed EOG by the loader), so ICA's
   `find_bads_eog` is active — the earlier "no EOG channels" note was about `eeg.json`.

#### 2026-10-17 — Track B real-data findings + fixes (25SusAttn sub-001)

**Unusable auxiliary channels (dataset-wide).** VEOG in sub-001 is dead
(mean −206 089 µV, std 1.4 µV); HEOG/EKG/EMG carry mV-scale DC offsets and
noise (HEOG std 13.8 mV). sub-002/003/010 are similar (VEOG std 1.5–9.2 mV).
The `eeg.json` says `EOGChannelCount: 0`; the loader types them EOG anyway.

**Fixes (all tested):**

1. **QC blink gate truthfulness** — the gate consumed the dead VEOG:
   `find_eog_events` thresholds a flat channel's own noise → 1018 "blinks"
   (92/min) with `blink_residual_pass: true`. Now candidates are ranked by
   1–10 Hz band-passed variance, each candidate's rate must be plausible
   (1–60/min), a dead candidate is retried (frontal proxy included), and a gate
   with no usable candidate returns `pass=None`
   (`reason='no_usable_blink_channel'`). Candidates the chain deliberately
   retyped away are excluded (`retyped_eog_channels`). Tests:
   `test_qc.py::TestBlinkResidualGate::{test_dead_eog_falls_back_to_frontal_proxy,
   test_no_usable_blink_channel_never_passes, test_retyped_eog_channels_are_not_used}`.
2. **Zapline frequency** — adaptive detection picked **87.9 Hz** on stroopvis
   (and reported it as an "additional frequency" on cptvis) and missed the
   50 Hz line. The 25SusAttn config now uses fixed `fline: 50.0`
   (`adaptive: false`): 17 components removed, 1.0 dB line reduction.
3. **QC line-noise metric** — was hardcoded to 60 Hz (HBN/US); now uses
   `info['line_freq']` (set by the loader from BIDS `PowerLineFrequency`) and
   records `line_noise_freq_hz`. Honest sub-001 cptvis residual at 50 Hz:
   **0.151** (the previous 0.033 was an irrelevant 60 Hz bin). Tests:
   `test_qc.py::TestRecordingQCLineFrequency`, `test_preprocess.py`
   (BrainVision sidecar → `info['line_freq'] == 50.0`).
4. **Loader** — trigger/trig/status channels → `stim` (regression test), BIDS
   `PowerLineFrequency` → `info['line_freq']`.
5. **New generic step `set_channel_types`** (`steps/set_channel_types.py`,
   registered; `tests/unit/test_set_channel_types.py`) — the 25SusAttn config
   retypes VEOG/HEOG/EKG/EMG to `misc`, so EOG-based ICA detection
   (`find_bads_eog`/`find_bads_ecg`) and the QC gate never consume them.
6. **Reference** — `projection: true` left the data projector-based (ICLabel
   warned "does not seem to be referenced to a common average reference");
   the config now applies a real CAR (`projection: false`).

**First-run numbers (sub-001, cptvis, final config):** 6 bad channels
(AF4, CP1, Fp1, Fp2, Fpz, PO3) interpolated; ASR repaired 20.5 % of samples,
window-rejected 0.3 %, output ratio 0.66 (no blow-up); ICA 63 components →
ICLabel rejected 7 (EOG/ECG auto-detection now a no-op by design); QC status
`excellent`, `recommended_action: accept`; epochs 598/600 trials (540 target +
60 nontarget, practice excluded via `phase=='main'`).

**Event-alignment check (2026-10-17).** The PsychoPy timing columns are a
constant offset from the `events.tsv` `onset` column within each task —
cptvis: `image_front.started - onset = +0.421 s` (std 0.000, n=600);
stroopvis: `stroop_arrow.started - onset = -7.679 s` (std 0.000, n=325).
Both are clock-offset artifacts between the PsychoPy session clock and the
EEG-time onsets (RTs computed from the PsychoPy columns match the logged
`response.rt`/`key_resp.rt`), not per-trial jitter, so they cannot smear
averages. The response-locked average on the cleaned cptvis data shows no
sharp press artifact (button presses are electrically small here), so the
absolute stimulus latency cannot be confirmed from EEG alone — flagged for
the lab's task-timing documentation if component latencies are to be
interpreted quantitatively.

#### 2026-10-17 — Track B pilot results (sub-001) + batch

**Final pipeline (config `~/proj/25SusAttn/configs/25susattn_optimized.yaml`):**
montage(sfp) → set_channel_types(VEOG/HEOG/EKG/EMG→misc) → 1 Hz HP → drop_flat →
variance bad channels (7 SD) → interpolate → zapline (fixed 50 Hz) → ASR (20,
EEGLAB path + pinv fix) → CAR (projection=False) → ICA infomax → ICLabel (0.5) →
40 Hz LP. Code commit `29c520e`.

**sub-001 results**

| task | bad channels | ASR repaired / rejected | ICA comps → rejected | line 50 Hz residual | blink gate | epochs |
|---|---|---|---|---|---|---|
| cptvis | 6 (AF4, CP1, Fp1, Fp2, Fpz, PO3) | 20.5 % / 0.3 % | 63 → 7 | 0.151 | Fp1 proxy, 326 blinks (~30/min), ratio 0.095, pass | 598/600 (540 target + 60 nontarget) |
| stroopvis | 4 (CP1, Fp1, Fp2, Fpz) | 22.8 % / 1.1 % | 63 → 3 | 0.131 | Fp1 proxy, 405 blinks, ratio 0.179, pass | 321/325 (163 congruent + 162 incongruent) |

Both QC `excellent` / `recommended_action: accept`. cptvis N1 −1.34 µV @ 127 ms
(POz/Oz/PO3/PO4); stroopvis shows the expected conflict direction (incongruent
more negative than congruent, −1.61 µV difference at ~251 ms at Fz/FCz). The
component windows in the task configs are a starting point for per-dataset
tuning, not a pipeline property.

**Resume caveat found:** the skip/reprocess decision compares the *config* hash
(R3), not the repository code revision — after a code change with an unchanged
config, `--force` is required to reprocess (observed on cptvis, 2026-10-17).

**Batch (in progress):** subjects 002/003/007 × cptvis/stroopvis
(preprocess + epochs), launched 2026-10-17 after the pilot checks passed.

#### 2026-10-17 — Track B batch (4 subjects) + gate veto

Batch `sub-001/002/003/007` × `cptvis`/`stroopvis` (preprocess + epochs), all 8
runs completed:

| subj | task | bad | ASR repaired | ICA rejected | 50 Hz residual | blink gate | epochs | action |
|---|---|---|---|---|---|---|---|---|
| 001 | cptvis | 6 | 20.5 % | 7 | 0.151 | Fp1, 326 (~30/min), 0.095 | 595 | accept |
| 001 | stroopvis | 4 | 22.8 % | 3 | 0.131 | Fp1, 405, 0.179 | 321 | accept |
| 002 | cptvis | 3 | 18.9 % | 4 | 0.210 | Fp1, 286, 0.122 | 585 | accept |
| 002 | stroopvis | 4 | 28.2 % | 3 | 0.252 | Fp1, 390, 0.107 | 325 | review¹ |
| 003 | cptvis | 2 | 10.6 % | 13 | 0.453 | Fp1, 255, 0.284 | 557 | accept |
| 003 | stroopvis | 2 | 21.0 % | 15 | 0.483 | Fp1, 553, 0.279 | 324 | accept |
| 007 | cptvis | 0 | 9.7 % | 17 | 0.257 | Fp1, 151, 0.247 | 594 | accept |
| 007 | stroopvis | 3 | 11.4 % | 12 | 0.225 | Fp1, 351, **0.374 FAIL** | 324 | **review** |

¹ pre-existing heuristic: `clustering=severe` → `quality_status='acceptable'`.

**Gate-veto fix (commit `336f4e9`).** 007/stroopvis failed the ocular gate while
`quality_metrics.recommended_action` still said `accept` — the action heuristic
(bad-channel % + ICA success) ignored the gate. `gates_ok=False` now downgrades
an `accept` to `review` (None keeps the old behaviour); `quality_metrics`
records `gates_ok`. Tests: `test_qc.py::TestRecommendedActionGateVeto` (4).

**Line-noise metric is relative — interpret with care.** 003's 48–52/20–40 Hz
ratio is 0.45–0.48 (highest of the batch) but its absolute 50 Hz amplitude is
**0.004 µV/√Hz, −40 dB below the 20–45 Hz broadband** (alpha at 10 Hz:
0.79–1.23 µV/√Hz). The ratio reflects the recording's low broadband power, not
residual line noise. No config change needed.

**Artifacts** were force-regenerated for all 8 recordings after the veto fix so
every QC JSON carries `gates_ok` and the corrected action.
