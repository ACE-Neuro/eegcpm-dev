# B6 audit briefing (prior state + items to adjudicate)

Companion to `plan/261013-2240-p1p2-remediation.md` §0/§B6. B6 = Checkpoint A (RPCA removal) +
golden resolved-config hash + B4-D2 blink-aware detector + tier-2 debt (tasks 1–20). HEAD `fd9bd16`
(B1–B5 committed); B6 uncommitted. Expected suite baseline **changes**: the RPCA fixture now passes,
so only the **2 `mne_connectivity`** tests (P4) should fail.

## Implementer summary (untrusted — verify)

- **Checkpoint A**: `RobustPCAStep` removed from `build_hbn_langer_chain()` → **9 steps**; class kept
  importable; `HBN_LANGER_SPEC_DEVIATIONS` surfaced via `get_config()`; chain docstring updated;
  9-step assertions fixed.
- **Task 1**: RPCA instrumentation (`return_info` → `n_iter`/`converged`; metadata `nnz_S_frac`,
  `energy_S_frac`, `n_iter`, `converged`).
- **Task 2**: fixture restored (`tests/fixtures/robust_pca/lin2011_exact_recovery.npz`, sha256 in
  INDEX.md, loaded via `Path(__file__)`).
- **Task 3**: `test_robust_pca_erp.py` — 2 × `xfail(strict=True)` on retention ≥ 0.95 + chain/registry
  assertion. Plan fixture (109 ch × 60 s): retention **0.675**, topo corr 0.979, `nnz_S_frac` 0.999,
  `energy_S_frac` 0.131, `n_iter` 200, `converged` False; 30-s variant: retention **0.188**, corr 0.834.
  **CORRECTION (B6-K8 note, 2026-10-16):** re-running the recorded probe (`/tmp/opencode/b6_probe_rpca_erp.py`)
  on the frozen fixture gives `n_iter` **31** / `converged` **True** for the 60-s fixture (30-s: 27 / True;
  retention 0.675 / 0.188 reproduce). The "n_iter 200 / converged False" claim was stale; the K8 fix
  (applied: False on `max_iter` exhaustion) is exercised by a forced `max_iter=1` unit test.
- **Tasks 4–8 golden config**: `get_config()` on all 9 chain steps; `library_versions()`;
  `resolved_config_hash`; `GOLDEN_RESOLVED_HASHES['hbn_langer'] =
  f3a40cd6ab825513b840ffed333f1de0d425c51f584023caa8b685e28613f378`; `load_golden_hashes()` still
  returns 5 stage keys (preprocessing → resolved hash; parity harness untouched otherwise); executor
  sidecar + DB `config_hash`; post-reference Cz variance check; tests (literal hash, per-step
  sensitivity, library-version sensitivity, `not-installed` fallback, parity path, all-required).
- **Tasks 9–16 tier-2**: zapline params/metadata (counts dict from `analytics_results`, missing key →
  `KeyError`); EOG single multi-target `lstsq` + z-scored condition number; `ReferenceStep`
  (`type='rest'` raises, `exclude_bads` removed, post-ref Cz check); `bad_channels.py` nanmedian
  neighbours + `:318,430` re-raise; ICA rank fix; pyproject `pipeline` extra pins; montage NaN/zero
  guard; `EXCLUSION_CRITERIA.md` regenerated.
- **Tasks 17–20**: `artifacts.py` 4 sites → `append_bad_annotations` (+ frame tests); `state.py:317`
  sanitizer + numpy round-trip test; N3 `run_identity_hash` + features provenance sidecar +
  `forward_source == 'template'` pin; A4 cache key + BEM/trans mtime/size.
- **B4-D2 detector**: blink-free kurtosis (EOG when present, else synthetic frontal proxy via
  `find_eog_events`; ±200 ms mask; reported mask/fallback), relative near-flat floor
  (`var < 1e-3 × median`), log-scale upper-tail variance/HF. Pilot5 × 3 re-run:

| recording | Rest | DM | FF |
|---|---|---|---|
| NDARAC853CR6 | **11** | 7 | 26 ✗ |
| NDARAC904DMU | **8** | 24 ✗ | 17 ✗ |
| NDARAM487XU3 | 96 ✗ (94 flat, corrupt) | 97 ✗ | 95 ✗ |
| NDARAM873GAC | **8** | 33 ✗ | 24 ✗ |
| NDARCD661GL0 | **8** | 12 ✗ | **11** |

  **7/11 non-corrupt recordings within budget** (Rest 4/4, DM 1/4, FF 2/4); remaining movie-task
  over-flagging is NOT blink-driven (masking variance/HF moves counts ≤2). Follow-ups: option (iv)
  pilot-50 recalibration and/or a user decision on extending the blink mask to variance/HF.
  **CORRECTION (B6-K1, 2026-10-16 — kept here as part of the record):** the table above sums to
  **6/12** non-corrupt recordings within budget (Rest 4/4, DM 1/4, FF **1/4** — CD661GL0/FF at 11);
  the "7/11, FF 2/4" sentence was a counting error. The table was re-checked row-by-row against the
  raw probe JSON (`/tmp/opencode/b6_pilot_full.json`) and is correct; `implementation/CHANGELOG.md`
  and `planning/VERIFICATION_TODO.md` now state 6/12.
- Suites: full = **2 failed (only connectivity P4) / 713 passed / 2 skipped / 2 xfailed**; B6-focused
  11 files = 181 passed / 2 xfailed; run_processor/CLI paths = 74 passed. `np.trapz` remains only in
  `erp_features.py:234` (B7); `info['temp']` grep 0.

## Items the auditor must adjudicate

1. **Kurtosis tail switch**: B4 pinned kurtosis as two-sided/raw; B6 switched to **upper-tail** per
   §12.5 ("log-scale / upper-tail kurtosis"; log undefined for signed excess kurtosis). Confirm or require
   a different rule (needs a user decision if changed).
2. **Near-flat floor**: only the `var < 1e-3 × median` disjunct implemented; the `log-variance z < −K`
   disjunct rejected with measured evidence (a healthy-fixture channel at 29% of median variance gets
   z = −54). Confirm the reasoning.
3. **ICLabel cross-check not implemented** (detector runs before any ICA in the chain) — confirm acceptable.
4. **ERP fixture retention 0.675 vs the audit's 2–28%**: confirm the xfail-strict regime still documents
   the removal correctly (30-s fixture gives 0.188).
5. **Golden hash semantics**: literal value correctness, library-version pinning, `get_config`
   completeness, parity-harness 5-key compatibility.
6. **Tier-2 correctness** for tasks 9–20 (esp. zapline counts, EOG lstsq equivalence, reference rest,
   bad-channels re-raise, artifacts append sites, state sanitizer, run_identity_hash).
7. **B4-D2 honesty**: the pilot table and the "movie tasks remain over-budget" residual.

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, k_id or new}],
risk_assessment,
reproducibility_data_integrity_notes,
residual_risks,
conditions_for_approval
```

---

## K3 pre-audit findings and rev1 dispositions (added 2026-10-16)

**B6-K1 (major, reporting integrity) — FIXED.** The pilot5 summary was inconsistent across three
documents (CHANGELOG "12/12", TODO "7/11, FF 2/4", table arithmetic 6/12). Re-derived row-by-row
against the recorded raw probe: **6/12 non-corrupt recordings within budget (Rest 4/4, DM 1/4,
FF 1/4)**; no table row was wrong (the prose claims were counting errors). CHANGELOG + TODO now both
state 6/12.

**B6-K2/K3 — FIXED (docstrings).** Near-flat rule is the `var < 1e-3 × median` disjunct only (the
log-variance z variant is rejected with the z=−54-at-29 %-of-median evidence); kurtosis documented as
upper-tail.

**B6-K4 — FIXED.** Chain docstring no longer claims a `STEP_REGISTRY` entry for RPCA.

**B6-K5 — FIXED.** Dead-EOG case now retries the frontal proxy before falling back; `blink_mask`
gains `attempts`; test added.

**B6-K6 — FIXED.** `preprocessed_qc` reads `n_components_removed_total` (dict-safe, legacy scalar
fallback); test added.

**B6-K7 — FIXED.** Zapline metadata is read/validated before the data write-back (a KeyError leaves
the raw unmutated); tests assert the non-mutation.

**B6-K8 — FIXED (preferred option).** `RobustPCAStep` returns `applied: False` + `reason:
'ialm_not_converged'` when IALM hits `max_iter` (write-back kept so the ERP xfail regime still
measures retention); forced-`max_iter=1` test added. Residual (recorded): a caller ignoring
`applied` still receives the non-converged L.

**B6-K9 — recorded.** Legacy `bad_channels.py:633,645` silent handlers → plan B7 task 15.

**Validator note:** one docstring contradiction (K2) — fixed; suite after rev1: **2 failed
(connectivity P4) / 717 passed / 2 skipped / 2 xfailed**.

---

## B6 rev2 — Opus final-audit conditions (NEW-1 … NEW-10), applied 2026-10-16

The rev1 record above is kept as-is; the following statements in it are **superseded** by the
final-audit findings and the rev2 fixes (details in `implementation/CHANGELOG.md` and
`planning/VERIFICATION_TODO.md`):

- The **"log-scale variant is undefined"** premise (adjudication item 1, and the rev1 docstrings)
  was **false**: it applied the log to the *signed excess* kurtosis. Sample Pearson kurtosis is
  `m4/m2² >= 1`, so `log(excess + 3)` is always defined; NEW-3 implements it (upper tail), and the
  auditor's probe measured 8/12 vs 6/12 pilot recordings within budget with log-kurtosis before the
  NEW-1/NEW-2 fixes.
- The **pilot table's chain order** omitted `ZaplineStep(fline=60)`; NEW-6 re-ran pilot5 in the true
  chain order (channel roles → zapline → 0.1–45 Hz filter (EEG+EOG) → detector) after the NEW-1/
  NEW-2/NEW-3 fixes. The old table's "6/12" summary is superseded by the rev2 table.
- The **filter step's effective picks** were the *data* channels (EEG only — `raw.filter()`'s
  default excludes EOG), so the EOG regressors kept their drift/line noise; NEW-1 filters EEG + EOG.
- The blink channel was ranked on **raw** variance; NEW-2 ranks it on 1–10 Hz band-passed variance
  (and records a plausibility reason for an implausibly low blink rate).
- `RobustPCAStep` now writes the low-rank component back **only on convergence** (NEW-10); the
  "write-back kept so the ERP xfail regime still measures retention" residual of K8 is gone (the
  fixtures converge).
