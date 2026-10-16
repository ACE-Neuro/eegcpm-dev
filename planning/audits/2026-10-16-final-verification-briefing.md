# Final verification audit package (P1 C1–C6 / P2 conditions 1–6)

For the final deep audit (plan §B8 task 2): verify the acceptance conditions on the code and
recorded evidence, flag anything unmet, and state exactly what remains for the unverified parts
(Checkpoint B). Work is on `remediation/p1p2`; B1–B7 committed (`79da12e`), B8 task 1 uncommitted.

> **Fix round F1–F7 applied after the first audit pass (2026-10-16, uncommitted):** CI file
> tracked, block-gate threshold + ASR window-tolerances/blow-up factor hashed (golden hash
> re-pinned `1ef8780d…` → `b47de382047676631d2d096b8553ba70b79734aaaebde6774ca00a08e83f1a52`),
> `forward.template` fsaverage-only, gate count corrected (19), QC display-copy annotations now
> appended, target-N1 dominance asserted. Full evidence in `VERIFICATION_TODO.md`
> §"Final-verification-audit fixes F1–F7".

## Test commands (all currently green except the documented P4 pair)

```bash
cd eegcpm-0.1
.venv/bin/python -m pytest tests/unit -q --no-cov -p no:cacheprovider \
  --ignore=tests/unit/test_specparam_basemodule.py --ignore=tests/unit/test_specparam_features.py
# → 2 failed / 876 passed / 2 xfailed / 0 skipped; failures = the two mne_connectivity P4 tests
.venv/bin/python -m pytest -m chain_e2e -q --no-cov -p no:cacheprovider   # 34 passed
.venv/bin/python -m pytest -m fsaverage -q --no-cov -p no:cacheprovider   # 59 passed / 0 skipped
.venv/bin/python -m pytest tests/unit/test_no_silent_continue.py -q --no-cov -p no:cacheprovider -rA -s
# 19 passed; prints "scanned 108 files across 11 core paths, 0 offenders"
```

## P1 conditions — evidence to verify

- **C1 fail-fast**: executor semantics (required step raise/skip/not-applied → `success=False`,
  no FIF, FAILED/EXCLUDED persisted on CLI + run_processor; empty list / missing dependency at
  construction). Tests: `test_executor_failure_semantics.py`, `test_preprocess_status_persistence.py`.
- **C2 event timing**: `test_asr_timeline.py` (events `np.array_equal` before/after, annotations
  survive both `meas_date` branches, `BAD_asr_window` marks) + `test_hbn_chain_e2e.py`
  (full 9-step chain on synthetic GSN-129: stim samples identical, annotations survive, FIF reload,
  sidecar statuses, `config_hash == GOLDEN_RESOLVED_HASHES['hbn_langer']`, Cz variance > 0;
  EXCLUDED variant).
- **C3 detector**: `test_bad_channel_detector.py` (flat Cz exempt + heavy-tailed detected;
  >cap → `RecordingExcluded`; blink mask; near-flat floor; log-Pearson upper tail) + pilot5 table
  (8/12 within budget; corrupt recording excluded; residual recorded).
- **C4 RPCA**: decision (a) recorded (plan §12.1); step removed from the chain (9 steps);
  `test_robust_pca_erp.py` xfail-strict documents the destruction; instrumentation + fixture.
- **C5 schema + hash**: `test_preprocessing_schema.py`, `test_config_hash.py` (literal
  `GOLDEN_RESOLVED_HASHES['hbn_langer'] = b47de382…`, per-step + algorithm-constant sensitivity
  incl. F2/F3, library-version sensitivity, parity-harness 5-key path), sidecar + DB `config_hash`.
- **C6 QC + gate + CI**: "Pipeline step status" metric (no false ok); no-silent-except gate
  (108 files, 0 offenders, no allow-list); `.github/workflows/pipeline-verification.yml`
  (P4 deselected, fsaverage-skip gate; **tracked with the F1–F7 round** — it was untracked
  through B7) — local dry run only; a green remote run requires pushing (user-gated).

## P2 conditions — evidence to verify

1. **Labels/names/signed/consumer**: `labels.py` + `test_source_labels.py` (30 labels, cerebellar
   excluded, midline BiHemi, min-vertices); unique names; `pick_ori='normal'` + `mean_flip`;
   `test_source_dipole_groundtruth.py`; `test_source_erp_features.py`.
2. **nave/loud/provenance**: `test_source.py` condition-selection + equalize tests; provenance
   JSON (`config_hash`, epochs sha256, ranks, flip summary, `excluded_reasons`, `forward_source`);
   cov/inv reload; consumer provenance gating + mixed-config gate.
3. **covariance/inverse/forward/montage**: `test_source_hardening.py` (shrunk cov + guards after
   forward; trial λ² + eLORETA `force_equal` spy; 3-layer BEM only + cache; pre-flight thresholds).
4. **source-ERP validity**: B2 dipole ground truth; B6 uniform-patch escalation resolved by the
   user's decision (a) keep `mean_flip` + document (plan §12.4 / §0.5 caveats); B8 synthetic
   end-to-end source-ERP (`test_e2e_synthetic_source_erp.py`: target N1 negative at 0.100 s,
   corr 0.964, amplitude retention 0.966 ARF-normalised with controls).
5. **Checkpoint B (real data)**: ⛔ **blocked** — (i) ASR high-pass design decision (plan §12.6);
   (ii) second dataset path; (iii) B8 checklist items (guard/blink/EOG real-data checks,
   amplitude retention, green CI).
6. **Re-audit**: this audit.

## Open gates (do not mistake for defects)

- **§12.6 ASR high-pass design** — user decision; blocks real-data runs (B7-N2 evidence: 76 %
  repaired on clean 1/f data, blow-up regime cond 9e4–1.6e6).
- **Second dataset path** — user to provide; blocks half of the B4-D2 validation and Checkpoint B.
- **Remote CI green run** — requires pushing; local dry run only.
- Accepted residuals (recorded): movie-task detector budget; blink under-detection (carried to B8);
  crosstalk fixed-orientation approximation; `load_processed_runs` detector-block reachability
  (RunProcessor-only); re-interpolation `bads_before` gap. (The former "guard factor not in
  `get_config`" residual is **fixed** by F3.)

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, condition_id or new}],
condition_status [{condition, met | unmet | pending-real-data, evidence_or_gap}],
risk_assessment,
residual_risks,
conditions_for_approval
```
