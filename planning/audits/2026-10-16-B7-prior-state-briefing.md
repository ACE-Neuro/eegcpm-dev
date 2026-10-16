# B7 audit briefing (QC truthfulness, test honesty, CI)

Companion to `plan/261013-2240-p1p2-remediation.md` §0/§B7. HEAD `17aec34` (B1–B6 committed);
B7 + rev1 uncommitted. Expected suite baseline: **2 failed only** (the two `mne_connectivity`
P4 tests); chain e2e green; no-silent-except gate green with no allow-list.

## Implementer summary (untrusted — verify)

- **QC truthfulness (tasks 1–3)**: "Pipeline step status" metric (bad on failed/excluded, warning
  on degraded; **rev1: warning "no chain metadata" when `step_status` is absent — no false ok**);
  `bad_channel_detection` keys; channel-role accounting; FAILED/EXCLUDED `*_qc_metrics.json`;
  `recording_qc_metrics` wired with `chain_metadata`/`raw_pre`; `hbn_recording_qc` block keys +
  detector count + `blink_residual_ratio` gate (old corr kept informational).
- **Test honesty (tasks 4–5)**: gate root fix (`parents[2]`), ≥1 file per CORE_PATH, >50 total,
  `modules/source` + `modules/qc` added (**rev1: + `cli`/`workflow`/`data`, 9 offenders fixed,
  108 files scanned, 0 offenders, no allow-list**); vacuous checks fixed.
- **Chain e2e (task 6)**: synthetic GSN-129 150 s recording; full 9-step chain; event samples
  identical; annotations survive; FIF reload; sidecar statuses; `config_hash == golden`; Cz > 0;
  EXCLUDED variant (contamination inside the ASR calibration window — documented).
- **CI (task 7)**: `.github/workflows/pipeline-verification.yml` (py3.13, `.[pipeline]`, backend
  imports, fsaverage cache+fetch, §0.1 command, `-m chain_e2e`).
- **Crosstalk (task 8) + source QC (task 9)**: free-orientation forward adaptation (mne constraint),
  per-label integer vertex lists (`get_cross_talk` rejects `BiHemiLabel`), numpy + `label_sign_flip`
  cross-check (N-3 discrepancy **pinned, not loosened**: max |ΔC| 2.08); cache key includes the
  noise-cov digest; `{subject}_crosstalk.npz`; QC metrics; **rev1: directed pair count**
  (`n_directed_above_0.3`).
- **test_source un-skips + non-vacuous crosstalk test (task 10)**; **argparse exit 2 (task 11)**;
  **A5** hidden temp patterns + single-writer note; **N4** `/`/`\` condition-name rejection;
  **N5** source-reconstruct exit codes + explicit task-config failure; **K9** legacy handlers fixed;
  **EOG guard** (carried from B6): refuses unfiltered regressors (`MAX_HF_POWER_FRACTION=0.05`
  above 50 Hz) + tests; golden hash re-pinned `ca91b7f2…` (superseded by the B7 rev2 re-pin to
  `1ef8780d…` — the audit conditions were applied afterwards; see the B7 rev2 evidence in
  `planning/VERIFICATION_TODO.md`; re-pinned again to `b47de382…` by the final-verification-audit
  fix round F1–F7, 2026-10-16, and to `b07bfe39…` by the ASR §12.6 EEGLAB-standard round,
  2026-10-16).
- **B7-N1 (critical, fixed)**: spherical-spline interpolation makes an interpolated channel an exact
  linear combination → ASR calibration covariance singular (cond 35 → ~1e15) → eegprep returned
  ~1e15 µV → recordings with ≥1 interpolated channel were excluded. Fix: ASR excludes interpolated
  channels from its **input** (they stay in the recording; reported
  `interpolated_excluded_from_asr`); not reordered (interpolation-before-ASR is deliberate).
- Suites: full = **2 failed (connectivity P4 only) / 833 passed / 2 xfailed / 0 skipped**;
  `chain_e2e` 13 passed; gate 19 passed / 108 files / 0 offenders.

## Items to adjudicate

1. **Crosstalk sign-cancellation (N-3)**: `get_cross_talk(mode='mean')` averages signed normals, so
   the saved `C` under-estimates leakage where sulcal-wall normals disagree (34 % flipped vertices,
   max |ΔC| 2.08). Documented + pinned; keep as the saved artifact, or promote the
   `label_sign_flip`-aligned summary?
2. **B7-N1 residual**: interpolated channels retain pre-ASR content (bursts present at interpolation
   time are never repaired in them); documented; ratify the minimal fix vs the chain reorder.
3. **B7-A4 coverage caveat**: the EXCLUDED e2e variant places contamination inside the calibration
   window (ASR repairs post-calibration contamination 22/75 → 0/75); is this acceptable for C2?
4. **B7-A5**: the EOG-guard threshold is synthetic-calibrated; validate on the real Checkpoint B
   subject (added to the B8 checklist).
5. Anything both prior passes missed.

## Prior passes

- **builder-validator → VALIDATED** (every claim reproduced; no vacuous tests; three scrutiny items
  above).
- **K3 pre-audit → revise** (B7-A1 major gate scope; A2 false-ok metric; A3 directed pair count;
  A4/A5 notes) — **all fixed in rev1** (A4/A5 recorded; see above).

## Output schema (required)

```
verdict (approve | revise | block),
critical_findings [{file_path:line, issue, impact, required_fix, k_id or new}],
risk_assessment,
reproducibility_data_integrity_notes,
residual_risks,
conditions_for_approval
```
