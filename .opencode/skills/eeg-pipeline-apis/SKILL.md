---
name: EEG Pipeline APIs
description: Use before touching MNE, eegprep, pyzaplineplus, eeglabio, or autoreject code in this repo, or when a pipeline test fails with an API error. Provides the verify-don't-guess protocol, verified signatures, and the library-specific gotchas found by the 2026-10-13 deep audits.
---

# EEG Pipeline API Verification

This skill exists because the 2026-10-13 deep audits found that most real bugs in
`eegcpm/modules/preprocessing` and `eegcpm/modules/source` were **unverified library
assumptions**: a wrong tuple index for `pyzaplineplus.run()`, eegprep kwargs never passed,
MNE defaults misread. Do not repeat that pattern — even if you are sure you remember an API.

## Rule 0 — verify, never trust memory

The installed package is the source of truth. Before using or changing any library call:

```bash
cd eegcpm-0.1

# signature checks
.venv/bin/python -c "import inspect, mne; print(inspect.signature(mne.compute_covariance))"
.venv/bin/python -c "import inspect, pyzaplineplus; print(inspect.signature(pyzaplineplus.run))"
.venv/bin/python -c "import inspect, eegprep; print(inspect.signature(eegprep.clean_asr))"

# locate source to grep behavior (never trust docstrings alone for return shapes)
.venv/bin/python -c "import mne, os; print(os.path.dirname(mne.__file__))"
```

For behavior (not just signatures), run a 5-line probe on synthetic data under
`/tmp/opencode/` — never edit files outside the repo for probing, and clean up after.

If a fact below is older than the installed version, re-verify before relying on it.
Entries tagged **[audit]** come from the 2026-10-13 audit probes and must be spot-checked
the first time they matter.

## Verified facts

- `references/mne-apis.md` — signatures + gotchas for every MNE function used by the
  preprocessing and source-reconstruction fixes.
- `references/other-libs.md` — eegprep, pyzaplineplus, eeglabio, autoreject.

## Repo invariants (audits F1–F5 / C1–C6 — do not regress)

1. **Fail loudly.** No silent `except: pass`. A required step that fails, skips, or
   reports `applied: False` must fail the run; never save `*_preprocessed_raw.fif` for
   failed runs; empty step lists must raise at construction.
2. **Annotation integrity.** Never `set_annotations(...)` over existing annotations —
   append. Event sample positions must be identical before/after ASR, zapline, and
   block rejection; insert `BAD_*`/boundary annotations for removed windows.
3. **Preserve `info`.** Modify data via `raw._data[picks] = ...` on a copy of the
   original `Raw`, not a fresh `RawArray` — otherwise `highpass`, `lowpass`,
   `meas_date`, `bads`, `subject_info`, `device_info` are silently lost.
4. **No state in `info['temp']`.** It does not survive FIF save. Use a JSON sidecar
   (config, versions, ranks, per-condition nave, label vertex counts).
5. **Signed source output.** `pick_ori='normal'` + `extract_label_time_course(mode='mean_flip')`;
   never return unsigned magnitude for ERP features.
6. **Configs:** one schema, `extra='forbid'`, nested sections consumed, resolved-parameter
   hash recorded.

## Where things live

- Verification state + conditions for approval: `planning/VERIFICATION_TODO.md`
- Approved plans: `plan/`
- Package under work: `eegcpm-0.1/eegcpm/`
