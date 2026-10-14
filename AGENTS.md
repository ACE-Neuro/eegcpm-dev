# EEGCPM — project guide for agents

EEG Connectome Predictive Modeling toolbox: an MNE-based EEG pipeline
(raw → preprocessing → epochs → source reconstruction → features → behavioral prediction).
**Primary scientific goal: source-based ERP.**

## Layout

| Path | Purpose |
|---|---|
| `eegcpm-0.1/` | Publishable Python package — **keep clean, no AI/dev artifacts** |
| `eegcpm-0.1/.venv/` | Working environment; use it for all commands |
| `planning/` | Architecture docs + `VERIFICATION_TODO.md` — source of truth for verification state |
| `plan/` | Approved remediation/implementation plans (per planner policy, `YYMMDD-HHMM-slug.md`) |
| `implementation/` | Historical records (changelog, comparisons) |
| `.claude/commands/` | Claude Code command definitions (critique/decide/plan/research) |

## Environment

- Package env: `eegcpm-0.1/.venv` (mne, mne-icalabel, eegprep, pyzaplineplus, eeglabio,
  autoreject, streamlit, pytest). **torch is intentionally not installed** — the neural
  extras are out of scope for pipeline verification.
- Tests: `cd eegcpm-0.1 && .venv/bin/python -m pytest tests/unit -q`
  (pytest addopts include coverage; add `-p no:cacheprovider` or `--no-cov` if needed).
- CLI: `.venv/bin/eegcpm --help`.

## Non-negotiable invariants (from 2026-10-13 deep audits — do not regress)

1. **Fail loudly.** No silent `except: pass`. A required step that fails, skips, or
   reports `applied: False` must fail the run. Never save `*_preprocessed_raw.fif`
   for failed or excluded recordings; never mark COMPLETED.
2. **Annotation integrity.** Never `set_annotations(...)` over existing annotations —
   append. Event sample positions must be identical before/after ASR, zapline, and
   block rejection. `BAD_*`/boundary annotations must be inserted for removed windows.
3. **Preserve `info`.** When modifying data, write into a copy of the original `Raw`
   (e.g. `raw._data[picks] = ...`), not a fresh `RawArray` — otherwise `highpass`,
   `lowpass`, `meas_date`, `bads`, `subject_info`, `device_info` are silently lost.
4. **No state in `info['temp']`.** It does not survive FIF save. Use a JSON sidecar
   next to outputs, and keep provenance (config, versions, ranks, nave) in it.
5. **Configs:** one schema with `extra='forbid'`; nested `forward`/`inverse` sections
   must actually be consumed; resolved parameters hashed — not raw YAML bytes.
6. **Source-ERP validity:** ROI extraction must be anatomically mapped (CONN MNI coords,
   labels via `vertex_to_mni` + `extract_label_time_course(mode='mean_flip')`); output
   must be signed (`pick_ori='normal'`), never unsigned magnitude.

## Library API discipline

Before using or editing any MNE/eegprep/pyzaplineplus/eeglabio/autoreject call, load the
`eeg-pipeline-apis` skill (`.opencode/skills/eeg-pipeline-apis/`) and verify signatures
against the installed package. The audits found most real bugs came from unverified API
assumptions — wrong tuple indices, kwargs never passed, misread defaults.

## Verification workflow

- Two-pass reviews: cheap triage (Kimi K3) → deep audit (Opus 5.5, `builder-deep-auditor`).
- Review agents are read-only; implementers never self-approve.
- Current state and conditions for approval: `planning/VERIFICATION_TODO.md`.
- Do not declare a phase "verified" until its audit conditions are met and re-audited.
