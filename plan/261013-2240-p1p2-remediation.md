# P1 + P2 Remediation Plan — preprocessing & source reconstruction

- **Created:** 2026-10-13 22:40 (planner: Opus 5.5, planner-single)
- **Spec (source of truth):** `planning/audits/2026-10-13-P1-P2-required-fixes.md` (F-IDs below refer to it)
- **Status tracking:** `planning/VERIFICATION_TODO.md`
- **Implementer:** works batch by batch, in order. Do not start batch N+1 before batch N's
  validation commands pass and its re-audit is requested. Never self-approve.

---

## 0. Environment, invariants, conventions (read before any batch)

### 0.1 Environment
- Work dir for all commands: `/home/clivewong/eegcpm/eegcpm-0.1`; interpreter `.venv/bin/python` (Python 3.13.9).
- Installed: **mne 1.13.2**, mne-icalabel 0.10.0, eegprep 0.4.0, pyzaplineplus 0.1.0, eeglabio 0.1.4,
  autoreject 0.5.1, numpy 2.5.4 (`np.trapezoid` exists; `np.trapz` is deprecated), scipy, pydantic v2.
  `torch` and `specparam` are intentionally NOT installed.
- fsaverage is cached at `~/mne_data/MNE-fsaverage-data/fsaverage` (`mne.get_config('SUBJECTS_DIR')`
  points to `~/mne_data/MNE-fsaverage-data`); BEM `bem/fsaverage-5120-5120-5120-bem-sol.fif` present.
- Standard test command (specparam modules cannot be collected; always ignore them):
  ```bash
  cd eegcpm-0.1 && .venv/bin/python -m pytest tests/unit -q --no-cov -p no:cacheprovider \
    --ignore=tests/unit/test_specparam_basemodule.py --ignore=tests/unit/test_specparam_features.py
  ```
- **Baseline (2026-10-13, before B1):** 12 failed / 441 passed / 2 skipped (plus 2 collection
  errors from `test_specparam_*` — specparam/fooof intentionally not installed; ignore those two
  files in every run, and record collection errors in the baseline wording, never as passes).
  Failing: `test_connectivity.py::test_wpli_close_to_mne_connectivity`, `::test_dwpli_close_to_mne_connectivity`
  (P4, out of scope — must stay exactly these two), `test_robust_pca.py::test_ialm_against_published_reference_fixture`
  (fixed in B6), 9 × `test_source.py` (fixed in B2/B4).

### 0.2 Library API discipline (mandatory)
- Load skill `eeg-pipeline-apis` before touching any MNE / eegprep / pyzaplineplus / eeglabio / autoreject call.
  Use only signatures from `.opencode/skills/eeg-pipeline-apis/references/{mne-apis,other-libs}.md`; anything
  not listed there → verify with `inspect.signature` + a 5-line probe under `/tmp/opencode/` first.
- **Version delta:** the audits probed mne **1.12.1**; the venv has **1.13.2**. In 1.13.2
  `make_inverse_operator`, `apply_inverse`, `apply_inverse_epochs`, `write_inverse_operator`,
  `read_inverse_operator`, `make_inverse_resolution_matrix`, `get_cross_talk`, `get_point_spread`
  exist **only** under `mne.minimum_norm`. In `modules/source/__init__.py` import them as
  `from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs, write_inverse_operator`
  (module-level names, so tests can monkeypatch `eegcpm.modules.source.apply_inverse_epochs`).
  Any audit recipe that names a top-level `mne.*` inverse function must be translated to this namespace.
- Verified return shapes you will rely on: `eegprep.clean_windows → (EEG, sample_mask)`;
  `eegprep.clean_asr → EEG dict`; `pyzaplineplus.zapline_plus(...) → (clean, config, analytics, plots)`
  with data shaped `(n_times, n_channels)`; `mne.read_epochs → EpochsFIF` (use `mne.BaseEpochs` for isinstance).

### 0.3 Non-negotiable invariants (AGENTS.md — every batch is reviewed against these)
1. Fail loudly: no `except …: pass/continue`; a required step that raises / skips / returns `applied: False`
   fails the run; never write `*_preprocessed_raw.fif` for failed/excluded runs; never mark COMPLETED.
2. Annotations are append-only. Use exactly this offset recipe whenever adding annotations:
   ```python
   off = raw.first_time if raw.annotations.orig_time is not None else 0.0
   new = mne.Annotations(rel_onsets + off, durs, descs, orig_time=raw.annotations.orig_time)
   raw.set_annotations(raw.annotations + new)
   ```
   (`rel_onsets` = seconds from the first data sample.) All new descriptions start with `BAD_`.
3. Preserve `info`: write data back with `raw._data[picks] = ...`; never rebuild a `RawArray`.
4. No state in `info['temp']`; per-run state goes to the JSON sidecar (§0.4).
5. One config schema per stage, `extra='forbid'`, nested sections consumed, resolved params hashed.
6. Source ERP: `pick_ori='normal'` + `extract_label_time_course(mode='mean_flip')`; signed output only.

### 0.4 Shared conventions introduced by this plan
- `RecordingExcluded(Exception)` lives in `eegcpm/modules/preprocessing/steps/base.py`;
  `__init__(self, reason: str, metrics: dict | None = None)`, attributes `.reason`, `.metrics`;
  exported from `steps/__init__.py` and `modules/preprocessing/__init__.py`.
- Step status vocabulary (executor `metadata['step_status'][step_name]`):
  `ok` | `noop` (allow-listed no-op) | `disabled` (`enabled=False`) | `failed` | `failed_tolerated` |
  `excluded` | `not_run` (after an abort).
- Preprocessing sidecar (written by the executor whenever `output_dir` is set), stem
  `{subject}{_ses-X}{_run-Y}`:
  success → `{stem}_preprocessing.json`; failure → `{stem}_preprocessing_FAILED.json`;
  exclusion → `{stem}_preprocessing_EXCLUDED.json`. Content: `step_status`, per-step metadata
  (JSON-sanitised with `eegcpm.workflow.state._make_json_serializable`), `errors`, `excluded`,
  `degraded`; from B6 also `resolved_config`, `config_hash`, `library_versions`.
  **Required before B3** (audit B1-rev1 condition 4): replace the sanitizer with a numpy-aware
  encoder (`np.integer→int`, `np.floating→float`, `np.bool_→bool`, `ndarray→list`) that **keeps
  `None`** — the current `_make_json_serializable` stringifies numpy scalars/arrays and drops
  `None`, which would corrupt provenance once the sidecar replaces `info['temp']` (B3) — plus a
  test that writes and re-reads a sidecar containing numpy scalars, an array and `None`.
- New pytest markers (add to `[tool.pytest.ini_options]` in `pyproject.toml`):
  `markers = ["slow: >30 s", "fsaverage: needs cached fsaverage", "chain_e2e: full hbn_langer chain"]`.
- New shared fixture file `tests/unit/conftest.py` with fixture `fsaverage_subjects_dir` that returns
  `Path(mne.datasets.fetch_fsaverage(verbose=False)).parent` **only if**
  `<subjects_dir>/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif` already exists, otherwise
  `pytest.skip("fsaverage not cached")`. Never download inside tests.
- After each batch: tick the batch in `planning/VERIFICATION_TODO.md` as `[~] implemented — awaiting audit`
  (never `[x] verified`), append one line to `implementation/CHANGELOG.md`. Keep `eegcpm-0.1/` free of
  planning/AI artifacts.

### 0.5 Approved design decisions restated (do not re-open)
| Topic | Decision |
|---|---|
| Executor | Required-by-default steps; abort on first required failure; optional failures tolerated and flagged `degraded`. |
| hbn_langer | All chain steps `required=True`. Named chain is code-defined; YAML only selects it (`steps: hbn_langer`). |
| Preprocessing config | One pydantic `PreprocessingSpec` (nested `params:` dialect); flat and keyed dialects rejected. |
| ASR | Repair with `eegprep.clean_asr` at full length; window rejection = **annotation only** (`BAD_asr_window` from `clean_windows` mask); never cut samples. hbn_langer calibration = fixed first 60 s via `clean_asr(ref_maxbadchannels=<ndarray>)` (canonical spec value); generic ASRStep default = data-driven `ref_maxbadchannels=0.075, ref_tolerances=(-inf, 5.5)`. |
| Bad channels | Robust (median/MAD) NaN-safe z-scores; exempt via explicit `exempt=(ref,)`; zero-variance non-exempt = bad; >11 → `RecordingExcluded`; no truncation, no channel dropping. |
| Source labels | Spheres around CONN MNI centres on `inv['src']` via `mne.vertex_to_mni`; radius 10 mm lateral, 15 mm midline (|x| ≤ 5 mm → `BiHemiLabel`); `min_vertices=3` else raise; Cerebellar network excluded and logged. |
| Source output | `pick_ori='normal'`, `mode='mean_flip'`, `loose=0.2`, `depth=0.8`; default `mode='evoked'`; evoked λ² = 1/snr² (snr 3), trial λ² = 1/snr_trial² (snr_trial 1). |
| Covariance | `method='shrunk', rank=None, keep_sample_mean=True, tmin=None, tmax=0.0` on baseline-corrected epochs. |
| BEM | 3-layer `fsaverage-5120-5120-5120-bem-sol.fif` only; missing → `FileNotFoundError`. No single-shell fallback. |

---

## 1. Batch overview and dependency rationale

| Batch | Goal | F-IDs | Approval conditions advanced |
|---|---|---|---|
| **B1** | Executor fail-fast, exclusion, dependency preflight, one preprocessing schema, status persistence | P1-F1, F6, F4 (pulled forward), F9 (blanket except only), F17 (ImportError only) | P1 C1, C5 (schema part) |
| **B2** | Source-ERP core: real anatomical ROI labels, unique names, cerebellar/midline geometry, signed output + simulated-dipole ground truth | P2-F1, F2, F3, F4, F11 (estimator+guards pulled forward), F13 (validate_input only), F16 (dipole test) | P2 cond 1 (part), 3 (F11), 4 (part) |
| **B3** | Timeline integrity: ASR without cutting, sidecar not `info['temp']`, ASR wrapper metadata | P1-F3, F8, F15 | P1 C2 (part) |
| **B4** | Source usability (config wiring, evoked/trials, consumer, nave, loud failures) + P1 bad-channel detector | P2-F5, F6, F7, F8, F9; P1-F5 | P2 cond 1–2 (part); P1 C3 |
| **B5** | Source hardening + provenance (covariance, trial inversion, eLORETA, BEM, montage preflight, files) | P2-F10, F11, F12, F13 | P2 cond 2–3 |
| **B6** | ⛔ Checkpoint A (RPCA) + RPCA instrumentation + golden resolved-config hash + P1 tier-2 debt | P1-F2, F7, F9, F13–F14, F16–F22 (F15 lands in B3) | P1 C4, C5 |
| **B7** | QC truthfulness, test-suite honesty, CI job | P1-F10, F11, F12; P2-F14, F15, F16 | P1 C6; P2 cond 4 |
| **B8** | End-to-end verification prep; ⛔ Checkpoint B (real EGI-129 subject) | P2 cond 5; audit re-request | P1 C1–C6 re-audit; P2 cond 5–6 |

Deviations from the suggested 7-batch order (and why):
- **P1-F4 (annotation append) moved into B1**: B1 already rewrites `_BlockRejectionStepWrapper` to write
  `BAD_BLOCK` before raising `RecordingExcluded`; doing the append there avoids editing the same 10 lines twice.
- **P2 simulated-dipole test moved from F16 into B2**: it is the only ground-truth check for F1/F4; landing
  it with the code it validates prevents another placeholder from passing review.
- **Minimal P2-F13 (`validate_input` → `mne.BaseEpochs`, montage check) moved into B2** so `test_source.py`
  is green at the end of B2; the full montage pre-flight stays in B5.
- **P2-F11–F13 get their own batch (B5)** — they were unassigned in the suggestion; they depend on B4's
  `mode`/`snr_trial` config fields, and B4 is already the largest batch.
- **Separate B8** for e2e prep + real-data checkpoint so B7 (CI/tests) can be audited without real data.
- P1 and P2 batches interleave as the spec's cross-cutting order requires; P1 and P2 code paths do not
  share files except `core/config.py`, so either track can proceed if the other is blocked at a checkpoint.

---

## B1 — Executor fail-fast, exclusion, preflight, one preprocessing schema

**Goal:** a failed/skipped required step or an excluded recording can never produce a FIF or a COMPLETED
row; the config cannot silently no-op.

**Files:** `eegcpm/modules/preprocessing/steps/{base.py,__init__.py,interpolate.py,zapline.py,asr.py,bad_channels.py,iclabel.py}`,
`eegcpm/modules/preprocessing/{pipeline_builder.py,hbn_langer_chain.py,bad_channels.py,__init__.py}`,
`eegcpm/workflow/state.py`, `eegcpm/core/config.py`, `eegcpm/cli/preprocess.py`, `eegcpm/pipeline/run_processor.py`,
`examples/configs/preprocessing/{minimal,standard,hbn_langer}.yaml`, `eegcpm/config/preprocessing/*.yaml`,
`config/preprocessing_config.yaml`, `config/preprocessing_mne_bids_equivalent.yaml`,
`config/preprocessing_mne_bids_with_iclabel.yaml` (delete), tests.

**Tasks**
1. **Step base (F1.1, F1.3, F1.5)** — `steps/base.py`:
   - class attribute `required: bool = True`; class attribute `ALLOWED_NOOP_REASONS: frozenset = frozenset()`.
     Do **not** add `required` to every subclass constructor; the builder sets `step.required` after
     construction (task 4).
   - add `RecordingExcluded` (§0.4).
   - add `def check_dependencies(self) -> None:` (default no-op).
   - `get_config()` also returns `'required': self.required`.
2. **Dependency preflight overrides (F1.5, F17-partial)**: `ZaplineStep.check_dependencies` →
   `from pyzaplineplus import zapline_plus`; `ASRStep.check_dependencies` → `import eegprep, eeglabio`
   when `method=='eegprep'`, `import asrpy` when `'asrpy'`; `ICLabelStep` → `import mne_icalabel`;
   `steps/bad_channels.py` `BadChannelDetectionStep` → `import autoreject` when `'ransac' in self.method`
   (`steps/bad_channels.py:74` normalizes `method` to a list). Each re-raises `ImportError` with a
   `pip install <pkg>` hint. In `modules/preprocessing/bad_channels.py` lines ~241 and ~374, replace the
   `except ImportError:` fallbacks (which silently yield 0 bad channels) with a re-raised `ImportError`.
3. **Allow-listed no-op**: `InterpolateBadChannelsStep.ALLOWED_NOOP_REASONS = frozenset({'no_bad_channels'})`.
4. **Executor** — `pipeline_builder.py`:
   - `__init__`: if `steps` is `None` or an empty list → `ValueError("Preprocessing pipeline has no steps")`.
     `_build_steps`: pop `required` (default `True`) from each spec dict, instantiate with `params`, then
     `step.required = required`. After building (also for named chains) call `s.check_dependencies()` for
     every step (exceptions propagate).
   - `process(raw, subject_id=None, **kwargs)` per step, in order:
     a. `not step.enabled` → status `disabled`, continue.
     b. `step.skip_step(...)` True → *not applied* (reason `conditional_skip`).
     c. `validate_inputs` returns False or raises → *not applied* (reason `validation_failed: <msg>`).
     d. `step.process` raises `RecordingExcluded` → status `excluded`; `metadata['excluded'] =
        {'step': name, 'reason': e.reason, **e.metrics}`; abort loop.
     e. other exception → *failure* with message + traceback in `metadata[step]`.
     f. returned meta has `skipped is True` or `applied is False` → *not applied* unless
        `meta.get('reason') in step.ALLOWED_NOOP_REASONS` (→ status `noop`). Missing `applied` key = applied.
     g. *not applied*/*failure*: required → status `failed`, append to `errors`, abort loop;
        optional → status `failed_tolerated`, append to `warnings`, `metadata['degraded'] = True`, continue.
     Remaining steps after an abort → `not_run`.
   - `success = not errors and 'excluded' not in metadata`. On failure set `outputs['data'] = None`,
     `outputs['ica'] = None`.
   - Save FIF/ICA **only** on success; always write the sidecar JSON (§0.4) when `output_dir` is set.
     `_save_outputs` keeps using `session_id`/`run_id` kwargs (callers must now pass them).
5. **Block rejection (F1, F4)** — `hbn_langer_chain._BlockRejectionStepWrapper.process`: compute bad blocks;
   if `n_bad > 0`, **append** `BAD_BLOCK` annotations with the §0.3 recipe (`rel_onsets = i*window_seconds`);
   then if `fraction > max_rejected_fraction` raise `RecordingExcluded('block_rejection_fraction_exceeded',
   {'n_blocks_rejected', 'n_blocks_total', 'rejected_fraction', 'max_rejected_fraction', 'bad_block_onsets_s'})`.
   Keep the `n_blocks == 0` branch returning `applied: False` (→ run fails; correct).
6. **Zapline (F1.6, F9-partial)** — `steps/zapline.py`: delete the `try:`/`except Exception` (lines 189,
   266-273) so errors propagate; rename `validate` → `validate_inputs(self, raw) -> bool` (raise as now,
   return `True`), guard the Nyquist check with `if self.fline is not None`.
7. **Status enum** — `workflow/state.py`: add `ProcessingStatus.EXCLUDED = "excluded"`.
8. **Schema (F6)** — `core/config.py` (pydantic v2, `model_config = ConfigDict(extra='forbid')`):
   - `StepSpec`: `name: str`, `params: Dict[str, Any] = {}`, `required: bool = True`; field validator:
     `name` must be a key of `eegcpm.modules.preprocessing.steps.STEP_REGISTRY` (lazy import inside the
     validator to avoid a core→modules import cycle).
   - `PreprocessingSpec`: `stage: Literal['preprocessing'] = 'preprocessing'`, `name: Optional[str]`,
     `variant: Optional[str]`, `description: Optional[str]`, `depends_on: Optional[Dict[str, str]]`,
     `task: Optional[str]`, `is_primary: bool = False`, `notes: Optional[str]`,
     `steps: Union[Literal['hbn_langer'], conlist(StepSpec, min_length=1)]`.
   - `load_preprocessing_spec(data: dict) -> PreprocessingSpec`: if `data.get('preprocessing')` is a dict,
     unwrap it (project-config envelope; the only accepted wrapper); then `PreprocessingSpec.model_validate`.
   - `PreprocessingSpec.pipeline_steps()` → `'hbn_langer'` or `[s.model_dump() for s in steps]`.
9. **Configs**: convert `examples/configs/preprocessing/{minimal,standard}.yaml` to nested `params:`; check
   every `eegcpm/config/preprocessing/*.yaml` validates (fix stale keys if any); strip
   `examples/configs/preprocessing/hbn_langer.yaml` to `stage`, `variant`, `description`, `is_primary`,
   `steps: hbn_langer` and turn the per-section blocks into YAML comments (documentation only; the
   chain is defined by `build_hbn_langer_chain()`). Delete the three keyed-dialect files under
   `eegcpm-0.1/config/` listed above (only referenced from `_legacy` docstrings; git history keeps them).
   Leave the multi-pipeline/batch YAMLs in `config/` untouched (not consumed by these entry points).
10. **CLI** — `cli/preprocess.py`:
    - replace the `yaml.safe_load` + `config['preprocessing']` + `variant == 'hbn_langer'` heuristic with
      `spec = load_preprocessing_spec(yaml.safe_load(...))`; on `ValidationError` print it and return non-zero.
    - remove the subject-level skip-check (lines ~108-116). Inside the run loop, after `session`/`run_num`
      are parsed, create **one `WorkflowState` per run** (`subject_id, task, pipeline, session, run`) and skip
      if `manager.load_state(subject_id, task, args.pipeline, session=session, run=run_num)` has status in
      `{COMPLETED, EXCLUDED}` and not `args.force`.
    - `module = PreprocessingPipeline(spec.pipeline_steps(), run_output)`;
      `module.process(raw, subject_id=subject_id, session_id=session, run_id=run_num)`.
    - final run status: success → COMPLETED (after QC); `result.metadata.get('excluded')` → EXCLUDED; else
      FAILED. Set both step and `workflow.status`, then `manager.save_state(workflow)`.
    - per-run statuses are collected in a list inside the run loop; after it, increment subject-level
      `completed`/`excluded`/`failed` counters (subject `completed` only if every run is COMPLETED); add the
      `excluded` counter to the summary; delete the unconditional `workflow.status = COMPLETED` (line ~368).
      Precedence across mixed runs: any FAILED → subject `failed`; else any EXCLUDED → subject `excluded`;
      else `completed`.
11. **run_processor** — `pipeline/run_processor.py` (~256-257, 375-385, ~505): build steps via
    `load_preprocessing_spec(self.config).pipeline_steps()`; pass `session_id`/`run_id`; status EXCLUDED when
    excluded; `processing_state.update_status('completed' | 'failed' | 'excluded')` conditional (was
    unconditional at ~505); `output_files['preprocessed']` taken from `preproc_result.output_files` and only
    on success.
12. Update `test_preprocessing_reachability.py::test_block_rejection_hard_fails_on_excessive_contamination`
    to `pytest.raises(RecordingExcluded)`.

**Tests to add** — `tests/unit/test_executor_failure_semantics.py` (register throwaway steps with
`monkeypatch.setitem(STEP_REGISTRY, 'boom', BoomStep)`; synthetic `RawArray` 8 EEG ch, 250 Hz, 20 s):
- `test_empty_steps_raises`: `PreprocessingPipeline([])` → `ValueError`.
- `test_missing_dependency_fails_at_construction`: monkeypatch `ZaplineStep.check_dependencies` to raise
  `ImportError` → constructing `[{'name': 'zapline', 'params': {}}]` raises `ImportError`.
- `test_required_exception_aborts`: filter → boom(raises RuntimeError) → filter, `output_dir=tmp_path`:
  `not result.success`; `step_status == {'filter_0'|'filter': 'ok', 'boom': 'failed', <last>: 'not_run'}`
  (use the executor's naming); `list(tmp_path.glob('*_preprocessed_raw.fif')) == []`;
  `*_preprocessing_FAILED.json` exists; `result.outputs['data'] is None`.
- `test_required_applied_false_fails`, `test_required_skipped_fails`, `test_validate_inputs_false_fails`.
- `test_optional_failure_tolerated`: `required: False` → success, `metadata['degraded'] is True`,
  status `failed_tolerated`, FIF written.
- `test_allowlisted_noop_ok`: interpolate with no bads → success, status `noop`.
- `test_recording_excluded`: step raising `RecordingExcluded('too_bad', {'x': 1})` →
  `metadata['excluded']['reason'] == 'too_bad'`, `*_EXCLUDED.json` exists, no FIF.
- `test_block_rejection_annotates_then_excludes`: raw with stim annotations, 30 % of 2-s blocks with
  median ptp > 100 µV → `RecordingExcluded`; `e.metrics['n_blocks_rejected'] > 0`; the raw passed in now
  contains the original stim annotations **plus** `BAD_BLOCK` entries.
- `tests/unit/test_preprocessing_schema.py`: flat dialect `{'steps': [{'name': 'filter', 'l_freq': 1}]}` →
  `ValidationError`; keyed dialect `{'filter': {...}}` → `ValidationError`; `{'steps': []}` → error; unknown
  step name → error; `{'preprocessing': {'steps': [...]}}` unwraps; `{'steps': 'hbn_langer'}` OK;
  parametrized over `examples/configs/preprocessing/*.yaml` + `eegcpm/config/preprocessing/*.yaml`: validates
  **and** `PreprocessingPipeline(spec.pipeline_steps())` constructs.
- `tests/unit/test_preprocess_status_persistence.py` (C1 on both entry points): tmp BIDS tree
  `sub-01/ses-01/eeg/sub-01_ses-01_task-T_run-1_eeg.fif` (a saved synthetic RawArray), YAML config using a
  registered failing step (and a second YAML using an excluding step). Call `preprocess_command(argparse.Namespace(...))`
  (copy the Namespace fields from `cli/main.py` preprocess parser / existing `tests/unit/test_preprocess.py`)
  and assert `WorkflowStateManager(db).load_state('01', 'T', pipeline, session='01', run='1').status` is
  `FAILED` / `EXCLUDED` and no `*_preprocessed_raw.fif` under the derivatives tree. Same assertions through
  `run_processor` (use its public run-processing method; mirror `tests/unit/test_pipeline.py` harness).

**Validation**
```bash
cd eegcpm-0.1
.venv/bin/python -m pytest tests/unit/test_executor_failure_semantics.py tests/unit/test_preprocessing_schema.py \
  tests/unit/test_preprocess_status_persistence.py tests/unit/test_preprocessing_reachability.py \
  tests/unit/test_preprocessing_chain.py -q --no-cov -p no:cacheprovider
# then the standard full command (§0.1): failures must be a subset of the baseline list
grep -rn "info\[.temp.\]" eegcpm --include=*.py   # unchanged in B1 (removed in B3)
```

**Exit criteria:** C1 fully (FAILED/EXCLUDED persisted, no FIF, on CLI + run_processor; empty list /
missing dep fail at construction); schema half of C5. **Re-audit scope:** `pipeline_builder.py`, `steps/base.py`,
`_BlockRejectionStepWrapper`, `zapline.py` (except/validate), `core/config.py` preprocessing schema,
`cli/preprocess.py`, `run_processor.py` status logic, new tests.

---

## B2 — Source-ERP core: anatomical labels, unique names, geometry, signed output

**Goal:** ROI time courses are anatomically mapped, uniquely named, signed (`mean_flip`), and proven by a
simulated-dipole ground-truth test.

**Files:** `eegcpm/data/conn_rois.py`, new `eegcpm/modules/source/labels.py`, `eegcpm/modules/source/__init__.py`,
`eegcpm/pipeline/base.py` (EpochsModule only), `eegcpm/core/config.py` (SourceConfig fields), `eegcpm/modules/qc/source_qc.py`
(name/metadata parsing only), `tests/unit/{test_conn_rois.py,test_source.py,conftest.py}`, new
`tests/unit/test_source_labels.py`, new `tests/unit/test_source_dipole_groundtruth.py`.

Probe facts (2026-10-13, fsaverage, before `mindist` pruning): with 10 mm lateral / 15 mm midline radii,
Cerebellar.Anterior/Posterior have **0** vertices; minimum non-cerebellar count is **12** at `oct6`
(PPC_L) but only **3** at `oct5` (FEF_R, PPC_L). ⇒ all fsaverage tests use `spacing='oct6'`.

**Tasks**
1. **Unique names (P2-F2)** — `conn_rois.py`: `ROI.full_name` = `f"{network}.{name}" + (f"_{hemisphere}" if hemisphere else "")`
   (midline names unchanged, e.g. `DefaultMode.MPFC`). Add `MIDLINE_X_MM = 5.0`, property
   `ROI.is_midline` (`abs(mni_coords[0]) <= MIDLINE_X_MM`), and
   `SOURCE_EXCLUDED_NETWORKS = {"Cerebellar": "no cortical surface source-space vertices within radius (fsaverage)"}`.
   Add a module-level assertion that `get_roi_names()` is unique. `grep -rn "\.LP\b\|Lateral\"\|AInsula\"" eegcpm`
   for hard-coded old names (connectivity/QC/UI) and update them.
2. **Labels (P2-F1, F3)** — new `modules/source/labels.py`:
   `build_conn_labels(src, subjects_dir, radius_mm=10.0, midline_radius_mm=15.0, radius_overrides=None, min_vertices=3, subject='fsaverage') -> (labels, info)`:
   - `mni = [mne.vertex_to_mni(src[h]['vertno'], h, subject, subjects_dir=subjects_dir) for h in (0, 1)]` (mm).
   - for each ROI in `CONN_ROIS` order: network in `SOURCE_EXCLUDED_NETWORKS` → append
     `{'name', 'reason'}` to `excluded_rois`, skip. Radius = `radius_overrides.get(full_name)` →
     else `midline_radius_mm` if `is_midline` → else `radius_mm`.
   - midline: `sel_h = src[h]['vertno'][np.linalg.norm(mni[h] - c, axis=1) <= r]` for both hemis; each hemi
     must be non-empty (else `ValueError`); `mne.BiHemiLabel(mne.Label(sel_lh, hemi='lh', name=f"{n}-lh", subject=subject), mne.Label(sel_rh, hemi='rh', name=f"{n}-rh", subject=subject), name=n)`.
   - lateral: hemisphere from `ROI.hemisphere` ('L'→lh=0, 'R'→rh=1), single `mne.Label`.
   - total vertices `< min_vertices` → `ValueError(f"ROI {n}: {k} vertices < min_vertices={min_vertices}")`.
   - `info = {'roi_names', 'roi_coords_mni' (n,3), 'roi_radius_mm' (n,), 'roi_vertex_counts' (n,),
     'roi_hemi' ('L'|'R'|'midline'), 'excluded_rois' (list of names), 'excluded_reasons' (dict)}`.
3. **SourceConfig fields needed now** (`core/config.py`, full schema work in B4): `roi_min_vertices: int = 3`,
   `roi_radius_midline: float = 15.0`, `roi_radius_overrides: Dict[str, float] = {}`,
   `subjects_dir: Optional[Path] = None`.
4. **Source module (P2-F1, F4)** — `modules/source/__init__.py`:
   - `version = "0.2.0"`; store `self.roi_radius = config.get('roi_radius', 10.0)` plus the three fields above;
     `self.subjects_dir` = config value or `Path(mne.datasets.fetch_fsaverage(verbose=False)).parent`.
   - at the top of `process`: `epochs = data.copy().pick('eeg', exclude='bads')` (EOG channels must not enter
     covariance/inverse); **preserve the existing average-reference-projector check (current lines ~64-68)
     unchanged.**
   - covariance (**moved forward from B5 task 1** so the ground-truth test runs under the final estimator):
     `mne.compute_covariance(epochs, tmin=None, tmax=0.0, method='shrunk', rank=None, keep_sample_mean=True, verbose=False)`,
     with guards (each raises `ValueError`, surfaced as `success=False`): `epochs.baseline is None` →
     "epochs must be baseline-corrected"; `epochs.tmin >= 0` → "no pre-stimulus interval"; non-finite data.
     Warning if `len(epochs) * n_prestim < 10 * n_eeg`.
     **Ordering:** forward acquisition stays first; run the estimator + guards after `_get_template_forward` —
     a pre-forward baseline guard would break `test_process_with_mock_forward` (it expects the forward error as
     `errors[0]`).
   - labels: `build_conn_labels(inverse_operator['src'], ...)` once per module instance (cache on `self`).
   - evoked: `apply_inverse(evoked, inv, lambda2=lambda2, method=self.method, pick_ori='normal')`, then
     `mne.extract_label_time_course(stc, labels, inv['src'], mode='mean_flip', allow_empty=False)` → `(n_rois, n_times)`.
   - trials: `stcs = apply_inverse_epochs(cond_epochs, inv, lambda2, method=self.method, pick_ori='normal', return_generator=True)`;
     `tcs = mne.extract_label_time_course(stcs, labels, inv['src'], mode='mean_flip', allow_empty=False, return_generator=False)`;
     `np.stack(tcs)` → `(n_trials, n_rois, n_times)` (λ² change to `snr_trial` happens in B5).
   - delete `_extract_conn_rois` modulo code and the per-trial modulo loop entirely.
   - npz always contains the `info` keys from task 2 (`roi_names`/`roi_hemi`/`excluded_rois` as `np.array(..., dtype=str)`).
5. **validate_input (P2-F13 partial)** — `pipeline/base.py` `EpochsModule.validate_input`: `isinstance(data, mne.BaseEpochs)`.
   In `SourceReconstructionModule.validate_input(data) -> bool`: return `False` (never raise) unless
   `isinstance(data, mne.BaseEpochs)` and every EEG channel has finite, non-zero `loc[:3]`. `process()` calls it
   first and returns `ModuleResult(success=False, errors=["Missing montage: EEG channel positions are absent or zero"])`.
   Update `test_source.py::test_validate_input_raw_with_montage` to expect `False` (module only accepts epochs;
   rationale in a test comment). Document in the docstring that the return-False behavior deviates from the base
   contract (other modules raise; `pipeline/executor.py:109` ignores the return value) — safe only because
   `process()` re-checks and fails loud. Add `test_source_validate_via_executor`: montage-less input through the
   executor path → `success=False`, no output written.
6. **QC parsing** — `source_qc.py` ~line 61: metadata keys = `{'roi_names', 'roi_coords_mni', 'roi_radius_mm',
   'roi_vertex_counts', 'roi_hemi', 'excluded_rois', 'sfreq'}`; also skip keys ending in `_times` or `_nave`.
   Lines ~524-540: delete the L/R guessing; `display_names = list(roi_names)` (already unique).

**Tests**
- `test_conn_rois.py`: add `test_roi_names_unique` (`len(set(names)) == 32`, `names[0] == "DefaultMode.MPFC"`,
  `"DefaultMode.LP_L"` and `"DefaultMode.LP_R"` present).
- `test_source_labels.py` (`fsaverage` marker; `src = mne.setup_source_space('fsaverage', 'oct6', subjects_dir=..., add_dist=False)`):
  30 labels; `excluded_rois == ['Cerebellar.Anterior', 'Cerebellar.Posterior']`; all counts ≥ 3; every midline
  ROI is a `BiHemiLabel` with both hemis non-empty; lateral labels have the right `hemi`; names unique;
  `min_vertices=10**6` → `ValueError`; an override of 20 mm increases that ROI's count.
- `test_source_dipole_groundtruth.py` (`fsaverage`, `slow`; module-scoped fixture builds the forward once):
  - info from `make_standard_montage('GSN-HydroCel-129')` (129 ch incl. `Cz`), 250 Hz, `set_montage`, average-ref projector.
  - `fwd = mne.make_forward_solution(info, 'fsaverage', src_oct6, bem_sol, eeg=True, meg=False, mindist=5.0)`;
    `fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, use_cps=True)`.
  - target `SensoriMotor.Superior` (midline; nearest other ROI centre is **DorsalAttention.FEF at 34.96 mm** —
    a prior ≥37 mm probe was wrong; treat ~35 mm as the leakage baseline when triaging a dominance failure);
    labels from
    `build_conn_labels(fwd['src'], ...)`; waveform `w(t) = -1.0·g(t, 0.100, 0.015) + 0.6·g(t, 0.200, 0.030)`
    (N1 negative, P2 positive; `g` Gaussian) on every target-label vertex multiplied by
    `mne.label_sign_flip(label, fwd['src'])` (verify BiHemiLabel support; if unsupported compute per hemi and
    concatenate) — so `mean_flip` extraction should recover `+w`.
  - `stc_sim` → `evoked = mne.apply_forward(fwd_fixed, stc_sim, info)`; 40 epochs = evoked + Gaussian sensor
    noise (`np.random.default_rng(seed)`, SD = 0.2 × evoked std), tmin −0.1 s, `baseline=(None, 0)`.
  - run `SourceReconstructionModule({'method': 'dSPM'}, tmp_path).process(epochs, forward=fwd)` (evoked mode)
    for seeds 0 and 1. Assert from the saved npz: (1) target ROI has the largest `max|tc|` in 0.05–0.25 s;
    (2) `corr(tc_target, w) > 0.8` and `tc_target` at its 0.05–0.15 s extremum is **negative**; (3) that
    extremum latency is within 1 sample of 0.100 s; (4) sign at the N1 latency is the same for both seeds;
    (5) `roi_tc.min() < 0` (signed); (6) `roi_vertex_counts.min() >= 3`, cerebellar names in `excluded_rois`,
    unique `roi_names`. **If (1) fails, do not weaken it.** Pre-registered triage: (i) check the ~35 mm
    DorsalAttention.FEF leakage first; (ii) retry with the most isolated ROI (compute candidates by maximum
    nearest-centre distance over all 32 ROI centres) before concluding; (iii) only then escalate to the auditor
    with the evidence. The P1 track (B3, B6) proceeds meanwhile.
  - covariance guards (moved from B5 tests): no-baseline epochs → `success False` with "baseline" in the error;
    non-finite data → fails; `tmin >= 0` → fails.
- `test_source.py`: all 9 baseline failures pass (`roi_radius`, `version == "0.2.0"`, validate_input, "montage"
  message).

**Validation**
```bash
.venv/bin/python -m pytest tests/unit/test_conn_rois.py tests/unit/test_source.py tests/unit/test_source_labels.py \
  tests/unit/test_source_dipole_groundtruth.py tests/unit/test_connectivity.py -q --no-cov -p no:cacheprovider
grep -n "i % \|% stc.data\|roi_idx %" eegcpm/modules/source/__init__.py   # must print nothing
grep -n "pick_ori" eegcpm/modules/source/__init__.py                      # every call has pick_ori='normal'
```
Connectivity: only the 2 baseline failures allowed.

**Exit criteria:** P2-F1–F4 implemented; P2 cond 4 partial (dipole test green). **Re-audit scope:**
`conn_rois.py`, `labels.py`, `source/__init__.py` extraction/inversion, `pipeline/base.py` EpochsModule, the two new tests.

---

## B3 — Timeline integrity: ASR, ASR wrapper, sidecar instead of `info['temp']`

**Goal:** after ASR the recording has identical length, identical event samples, intact `info` and annotations,
and `BAD_asr_window` marks; ASR metadata is truthful.

**Files:** `eegcpm/modules/preprocessing/steps/asr.py`, `eegcpm/modules/preprocessing/hbn_langer_chain.py`
(`_ASRStepWrapper`, `_derive_asr_burden`), `eegcpm/modules/qc/hbn_recording_qc.py` (ASR fields only), new
`tests/unit/test_asr_timeline.py`, `tests/unit/test_preprocessing_reachability.py` (burden test).

**Tasks**
1. **Probe first** (skill Rule 0): confirm `eegprep.eeg_mne2eeg` µV scaling, `clean_asr` signature/return,
   `clean_windows(EEG, max_bad_channels=0.25, zthresholds=(-np.inf, 7))` returns `(EEG, sample_mask)` with
   `sample_mask.shape == (n_times,)` (True = clean), and that `clean_asr(..., ref_maxbadchannels=<ndarray>)`
   expects `(n_channels, n_samples)` in µV. Record findings as a comment in `asr.py`.
2. **ASRStep rewrite (P1-F3, F15)** — `steps/asr.py`:
   - params: `cutoff=20.0`, `method='eegprep'`, `window_criterion: float = 0.25` (fraction of bad channels per
     window, passed as `clean_windows(max_bad_channels=...)`; never falls back to `window_length`),
     `window_tolerances=(-np.inf, 7.0)`, `calibration: Literal['data_driven', 'fixed_first_60s'] = 'data_driven'`,
     `train_duration=60.0`, `window_length: Optional[float] = None` (passed to `clean_asr(window_len=...)` only if
     set), `exclude_channels: tuple = ()`.
   - picks: `mne.pick_types(raw.info, eeg=True, exclude='bads')` minus `exclude_channels` minus
     `metadata.get('channel_roles', {}).get('reference_channel')` (flat Cz before re-reference must not enter ASR).
     Fewer than 2 picks → `raise ValueError` (no `skipped` return).
   - `eeg = eegprep.eeg_mne2eeg(raw.copy().pick(picks))`.
   - calibration:
     - `data_driven` → `EEG_rep = clean_asr(eeg, cutoff=cutoff, ref_maxbadchannels=0.075, ref_tolerances=(-np.inf, 5.5))`;
     - `fixed_first_60s` → `ref = eeg['data'][:, :int(train_duration*sfreq)]`,
       `EEG_rep = clean_asr(eeg, cutoff=cutoff, ref_maxbadchannels=ref)` (`ref_tolerances` is a no-op when
       `ref_maxbadchannels` is an ndarray — left at the library default intentionally).
   - `repaired = EEG_rep['data']`; assert shape equals input shape, else `RuntimeError`; non-finite values →
     `RuntimeError` (delete `np.nan_to_num` and the whole `BAD_asr_naninf` branch).
   - `_, mask = eegprep.clean_windows(EEG_rep, max_bad_channels=window_criterion, zthresholds=window_tolerances)`;
     `len(mask) != raw.n_times` → `RuntimeError`.
   - write back: `raw._data[picks] = repaired * 1e-6` (the executor already passed a copy; never `RawArray`;
     delete the montage-restore code — no longer needed).
   - runs of `~mask` → `BAD_asr_window` annotations via the §0.3 recipe (onset = start_sample / sfreq,
     duration = n_samples / sfreq).
   - repaired-sample mask: per-sample RMS over picks of `(post - pre)` > 0.5 × per-sample RMS of `pre`
     (moved from `_derive_asr_burden`).
   - metadata: `applied: True`, `implementation`, `calibration`, `n_calibration_samples`, `cutoff`,
     `window_criterion`, `window_tolerances`, `n_channels_asr`, `excluded_from_asr`, `window_rejected_fraction`
     (`(~mask).mean()`), `n_bad_windows`, `bad_window_intervals_s` (list of `[onset, duration]`),
     `repaired_fraction`, `repaired_intervals_s`, `asr_burden` (fraction of samples rejected **or** repaired),
     `n_samples_before == n_samples_after`.
   - delete all `info['temp']` writes. `validate_inputs`: only enforce `train_duration` when
     `calibration == 'fixed_first_60s'`. Leave the `asrpy` branch functional but it is not part of any golden chain.
3. **Wrapper (P1-F8)** — `_ASRStepWrapper.process`: delegate to
   `ASRStep(cutoff=self.cutoff, method='eegprep', window_criterion=self.window_criterion, calibration='fixed_first_60s', train_duration=60.0)`;
   return the inner metadata unchanged plus `retain_burst_masks`. Remove `_derive_asr_burden` and every
   `info['temp']` line; `check_dependencies` delegates to `ASRStep`.
4. **hbn_recording_qc.py** lines ~92-97: read `asr_burden`, `window_rejected_fraction`, `repaired_fraction`
   from `chain_metadata['asr']`; delete the `info['temp']` read.
5. Rewrite `test_preprocessing_reachability.py::test_derive_asr_burst_mask_non_zero_on_burst` against
   `ASRStep` metadata (`repaired_fraction > 0` on an injected burst).

**Tests** — `tests/unit/test_asr_timeline.py` (real eegprep, no `importorskip`):
- Fixture: `RawArray` 32 EEG ch (`standard_1020` names + montage), 250 Hz, 180 s, `first_samp=1000`,
  `set_meas_date(datetime(2020,1,1,tzinfo=timezone.utc))`, 1/f-ish noise 10 µV, `raw.filter(0.5, None)` (sets
  `info['highpass']`), one channel in `info['bads']`, `stim` annotations every 2 s (duration 0), one
  pre-existing `BAD_manual` annotation; artifact window 100–103 s (+500 µV broadband on all channels).
- `test_events_identical_after_asr`: `mne.events_from_annotations` before/after → `np.array_equal` on sample
  columns; `n_times` equal; `meas_date`, `highpass`, `lowpass`, `bads` equal; `BAD_manual` kept.
- `test_bad_asr_window_marks_artifact`: ≥1 `BAD_asr_window` overlapping 100–103 s; none of them covers more
  than 20 % of the recording.
- `test_meas_date_none_also_works`: same with `meas_date=None` (offset recipe branch).
- `test_nan_output_raises`: monkeypatch `eegprep.clean_asr` to return NaN data → `RuntimeError`.
- `test_reference_channel_excluded`: add zero `Cz` + `metadata={'channel_roles': {'reference_channel': 'Cz'}}`
  → Cz still all zeros, `'Cz' in meta['excluded_from_asr']`.
- `test_wrapper_metadata_truthful`: `calibration == 'fixed_first_60s'`,
  `n_calibration_samples == 60*sfreq`, `'temp' not in raw.info or not raw.info['temp']`.
- `test_block_rejection_appends_with_offsets` (F4): stim annotations + one contaminated block at data seconds
  10–12 (fraction < 0.2) → all original annotations present; a `BAD_BLOCK` whose onset minus
  (`raw.first_time` if `orig_time` else 0) == 10.0; run for meas_date set and None.

**Validation**
```bash
.venv/bin/python -m pytest tests/unit/test_asr_timeline.py tests/unit/test_preprocessing_reachability.py \
  tests/unit/test_preprocessing_chain.py tests/unit/test_executor_failure_semantics.py -q --no-cov -p no:cacheprovider
grep -rn "info\[.temp.\]\|info.get(.temp" eegcpm --include=*.py   # must print nothing
grep -n "RawArray\|nan_to_num" eegcpm/modules/preprocessing/steps/asr.py   # must print nothing
grep -n "except" eegcpm/modules/preprocessing/steps/asr.py   # must print nothing (or only re-raises)
```

**Exit criteria:** P1-F3/F4/F8/F15 done; C2 unit-level (full-chain e2e lands in B7).
**Re-audit scope:** `asr.py`, `_ASRStepWrapper`, block-rejection append, `hbn_recording_qc.py` ASR fields, new tests.

---

## B4 — Source usability (config, modes, consumer, nave, loud failures) + P1 bad-channel detector

**Goal:** the source config fully reaches the module, an evoked ERP path exists, features can consume the
output, trial-count confounds and missing conditions are explicit; the P1 detector is correct.

### B4-P2 files
`eegcpm/core/config.py`, `eegcpm/modules/source/__init__.py`, `eegcpm/cli/source_reconstruct.py`,
`eegcpm/modules/features/source_erp_features.py`, `eegcpm/cli/features.py`, `eegcpm/cli/main.py` (features
parser), `examples/configs/source/dspm-conn32.yaml`, `eegcpm/config/source/{dspm_conn32,sloreta_conn32}.yaml`,
`tests/unit/test_source.py`, new `tests/unit/test_source_config.py`, new `tests/unit/test_source_erp_features.py`.

**Tasks (P2)**
1. **Schema (P2-F5)** — `model_config = ConfigDict(extra='forbid')` on `ForwardConfig`, `InverseConfig`,
   `SourceOutputConfig`, `SourceConfig` (source models only; other stages are P3/P4 scope).
   - `InverseConfig`: `snr: float = 3.0`, `snr_trial: float = 1.0`, `loose: float = 0.2`, `depth: float = 0.8`,
     `method_params: Optional[Dict[str, Any]] = None`.
   - `SourceOutputConfig`: add `compute_crosstalk: bool = True` (used in B7).
   - `SourceConfig`: `method: Literal['dSPM', 'sLORETA', 'eLORETA', 'MNE'] = 'sLORETA'`;
     `variant: Optional[str] = None`; `description: Optional[str] = None`;
     `parcellation: Literal['conn_networks'] = 'conn_networks'`;
     `mode: Literal['evoked', 'trials', 'both'] = 'evoked'`; `min_trials: int = 10`;
     `equalize_trials: Optional[int] = None`; `random_seed: int = 42`; plus B2 fields.
     `model_validator(mode='after')`: `variant` None → `f"{method}-CONN32"`; given but
     `variant.split('-')[0].lower() != method.lower()` → `ValueError`.
2. **Module wiring** — `SourceReconstructionModule.__init__`: `self.cfg = SourceConfig.model_validate(config)`
   (flat keys like `snr`, `trial_level` now raise). Keep attributes for tests:
   `method, parcellation, snr (=cfg.inverse.snr), snr_trial, loose, depth, spacing (=cfg.forward.spacing),
   roi_radius, mode`. Honour `output.save_stc` (evoked only), `output.save_roi_tc`, `output.generate_qc`.
   Outputs: evoked → `{subject}_roi_tc.npz` (`{cond}`: `(n_rois, n_times)`); trials →
   `{subject}_roi_tc_trials.npz` (`{cond}`: `(n_trials, n_rois, n_times)`); `both` → both files. Each file
   has `{cond}_times`, `{cond}_nave`, `sfreq` and the label metadata keys.
3. **Conditions, nave, loud failures (P2-F7, F8, F9)** — one helper `_select_condition_epochs(epochs)`:
   - requested = task-config condition names (or all `event_id` keys if no task config).
   - no matching events → warning `"condition '<c>': no matching events"`, add to `conditions_missing`.
   - `n < min_trials` → warning, `conditions_skipped[c] = {'n_trials': n, 'reason': 'below_min_trials'}`.
   - `equalize_trials=N`: `rng = np.random.default_rng(cfg.random_seed)` created once per subject; iterate
     conditions in sorted order; `n < N` → skipped (`reason: 'below_equalize_trials'`); else
     `idx = np.sort(rng.choice(n, N, replace=False))`.
   - zero produced → `ModuleResult(success=False, errors=["no conditions produced; requested=…, missing=…, skipped=…"])`;
     nothing written. Never call `.average()` on an empty selection; never write a zero-trial array.
   - metadata: `conditions_requested`, `conditions_produced`, `conditions_missing`, `conditions_skipped`,
     `nave` (dict); `nave_confound_warning = method in {'dSPM','sLORETA','eLORETA'} and equalize_trials is None`.
4. **CLI** — `cli/source_reconstruct.py:249` passes `source_config.model_dump()` (already); the CLI must exit
   non-zero on `ValidationError`.
5. **Config files**: rewrite `examples/configs/source/dspm-conn32.yaml` to the nested schema (`variant: dSPM-CONN32`,
   `forward: {template, spacing}`, `inverse: {snr, loose, depth}`, drop `name`, `use_conn_rois`, `conductivity`,
   flat `snr/loose/depth/spacing`); verify both `eegcpm/config/source/*.yaml` validate (variant matches method).
6. **Feature consumer (P2-F6)** — `source_erp_features.py`:
   - accept `Mapping`, `np.lib.npyio.NpzFile` (→ `{k: f[k] for k in f.files}`) or a path.
   - `roi_names = [str(x) for x in data['roi_names']]`; conditions = keys `k` with `f"{k}_times"` present;
     any condition array with `ndim != 2` → `ValueError("trial-level array; use the evoked *_roi_tc.npz")`.
   - time axis = `data[f"{cond}_times"]` (keeps negative tmin); sfreq derived from it (delete the 500.0 default).
   - per condition × ROI × component × metric column `f"{cond}_{roi}_{comp}_{metric}"`; no averaging across
     conditions; add `f"{cond}_nave"` columns when present.
   - `np.trapezoid` instead of `np.trapz`.
   - zero matched conditions/ROIs → `ValueError` **before** any file is written; write CSV once at the end.
   - CLI: add `source_erp` to `--feature-type` choices in `cli/features.py` (~line 177) and a `--source <variant>`
     argument (mirror the connectivity parser); input path
     `paths.get_source_dir(preprocessing=…, task=…, variant=…, subject=…, session=…) / f"{subject}_roi_tc.npz"`.

**Tests (P2)**
- `test_source_config.py`: flat `{'snr': 2}` → `ValidationError`; `{'trial_level': True}` → error;
  `{'method': 'dSPM', 'variant': 'sLORETA-CONN32'}` → error; `{'method': 'dSPM'}` → variant `dSPM-CONN32`;
  nested `{'forward': {'spacing': 'oct5'}, 'inverse': {'snr': 2.0, 'snr_trial': 1.5}, 'roi_radius': 12}` reaches
  module attributes; parametrized: every YAML in `examples/configs/source/` and `eegcpm/config/source/` validates.
- `test_source.py`: migrate tests to nested configs (`{'inverse': {'snr': s}}`, `{'forward': {'spacing': …}}`);
  `test_parcellation_options` → non-`conn_networks` raises.
- Loud-failure tests (synthetic epochs + forward from the B2 fixture, `fsaverage` marker): one condition with
  0 trials and one with 3 (`min_trials=10`) → both listed, no arrays for them; all conditions missing →
  `success is False`, no npz; `equalize_trials=5` twice with the same seed → identical outputs and
  `{cond}_nave == 5`; `{cond}_nave` always present.
- `test_source_erp_features.py`: synthetic npz with 2 conditions, 3 ROIs, times −0.2…0.6 s, known negative
  peak at 0.1 s in one ROI → column `"<cond>_<roi>_N1_peak_latency"` (use the module's real component/metric
  names) ≈ 0.100; per-condition columns distinct; 3-D input → `ValueError` and no CSV in `output_dir`;
  `NpzFile` input works.

### B4-P1 files
`steps/channel_roles.py`, `steps/bad_channels_pinned.py`, `steps/interpolate.py`, `hbn_langer_chain.py`
(`_BadChannelDetectorWrapper`), `tests/unit/test_preprocessing_chain.py`, `tests/unit/test_preprocessing_reachability.py`,
new `tests/unit/test_bad_channel_detector.py`.

**Tasks (P1-F5)**
1. `channel_roles.py`: delete the `raw._eegcpm_reference_channels` hack (lines ~96-100); `metadata['channel_roles']['reference_channel']` is the only source of truth.
2. `bad_channels_pinned.py`: `detect_bad_channels(raw, threshold_sd=3.0, exempt=(), max_bad=MAX_BAD_CHANNELS) -> (bads, report)`;
   delete `_is_cz`, `_is_cz_by_name`, all `try/except` blocks; import `scipy.stats`/`scipy.signal` at module top.
   - eligible = EEG picks (exclude bads) not in `exempt`.
   - flat: `np.ptp(x) == 0` or `var <= 1e-20` (V²) on an eligible channel → bad (reason `flat`), removed from
     the statistics below.
   - metrics on remaining eligible channels: variance, `scipy.stats.kurtosis(fisher=True)`, high-frequency
     power from `scipy.signal.welch(nperseg=min(2*sfreq, n_times))` in **25–40 Hz** (inside the 0.1–45 Hz
     passband the detector runs after; record `hf_band_hz=[25, 40]` in the report — documented deviation from
     Langer's band, P1-F5 "document the band").
   - robust z: `med = np.nanmedian(m)`; `mad = 1.4826*np.nanmedian(|m - med|)`; if `mad == 0` → use `np.nanstd`;
     if that is 0 too → metric marked degenerate in the report, z = 0. Non-finite z on an eligible channel → bad.
     `z > threshold_sd` → bad (reason per criterion).
   - `len(bads) > max_bad` → `raise RecordingExcluded('too_many_bad_channels', {'n_detected': n, 'max_bad_channels': max_bad, 'detected': bads})`.
   - report: `detected`, `reasons` (name → list), `z_scores` (name → {var, kurtosis, hf}), `n_detected_precap`,
     `flat`, `exempt`, `hf_band_hz`, `degenerate_metrics`.
   - `apply_bad_channels(raw, bads, exempt=())`: appends to `info['bads']`, never adds exempt names.
3. `_BadChannelDetectorWrapper.process`: `exempt = (metadata['channel_roles']['reference_channel'],)` when
   `exclude_cz` (KeyError if channel_roles metadata missing → run fails, correct); return report + `n_bads`.
4. `interpolate.py`: delete the drop branch; `pct_bad > max_bad_percent` →
   `raise RecordingExcluded('too_many_bad_channels_for_interpolation', {'n_bad', 'pct_bad', 'max_bad_percent'})`.

**Tests (P1, C3)** — `test_bad_channel_detector.py` (synthetic 110-ch EGI-like raw incl. `Cz`):
- flat Cz + one heavy-tailed channel (Gaussian + 50 spikes at 20× SD) + Gaussian rest, `exempt=('Cz',)` →
  heavy-tailed channel ∈ bads, `'Cz' not in bads`, all non-exempt kurtosis z finite.
- flat non-exempt channel → bad with reason `flat`.
- `MAX_BAD_CHANNELS + 5` high-amplitude channels → `RecordingExcluded` with `metrics['n_detected'] >= MAX_BAD_CHANNELS + 1`.
- chain path: `ChannelRolesStep` → wrapper with its metadata → Cz exempt (no info flag involved).
- interpolate over budget → `RecordingExcluded`; channel count unchanged.
- Replace the body of `test_preprocessing_chain.py::test_cz_exempt_from_bad_channel_detection` with the first
  case; `test_preprocessing_reachability.py::test_pinned_detector_caps_at_max_bad_channels` →
  `pytest.raises(RecordingExcluded)`; `test_pinned_detector_exempts_cz_before_detection` uses `exempt=`.

**Validation**
```bash
.venv/bin/python -m pytest tests/unit/test_source.py tests/unit/test_source_config.py tests/unit/test_source_erp_features.py \
  tests/unit/test_source_dipole_groundtruth.py tests/unit/test_bad_channel_detector.py tests/unit/test_preprocessing_chain.py \
  tests/unit/test_preprocessing_reachability.py -q --no-cov -p no:cacheprovider
grep -n "trapz\|500.0" eegcpm/modules/features/source_erp_features.py                 # nothing
grep -n "except" eegcpm/modules/preprocessing/steps/bad_channels_pinned.py             # nothing
```

**Exit criteria:** P2 cond 1 complete (F1–F6); F7–F9 of cond 2; P1 C3. **Re-audit scope:** SourceConfig,
module config/condition logic, consumer + CLI entry, detector/interpolate/channel_roles, new tests.

---

## B5 — Source hardening + provenance (P2-F10, F11, F12, F13)

**Files:** `eegcpm/modules/source/__init__.py`, `eegcpm/cli/source_reconstruct.py`, new `tests/unit/test_source_hardening.py`.

**Tasks**
1. **Covariance (F11)** — estimator + guards moved forward to **B2 task 4** so B2's ground-truth test runs
   under the final estimator (`method='shrunk'`, `rank=None`, `keep_sample_mean=True`, baseline/non-finite
   guards, sample-count warning). Remaining here: rank logging `data_rank = mne.compute_rank(epochs, rank=None)`
   and `cov_rank = mne.compute_rank(noise_cov, info=epochs.info)`, surfaced in the provenance JSON.
2. **Inversion (F12)**: `lambda2_evoked = 1/cfg.inverse.snr**2`; trials use `lambda2_trial = 1/cfg.inverse.snr_trial**2`
   in `apply_inverse_epochs`. Helper `_resolve_method_params(method, loose, user)` → for `eLORETA` with
   `0 < loose < 1` returns `{'force_equal': True, **(user or {})}`, else `user` (may be None). Pass
   `method_params=` to both `apply_inverse` and `apply_inverse_epochs`.
3. **Forward (F13)**: delete the single-shell branch and the dead `bem_path` reassignment; helper
   `_load_bem(subjects_dir)` raises `FileNotFoundError` if `fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif`
   is missing. `make_forward_solution(info, trans='fsaverage', src=src, bem=bem_sol, eeg=True, meg=False, mindist=5.0)`.
   Cache `(src, fwd)` in a class-level dict keyed by `sha256(ch_names + rounded loc[:3] bytes) + spacing + BEM filename + mne.__version__`.
4. **Montage pre-flight (F13)** `_preflight_montage(info, subjects_dir) -> dict`, called in `process` before the
   forward: (a) EEG `loc[:3]` finite and non-zero (message contains "montage"); (b) `info['dig']` contains
   LPA, Nasion, RPA cardinal points (`FIFFV_POINT_CARDINAL`) and every dig point is in `FIFFV_COORD_HEAD`;
   (c) electrode positions transformed with `mne.read_trans(<subjects_dir>/fsaverage/bem/fsaverage-trans.fif)`
   (= what `trans='fsaverage'` uses), nearest distance to `fsaverage-head.fif` scalp vertices
   (`scipy.spatial.cKDTree`): **warn if median > 10 mm, fail if max > 25 mm**. Return
   `{'median_mm', 'max_mm'}` for provenance. (Probe 2026-10-13: standard GSN-HydroCel-129 gives median 7.2 mm,
   max 18.7 mm, so a per-electrode >10 mm warning would fire on every subject; see Deviations.)
5. **Provenance (F10)** per subject in `output_dir`: `{subject}_source_config.yaml`
   (`yaml.safe_dump(cfg.model_dump(mode='json'))`); `{subject}-cov.fif` (`noise_cov.save(..., overwrite=True)`);
   `{subject}-inv.fif` (`write_inverse_operator(fname, inv, overwrite=True)`); `{subject}_provenance.json` with:
   `eegcpm`, `mne`, `numpy`, `scipy` versions; `subjects_dir`; BEM filename; spacing; `n_sources`
   (`sum(len(s['vertno']) for s in inv['src'])`); cov method; `cov_rank`; `data_rank`; `n_channels`;
   `lambda2_evoked`/`lambda2_trial`; `method_params`; `nave`; label info (counts, radii, hemi, excluded);
   `excluded_reasons`; per-ROI flip summary (mean(dots) or flip axis) so silent polarity flips across
   mne versions/configs are detectable (Opus B2-audit condition 5);
   montage distances; condition bookkeeping (B4); `nave_confound_warning`; `epochs_path` + `epochs_sha256`
   (streamed 1 MiB chunks; path passed by the CLI as `process(..., epochs_path=...)`); `config_hash` =
   sha256 of `json.dumps(cfg.model_dump(mode='json'), sort_keys=True)`.
6. **CLI**: `sorted()` every glob (subjects, sessions, task configs, epoch files); helper
   `_select_epoch_file(session_dir) -> Path` raises on 0 or >1 `*_epo.fif` (session counted failed with that
   message); pass `epochs_path`. Remove the `except Exception: continue` in the task-config scan (line ~92):
   report the unreadable file and fail.

**Tests** — `test_source_hardening.py` (covariance guard tests live in B2):
`_resolve_method_params('eLORETA', 0.2, None) == {'force_equal': True}`, `('dSPM', 0.2, None) is None`;
`_load_bem(tmp_path)` → `FileNotFoundError`; pre-flight: no montage → "montage"; `make_dig_montage(ch_pos=..., coord_frame='head')`
without fiducials → "fiducials"; standard GSN-HydroCel-129 passes with `max_mm < 25`; all positions shifted
+40 mm → fails; spy on `eegcpm.modules.source.apply_inverse_epochs` (wrap original) in `mode='trials'` captures
`lambda2 == 1.0` and `pick_ori == 'normal'`; after a dipole-fixture run, provenance JSON contains every key of
task 5, `mne.read_cov` and `mne.minimum_norm.read_inverse_operator` reload the files;
`_select_epoch_file` raises for two files. eLORETA smoke run (`fsaverage`, `slow`) succeeds.

**Validation:** `.venv/bin/python -m pytest tests/unit/test_source*.py -q --no-cov -p no:cacheprovider`;
`grep -n "conductivity=(0.3,)\|make_bem_model" eegcpm/modules/source/__init__.py` prints nothing.

**Exit criteria:** P2 cond 2 (F10) and cond 3 (F11–F13). **Re-audit scope:** covariance/inverse/forward/
pre-flight/provenance code, CLI globbing, new tests.

---

## B6 — ⛔ Checkpoint A (RPCA) + golden resolved-config hash + P1 tier-2

### ⛔ CHECKPOINT A — RPCA gate (P1-F2) — needs a user decision, do not pick one
Before any B6 code, the user records in `planning/VERIFICATION_TODO.md` (decisions log) one of:
- **(a) Remove** `RobustPCAStep` from `build_hbn_langer_chain()` (chain becomes 9 steps), with a spec-deviation
  note in the chain docstring and `PreprocessingPipeline.get_config()['spec_deviations'] = ['P1-F2: robust_pca removed (destroys ERP; synthetic retention 2–28%)']`.
- **(b) Validate** on ≥5 real HBN pilot recordings with `scripts/validate_rpca_erp.py` (chain with vs without
  RPCA via `required`/chain variant; identical epoching). Definitions: `e_w`, `e_wo` = grand-average ERP
  (channels × 0–600 ms); **retention** = `<e_w, e_wo> / <e_wo, e_wo>`; **topography corr** = Pearson r of the
  channel vectors at the GFP peak of `e_wo`. Accept RPCA only if retention ∈ [0.95, 1.05] and topography
  corr ≥ 0.95 on every recording; write results to `planning/audits/rpca_validation_<date>.md`. If rejected → (a).
  Needs real data → blocks until available.

**Tasks independent of the decision**
1. RPCA instrumentation (F2): `ialm_decompose(..., return_info=False)`; with `return_info=True` also return
   `{'n_iter', 'converged'}`. `RobustPCAStep` metadata adds `nnz_S_frac = np.count_nonzero(S)/S.size`,
   `energy_S_frac = ||S||_F² / ||M||_F²`, `n_iter`, `converged`.
2. Restore fixture (F12): copy `/home/clivewong/proj/eegcpm-dev/eegcpm-0.1/tests/fixtures/robust_pca/lin2011_exact_recovery.npz`
   → `eegcpm-0.1/tests/fixtures/robust_pca/` (if absent, regenerate per `tests/fixtures/INDEX.md`, seed 2011);
   add its sha256 to `INDEX.md`; ensure `.gitignore` does not exclude it; `test_robust_pca.py` loads it via
   `Path(__file__).resolve().parents[1] / "fixtures" / "robust_pca" / "lin2011_exact_recovery.npz"`.
3. ERP-injection test `tests/unit/test_robust_pca_erp.py`: 109 ch × 60 s at 250 Hz, rank-10 mixed AR noise
   + fixed-topography ERP (N1/P2 waveform) at 25 onsets; compute retention (definition above) of the
   epoch-averaged ERP after `RobustPCAStep()`. Decision (a): `@pytest.mark.xfail(strict=True)` on
   `retention >= 0.95` (documents why it was removed) + assert `'robust_pca' not in [s.name for s in build_hbn_langer_chain()]`.
   Decision (b): assert metadata keys present and retention recorded; the real-data gate governs acceptance.

**Golden config (P1-F7)**
4. Every hbn chain step/wrapper implements `get_config()` returning **all** constructor params + `name`,
   `version`, `enabled`, `required` (wrappers currently return only base fields).
5. `core/config_hash.py`: `library_versions()` via `importlib.metadata.version` for eegcpm, mne, numpy, scipy,
   eegprep, pyzaplineplus, eeglabio, autoreject, mne-icalabel (`'not-installed'` if missing);
   `resolved_config_hash(pipeline) = sha256(json.dumps({'pipeline': pipeline.get_config(), 'libraries': library_versions()}, sort_keys=True, default=str))`;
   literal `GOLDEN_RESOLVED_HASHES = {'hbn_langer': '<64-hex>'}` (computed once in the pinned venv after
   Checkpoint A). **Keep the `'preprocessing'` stage key resolvable**: `parity_harness.py:110-116,150-161`
   asserts `set(hashes) == set(GOLDEN_CONFIG_NAMES)` over 5 stages, so `load_golden_hashes()` must keep
   returning 5 keys — map the `'preprocessing'` stage to `GOLDEN_RESOLVED_HASHES['hbn_langer']` and drop the
   YAML-bytes path for that stage only (`check_golden_config_hashes` keeps its signature; the other 4 stages
   untouched — P4). Add a test asserting `set(load_golden_hashes()) == set(GOLDEN_CONFIG_NAMES)` and that the
   `run_parity_harness()` golden-hash assertion path (parity_harness.py:150-161) does not raise.
6. Executor sidecar adds `resolved_config`, `config_hash`, `library_versions`; CLI and run_processor set
   `workflow.config_hash` before `save_state`.
7. Post-reference Cz check: `ReferenceStep(check_reference_channel_variance: Optional[str] = None)`; hbn chain
   passes `'Cz'`; after re-reference `var(Cz)` must be finite and > 0 else `RuntimeError`; metadata
   `reference_channel_post_ref_variance`.
8. `test_config_hash.py`: `resolved_config_hash(PreprocessingPipeline('hbn_langer')) == GOLDEN_RESOLVED_HASHES['hbn_langer']`;
   changing any step param changes the hash. Add `test_hbn_chain_all_required`:
   `all(s.required for s in build_hbn_langer_chain())` (P1-F1 "all 10 steps required").

**Tier-2 (F9, F13–F14, F16–F22 — F15 lands in B3)**
9. Zapline (F9): unpack `clean, zap_config, analytics, _ = zapline_plus(...)`; pass `noisefreqs=[self.fline]`
   when `fline` is set (else `self.noisefreqs`, `[]` = auto); probe `analytics` keys first, then
   `n_components_removed = {f: analytics[k]['n_remove_final'] …}` — missing key → `KeyError` (no default 1);
   detected freqs from `zap_config` (probe the key first; missing key → `KeyError`, no default).
10. EOG regression (F13): single `np.linalg.lstsq(X.T, Y.T, rcond=None)` (X = EOG + intercept rows, Y = EEG);
    condition number of the z-scored EOG rows (+ intercept) computed once; metadata
    `spec_deviation: 'EOG regression without ICA; frontal ERP attenuation risk (checked in B8)'`.
    Tests: equals the old per-channel solution within 1e-10; scaling EOG ×1e6 leaves the reported condition number unchanged.
11. Reference (F16): `type='rest'` → `NotImplementedError`; remove `exclude_bads` (update the hbn chain call).
12. `steps/bad_channels.py:~197-206` (F18): `np.nanmedian` over montage neighbours; flat channel = bad.
    Also replace the silent `except Exception: print(...); return set(), {}` handlers at
    `eegcpm/modules/preprocessing/bad_channels.py:318,430` (audit B1-rev1 escalation) with re-raise,
    and add a test that a failing RANSAC/PREP path fails the step instead of returning zero bad
    channels. (Note: the full path — `steps/bad_channels.py` does not contain these handlers.)
13. ICA (F20) `ica.py:~372`: `min(max(5, r - 1), r - 1)`; `r - 1 < 1` → `ValueError`. ICLabel/ICA double
    apply (F19): comment only (idempotent; refuted by audit).
14. Dependencies (F21): `pyproject.toml` optional extra
    `pipeline = ["mne==1.13.2", "mne-icalabel==0.10.0", "eegprep==0.4.0", "pyzaplineplus==0.1.0", "eeglabio==0.1.4", "autoreject==0.5.1"]`.
15. Montage guard (F22) `steps/montage.py`: after `set_montage`, any remaining EEG channel with NaN/zero
    position → `ValueError` listing channels.
16. Regenerate `planning/EXCLUSION_CRITERIA.md` from the code: `too_many_bad_channels`,
    `too_many_bad_channels_for_interpolation`, `block_rejection_fraction_exceeded`, required-step failure.

**Validation:** full standard command; `test_robust_pca.py` now passes; `test_config_hash.py` green.
**Exit criteria:** C4 (decision recorded; if (b), real-data report attached), C5 complete.
**Re-audit scope:** chain composition, RPCA metadata/tests, `config_hash.py`, `get_config` implementations,
reference/zapline/EOG/ICA/montage edits.

---

## B7 — QC truthfulness, test honesty, CI (P1-F10/F11/F12, P2-F14/F15/F16)

**Files:** `modules/qc/{preprocessed_qc,hbn_recording_qc,source_qc}.py`, `cli/preprocess.py`,
`modules/source/__init__.py` (crosstalk inputs), `tests/unit/test_no_silent_continue.py`, offenders found by it,
new `tests/unit/test_hbn_chain_e2e.py`, `tests/unit/test_source.py`, `.github/workflows/pipeline-verification.yml`
(repo root; `working-directory: eegcpm-0.1`).

**Tasks — P1 QC (F10)**
1. `preprocessed_qc.py`: metric **"Pipeline step status"** from `metadata['step_status']`: `bad` if any required
   step is `failed`/`excluded`, `warning` if `metadata['degraded']`, else `ok`. Line ~289: read
   `metadata['bad_channel_detection']` (`detected`, `n_detected_precap`). Lines ~333-348: subtract
   `channel_roles` exclusions (`neck_face_dropped` + EOG retyped) from `n_dropped`; add informational metric
   "Channel-role exclusions"; "Bad Channels" counts detected/interpolated only.
2. `cli/preprocess.py`: line ~247 also reads `bad_channel_detection`; for FAILED/EXCLUDED runs write
   `..._qc_metrics.json` with `status` and `reason` (no HTML needed); call
   `recording_qc_metrics(raw_post, task, subject_id, chain_metadata=result.metadata, raw_pre=raw)` and store it
   under `recording_qc` in the metrics JSON.
3. `hbn_recording_qc.py`: block keys `rejected_fraction`, `n_blocks_rejected`; `bad_channels_n` from
   `chain_metadata['bad_channel_detection']`; replace the in-sample `residual_ocular_maxcorr` gate (vacuous
   after OLS regression) with `blink_residual_ratio`: blinks found on `raw_pre` VEOG
   (`mne.preprocessing.find_eog_events(raw_pre, ch_name=<first EOG name present>)`), ratio of blink-locked
   frontal ptp post/pre; `pass = ratio < OCULAR_GATE`; NaN/None if `raw_pre` missing. Keep the old corr as
   an informational field named `residual_ocular_maxcorr_insample`.

**Tasks — tests (F11, F12)**
4. `test_no_silent_continue.py`: root = `Path(__file__).resolve().parents[2]`; assert ≥1 file scanned per
   CORE_PATH and >50 in total; add `eegcpm/modules/source` **and `eegcpm/modules/qc`** to CORE_PATHS; exclude
   `_legacy` dirs **and** add a test that no module imports `_legacy`. Fix every offender (re-raise, narrow +
   record in metadata, or log with a recorded error) — no allow-list.
5. Grep tests for remaining vacuous checks (e.g. `is not` copy assertions) and make them assert data.
6. `test_hbn_chain_e2e.py` (`chain_e2e`, `slow`): synthetic GSN-HydroCel-129 recording (129 names incl. `Cz`),
   250 Hz, 150 s, `first_samp != 0`, meas_date set, 1/f noise + 60 Hz line + blinks on EOG names + ERP at
   2-s `stim` annotations, Cz all zeros. Run `PreprocessingPipeline('hbn_langer', output_dir=tmp)` with
   session/run → success; `stim` event samples identical input vs output; all input annotations present;
   FIF reloads with same annotations and meas_date; sidecar `step_status` all `ok`/`noop`;
   `config_hash == GOLDEN_RESOLVED_HASHES['hbn_langer']`; Cz variance > 0. Variant with 30 % contaminated
   blocks → EXCLUDED sidecar, no FIF.
7. CI workflow: python 3.13; `pip install -e "eegcpm-0.1[pipeline]" pytest pytest-cov`; step
   `python -c "import eegprep, pyzaplineplus, eeglabio, autoreject, mne_icalabel"` (fails the job if missing);
   cache `~/mne_data` and run `python -c "import mne; mne.datasets.fetch_fsaverage()"` before tests (the only
   allowed download); run the §0.1 command, then `pytest -m chain_e2e`.

**Tasks — source QC (P2-F14, F15, F16)**
8. Crosstalk: if `cfg.output.compute_crosstalk`, build a QC-only fixed-orientation pair
   (`convert_forward_solution(fwd, surf_ori=True, force_fixed=True)`, `make_inverse_operator(info, fwd_fixed, noise_cov, fixed=True, depth=0.8)`;
   documented approximation of loose=0.2), `resmat = make_inverse_resolution_matrix(fwd_fixed, inv_fixed, method=method, lambda2=lambda2_evoked)`.
   `get_cross_talk(resmat, src, labels, mode='mean', norm=False)` returns **one whole-source-space
   `SourceEstimate` per label** (probe the layout first; a bare `.data.mean()` over the whole brain is wrong).
   Two-step per pair: for label `i`, `ctf_i = get_cross_talk(...)[i]`; then
   `C[i, j] = mean(|ctf_i.in_label(labels[j]).data|) / mean(|ctf_i.in_label(labels[i]).data|)`.
   **Cross-check (critic N-3):** also compute the matrix with plain numpy + `mne.label_sign_flip` — the
   `get_cross_talk` signed mean can cancel across sulcal-wall normals — and assert `np.allclose` against the
   library result. If assertion (a) in task 10 fails physically, report the value — never loosen it.
   Cache the ROI×ROI matrix per montage hash; save `{subject}_crosstalk.npz`; pass `labels`, `crosstalk_matrix`
   into `qc_data`. Delete the delta-vs-reconstructed code and the `vertices[len//2]` "centre" (source_qc ~925-1001).
9. QC metrics: `bad` if any `roi_vertex_counts < roi_min_vertices` or any non-finite ROI value; `bad` if a
   `{cond}_times` key is missing (delete the fabricated time axis and the `sfreq=500.0` default, ~523-550);
   ERP plots from the evoked file.
10. `test_source.py`: replace both `@pytest.mark.skip` with the `fsaverage_subjects_dir` fixture
    (`oct6`, nested config) and assert signed outputs, unique names, cerebellar exclusion, provenance files.
    Crosstalk test (non-vacuous — diagonal == 1 is tautological): (a) the dipole-fixture target ROI's strongest
    off-diagonal entry is finite and < 1; (b) permuting two label indices permutes the corresponding
    rows/columns of `C` (proves per-label vertex indexing; fails for whole-brain averaging). If (a) fails
    physically (leakage), report the value to the auditor — never loosen the assertion.

**Validation:** full standard command (only the 2 baseline connectivity failures may remain);
`.venv/bin/python -m pytest tests/unit/test_no_silent_continue.py -q --no-cov -p no:cacheprovider -rA` shows files scanned > 50.
**Exit criteria:** C6; P2 cond 4. **Re-audit scope:** QC modules, gate + fixed offenders, e2e test, CI file.

---

## B8 — End-to-end verification prep; ⛔ Checkpoint B

1. `tests/unit/test_e2e_synthetic_source_erp.py` (`fsaverage`, `chain_e2e`, `slow`): B7 synthetic recording with
   the ERP generated from the B2 dipole (SensoriMotor.Superior) via `apply_forward` → hbn_langer chain →
   `mne.Epochs(raw, events, tmin=-0.2, tmax=0.6, baseline=(None, 0), reject_by_annotation=True)` (plain MNE;
   P3 module deliberately not used) → source (evoked, dSPM) → `SourceERPFeatureModule`. Assert target-ROI
   N1 negative and latency within 1 sample; feature CSV columns per condition.
2. Re-audit package: per batch, list changed files + test commands; request Opus deep audit of P1 F1–F5 +
   C1–C4 and P2 changed areas (cond 6).
3. ⛔ **CHECKPOINT B — real EGI-129 subject** (needs real data; P0 "representative subject" still open).
   Preconditions: P1 C1–C4 and P2 cond 1–4 re-audited. Then: `eegcpm preprocess --config examples/configs/preprocessing/hbn_langer.yaml …`;
   epochs; `eegcpm source-reconstruct --config eegcpm/config/source/dspm_conn32.yaml …`; save
   `mne.viz.plot_alignment(info, trans='fsaverage', subject='fsaverage', subjects_dir=…, surfaces='head-dense', eeg=['original', 'projected'], coord_frame='mri')`
   screenshot and pre-flight distances to `planning/audits/`; confirm crosstalk matrix in the QC HTML;
   review provenance JSON; sensor-vs-source N1/P2 polarity sanity; frontal-ERP EOG sensitivity (F13): frontal
   ERP with vs without EOG regression; RPCA data if Checkpoint A = (b). Record in `VERIFICATION_TODO.md`.

---

## 9. Acceptance criteria (plan-level, testable)

| Audit condition | Evidence |
|---|---|
| P1 C1 | B1 tests (executor, schema, status persistence on CLI + run_processor) green |
| P1 C2 | `test_asr_timeline.py` + `test_hbn_chain_e2e.py` green (events identical, annotations survive, `BAD_*` marks, meas_date both branches) |
| P1 C3 | `test_bad_channel_detector.py` green (flat Cz exempt, heavy-tailed detected, >11 → `RecordingExcluded`) |
| P1 C4 | Checkpoint A decision in VERIFICATION_TODO; (b) → real-data report with retention ∈ [0.95, 1.05], topo r ≥ 0.95 |
| P1 C5 | `test_preprocessing_schema.py` + `test_config_hash.py` (literal golden resolved hash) green; hash in DB + sidecar |
| P1 C6 | "Pipeline step status" QC metric; no-silent-except gate scans >50 files with 0 offenders and covers `modules/source` **and** `modules/qc`; CI workflow present and green |
| P1 F1 detail | `test_hbn_chain_all_required` (B6) + required-step executor tests (B1) green; subject-status precedence (failed > excluded > completed) asserted |
| Parity harness | `load_golden_hashes()` keeps all 5 `GOLDEN_CONFIG_NAMES` stage keys; `run_parity_harness()` golden path does not raise (B6) |
| P2 1 | B2 + B4 tests; no modulo extraction; `pick_ori='normal'` everywhere; consumer test green |
| P2 2 | nave saved/equalised; loud-failure tests; provenance/cov/inv files test |
| P2 3 | `test_source_hardening.py` green; no single-shell code |
| P2 4 | full suite: only the 2 baseline connectivity failures remain; dipole + un-skipped e2e pass locally with cached fsaverage |
| P2 F14 detail | crosstalk test non-vacuous, incl. numpy + `mne.label_sign_flip` cross-check (B7) |
| P2 5 | Checkpoint B artifacts in `planning/audits/` |
| P2 6 / P1 re-audit | Opus audit verdict ≠ BLOCK on the changed areas |

Global greps that must print nothing after B7: `grep -rn "info\[.temp.\]" eegcpm --include=*.py`;
`grep -rn "set_annotations(annots)" eegcpm --include=*.py`; `grep -rn "np.trapz(" eegcpm --include=*.py`;
`grep -rn "mne\.make_inverse_operator\|mne\.apply_inverse" eegcpm --include=*.py`.

## 10. Known risks

- **`parity_harness.py` golden-hash compatibility (critic F-1)**: `load_golden_hashes()` must keep returning
  all 5 `GOLDEN_CONFIG_NAMES` keys after B6's hash rework; B6 task 5 maps `'preprocessing'` to the resolved
  hash. Covered by a test — do not regress.
- **Connectivity (P4) silently consumes the new format**: `cli/connectivity.py` globs `*_roi_tc.npz` and assumes
  3-D trials; after B4 that file is evoked 2-D and trials live in `*_roi_tc_trials.npz`; ROI count is 30.
  Old magnitude-based connectivity outputs are invalid anyway (backlog: regenerate). Add to VERIFICATION_TODO P4.
- **Dipole dominance assertion (B2)** may fail from physical leakage; instructed to escalate, not weaken.
- **Fixed-first-60 s ASR calibration** depends on a clean first minute in HBN recordings; data-driven is the
  fallback the spec allowed; revisit at Checkpoint B if calibration looks contaminated.
- **Golden hash includes library versions** → it changes on any dependency bump (intended pin; re-pin per release).
- **Crosstalk resolution matrix** at oct6 ≈ 8k×8k floats (~0.5 GB) per montage; cached once per process.
- **`min_trials=10` default** excludes sparse conditions; configurable.
- **25–40 Hz HF band** for the detector differs from Langer's band (post-filter constraint); recorded in the report.
- RPCA option (b) and Checkpoint B block on real data.
- Deleting the three keyed-dialect YAMLs could affect undocumented user workflows (recoverable from git).

## 11. Out of scope

P3 epoching/trial rejection (`modules/epochs/`, `cli/epochs.py`, `epochs_qc.py`) — B8 uses plain `mne.Epochs`;
P4 features/prediction beyond the `source_erp_features` consumer (connectivity, specparam, ISC, drowsiness,
CPM, the 2 failing connectivity tests, feature/prediction golden configs); Streamlit UI (`eegcpm/ui/`) beyond
renaming hard-coded ROI names; REST API; user management; `_legacy` code (excluded from the gate, not fixed);
`extra='forbid'` for non-source/non-preprocessing config models; neural extras / torch.

## 12. Unresolved questions (block or change the plan)

1. **Checkpoint A (RPCA):** remove from the ERP chain (a) or validate on ≥5 real HBN recordings (b)? Blocks B6.
2. **Checkpoint B:** which real EGI-129 subject/session (P0 open item), and where is the data? Blocks B8 step 3.
3. OK to **delete** the three keyed-dialect YAMLs in `eegcpm-0.1/config/` (B1 task 9), or move them?
   Default if no answer: delete. **Resolved 2026-10-14: deleted** (recoverable from git).
4. **Source-ERP extraction mode (Opus B2 audit escalation, 2026-10-14 — blocks P2 condition 4,
   not B2):** with a uniform-patch simulation (same dSPM/loose-0.2 pipeline, 40 epochs),
   `mean_flip` makes the injected ROI dominant in only **11/30** ROIs vs **22/30** for
   `pca_flip`; midline flip axes are near-tied (|mean(dots)| 0.008–0.21; MPFC lh −0.70 / rh +0.71)
   and one ROI (`Language.IFG_R`) comes out polarity-inverted under both modes. Mean absolute
   source energy still ranks the true ROIs first — the loss is cancellation during extraction,
   not the inverse. Options: (a) keep `mean_flip` and document attenuation + polarity convention,
   (b) switch to `pca_flip` (still signed), (c) redefine midline ROIs (split by hemisphere or use
   surface patches instead of spheres). **User decision required before P2 condition 4 is
   marked verified.**
