"""
Modular preprocessing pipeline builder.

Builds flexible preprocessing pipelines from configuration.

Author: EEGCPM Development Team
Created: 2025-12
"""

import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
import mne

from eegcpm.pipeline.base import ModuleResult
from eegcpm.workflow.state import _make_json_serializable
from .steps import STEP_REGISTRY, ProcessingStep
from .steps.base import RecordingExcluded


def _resolve_named_chain(name: str):
    """Resolve a named canonical chain to prebuilt step instances."""
    if name == "hbn_langer":
        from eegcpm.modules.preprocessing.hbn_langer_chain import (
            build_hbn_langer_chain,
        )
        return build_hbn_langer_chain()
    raise ValueError(
        f"Unknown named chain {name!r}. Available: ['hbn_langer']"
    )


class PreprocessingPipeline:
    """
    Flexible preprocessing pipeline builder.

    Builds a pipeline from a list of step configurations, where each
    step is an independent processing unit.

    Parameters
    ----------
    steps : list of dict
        List of step configurations, each with 'name' and optional 'params'
    output_dir : Path, optional
        Directory for saving outputs

    Examples
    --------
    Create a simple pipeline:
    >>> config = {
    ...     'steps': [
    ...         {'name': 'filter', 'params': {'l_freq': 1.0, 'h_freq': 40.0}},
    ...         {'name': 'ica', 'params': {'method': 'picard'}},
    ...     ]
    ... }
    >>> pipeline = PreprocessingPipeline(config['steps'])
    >>> result = pipeline.process(raw)

    Create EEGLab-style three-stage ASR pipeline:
    >>> config = {
    ...     'steps': [
    ...         {'name': 'filter', 'params': {'l_freq': 0.5, 'h_freq': 40.0}},
    ...         {'name': 'asr', 'params': {'cutoff': 40, 'mode': 'mild'}},
    ...         {'name': 'ica', 'params': {'method': 'picard'}},
    ...         {'name': 'iclabel', 'params': {'threshold': 0.8}},
    ...         {'name': 'asr', 'params': {'cutoff': 20, 'mode': 'aggressive'}},
    ...     ]
    ... }
    """

    def __init__(
        self,
        steps,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize pipeline.

        Parameters
        ----------
        steps : list of dict, or str
            Step configurations, OR the name of a named canonical chain
            (currently: "hbn_langer" — the deterministic Langer+ASR
            chain; built via build_hbn_langer_chain()).
        output_dir : Path, optional
            Output directory

        Raises
        ------
        ValueError
            If no steps are supplied, a step name is unknown, or a
            step's parameters are invalid.
        ImportError
            If a step's backend dependency is missing (fail-fast
            dependency preflight, see check_dependencies).
        """
        if isinstance(steps, str):
            steps = _resolve_named_chain(steps)
        if not steps:
            raise ValueError("Preprocessing pipeline has no steps")
        self.step_configs = steps
        self.output_dir = output_dir
        if isinstance(steps[0], ProcessingStep):
            self.steps = list(steps)  # prebuilt instances (named chains)
        else:
            self.steps = self._build_steps(steps)

        # Fail-fast dependency preflight: a missing backend must fail
        # here rather than silently degrading a step at runtime.
        for step in self.steps:
            step.check_dependencies()

    def _build_steps(self, step_configs: List[Dict]) -> List[ProcessingStep]:
        """
        Build step instances from configurations.

        Parameters
        ----------
        step_configs : list of dict
            List of step configurations. Each entry must carry 'name'
            and may carry 'params' (constructor kwargs, nested dialect)
            and 'required' (bool, default True).

        Returns
        -------
        steps : list of ProcessingStep
            Instantiated step objects

        Raises
        ------
        ValueError
            If step name is not recognized
        """
        steps = []

        for i, step_config in enumerate(step_configs):
            step_name = step_config.get('name')

            if not step_name:
                raise ValueError(f"Step {i} missing 'name' field")

            if step_name not in STEP_REGISTRY:
                available = ', '.join(STEP_REGISTRY.keys())
                raise ValueError(
                    f"Unknown step '{step_name}'. "
                    f"Available steps: {available}"
                )

            # Get step class
            step_class = STEP_REGISTRY[step_name]

            # Get parameters
            params = step_config.get('params', {}) or {}
            required = step_config.get('required', True)

            # Instantiate step
            try:
                step = step_class(**params)
            except TypeError as e:
                raise ValueError(
                    f"Invalid parameters for step '{step_name}': {e}"
                ) from e
            step.required = bool(required)
            steps.append(step)

        return steps

    def process(
        self,
        raw: mne.io.BaseRaw,
        subject_id: Optional[str] = None,
        **kwargs
    ) -> ModuleResult:
        """
        Execute preprocessing pipeline.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw EEG data
        subject_id : str, optional
            Subject identifier
        **kwargs : dict
            Additional metadata (session, task, etc.)

        Returns
        -------
        result : ModuleResult
            Processing result with:
            - success: bool (False on any required-step failure or on
              recording exclusion)
            - outputs: {'data': raw_processed, 'ica': ica_object} (both
              None when the run is not successful)
            - metadata: dict with 'step_status' (one entry per step),
              per-step metadata, plus 'excluded' / 'degraded' when set
            - warnings: list of tolerable (optional-step) failures
            - errors: list of fatal (required-step) failures

        Notes
        -----
        Failure semantics (fail loudly):

        - a disabled step is recorded as ``disabled`` and skipped;
        - a conditionally skipped, validation-failed or not-applied
          step is recorded as ``failed`` (required) or
          ``failed_tolerated`` (optional) unless its reason is in
          ``step.ALLOWED_NOOP_REASONS`` (then ``noop``);
        - any required failure aborts the remaining steps
          (``not_run``) and clears the outputs;
        - ``RecordingExcluded`` marks the run ``excluded`` (no FIF);
        - the FIF/ICA are saved **only** on success; the JSON sidecar
          is always written when ``output_dir`` is set.
        """
        start_time = time.time()

        metadata: Dict[str, Any] = {'step_status': {}}
        warnings = []
        errors = []
        ica = None  # Store ICA object if created

        raw_processed = raw.copy()
        aborted = False

        # Execute steps in sequence
        for i, step in enumerate(self.steps):
            step_name = f"{step.name}_{i}" if self._count_step_occurrences(step.name) > 1 else step.name

            if aborted:
                metadata['step_status'][step_name] = 'not_run'
                metadata[step_name] = {'not_run': True, 'reason': 'aborted'}
                continue

            # (a) Disabled steps are skipped explicitly.
            if not step.enabled:
                metadata['step_status'][step_name] = 'disabled'
                metadata[step_name] = {
                    'enabled': False,
                    'applied': False,
                    'reason': 'disabled',
                }
                continue

            step_meta: Dict[str, Any] = {}
            not_applied_reason: Optional[str] = None
            failure: Optional[Dict[str, Any]] = None
            excluded: Optional[RecordingExcluded] = None

            # (b) Conditional skip.
            try:
                if step.skip_step(raw_processed, metadata):
                    not_applied_reason = 'conditional_skip'
            except RecordingExcluded as e:
                excluded = e
            except Exception as e:
                failure = {'error': str(e), 'traceback': traceback.format_exc()}

            # (c) Input validation — False or a raise both mean
            # "not applied" (a required step then fails the run).
            if not_applied_reason is None and failure is None and excluded is None:
                try:
                    if not step.validate_inputs(raw_processed):
                        not_applied_reason = 'validation_failed'
                except RecordingExcluded as e:
                    excluded = e
                except Exception as e:
                    not_applied_reason = f"validation_failed: {e}"

            # (d)/(e) Run the step.
            if not_applied_reason is None and failure is None and excluded is None:
                try:
                    print(f"  Running: {step_name}")
                    new_raw, step_meta = step.process(raw_processed, metadata)
                    # A step must return (raw, dict). Trusting the
                    # contract blindly would leak a non-Raw downstream
                    # (or crash on `step_meta.get` below).
                    if not isinstance(new_raw, mne.io.BaseRaw) or not isinstance(step_meta, dict):
                        raise TypeError(
                            f"{step_name}.process must return "
                            f"(mne.io.BaseRaw, dict); got "
                            f"({type(new_raw).__name__}, {type(step_meta).__name__})"
                        )
                    raw_processed = new_raw
                except RecordingExcluded as e:
                    excluded = e
                except Exception as e:
                    failure = {
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                    }

            # (d) Recording excluded — abort the run immediately.
            if excluded is not None:
                metadata['step_status'][step_name] = 'excluded'
                metadata[step_name] = {
                    'applied': False,
                    'excluded': True,
                    'reason': excluded.reason,
                }
                metadata['excluded'] = {
                    'step': step_name,
                    'reason': excluded.reason,
                    **excluded.metrics,
                }
                print(f"  ⛔ {step_name} excluded recording: {excluded.reason}")
                aborted = True
                continue

            if failure is not None:
                # (e) Unexpected failure.
                metadata[step_name] = failure
                message = f"{step_name} failed: {failure['error']}"
                status = 'failed' if step.required else 'failed_tolerated'
            elif not_applied_reason is not None:
                # (b)/(c) Not applied.
                metadata[step_name] = {
                    'applied': False,
                    'reason': not_applied_reason,
                }
                message = f"{step_name} not applied: {not_applied_reason}"
                status = 'failed' if step.required else 'failed_tolerated'
            elif bool(step_meta.get('skipped', False)) or (
                'applied' in step_meta and not bool(step_meta['applied'])
            ):
                # (f) Step reported itself as not applied. `bool(...)`
                # because backends legitimately return numpy booleans
                # (`np.bool_(True)` is not `is True`).
                reason = step_meta.get('reason', 'not_applied')
                metadata[step_name] = step_meta
                if reason in step.ALLOWED_NOOP_REASONS:
                    status = 'noop'
                    message = None
                else:
                    message = f"{step_name} not applied: {reason}"
                    status = 'failed' if step.required else 'failed_tolerated'
            else:
                # Success.
                metadata[step_name] = step_meta
                status = 'ok'
                message = None
                # Special handling for ICA step - save ICA object
                if step.name == 'ica' and isinstance(step_meta, dict) and 'ica_object' in step_meta:
                    ica = step_meta['ica_object']

            metadata['step_status'][step_name] = status

            # (g) Required failures abort; optional failures degrade.
            if status == 'failed':
                errors.append(message)
                print(f"  ⚠️  {message}")
                aborted = True
            elif status == 'failed_tolerated':
                warnings.append(message)
                metadata['degraded'] = True
                print(f"  ⚠️  {message} (tolerated: step is optional)")

        # Determine success: any fatal error or exclusion fails the run.
        success = not errors and 'excluded' not in metadata

        # Save outputs only on success; always write the provenance
        # sidecar when an output directory is configured.
        output_files: List[Path] = []
        if self.output_dir:
            if success:
                try:
                    output_files = self._save_outputs(
                        raw_processed,
                        ica,
                        subject_id,
                        **kwargs
                    )
                except Exception as save_error:
                    # A half-saved run must never masquerade as success:
                    # drop whatever was written, record the error and let
                    # the FAILED sidecar be written below.
                    self._delete_saved_outputs(
                        self._output_stem(subject_id, **kwargs))
                    message = (
                        f"failed to save preprocessed outputs: {save_error}")
                    errors.append(message)
                    metadata['save_error'] = {
                        'error': str(save_error),
                        'traceback': traceback.format_exc(),
                    }
                    print(f"  ⚠️  {message}")
                    success = False
                    output_files = []
            try:
                self._write_sidecar(
                    metadata, errors, warnings, success, subject_id, **kwargs
                )
            except Exception:
                # A saved FIF without its provenance sidecar is not a
                # trustworthy output: delete it and re-raise.
                for path in output_files:
                    try:
                        Path(path).unlink()
                    except FileNotFoundError:
                        pass
                raise

        if not success:
            raw_processed = None
            ica = None

        return ModuleResult(
            success=success,
            module_name="preprocessing_pipeline",
            execution_time_seconds=time.time() - start_time,
            outputs={'data': raw_processed, 'ica': ica},
            output_files=output_files,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )

    def _count_step_occurrences(self, step_name: str) -> int:
        """Count how many times a step appears in the pipeline."""
        return sum(1 for step in self.steps if step.name == step_name)

    def _output_stem(self, subject_id: Optional[str], **kwargs) -> str:
        """Build the output stem ``{subject}{_ses-X}{_run-Y}``."""
        subject_str = subject_id if subject_id else "unknown"
        session_id = kwargs.get('session_id', '')
        run_id = kwargs.get('run_id', '')

        session_str = f"_ses-{session_id}" if session_id else ""
        run_str = f"_run-{run_id}" if run_id else ""

        return f"{subject_str}{session_str}{run_str}"

    def _write_sidecar(
        self,
        metadata: Dict[str, Any],
        errors: List[str],
        warnings: List[str],
        success: bool,
        subject_id: Optional[str] = None,
        **kwargs
    ) -> Optional[Path]:
        """Write the per-run provenance/status sidecar JSON.

        Naming (see plan §0.4): ``{stem}_preprocessing.json`` on
        success, ``{stem}_preprocessing_FAILED.json`` on failure,
        ``{stem}_preprocessing_EXCLUDED.json`` on exclusion. Always
        written when ``output_dir`` is set — including for failed and
        excluded runs, which have no FIF.
        """
        if self.output_dir is None:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if 'excluded' in metadata:
            suffix = '_EXCLUDED'
        elif not success:
            suffix = '_FAILED'
        else:
            suffix = ''

        path = self.output_dir / f"{self._output_stem(subject_id, **kwargs)}_preprocessing{suffix}.json"

        payload: Dict[str, Any] = dict(metadata)
        payload.update({
            'subject_id': subject_id,
            'session_id': kwargs.get('session_id'),
            'run_id': kwargs.get('run_id'),
            'success': bool(success),
            'errors': list(errors),
            'warnings': list(warnings),
            'excluded': metadata.get('excluded'),
            'degraded': bool(metadata.get('degraded', False)),
        })

        # Per-step metadata can contain numpy scalars / arrays and MNE
        # objects; reuse the shared JSON sanitiser.
        with open(path, 'w') as f:
            json.dump(_make_json_serializable(payload), f, indent=2, default=str)

        return path

    def _delete_saved_outputs(self, stem: str) -> None:
        """Delete this run's saved FIF outputs (used on a failed run).

        Never touches other runs: only ``{stem}_preprocessed_raw.fif``
        and ``{stem}_ica.fif`` in this pipeline's output directory.
        """
        if self.output_dir is None:
            return
        for name in (f"{stem}_preprocessed_raw.fif", f"{stem}_ica.fif"):
            try:
                (self.output_dir / name).unlink()
            except FileNotFoundError:
                pass

    def _save_outputs(
        self,
        raw: mne.io.BaseRaw,
        ica: Optional[mne.preprocessing.ICA],
        subject_id: Optional[str],
        **kwargs
    ) -> List[Path]:
        """
        Save preprocessed data and ICA object.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Preprocessed data
        ica : mne.preprocessing.ICA or None
            ICA object (if computed)
        subject_id : str
            Subject identifier
        **kwargs : dict
            Additional metadata (session, run, etc.)

        Returns
        -------
        output_files : list of Path
            Paths to saved files
        """
        output_files = []

        # Build filename
        stem = self._output_stem(subject_id, **kwargs)

        # Save preprocessed raw
        raw_filename = f"{stem}_preprocessed_raw.fif"
        raw_path = self.output_dir / raw_filename
        raw.save(raw_path, overwrite=True, verbose=False)
        output_files.append(raw_path)

        # Save ICA if available
        if ica is not None:
            ica_filename = f"{stem}_ica.fif"
            ica_path = self.output_dir / ica_filename
            ica.save(ica_path, overwrite=True)
            output_files.append(ica_path)

        return output_files

    def get_config(self) -> Dict[str, Any]:
        """
        Get pipeline configuration.

        Returns
        -------
        config : dict
            Configuration that can reproduce this pipeline
        """
        return {
            'steps': [
                {
                    'name': step.name,
                    'params': step.get_config()
                }
                for step in self.steps
            ]
        }

    def __repr__(self) -> str:
        """String representation."""
        step_names = [step.name for step in self.steps]
        return f"PreprocessingPipeline({' → '.join(step_names)})"
