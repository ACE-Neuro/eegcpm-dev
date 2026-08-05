"""
Modular preprocessing pipeline builder.

Builds flexible preprocessing pipelines from configuration.

Author: EEGCPM Development Team
Created: 2025-12
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import mne

from eegcpm.pipeline.base import ModuleResult
from .steps import STEP_REGISTRY, ProcessingStep


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
        """
        if isinstance(steps, str):
            steps = _resolve_named_chain(steps)
        self.step_configs = steps
        self.output_dir = output_dir
        if steps and isinstance(steps[0], ProcessingStep):
            self.steps = steps  # prebuilt instances (named chains)
        else:
            self.steps = self._build_steps(steps)

    def _build_steps(self, step_configs: List[Dict]) -> List[ProcessingStep]:
        """
        Build step instances from configurations.

        Parameters
        ----------
        step_configs : list of dict
            List of step configurations

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
            params = step_config.get('params', {})

            # Instantiate step
            try:
                step = step_class(**params)
                steps.append(step)
            except TypeError as e:
                raise ValueError(
                    f"Invalid parameters for step '{step_name}': {e}"
                ) from e

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
            - success: bool
            - outputs: {'data': raw_processed, 'ica': ica_object (if applicable)}
            - metadata: dict with results from each step
            - warnings: list of warning messages
            - errors: list of error messages
        """
        start_time = time.time()

        metadata = {}
        warnings = []
        errors = []
        ica = None  # Store ICA object if created

        raw_processed = raw.copy()

        # Execute steps in sequence
        for i, step in enumerate(self.steps):
            step_name = f"{step.name}_{i}" if self._count_step_occurrences(step.name) > 1 else step.name

            try:
                # Check if step should be skipped
                if step.skip_step(raw_processed, metadata):
                    metadata[step_name] = {
                        'skipped': True,
                        'reason': 'disabled or conditional skip'
                    }
                    continue

                # Validate inputs
                if not step.validate_inputs(raw_processed):
                    warnings.append(
                        f"{step_name}: Input validation failed, skipping"
                    )
                    metadata[step_name] = {'skipped': True, 'reason': 'validation_failed'}
                    continue

                # Process
                print(f"  Running: {step_name}")
                raw_processed, step_meta = step.process(raw_processed, metadata)

                # Store metadata
                metadata[step_name] = step_meta

                # Special handling for ICA step - save ICA object
                if step.name == 'ica' and 'ica_object' in step_meta:
                    ica = step_meta['ica_object']

            except Exception as e:
                import traceback
                error_msg = f"{step_name} failed: {str(e)}"
                errors.append(error_msg)
                warnings.append(error_msg)

                metadata[step_name] = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }

                # Decide whether to continue or abort
                # For now, continue with remaining steps
                print(f"  ⚠️  {error_msg}")

        # Determine success
        success = len(errors) == 0 or raw_processed is not None

        # Save outputs if requested
        output_files = []
        if self.output_dir and success:
            output_files = self._save_outputs(
                raw_processed,
                ica,
                subject_id,
                **kwargs
            )

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
        subject_str = subject_id if subject_id else "unknown"
        session_id = kwargs.get('session_id', '')
        run_id = kwargs.get('run_id', '')

        session_str = f"_ses-{session_id}" if session_id else ""
        run_str = f"_run-{run_id}" if run_id else ""

        # Save preprocessed raw
        raw_filename = f"{subject_str}{session_str}{run_str}_preprocessed_raw.fif"
        raw_path = self.output_dir / raw_filename
        raw.save(raw_path, overwrite=True, verbose=False)
        output_files.append(raw_path)

        # Save ICA if available
        if ica is not None:
            ica_filename = f"{subject_str}{session_str}{run_str}_ica.fif"
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
