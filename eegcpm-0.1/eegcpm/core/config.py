"""
Global configuration for EEGCPM.

Stage-First Architecture:
- Each stage config can specify dependencies on upstream stages
- Config locking prevents accidental reuse across different dependencies
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    conlist,
    field_validator,
    model_validator,
)


class BaseStageConfig(BaseModel):
    """Base configuration for any processing stage.

    This provides common fields for stage-first architecture, including
    dependency tracking and config locking.

    Attributes
    ----------
    stage : str
        Stage name (preprocessing, epochs, source, features, prediction)
    name : str
        Variant name for this configuration
    depends_on : dict, optional
        Upstream dependencies. Keys are stage names, values are variant names.
        Example: {"preprocessing": "standard", "task": "rest"}
    description : str, optional
        Human-readable description of this configuration
    """

    stage: str = Field(..., description="Processing stage name")
    name: str = Field(..., description="Configuration variant name")
    depends_on: Optional[Dict[str, str]] = Field(
        default=None,
        description="Upstream stage dependencies (stage_name -> variant_name)"
    )
    description: Optional[str] = Field(default=None, description="Config description")


class PreprocessingConfig(BaseModel):
    """Preprocessing pipeline configuration."""

    # Filtering
    l_freq: Optional[float] = Field(default=0.5, description="High-pass filter cutoff (Hz)")
    h_freq: Optional[float] = Field(default=40.0, description="Low-pass filter cutoff (Hz)")
    notch_freq: Optional[float] = Field(default=None, description="Notch filter frequency")

    # ICA
    ica_method: str = Field(
        default="infomax",
        description="ICA algorithm: infomax, extended_infomax, picard, fastica"
    )
    ica_n_components: Optional[int] = Field(default=None, description="Number of ICA components")
    ica_random_state: int = Field(default=42)

    # ASR (Artifact Subspace Reconstruction)
    use_asr: bool = Field(default=False, description="Apply ASR for artifact removal")
    asr_cutoff: float = Field(default=20.0, description="ASR cutoff parameter")

    # Artifact rejection
    reject_criteria: Optional[Dict[str, float]] = Field(
        default=None,
        description="Rejection thresholds (e.g., {'eeg': 100e-6})"
    )
    use_autoreject: bool = Field(default=False, description="Use autoreject for epoch rejection")


class StepSpec(BaseModel):
    """One preprocessing step in the single accepted nested dialect.

    ``extra='forbid'`` makes the two legacy dialects hard errors
    instead of silent no-ops: the *flat* dialect
    (``{'name': 'filter', 'l_freq': 1.0}``) and the *keyed* dialect
    (``{'filter': {...}}``) are both rejected.

    Attributes
    ----------
    name : str
        A key of ``eegcpm.modules.preprocessing.steps.STEP_REGISTRY``.
    params : dict
        Constructor keyword arguments for the step.
    required : bool
        Whether a failure / skip / not-applied result of this step must
        fail the run (default: True).
    """

    model_config = ConfigDict(extra='forbid')

    name: str = Field(..., description="Registered preprocessing step name")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step constructor keyword arguments",
    )
    required: bool = Field(
        default=True,
        description="Whether failure/skip/no-op of this step fails the run",
    )

    @field_validator('name')
    @classmethod
    def _validate_step_name(cls, value: str) -> str:
        """Reject unknown step names.

        The import is deliberately lazy: ``eegcpm.modules.preprocessing``
        imports pipeline machinery that itself imports core helpers, so a
        module-level import here would create a core -> modules cycle.
        """
        from eegcpm.modules.preprocessing.steps import STEP_REGISTRY

        if value not in STEP_REGISTRY:
            raise ValueError(
                f"Unknown preprocessing step {value!r}. "
                f"Available steps: {sorted(STEP_REGISTRY)}"
            )
        return value


class HbnLangerParams(BaseModel):
    """Per-dataset parameters of the code-defined ``hbn_langer`` chain.

    The chain's steps and their pinned parameters live in code
    (:func:`build_hbn_langer_chain`); this schema exposes the parameters a
    dataset may tune without editing the chain — currently the ASR ones
    (plan §12.6, 2026-10-16 EEGLAB-standard decision). Every field name is
    exactly a ``build_hbn_langer_chain`` keyword, so
    ``build_hbn_langer_chain(**params.model_dump())`` is the forwarding path.

    ``extra='forbid'``: an unknown/misspelled key is a hard error, never a
    silent no-op. The applied values are recorded in the resolved config and
    the per-run sidecar (via the ASR step metadata), so a non-default dataset
    is identifiable from its provenance.

    Attributes
    ----------
    asr_cutoff : float
        ASR cutoff in standard deviations. EEGLAB semantics: aggressive 3,
        default 5, conservative 20 (GUI default 20; Chang et al. 2019
        optimal 20-30). Default 20 (the chain's conservative setting).
        Bounded to [1, 200] (R6).
    asr_drift_transition_hz : (float, float) or None
        Transition band (Hz) of the EEGLAB-standard ``clean_drifts``
        high-pass applied to the ASR **input** (default ``(0.25, 0.75)``,
        EEGLAB's own ``clean_artifacts`` band). ``clean_asr`` requires
        high-passed input; the write-back is the high-passed repaired data
        (no add-back) and the step updates ``info['highpass']`` to the
        passband edge. ``None`` disables the filter (the non-EEGLAB regime;
        the step warns). Bounds (R6): ``0 < low < high`` and ``high <= 5 Hz``
        here; the upper edge must also stay below the recording's Nyquist
        frequency, which :class:`ASRStep` checks at runtime.
    asr_max_dims : float
        Maximum fraction of dimensions ASR may reconstruct (eegprep/EEGLAB
        default 0.66; must be in (0, 1]).
    asr_calibration : {'fixed_first_60s', 'data_driven'}
        ``'fixed_first_60s'`` (default): the canonical HBN-Langer
        calibration — the first 60 s of the recording are the calibration
        reference. ``'data_driven'``: EEGLAB's automatic clean-window
        calibration (``clean_windows`` selects low-artifact windows
        internally) — the built-in per-recording adaptivity, opted into per
        dataset. The spec default stays ``fixed_first_60s``.
    """

    model_config = ConfigDict(extra='forbid')

    asr_cutoff: float = Field(
        default=20.0, ge=1.0, le=200.0,
        description=(
            "ASR cutoff (SD); EEGLAB: aggressive 3, default 5, conservative 20. "
            "R6 (ASR scoped audit): bounded to [1, 200] — outside that range the "
            "cutoff is not a usable EEGLAB setting"
        ),
    )
    asr_drift_transition_hz: Optional[Tuple[float, float]] = Field(
        default=(0.25, 0.75),
        description=(
            "Transition band (Hz) of the clean_drifts high-pass applied to the "
            "ASR input (EEGLAB-standard); None disables it (non-EEGLAB regime)"
        ),
    )
    asr_max_dims: float = Field(
        default=0.66, gt=0.0, le=1.0,
        description="Maximum fraction of dimensions ASR may reconstruct",
    )
    asr_calibration: Literal['fixed_first_60s', 'data_driven'] = Field(
        default='fixed_first_60s',
        description=(
            "ASR calibration: fixed first-60-s reference (canonical) or "
            "EEGLAB's automatic clean-window calibration (per-recording adaptivity)"
        ),
    )

    @field_validator('asr_drift_transition_hz')
    @classmethod
    def _check_drift_transition(
        cls, value: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """Require a strictly increasing, positive transition band below 5 Hz.

        R6 (ASR scoped audit): ``0 < low < high`` (a non-increasing band is a
        hard error, not a silent no-op) **and** ``high <= 5 Hz`` — the band is a
        drift high-pass, and an edge above 5 Hz is not an EEGLAB transition
        band. The upper edge must also stay below the recording's Nyquist
        frequency; that is checked at runtime in :class:`ASRStep`
        (``_check_drift_nyquist``), because Nyquist is a property of the data,
        not of the schema.
        """
        if value is None:
            return None
        low, high = value
        if not 0.0 < low < high:
            raise ValueError(
                "asr_drift_transition_hz must be a (low, high) pair with "
                f"0 < low < high (Hz); got {value!r}"
            )
        if high > 5.0:
            raise ValueError(
                "asr_drift_transition_hz upper edge must be <= 5 Hz (it is a "
                f"drift high-pass transition band); got {value!r}"
            )
        return (float(low), float(high))


class PreprocessingSpec(BaseModel):
    """The single preprocessing configuration schema.

    ``steps`` is the only selector of the processing chain: either the
    named canonical chain (``'hbn_langer'``, defined in code by
    :func:`build_hbn_langer_chain`) or a non-empty list of
    :class:`StepSpec` entries. A ``variant:`` name never selects the
    chain — ``variant: hbn_langer`` next to an explicit step list is
    rejected (older configs relied on the variant alone, which silently
    changed meaning once the chain became code-defined).

    ``chain_params`` tunes the named chain per dataset (currently the ASR
    parameters, :class:`HbnLangerParams`). It is only meaningful with
    ``steps: hbn_langer``; an explicit step list carries its parameters
    per step, so ``chain_params`` next to a list is rejected instead of
    being silently ignored.
    """

    model_config = ConfigDict(extra='forbid')

    stage: Literal['preprocessing'] = 'preprocessing'
    name: Optional[str] = Field(default=None, description="Variant name")
    variant: Optional[str] = Field(default=None, description="Variant name (legacy alias of name)")
    description: Optional[str] = Field(default=None, description="Config description")
    depends_on: Optional[Dict[str, str]] = Field(
        default=None, description="Upstream stage dependencies"
    )
    task: Optional[str] = Field(default=None, description="Task name override")
    is_primary: bool = Field(default=False, description="Primary variant for this stage")
    notes: Optional[str] = Field(default=None, description="Free-form notes")
    steps: Union[Literal['hbn_langer'], conlist(StepSpec, min_length=1)] = Field(
        ..., description="Named chain or non-empty list of step specs"
    )
    chain_params: HbnLangerParams = Field(
        default_factory=HbnLangerParams,
        description=(
            "Named-chain parameters (only meaningful with steps: hbn_langer); "
            "forwarded to build_hbn_langer_chain"
        ),
    )

    @model_validator(mode='after')
    def _validate_variant_chain_consistency(self):
        """Reject ``variant: hbn_langer`` with an explicit step list.

        Only ``steps:`` selects the chain. Older configs declared
        ``variant: hbn_langer`` and were silently executed as their
        (different) explicit step list, or vice versa — that silent
        meaning change is now a hard error. Likewise ``chain_params``
        next to an explicit step list would be silently ignored, so it is
        rejected too (``model_fields_set`` distinguishes "explicitly given"
        from the schema default).
        """
        if self.variant == 'hbn_langer' and self.steps != 'hbn_langer':
            raise ValueError(
                "variant: hbn_langer requires steps: hbn_langer "
                "(only `steps` selects the processing chain; a variant "
                "name alone does not)"
            )
        if 'chain_params' in self.model_fields_set and self.steps != 'hbn_langer':
            raise ValueError(
                "chain_params is only meaningful with steps: hbn_langer "
                "(an explicit step list carries its parameters per step; "
                "chain_params would be silently ignored)"
            )
        return self

    def pipeline_steps(self):
        """Return the ``PreprocessingPipeline`` input for these steps.

        For the named chain the steps are built here (with
        ``chain_params`` forwarded to :func:`build_hbn_langer_chain`) and
        returned as a :class:`ResolvedChainSteps` — a prebuilt step list that
        also carries the chain's recorded spec deviations, so
        ``PreprocessingPipeline(spec.pipeline_steps())`` resolves to the same
        config (and hash) as ``PreprocessingPipeline('hbn_langer')`` for
        default parameters.

        Returns
        -------
        ResolvedChainSteps or list of dict
            Prebuilt named-chain step instances (plus spec deviations), else
            a list of ``{'name', 'params', 'required'}`` dicts.
        """
        if self.steps == 'hbn_langer':
            from eegcpm.modules.preprocessing.hbn_langer_chain import (
                build_hbn_langer_chain,
            )

            # Returns a ResolvedChainSteps: prebuilt step instances that carry
            # the chain's spec deviations, so the resolved config (and hash)
            # matches `PreprocessingPipeline('hbn_langer')`.
            return build_hbn_langer_chain(**self.chain_params.model_dump())
        return [step.model_dump() for step in self.steps]


def load_preprocessing_spec(data: Dict[str, Any]) -> PreprocessingSpec:
    """Validate a preprocessing configuration mapping.

    Accepts either a bare preprocessing config or a project-config
    envelope whose ``preprocessing`` key holds one; the envelope is the
    only wrapping accepted. A mapping that carries **both** a
    ``preprocessing`` envelope and a top-level ``steps`` key is
    ambiguous (which one selects the chain?) and is rejected.

    Parameters
    ----------
    data : dict
        Parsed YAML mapping.

    Returns
    -------
    PreprocessingSpec
        Validated specification.

    Raises
    ------
    pydantic.ValidationError
        If the mapping uses a legacy dialect, has no/empty steps,
        references an unknown step, or mixes the envelope with a
        top-level ``steps`` key.
    """
    if isinstance(data, dict) and isinstance(data.get('preprocessing'), dict):
        if 'steps' in data:
            raise ValidationError.from_exception_data(
                title='PreprocessingSpec',
                line_errors=[{
                    'type': 'value_error',
                    'loc': ('steps',),
                    'input': data.get('steps'),
                    'ctx': {'error': ValueError(
                        "ambiguous preprocessing config: both the "
                        "`preprocessing:` envelope and a top-level `steps:` "
                        "key are present — use one or the other"
                    )},
                }],
            )
        data = data['preprocessing']
    return PreprocessingSpec.model_validate(data)


class RejectionConfig(BaseModel):
    """Epoch rejection configuration."""
    # Amplitude-based rejection thresholds (max peak-to-peak)
    reject: Optional[Dict[str, float]] = Field(
        default=None,
        description="Max amplitude thresholds by channel type (e.g., {'eeg': 150e-6})"
    )

    # Flat signal detection thresholds (min peak-to-peak)
    flat: Optional[Dict[str, float]] = Field(
        default=None,
        description="Min amplitude thresholds for flat signal detection (e.g., {'eeg': 1e-6})"
    )

    # Rejection time window (within epoch)
    reject_tmin: Optional[float] = Field(
        default=None,
        description="Start of rejection window (None = epoch start)"
    )
    reject_tmax: Optional[float] = Field(
        default=None,
        description="End of rejection window (None = epoch end)"
    )

    # Annotation-based rejection
    reject_by_annotation: bool = Field(
        default=True,
        description="Reject epochs overlapping BAD annotations"
    )

    # AutoReject (data-driven)
    use_autoreject: bool = Field(
        default=False,
        description="Use autoreject library for data-driven rejection"
    )
    autoreject_n_interpolate: List[int] = Field(
        default=[1, 4, 8, 16],
        description="Number of channels to interpolate in autoreject"
    )

    # Global rejection strategy
    strategy: str = Field(
        default="threshold",
        description="Rejection strategy: threshold, autoreject, both"
    )

    # Default presets
    @classmethod
    def lenient(cls) -> "RejectionConfig":
        """Lenient rejection: keep more data."""
        return cls(
            reject={"eeg": 200e-6},
            flat={"eeg": 0.5e-6},
            strategy="threshold"
        )

    @classmethod
    def strict(cls) -> "RejectionConfig":
        """Strict rejection: cleaner data."""
        return cls(
            reject={"eeg": 100e-6},
            flat={"eeg": 1e-6},
            strategy="threshold"
        )

    @classmethod
    def adaptive(cls) -> "RejectionConfig":
        """Adaptive rejection using autoreject."""
        return cls(
            use_autoreject=True,
            strategy="autoreject"
        )


class EpochsConfig(BaseModel):
    """Epoch extraction configuration."""

    # Stage and variant info
    stage: str = Field(default="epochs", description="Processing stage")
    variant: str = Field(default="standard", description="Configuration variant name")

    # Dependencies
    depends_on: Dict[str, str] = Field(
        default_factory=dict,
        description="Dependencies on previous stages (preprocessing, task)"
    )

    tmin: float = Field(default=-0.5, description="Epoch start time (seconds)")
    tmax: float = Field(default=1.0, description="Epoch end time (seconds)")
    baseline: Optional[tuple[float, float]] = Field(
        default=(-0.2, 0.0),
        description="Baseline correction window"
    )
    event_id: Dict[str, int] = Field(
        default_factory=dict,
        description="Event name to code mapping"
    )

    # Rejection configuration
    rejection: RejectionConfig = Field(
        default_factory=RejectionConfig,
        description="Epoch rejection settings"
    )

    # Detrending
    detrend: Optional[int] = Field(
        default=None,
        description="Detrend type: 0=constant (DC), 1=linear, None=no detrend"
    )

    # Decimation
    decim: int = Field(
        default=1,
        description="Decimation factor to reduce sampling rate"
    )

    # QC generation
    generate_qc: bool = Field(
        default=True,
        description="Generate QC reports"
    )

    # Notes
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes about this configuration"
    )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "EpochsConfig":
        """Load EpochsConfig from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file

        Returns
        -------
        EpochsConfig
            Loaded configuration
        """
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ForwardConfig(BaseModel):
    """Forward model configuration for source reconstruction.

    ``extra='forbid'`` (P2-F5): the nested ``forward`` section must actually be
    consumed; unknown/misplaced keys (e.g. a flat top-level ``spacing``) raise
    instead of being silently ignored.

    F4 (final-verification-audit fix round): ``template`` is constrained to
    ``Literal['fsaverage']``. The source module's template forward, BEM,
    head->MRI transform and CONN labels are all fsaverage-specific and the
    module never read the field, so any other value was silently ignored — now
    it is rejected at validation instead of pretending to select a template.
    """

    model_config = ConfigDict(extra="forbid")

    template: Literal["fsaverage"] = Field(
        default="fsaverage",
        description="Template head model (fsaverage only)",
    )
    spacing: str = Field(default="oct6", description="Source space spacing (oct5, oct6, ico4, ico5)")


class InverseConfig(BaseModel):
    """Inverse operator configuration for source reconstruction.

    ``extra='forbid'`` (P2-F5). ``snr`` drives the evoked lambda2 (1/snr**2);
    ``snr_trial`` drives the single-trial lambda2 (wired in B5).
    """

    model_config = ConfigDict(extra="forbid")

    snr: float = Field(default=3.0, description="Signal-to-noise ratio (evoked)")
    snr_trial: float = Field(default=1.0, description="Signal-to-noise ratio (single trials)")
    loose: float = Field(default=0.2, description="Loose orientation constraint (0-1, 0=fixed, 1=free)")
    depth: float = Field(default=0.8, description="Depth weighting (0-1)")
    method_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extra method parameters passed to apply_inverse (e.g. eLORETA force_equal)",
    )


class SourceOutputConfig(BaseModel):
    """Source reconstruction output configuration (``extra='forbid'``, P2-F5)."""

    model_config = ConfigDict(extra="forbid")

    save_stc: bool = Field(default=True, description="Save full source estimates")
    save_roi_tc: bool = Field(default=True, description="Save ROI time courses")
    generate_qc: bool = Field(default=True, description="Generate QC report")
    compute_crosstalk: bool = Field(
        default=True, description="Compute ROI x ROI crosstalk for QC (B7)"
    )


class SourceConfig(BaseModel):
    """Source reconstruction configuration.

    One schema for the source stage (P2-F5): ``extra='forbid'`` so the two legacy
    dialects (flat ``snr``/``spacing``/``trial_level`` keys, or unlisted keys) are
    hard errors instead of silently-ignored parameters. ``variant`` defaults to
    ``f"{method}-CONN32"`` and must match ``method`` when given.
    """

    model_config = ConfigDict(extra="forbid")

    stage: str = Field(default="source", description="Processing stage")
    variant: Optional[str] = Field(
        default=None, description="Variant name (e.g., 'dSPM-CONN32'); defaults to '{method}-CONN32'"
    )
    description: Optional[str] = Field(default=None, description="Config description")

    depends_on: Dict[str, str] = Field(
        default_factory=dict,
        description="Dependencies: {preprocessing: variant, task: name, epochs: variant}"
    )

    method: Literal["dSPM", "sLORETA", "eLORETA", "MNE"] = Field(
        default="sLORETA",
        description="Inverse method: dSPM, sLORETA, eLORETA, MNE"
    )

    forward: ForwardConfig = Field(default_factory=ForwardConfig, description="Forward model parameters")
    inverse: InverseConfig = Field(default_factory=InverseConfig, description="Inverse operator parameters")

    parcellation: Literal["conn_networks"] = Field(
        default="conn_networks",
        description="Parcellation: conn_networks (32 ROIs) is the only implemented scheme",
    )
    roi_radius: float = Field(default=10.0, description="ROI extraction radius in mm")
    roi_radius_midline: float = Field(
        default=15.0, description="ROI extraction radius in mm for midline ROIs (|x| <= 5 mm)"
    )
    roi_radius_overrides: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-ROI extraction radius overrides keyed by ROI full name",
    )
    roi_min_vertices: int = Field(
        default=3, description="Minimum vertices per ROI label; smaller labels raise"
    )
    subjects_dir: Optional[Path] = Field(
        default=None, description="Subjects directory for the source space (default: cached fsaverage)"
    )

    mode: Literal["evoked", "trials", "both"] = Field(
        default="evoked",
        description="Output mode: evoked ROI ERP (2-D), single trials (3-D), or both",
    )
    min_trials: int = Field(
        default=10,
        ge=1,
        description="Minimum trials per condition; fewer are skipped loudly",
    )
    equalize_trials: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Randomly select this many trials per condition (seeded) to remove "
            "the sqrt(nave) confound; must be >= 1 (0 would write zero-trial arrays)"
        ),
    )
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for equalize_trials selection (numpy Generator seeds are non-negative)",
    )

    subjects: Any = Field(default="all", description="Subject selection ('all' or list of subject IDs)")
    output: SourceOutputConfig = Field(default_factory=SourceOutputConfig, description="Output options")

    @model_validator(mode="after")
    def _derive_and_check_variant(self) -> "SourceConfig":
        """Default ``variant`` to ``{method}-CONN32``; reject method mismatches."""
        if self.variant is None:
            self.variant = f"{self.method}-CONN32"
        elif self.variant.split("-")[0].lower() != self.method.lower():
            raise ValueError(
                f"variant {self.variant!r} does not match method {self.method!r}; "
                f"the variant must start with the method (e.g. '{self.method}-CONN32')"
            )
        return self

    @classmethod
    def from_yaml(cls, config_path: Path) -> "SourceConfig":
        """Load SourceConfig from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to YAML configuration file

        Returns
        -------
        SourceConfig
            Loaded and validated configuration
        """
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ConnectivityConfig(BaseModel):
    """Connectivity analysis configuration."""

    methods: List[str] = Field(
        default=["plv", "correlation"],
        description="Connectivity methods to compute"
    )
    frequency_bands: Dict[str, tuple[float, float]] = Field(
        default={
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "gamma": (30, 45),
        }
    )
    time_windows: List[Dict[str, Any]] = Field(
        default=[
            {"name": "prestim", "tmin": -0.5, "tmax": 0.0},
            {"name": "poststim", "tmin": 0.0, "tmax": 0.5},
        ]
    )

    # ROI selection
    use_conn_rois: bool = Field(
        default=True,
        description="Use CONN 32-ROI template"
    )


class PredictionConfig(BaseModel):
    """Prediction/CPM configuration."""

    prediction_type: str = Field(
        default="between_subject",
        description="within_subject, between_subject, or between_group"
    )
    target_variable: str = Field(..., description="Target variable name in behavioral_scores")

    # Features
    feature_sources: List[str] = Field(
        default=["connectivity"],
        description="Feature sources: connectivity, erp, timefreq"
    )
    feature_windows: List[str] = Field(
        default=["poststim"],
        description="Which time windows to use for features"
    )

    # Cross-validation
    cv_strategy: str = Field(
        default="kfold",
        description="kfold, leave_one_out, stratified_kfold"
    )
    n_folds: int = Field(default=5)

    # Edge selection (CPM-specific)
    edge_selection_threshold: float = Field(
        default=0.05,
        description="P-value threshold for edge selection"
    )


class EvaluationConfig(BaseModel):
    """Model evaluation configuration."""

    models: List[Dict[str, Any]] = Field(
        default=[
            {"type": "ridge", "alpha": [0.1, 1.0, 10.0, 100.0]},
            {"type": "svm", "C": [0.1, 1.0, 10.0]},
        ]
    )
    metrics: List[str] = Field(
        default=["r2", "mse", "pearson_r"],
        description="Evaluation metrics"
    )
    permutation_test: bool = Field(default=False)
    n_permutations: int = Field(default=1000)


class PartitionConfig(BaseModel):
    """One SLURM partition with its per-array-task resources.

    Used by the batch config (``HpcBatchConfig.slurm.partitions``): each
    partition carries the resources its array tasks request and the
    concurrency cap (``%k``) for it. There is exactly one partition schema —
    do not add a second one.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="SLURM partition name")
    max_concurrent: int = Field(
        default=1, ge=1,
        description="Maximum array tasks running at once on this partition (%k)",
    )
    cpus: int = Field(default=4, ge=1, description="CPUs per array task")
    mem: str = Field(default="16G", description="Memory per array task")
    time: str = Field(default="02:00:00", description="Wall time per array task (HH:MM:SS)")


class SlurmConfig(BaseModel):
    """SLURM HPC configuration.

    ``extra='forbid'`` (the repo-wide config invariant): a misspelled key in a
    cluster YAML is a hard error, never a silently ignored default.
    """

    model_config = ConfigDict(extra="forbid")

    partition: str = Field(default="normal")
    time: str = Field(default="04:00:00")
    mem: str = Field(default="32G")
    cpus_per_task: int = Field(default=8)
    gpus: Optional[int] = Field(default=None)
    modules: List[str] = Field(default_factory=list, description="Modules to load")
    partitions: List[PartitionConfig] = Field(
        default_factory=list,
        description="Per-partition resources/concurrency for the batch config",
    )
    venv: Optional[str] = Field(
        default=None,
        description="Virtualenv to activate/use (None = the submitter's interpreter)",
    )
    exports: Dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables exported by the generated scripts",
    )


class HpcBatchConfig(BaseModel):
    """Batch YAML schema for ``eegcpm hpc`` (the only HPC batch config).

    Unknown keys are **hard errors** (``extra='forbid'``): a silently ignored
    typo in a cluster config would submit the wrong job. Paths are resolved to
    absolute paths against ``project`` (``project`` itself is resolved against
    the YAML file's directory by :meth:`from_yaml`), because the generated
    sbatch scripts run on compute nodes with an arbitrary cwd.

    ``stages`` are the per-unit array stages; ``features`` is deliberately not
    allowed there — it is the task-level ``hpc finalize`` step.
    """

    model_config = ConfigDict(extra="forbid")

    slurm: SlurmConfig = Field(default_factory=SlurmConfig)
    project: Path = Field(..., description="Project root directory")
    preprocessing_config: Path = Field(..., description="Preprocessing config YAML")
    source_config: Path = Field(..., description="Source reconstruction config YAML")
    features_config: Path = Field(..., description="Features config YAML (finalize step)")
    stages: List[Literal["preprocess", "epochs", "source"]] = Field(
        ..., description="Per-unit array stages (features runs via finalize)"
    )
    logs_dir: Optional[Path] = Field(
        default=None, description="SLURM log dir (default: <project>/hpc/logs)"
    )
    state_db_dir: Optional[Path] = Field(
        default=None, description="Per-unit state DB dir (default: <project>/hpc/state)"
    )
    submissions_dir: Optional[Path] = Field(
        default=None,
        description="Chunk/script dir (default: <project>/hpc/submissions)",
    )

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> "HpcBatchConfig":
        """Make every path absolute; fill the optional dirs from the project."""
        project = Path(self.project).expanduser()
        self.project = project

        for field_name, default in (
            ("logs_dir", project / "hpc" / "logs"),
            ("state_db_dir", project / "hpc" / "state"),
            ("submissions_dir", project / "hpc" / "submissions"),
            ("preprocessing_config", None),
            ("source_config", None),
            ("features_config", None),
        ):
            value = getattr(self, field_name)
            value = Path(value).expanduser() if value is not None else default
            if value is not None and not value.is_absolute():
                value = project / value
            setattr(self, field_name, value)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "HpcBatchConfig":
        """Load and validate a batch YAML file.

        A relative ``project`` is resolved against the YAML file's directory
        (so a config shipped next to a project still points at that project),
        then every other path is resolved against the project.
        """
        import yaml

        path = Path(path).expanduser()
        try:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
        except OSError as exc:
            raise FileNotFoundError(f"batch config not found: {path} ({exc})") from exc
        except yaml.YAMLError as exc:  # YAMLError is not a ValueError
            raise ValueError(f"{path}: invalid YAML: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: expected a YAML mapping, got {type(data).__name__}"
            )
        project = data.get("project")
        if project is not None:
            project = Path(str(project)).expanduser()
            if not project.is_absolute():
                project = (path.parent / project).resolve()
            data["project"] = project
        return cls(**data)



class Config(BaseModel):
    """Master configuration combining all sub-configs."""

    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    epochs: EpochsConfig = Field(default_factory=EpochsConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    connectivity: ConnectivityConfig = Field(default_factory=ConnectivityConfig)
    prediction: PredictionConfig = Field(default_factory=lambda: PredictionConfig(target_variable="score"))
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    slurm: SlurmConfig = Field(default_factory=SlurmConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
