"""
Global configuration for EEGCPM.

Stage-First Architecture:
- Each stage config can specify dependencies on upstream stages
- Config locking prevents accidental reuse across different dependencies
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
    """Forward model configuration for source reconstruction."""
    template: str = Field(default="fsaverage", description="Template head model")
    spacing: str = Field(default="oct6", description="Source space spacing (oct5, oct6, ico4, ico5)")


class InverseConfig(BaseModel):
    """Inverse operator configuration for source reconstruction."""
    snr: float = Field(default=3.0, description="Signal-to-noise ratio")
    loose: float = Field(default=0.2, description="Loose orientation constraint (0-1, 0=fixed, 1=free)")
    depth: float = Field(default=0.8, description="Depth weighting (0-1)")


class SourceOutputConfig(BaseModel):
    """Source reconstruction output configuration."""
    save_stc: bool = Field(default=True, description="Save full source estimates")
    save_roi_tc: bool = Field(default=True, description="Save ROI time courses")
    generate_qc: bool = Field(default=True, description="Generate QC report")


class SourceConfig(BaseModel):
    """Source reconstruction configuration."""

    stage: str = Field(default="source", description="Processing stage")
    variant: str = Field(default="sLORETA-CONN32", description="Variant name (e.g., 'dSPM-CONN32')")

    depends_on: Dict[str, str] = Field(
        default_factory=dict,
        description="Dependencies: {preprocessing: variant, task: name, epochs: variant}"
    )

    method: str = Field(
        default="sLORETA",
        description="Inverse method: dSPM, sLORETA, eLORETA, MNE"
    )

    forward: ForwardConfig = Field(default_factory=ForwardConfig, description="Forward model parameters")
    inverse: InverseConfig = Field(default_factory=InverseConfig, description="Inverse operator parameters")

    parcellation: str = Field(
        default="conn_networks",
        description="Parcellation: conn_networks (32 ROIs), aparc (68), schaefer100, etc."
    )
    roi_radius: float = Field(default=10.0, description="ROI extraction radius in mm")

    subjects: Any = Field(default="all", description="Subject selection ('all' or list of subject IDs)")
    output: SourceOutputConfig = Field(default_factory=SourceOutputConfig, description="Output options")

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


class SlurmConfig(BaseModel):
    """SLURM HPC configuration."""

    partition: str = Field(default="normal")
    time: str = Field(default="04:00:00")
    mem: str = Field(default="32G")
    cpus_per_task: int = Field(default=8)
    gpus: Optional[int] = Field(default=None)
    modules: List[str] = Field(default_factory=list, description="Modules to load")


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
