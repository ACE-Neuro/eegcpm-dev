"""
Base module interface for EEGCPM pipeline.

All analysis modules inherit from BaseModule.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic

import mne


@dataclass
class ModuleResult:
    """Result container for module execution."""

    success: bool
    module_name: str
    execution_time_seconds: float
    outputs: Dict[str, Any] = field(default_factory=dict)
    output_files: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"ModuleResult({self.module_name}: {status}, {self.execution_time_seconds:.2f}s)"


class BaseModule(ABC):
    """
    Abstract base class for all analysis modules.

    Subclasses must implement:
        - name: Module identifier
        - process(): Main processing logic
        - validate_input(): Input validation

    Optional overrides:
        - get_output_spec(): Define expected outputs
        - cleanup(): Resource cleanup
    """

    name: str = "base_module"
    version: str = "0.1.0"
    description: str = "Base module"

    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize module.

        Args:
            config: Module configuration dict
            output_dir: Directory for output files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data before processing.

        Args:
            data: Input data (type depends on module)

        Returns:
            True if valid, raises ValueError otherwise
        """
        pass

    @abstractmethod
    def process(self, data: Any, **kwargs) -> ModuleResult:
        """
        Execute the module's main processing.

        Args:
            data: Input data
            **kwargs: Additional arguments

        Returns:
            ModuleResult with outputs and status
        """
        pass

    def get_output_spec(self) -> Dict[str, str]:
        """
        Define expected outputs.

        Returns:
            Dict mapping output name to description
        """
        return {}

    def cleanup(self) -> None:
        """Release resources after processing."""
        pass

    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get data needed to resume from checkpoint.

        Returns:
            Serializable dict for checkpoint
        """
        return {
            "module_name": self.name,
            "version": self.version,
            "config": self.config,
            "output_dir": str(self.output_dir),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_data: Dict[str, Any]) -> "BaseModule":
        """
        Restore module from checkpoint data.

        Args:
            checkpoint_data: Data from get_checkpoint_data()

        Returns:
            Restored module instance
        """
        return cls(
            config=checkpoint_data["config"],
            output_dir=Path(checkpoint_data["output_dir"]),
        )


class RawDataModule(BaseModule):
    """Base class for modules that process Raw data."""

    def validate_input(self, data: Any) -> bool:
        if not isinstance(data, mne.io.BaseRaw):
            raise ValueError(f"Expected mne.io.Raw, got {type(data)}")
        return True


class EpochsModule(BaseModule):
    """Base class for modules that process Epochs."""

    def validate_input(self, data: Any) -> bool:
        # read_epochs() returns EpochsFIF, which is not an instance of mne.Epochs
        # (P2-F13); mne.BaseEpochs covers Epochs, EpochsArray, EpochsFIF.
        if not isinstance(data, mne.BaseEpochs):
            raise ValueError(f"Expected mne.Epochs (mne.BaseEpochs), got {type(data)}")
        return True


class SourceModule(BaseModule):
    """Base class for modules that process source estimates."""

    def validate_input(self, data: Any) -> bool:
        if not isinstance(data, (mne.SourceEstimate, mne.VectorSourceEstimate)):
            raise ValueError(f"Expected SourceEstimate, got {type(data)}")
        return True
