"""
Processing State Management

Handles reading/writing processing_state.json files for each run.
These files enable distributed processing (HPC) and UI synchronization.

Author: EEGCPM Development Team
Created: 2025-01
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class ProcessingState:
    """
    Processing state for a single run.

    Written to processing_state.json in each run's output directory.
    Enables CLI/UI synchronization and distributed HPC processing.

    Attributes
    ----------
    subject_id : str
        Subject identifier
    session : str
        Session identifier (e.g., 'ses-01')
    task : str
        Task name (e.g., 'contdet', 'rest')
    run : str
        Run identifier (e.g., 'run-1')
    status : str
        Processing status: 'pending', 'in_progress', 'completed',
        'failed', 'excluded'
    started_at : str
        ISO timestamp when processing started
    completed_at : str, optional
        ISO timestamp when processing reached a final status
        ('completed', 'failed' or 'excluded')
    source : str
        Processing source: 'CLI' or 'UI'
    logs : list of str
        Processing log messages
    metadata : dict
        Step-by-step processing metadata
    quality : str, optional
        Quality assessment: 'excellent', 'good', 'acceptable', 'poor', 'reject'
    error : str, optional
        Error message for a final non-completed status ('failed' or
        'excluded')
    config : dict, optional
        Preprocessing configuration used
    output_files : dict, optional
        Output file paths
    """

    subject_id: str
    session: str
    task: str
    run: str
    status: str
    started_at: str
    source: str = "UI"
    completed_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality: Optional[str] = None
    error: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    output_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, filtering out non-JSON-serializable objects."""
        data = asdict(self)

        # Filter out non-serializable objects in metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = self._filter_serializable(data['metadata'])

        return data

    def _filter_serializable(self, obj: Any) -> Any:
        """Recursively filter out non-JSON-serializable objects."""
        if isinstance(obj, dict):
            return {
                k: self._filter_serializable(v)
                for k, v in obj.items()
                if not k.endswith('_object')  # Skip *_object keys (like 'ica_object')
            }
        elif isinstance(obj, (list, tuple)):
            return [self._filter_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Non-serializable object, convert to string representation
            return str(obj)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingState':
        """Create from dictionary."""
        return cls(**data)

    def save(self, output_dir: Path) -> Path:
        """
        Save state to processing_state.json in output directory.

        Parameters
        ----------
        output_dir : Path
            Run output directory (e.g., .../sub-001/ses-01/task-rest/run-1/)

        Returns
        -------
        state_file : Path
            Path to saved state file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        state_file = output_dir / "processing_state.json"

        with open(state_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

        return state_file

    @classmethod
    def load(cls, output_dir: Path) -> Optional['ProcessingState']:
        """
        Load state from processing_state.json.

        Parameters
        ----------
        output_dir : Path
            Run output directory

        Returns
        -------
        state : ProcessingState or None
            Loaded state, or None if file doesn't exist
        """
        state_file = Path(output_dir) / "processing_state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Warning: Failed to load state from {state_file}: {e}")
            return None

    @classmethod
    def scan_directory(cls, root_dir: Path, pattern: str = "**/processing_state.json") -> List['ProcessingState']:
        """
        Scan directory tree for all processing_state.json files.

        Parameters
        ----------
        root_dir : Path
            Root directory to scan
        pattern : str
            Glob pattern for state files

        Returns
        -------
        states : list of ProcessingState
            All found processing states
        """
        root_dir = Path(root_dir)
        states = []

        for state_file in root_dir.glob(pattern):
            state = cls.load(state_file.parent)
            if state is not None:
                states.append(state)

        return states

    def update_status(self, status: str, error: Optional[str] = None):
        """Update processing status.

        A final status (``completed``/``failed``/``excluded``) always
        stamps ``completed_at``, and a passed ``error`` is always
        stored — including for an exclusion, which some consumers show
        as the run's failure reason. A ``None`` error never clobbers a
        previous one.
        """
        self.status = status

        if error:
            self.error = error

        if status in ('completed', 'failed', 'excluded'):
            self.completed_at = datetime.now().isoformat()

    def add_log(self, message: str):
        """Add log message."""
        self.logs.append(message)

    def set_metadata(self, metadata: Dict[str, Any]):
        """Set processing metadata."""
        self.metadata = metadata

    def set_quality(self, quality: str):
        """Set quality assessment."""
        self.quality = quality


def create_processing_state(
    subject_id: str,
    session: str,
    task: str,
    run: str,
    source: str = "UI",
    config: Optional[Dict[str, Any]] = None
) -> ProcessingState:
    """
    Create a new processing state.

    Parameters
    ----------
    subject_id : str
        Subject identifier
    session : str
        Session identifier
    task : str
        Task name
    run : str
        Run identifier
    source : str
        Processing source ('CLI' or 'UI')
    config : dict, optional
        Preprocessing configuration

    Returns
    -------
    state : ProcessingState
        New processing state with status='in_progress'
    """
    return ProcessingState(
        subject_id=subject_id,
        session=session,
        task=task,
        run=run,
        status='in_progress',
        started_at=datetime.now().isoformat(),
        source=source,
        config=config,
    )
