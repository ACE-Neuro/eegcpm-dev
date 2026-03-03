"""Project management for EEGCPM UI."""

from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yaml


class Project:
    """EEGCPM project configuration."""

    def __init__(
        self,
        name: str,
        bids_root: str,
        eegcpm_root: str,
        last_accessed: Optional[str] = None
    ):
        self.name = name
        self.bids_root = str(Path(bids_root).resolve())
        self.eegcpm_root = str(Path(eegcpm_root).resolve())
        self.last_accessed = last_accessed or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'bids_root': self.bids_root,
            'eegcpm_root': self.eegcpm_root,
            'last_accessed': self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Project':
        """Create from dictionary."""
        return cls(**data)

    def validate(self) -> tuple[bool, Optional[str]]:
        """Validate project paths exist.

        Returns:
            (is_valid, error_message)
        """
        bids_path = Path(self.bids_root)
        eegcpm_path = Path(self.eegcpm_root)

        if not bids_path.exists():
            return False, f"BIDS root not found: {self.bids_root}"

        # EEGCPM root can be created if it doesn't exist
        if not eegcpm_path.exists():
            try:
                eegcpm_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"Cannot create EEGCPM root: {e}"

        return True, None


class ProjectManager:
    """Manages EEGCPM project configurations."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize project manager.

        Args:
            config_path: Path to projects.yaml (default: ~/.eegcpm/projects.yaml)
        """
        if config_path is None:
            config_path = Path.home() / ".eegcpm" / "projects.yaml"

        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing projects
        self.projects: List[Project] = []
        self.load()

    def load(self):
        """Load projects from config file."""
        if not self.config_path.exists():
            self.projects = []
            return

        try:
            with open(self.config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            projects_data = data.get('projects', [])
            self.projects = [Project.from_dict(p) for p in projects_data]

        except Exception as e:
            print(f"Warning: Could not load projects: {e}")
            self.projects = []

    def save(self):
        """Save projects to config file."""
        data = {
            'projects': [p.to_dict() for p in self.projects]
        }

        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving projects: {e}")

    def add_project(self, project: Project) -> bool:
        """Add or update a project.

        Args:
            project: Project to add

        Returns:
            True if added successfully
        """
        # Check if project name already exists
        existing = self.get_project(project.name)
        if existing:
            # Update existing project
            self.projects.remove(existing)

        self.projects.append(project)
        self.save()
        return True

    def get_project(self, name: str) -> Optional[Project]:
        """Get project by name.

        Args:
            name: Project name

        Returns:
            Project if found, None otherwise
        """
        for project in self.projects:
            if project.name == name:
                return project
        return None

    def get_projects_sorted(self) -> List[Project]:
        """Get projects sorted by last accessed (most recent first).

        Returns:
            List of projects sorted by recency
        """
        return sorted(
            self.projects,
            key=lambda p: p.last_accessed,
            reverse=True
        )

    def update_last_accessed(self, name: str):
        """Update last accessed timestamp for a project.

        Args:
            name: Project name
        """
        project = self.get_project(name)
        if project:
            project.last_accessed = datetime.now().isoformat()
            self.save()

    def delete_project(self, name: str) -> bool:
        """Delete a project from history (does not delete files).

        Args:
            name: Project name

        Returns:
            True if deleted successfully
        """
        project = self.get_project(name)
        if project:
            self.projects.remove(project)
            self.save()
            return True
        return False

    def rename_project(self, old_name: str, new_name: str) -> bool:
        """Rename a project.

        Args:
            old_name: Current project name
            new_name: New project name

        Returns:
            True if renamed successfully
        """
        # Check if new name already exists
        if self.get_project(new_name):
            return False

        project = self.get_project(old_name)
        if project:
            project.name = new_name
            self.save()
            return True
        return False
