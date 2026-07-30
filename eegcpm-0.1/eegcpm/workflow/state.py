"""Workflow state management for tracking processing progress.

This module provides a SQLite-based state tracking system for EEGCPM pipelines,
enabling resume functionality, progress monitoring, and workflow coordination.
"""

import sqlite3
import json
import copy
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from enum import Enum


def _make_json_serializable(obj: Any) -> Any:
    """Recursively remove non-JSON-serializable objects from nested structures.

    Removes:
    - MNE ICA objects
    - Any other non-serializable objects (converted to type name string)
    """
    if obj is None:
        return None

    # Check if it's an MNE ICA object
    if hasattr(obj, '__class__') and obj.__class__.__name__ == 'ICA':
        return None  # Remove ICA objects

    # Handle dictionaries recursively
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items() if _make_json_serializable(v) is not None}

    # Handle lists/tuples recursively
    if isinstance(obj, (list, tuple)):
        filtered = [_make_json_serializable(item) for item in obj]
        # Remove None values from lists
        return [item for item in filtered if item is not None]

    # Handle primitives and JSON-serializable types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Try to serialize - if it fails, return type name
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return f"<non-serializable: {type(obj).__name__}>"


class ProcessingStatus(str, Enum):
    """Status of a processing workflow."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    """Record of a single processing step."""
    step_name: str
    status: ProcessingStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Complete state of a subject-task-run processing workflow."""
    subject_id: str
    task: str
    pipeline: str
    status: ProcessingStatus
    session: Optional[str] = None  # e.g., "01"
    run: Optional[str] = None  # e.g., "01", "02", or None for combined
    steps: List[StepRecord] = field(default_factory=list)
    config_hash: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # NEW: Stage tracking for stage-first architecture
    current_stage: str = "preprocessing"  # preprocessing, epochs, source, features, prediction
    metadata: Dict[str, Any] = field(default_factory=dict)  # Stage-specific metadata

    # R-015: condition-level and run-level identity. `condition` separates
    # feature conditions (resting_ec / resting_eo / despicable_me ...) for
    # subject-level stages; `run_id` separates batch runs (e.g. prediction
    # runs) that have no natural subject. Both default to "" (legacy rows).
    condition: Optional[str] = None
    run_id: Optional[str] = None

    @property
    def is_combined(self) -> bool:
        """Check if this is a combined workflow (aggregates multiple runs)."""
        return self.run is None or self.run == "combined"

    @property
    def workflow_key(self) -> str:
        """Get unique workflow key."""
        parts = [self.subject_id, self.task, self.pipeline]
        if self.session:
            parts.insert(1, f"ses-{self.session}")
        if self.run:
            parts.insert(-1, f"run-{self.run}")
        return "/".join(parts)

    def get_step(self, step_name: str) -> Optional[StepRecord]:
        """Get step record by name."""
        for step in self.steps:
            if step.step_name == step_name:
                return step
        return None

    def add_step(self, step: StepRecord) -> None:
        """Add or update a step record."""
        existing = self.get_step(step.step_name)
        if existing:
            self.steps.remove(existing)
        self.steps.append(step)
        self.updated_at = datetime.now()

    def get_completed_steps(self) -> List[str]:
        """Get list of completed step names."""
        return [s.step_name for s in self.steps if s.status == ProcessingStatus.COMPLETED]

    def get_failed_steps(self) -> List[str]:
        """Get list of failed step names."""
        return [s.step_name for s in self.steps if s.status == ProcessingStatus.FAILED]


class WorkflowStateManager:
    """Manager for workflow state persistence using SQLite."""

    def __init__(self, db_path: Path):
        """
        Initialize state manager.

        Parameters
        ----------
        db_path : Path
            Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # R-015 migration: the identity key must include condition and
            # run_id. SQLite cannot alter a table-level UNIQUE constraint,
            # so migrate by rebuild when the columns are missing.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(workflows)")}
            if cols and "condition" not in cols:
                conn.execute("ALTER TABLE workflows RENAME TO workflows_old")
                conn.execute("""
                    CREATE TABLE workflows (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject_id TEXT NOT NULL,
                        session TEXT,
                        task TEXT NOT NULL,
                        run TEXT,
                        pipeline TEXT NOT NULL,
                        status TEXT NOT NULL,
                        config_hash TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        current_stage TEXT DEFAULT 'preprocessing',
                        metadata TEXT,
                        condition TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL DEFAULT '',
                        UNIQUE(subject_id, session, task, run, pipeline, condition, run_id)
                    )
                """)
                conn.execute("""
                    INSERT INTO workflows (id, subject_id, session, task, run,
                        pipeline, status, config_hash, created_at, updated_at,
                        current_stage, metadata, condition, run_id)
                    SELECT id, subject_id, session, task, run, pipeline,
                        status, config_hash, created_at, updated_at,
                        current_stage, metadata, '', ''
                    FROM workflows_old
                """)
                conn.execute("DROP TABLE workflows_old")

            # Create workflows table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject_id TEXT NOT NULL,
                    session TEXT,
                    task TEXT NOT NULL,
                    run TEXT,
                    pipeline TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_stage TEXT DEFAULT 'preprocessing',
                    metadata TEXT,
                    condition TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    UNIQUE(subject_id, session, task, run, pipeline, condition, run_id)
                )
            """)

            # Migration: Add new columns to existing tables if they don't exist
            try:
                conn.execute("ALTER TABLE workflows ADD COLUMN current_stage TEXT DEFAULT 'preprocessing'")
            except sqlite3.OperationalError:
                pass  # Column already exists

            try:
                conn.execute("ALTER TABLE workflows ADD COLUMN metadata TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists

            for col, ddl in (("condition", "TEXT NOT NULL DEFAULT ''"),
                             ("run_id", "TEXT NOT NULL DEFAULT ''")):
                try:
                    conn.execute(f"ALTER TABLE workflows ADD COLUMN {col} {ddl}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            conn.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id INTEGER NOT NULL,
                    step_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    output_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY (workflow_id) REFERENCES workflows(id),
                    UNIQUE(workflow_id, step_name)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subject_task
                ON workflows(subject_id, session, task)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON workflows(status)
            """)

            conn.commit()

    def save_state(self, state: WorkflowState) -> None:
        """
        Save workflow state to database.

        Parameters
        ----------
        state : WorkflowState
            Workflow state to save
        """
        with sqlite3.connect(self.db_path) as conn:
            # Update timestamp
            state.updated_at = datetime.now()
            if state.created_at is None:
                state.created_at = state.updated_at

            # Serialize metadata
            metadata_json = json.dumps(state.metadata) if state.metadata else None

            # Insert or update workflow
            cursor = conn.execute("""
                INSERT INTO workflows (subject_id, session, task, run, pipeline, status, config_hash,
                                      created_at, updated_at, current_stage, metadata, condition, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id, session, task, run, pipeline, condition, run_id)
                DO UPDATE SET
                    status=excluded.status,
                    config_hash=excluded.config_hash,
                    updated_at=excluded.updated_at,
                    current_stage=excluded.current_stage,
                    metadata=excluded.metadata
                RETURNING id
            """, (
                state.subject_id,
                # NULL session/run defeat the UNIQUE key (SQLite treats
                # NULLs as distinct) — store '' instead (R-015).
                state.session if state.session is not None else "",
                state.task,
                state.run if state.run is not None else "",
                state.pipeline,
                state.status.value,
                state.config_hash,
                state.created_at,
                state.updated_at,
                state.current_stage,
                metadata_json,
                state.condition or "",
                state.run_id or "",
            ))

            workflow_id = cursor.fetchone()[0]

            # Delete existing steps
            conn.execute("DELETE FROM steps WHERE workflow_id = ?", (workflow_id,))

            # Insert steps
            for step in state.steps:
                # Prepare metadata - remove non-JSON-serializable objects
                serializable_metadata = None
                if step.metadata:
                    # Deep copy and recursively filter non-serializable objects
                    metadata_copy = copy.deepcopy(step.metadata)
                    metadata_clean = _make_json_serializable(metadata_copy)
                    serializable_metadata = json.dumps(metadata_clean)

                conn.execute("""
                    INSERT INTO steps
                    (workflow_id, step_name, status, started_at, completed_at,
                     error_message, output_path, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    workflow_id,
                    step.step_name,
                    step.status.value,
                    step.started_at,
                    step.completed_at,
                    step.error_message,
                    step.output_path,
                    serializable_metadata
                ))

            conn.commit()

    def load_state(
        self,
        subject_id: str,
        task: str,
        pipeline: str,
        session: Optional[str] = None,
        run: Optional[str] = None,
        condition: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[WorkflowState]:
        """
        Load workflow state from database.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        task : str
            Task name
        pipeline : str
            Pipeline name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier

        Returns
        -------
        WorkflowState or None
            Loaded state, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get workflow
            cursor = conn.execute("""
                SELECT * FROM workflows
                WHERE subject_id = ? AND task = ? AND pipeline = ?
                AND (session = ? OR (session IS NULL AND ? IS NULL)
                     OR (session = '' AND ? IS NULL))
                AND (run = ? OR (run IS NULL AND ? IS NULL)
                     OR (run = '' AND ? IS NULL))
                AND condition = ? AND run_id = ?
            """, (subject_id, task, pipeline, session, session, session,
                  run, run, run, condition or "", run_id or ""))

            row = cursor.fetchone()
            if row is None:
                return None

            workflow_id = row['id']

            # Get steps
            cursor = conn.execute("""
                SELECT * FROM steps WHERE workflow_id = ?
                ORDER BY id
            """, (workflow_id,))

            steps = []
            for step_row in cursor.fetchall():
                steps.append(StepRecord(
                    step_name=step_row['step_name'],
                    status=ProcessingStatus(step_row['status']),
                    started_at=datetime.fromisoformat(step_row['started_at']) if step_row['started_at'] else None,
                    completed_at=datetime.fromisoformat(step_row['completed_at']) if step_row['completed_at'] else None,
                    error_message=step_row['error_message'],
                    output_path=step_row['output_path'],
                    metadata=json.loads(step_row['metadata']) if step_row['metadata'] else {}
                ))

            # Deserialize metadata
            metadata = json.loads(row['metadata']) if row['metadata'] else {}

            return WorkflowState(
                subject_id=row['subject_id'],
                session=row['session'] if row['session'] else None,
                task=row['task'],
                run=row['run'] if row['run'] else None,
                pipeline=row['pipeline'],
                status=ProcessingStatus(row['status']),
                config_hash=row['config_hash'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                current_stage=row['current_stage'] if 'current_stage' in row.keys() else 'preprocessing',
                condition=row['condition'] if 'condition' in row.keys() else None,
                run_id=row['run_id'] if 'run_id' in row.keys() else None,
                metadata=metadata,
                steps=steps
            )

    def get_all_states(
        self,
        subject_id: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None,
        run: Optional[str] = None,
        pipeline: Optional[str] = None,
        status: Optional[ProcessingStatus] = None,
        condition: Optional[str] = None,
        run_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[WorkflowState]:
        """
        Get all workflow states matching filters.

        Parameters
        ----------
        subject_id : str, optional
            Filter by subject ID
        session : str, optional
            Filter by session
        task : str, optional
            Filter by task
        run : str, optional
            Filter by run
        pipeline : str, optional
            Filter by pipeline
        status : ProcessingStatus, optional
            Filter by status

        Returns
        -------
        List[WorkflowState]
            List of matching workflow states
        """
        query = "SELECT * FROM workflows WHERE 1=1"
        params = []

        if subject_id:
            query += " AND subject_id = ?"
            params.append(subject_id)
        if session:
            query += " AND session = ?"
            params.append(session)
        if task:
            query += " AND task = ?"
            params.append(task)
        if run:
            query += " AND run = ?"
            params.append(run)
        if pipeline:
            query += " AND pipeline = ?"
            params.append(pipeline)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        if condition is not None:
            query += " AND condition = ?"
            params.append(condition)
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if stage is not None:
            query += " AND current_stage = ?"
            params.append(stage)

        query += " ORDER BY subject_id, session, task, run"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            states = []
            for row in cursor.fetchall():
                state = self.load_state(
                    row['subject_id'],
                    row['task'],
                    row['pipeline'],
                    session=row['session'],
                    run=row['run'],
                    condition=row['condition'],
                    run_id=row['run_id'],
                )
                if state:
                    states.append(state)

            return states

    def delete_state(
        self,
        subject_id: str,
        task: str,
        pipeline: str,
        session: Optional[str] = None,
        run: Optional[str] = None
    ) -> bool:
        """
        Delete workflow state.

        Parameters
        ----------
        subject_id : str
            Subject identifier
        task : str
            Task name
        pipeline : str
            Pipeline name
        session : str, optional
            Session identifier
        run : str, optional
            Run identifier

        Returns
        -------
        bool
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM workflows
                WHERE subject_id = ? AND task = ? AND pipeline = ?
                AND (session = ? OR (session IS NULL AND ? IS NULL))
                AND (run = ? OR (run IS NULL AND ? IS NULL))
            """, (subject_id, task, pipeline, session, session, run, run))
            conn.commit()
            return cursor.rowcount > 0

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all workflows.

        Returns
        -------
        dict
            Summary with counts by status, subject, task, pipeline
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Count by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM workflows
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Count subjects
            cursor = conn.execute("SELECT COUNT(DISTINCT subject_id) FROM workflows")
            n_subjects = cursor.fetchone()[0]

            # Count tasks
            cursor = conn.execute("SELECT COUNT(DISTINCT task) FROM workflows")
            n_tasks = cursor.fetchone()[0]

            # Count pipelines
            cursor = conn.execute("SELECT COUNT(DISTINCT pipeline) FROM workflows")
            n_pipelines = cursor.fetchone()[0]

            # Total workflows
            cursor = conn.execute("SELECT COUNT(*) FROM workflows")
            n_total = cursor.fetchone()[0]

            return {
                'total_workflows': n_total,
                'n_subjects': n_subjects,
                'n_tasks': n_tasks,
                'n_pipelines': n_pipelines,
                'status_counts': status_counts
            }
