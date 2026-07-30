"""R-015: condition/run_id state identity, migration, and stage filter."""

import sqlite3
from pathlib import Path

from eegcpm.workflow.state import (
    ProcessingStatus, WorkflowState, WorkflowStateManager,
)


def _state(subject="sub-1", condition=None, run_id=None, stage="features"):
    return WorkflowState(
        subject_id=subject, task="RestingState", pipeline="hbn",
        status=ProcessingStatus.COMPLETED, condition=condition,
        run_id=run_id, current_stage=stage,
    )


def test_condition_distinct_states_coexist(tmp_path):
    """Two states differing only by condition must NOT collide."""
    mgr = WorkflowStateManager(tmp_path / "state.db")
    mgr.save_state(_state(condition="resting_ec"))
    mgr.save_state(_state(condition="resting_eo"))
    ec = mgr.get_all_states(condition="resting_ec")
    eo = mgr.get_all_states(condition="resting_eo")
    assert len(ec) == 1 and len(eo) == 1
    assert ec[0].condition == "resting_ec"
    assert eo[0].condition == "resting_eo"


def test_same_full_key_upserts(tmp_path):
    """Same (subject, task, pipeline, condition, run_id) updates in place."""
    mgr = WorkflowStateManager(tmp_path / "state.db")
    mgr.save_state(_state(condition="resting_ec"))
    s2 = _state(condition="resting_ec")
    s2.status = ProcessingStatus.FAILED
    mgr.save_state(s2)
    rows = mgr.get_all_states(condition="resting_ec")
    assert len(rows) == 1
    assert rows[0].status == ProcessingStatus.FAILED


def test_run_id_distinct_states_coexist(tmp_path):
    """Batch prediction runs (no natural subject) distinct by run_id."""
    mgr = WorkflowStateManager(tmp_path / "state.db")
    mgr.save_state(_state(subject="batch", run_id="run-001", stage="prediction"))
    mgr.save_state(_state(subject="batch", run_id="run-002", stage="prediction"))
    rows = mgr.get_all_states(stage="prediction")
    assert len(rows) == 2


def test_stage_filter(tmp_path):
    mgr = WorkflowStateManager(tmp_path / "state.db")
    mgr.save_state(_state(condition="resting_ec", stage="features"))
    mgr.save_state(_state(subject="sub-2", condition=None, stage="preprocessing"))
    feats = mgr.get_all_states(stage="features")
    assert len(feats) == 1 and feats[0].current_stage == "features"


def test_migration_from_legacy_schema(tmp_path):
    """A pre-R-015 database (no condition/run_id columns, old UNIQUE)
    migrates by rebuild, preserving rows."""
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id TEXT NOT NULL, session TEXT, task TEXT NOT NULL,
                run TEXT, pipeline TEXT NOT NULL, status TEXT NOT NULL,
                config_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                current_stage TEXT DEFAULT 'preprocessing',
                metadata TEXT,
                UNIQUE(subject_id, session, task, run, pipeline)
            )
        """)
        conn.execute("""
            INSERT INTO workflows (subject_id, session, task, run, pipeline,
                                   status, config_hash)
            VALUES ('sub-legacy', NULL, 'RestingState', NULL, 'hbn',
                    'completed', 'abc')
        """)
    mgr = WorkflowStateManager(db)  # triggers migration
    rows = mgr.get_all_states()
    assert len(rows) == 1
    assert rows[0].subject_id == "sub-legacy"
    # Post-migration: condition-distinct rows coexist
    mgr.save_state(_state(subject="sub-legacy", condition="resting_ec"))
    mgr.save_state(_state(subject="sub-legacy", condition="resting_eo"))
    assert len(mgr.get_all_states()) == 3
