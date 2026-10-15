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


# ---------------------------------------------------------------- encoder (B3)

def test_json_encoder_keeps_none_and_converts_numpy():
    """The sidecar encoder keeps None and converts numpy types (B1-rev1 cond. 4)."""
    import json
    import numpy as np
    from eegcpm.workflow.state import _make_json_serializable

    payload = {
        "n_calibration_samples": None,
        "counts": np.array([1, 2, 3], dtype=np.int64),
        "flag": np.bool_(True),
        "ratio": np.float32(0.5),
        "nested": {"value": np.int64(7), "nothing": None, "items": [None, np.float64(1.5)]},
        "path": Path("/tmp/x"),
    }
    out = _make_json_serializable(payload)

    assert out["n_calibration_samples"] is None
    assert out["counts"] == [1, 2, 3]
    assert out["flag"] is True
    assert out["ratio"] == 0.5
    assert out["nested"] == {"value": 7, "nothing": None, "items": [None, 1.5]}
    assert out["path"] == "/tmp/x"

    # Strict JSON round-trip: no NaN/Infinity tokens, None survives.
    text = json.dumps(out, allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["n_calibration_samples"] is None


def test_json_encoder_non_finite_floats_become_strings():
    """±inf/nan must not produce non-strict JSON tokens."""
    import json
    import math
    import numpy as np
    from eegcpm.workflow.state import _make_json_serializable

    out = _make_json_serializable(
        {"tol": (float("-inf"), 7.0), "plus": np.float64(np.inf), "nan": float("nan")}
    )
    assert out["tol"] == ["-inf", 7.0]
    assert out["plus"] == "inf"
    assert out["nan"] == "nan"
    json.dumps(out, allow_nan=False)  # must not raise


def test_save_state_round_trips_numpy_scalar_metadata(tmp_path):
    """B3-K5: ``save_state`` must not crash on numpy metadata.

    The old ``json.dumps(state.metadata)`` bypassed the numpy-aware encoder:
    a numpy scalar raised ``TypeError`` (crashing the save) and a non-finite
    float wrote ``-Infinity`` (non-strict JSON). Exposure is live since the
    preprocessing sidecar records ``resolved_config``/``library_versions``.
    """
    import json
    import sqlite3
    import numpy as np

    db_path = tmp_path / "state.db"
    mgr = WorkflowStateManager(db_path)
    state = _state(stage="preprocessing")
    state.metadata = {
        "n_detected": np.int64(7),
        "fraction": np.float64(0.25),
        "flag": np.bool_(True),
        "per_channel": np.array([1.5, 2.5]),
        "tol": float("-inf"),
        "nothing": None,
        "resolved_config": {"spec_deviations": [], "steps": []},
    }
    mgr.save_state(state)

    with sqlite3.connect(db_path) as conn:
        raw = conn.execute(
            "SELECT metadata FROM workflows WHERE subject_id = 'sub-1'"
        ).fetchone()[0]
    assert "Infinity" not in raw and "NaN" not in raw
    stored = json.loads(raw)
    assert stored["n_detected"] == 7
    assert stored["fraction"] == 0.25
    assert stored["flag"] is True
    assert stored["per_channel"] == [1.5, 2.5]
    assert stored["tol"] == "-inf"
    assert stored["nothing"] is None

    # The loaded state round-trips the same values.
    loaded = mgr.load_state("sub-1", "RestingState", "hbn")
    assert loaded is not None
    assert loaded.metadata["n_detected"] == 7
    assert loaded.metadata["tol"] == "-inf"
