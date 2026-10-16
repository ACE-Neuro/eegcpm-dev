"""B7-A1: `data/hbn.py` must record what it skips, not swallow it.

`add_behavioral_scores` converted phenotype values with ``float(...)`` inside
``except (ValueError, TypeError): pass`` — the non-numeric values vanished
without a trace. The narrowed handlers now log per value (debug) and a summary
(warning); these tests pin both the skip behaviour and the record.
"""

import logging
from pathlib import Path

import pytest

from eegcpm.core.models import Project, Subject
from eegcpm.data.hbn import add_behavioral_scores


def _project(subject_ids=("sub-01",)):
    return Project(
        name="HBN_Dataset",
        root_path=Path("/tmp/hbn"),
        subjects=[Subject(id=sub_id) for sub_id in subject_ids],
    )


def test_numeric_scores_are_added(caplog):
    project = _project()
    phenotype = {"sub-01": {"age": "12", "wasi_iq": 105.0, "sex": "M"}}

    with caplog.at_level(logging.DEBUG, logger="eegcpm.data.hbn"):
        add_behavioral_scores(project, phenotype)

    scores = project.subjects[0].behavioral_scores
    assert scores == {"age": 12.0, "wasi_iq": 105.0}
    assert "sex" not in scores  # non-numeric, skipped
    # The skip is recorded (debug line + warning summary), not swallowed.
    messages = [record.getMessage() for record in caplog.records]
    assert any("sub-01.sex is not numeric" in msg for msg in messages)
    assert any("skipped 1 non-numeric score(s)" in msg for msg in messages)
    assert any(record.levelno == logging.WARNING for record in caplog.records)


def test_score_columns_subset_is_respected(caplog):
    project = _project()
    phenotype = {"sub-01": {"age": "12", "wasi_iq": "n/a", "sex": "M"}}

    with caplog.at_level(logging.DEBUG, logger="eegcpm.data.hbn"):
        add_behavioral_scores(project, phenotype, score_columns=["age", "wasi_iq"])

    assert project.subjects[0].behavioral_scores == {"age": 12.0}
    messages = [record.getMessage() for record in caplog.records]
    assert any("sub-01.wasi_iq is not numeric" in msg for msg in messages)
    assert not any("sub-01.sex" in msg for msg in messages)


def test_requested_but_absent_score_columns_are_logged(caplog):
    """A requested column the phenotype table lacks is recorded (B7-A1)."""
    project = _project()
    phenotype = {"sub-01": {"age": "12"}}

    with caplog.at_level(logging.DEBUG, logger="eegcpm.data.hbn"):
        add_behavioral_scores(project, phenotype, score_columns=["age", "wasi_iq"])

    assert project.subjects[0].behavioral_scores == {"age": 12.0}
    messages = [record.getMessage() for record in caplog.records]
    assert any("no column(s) wasi_iq" in msg for msg in messages)


def test_unknown_subjects_are_untouched(caplog):
    project = _project()
    with caplog.at_level(logging.DEBUG, logger="eegcpm.data.hbn"):
        add_behavioral_scores(project, {"sub-99": {"age": "12"}})
    assert project.subjects[0].behavioral_scores == {}
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.parametrize("bad_value", [None, "unknown", "1,2"])
def test_non_numeric_values_are_skipped(bad_value, caplog):
    """None → TypeError, "unknown"/"1,2" → ValueError: skipped and logged."""
    project = _project()
    with caplog.at_level(logging.DEBUG, logger="eegcpm.data.hbn"):
        add_behavioral_scores(project, {"sub-01": {"score": bad_value}})
    assert project.subjects[0].behavioral_scores == {}
    assert any("is not numeric" in r.getMessage() for r in caplog.records)


def test_nan_is_stored_unchanged():
    """``float('nan')`` parses, so it is stored — recorded behaviour boundary.

    B7-A1 only fixed the *silent* handling of conversion errors; whether NaN
    should be filtered out of behavioral scores is a separate decision.
    """
    project = _project()
    add_behavioral_scores(project, {"sub-01": {"score": float("nan")}})
    score = project.subjects[0].behavioral_scores["score"]
    assert score != score  # NaN
