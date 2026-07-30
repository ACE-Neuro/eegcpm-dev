"""Tests for the three-layer leakage architecture (per S01 + R2)."""

import logging
from pathlib import Path

import pandas as pd
import pytest

from eegcpm.core.leakage import (
    L2_BLOCK_LIST_EXACT,
    L2_BLOCK_LIST_PATTERNS,
    PARTICIPANTS_TSV_ALLOW_LIST,
    l1_load_participants_tsv,
    l2_assert_no_blocked_columns,
    l3_load_archive_scores,
)


# --------------------------------------------------------------- L1 tests

def test_l1_load_drops_blocked_column_logs_assertion(tmp_path, caplog):
    """S01: L1 projects to the allow-list and logs the dropped
    columns. It does NOT raise on source contents."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tage\tp_factor\n"
        "sub-01\t10\t0.5\n"
    )
    log = tmp_path / "leakage.log"
    with caplog.at_level(logging.WARNING, logger="eegcpm.core.leakage"):
        df = l1_load_participants_tsv(
            path=tsv, allow_list=PARTICIPANTS_TSV_ALLOW_LIST, log_path=log)
    # The forbidden column was DROPPED (not present in df)
    assert "p_factor" not in df.columns
    assert "participant_id" in df.columns
    assert "age" in df.columns
    # The drop is LOGGED to the log file
    log_text = log.read_text()
    assert "p_factor" in log_text
    assert "[L1 LOAD]" in log_text


def test_l1_load_does_not_raise_on_source_contents(tmp_path):
    """S01: L1 never raises on source-file contents (the source IS
    what it is; the assertion is the log, not a raise)."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tp_factor\tattention\n"
        "sub-01\t0.5\t0.3\n"
    )
    # No raise
    df = l1_load_participants_tsv(path=tsv,
                                    allow_list=PARTICIPANTS_TSV_ALLOW_LIST)
    # Only allowed columns present
    assert "p_factor" not in df.columns
    assert "attention" not in df.columns
    assert "participant_id" in df.columns


def test_l1_load_preserves_exploratory_specifics_in_frozen_scores(tmp_path):
    """R-003: un-prefixed specific factors (internalizing, externalizing,
    adhd_attention) are in the FROZEN-SCORE allow-list (NOT the
    participants.tsv allow-list). They are archive factor scores
    in participants.tsv; they belong to the frozen-score file."""
    parquet_path = tmp_path / "frozen.parquet"
    import pandas as pd
    pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "d": [0.5, 0.6],
        "internalizing": [0.3, 0.4],
        "externalizing": [0.2, 0.3],
        "adhd_attention": [0.1, 0.2],
    }).to_parquet(parquet_path)
    from eegcpm.core.leakage import l1_load_frozen_scores
    df = l1_load_frozen_scores(path=parquet_path)
    assert "internalizing" in df.columns
    assert "externalizing" in df.columns
    assert "adhd_attention" in df.columns
    assert "d" in df.columns


def test_l1_drops_specifics_in_participants_tsv(tmp_path):
    """R-003: internalizing/externalizing/adhd_attention in
    participants.tsv are dropped (they belong to the frozen-score
    file, not participants.tsv)."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tinternalizing\texternalizing\tadhd_attention\n"
        "sub-01\t0.5\t0.3\t0.4\n"
    )
    df = l1_load_participants_tsv(path=tsv,
                                    allow_list=PARTICIPANTS_TSV_ALLOW_LIST)
    assert "internalizing" not in df.columns
    assert "externalizing" not in df.columns
    assert "adhd_attention" not in df.columns
    assert "participant_id" in df.columns


# --------------------------------------------------------------- L2 tests

def test_l2_hard_raises_on_blocked_name_in_feature_frame():
    """S01: L2 raises HARD if a blocked name reaches a feature frame."""
    df = pd.DataFrame({"subject_id": ["S1"], "p_factor": [0.5]})
    with pytest.raises(ValueError, match="\\[L2 BLOCK\\]"):
        l2_assert_no_blocked_columns(df, frame_label="feature_frame")


def test_l2_hard_raises_on_alias_pattern():
    df = pd.DataFrame({"subject_id": ["S1"], "p_factor_v1": [0.5]})
    with pytest.raises(ValueError, match="matches pattern"):
        l2_assert_no_blocked_columns(df, frame_label="feature_frame")


def test_l2_hard_raises_on_archive_column_without_l3_provenance():
    """S01: archive_* columns are produced by L3 only. If a frame
    carries archive_* columns WITHOUT the L3 provenance, L2 raises."""
    df = pd.DataFrame({"subject_id": ["S1"], "archive_p_factor": [0.5]})
    with pytest.raises(ValueError, match="L3 provenance"):
        l2_assert_no_blocked_columns(df, frame_label="feature_frame")


def test_l2_accepts_internalizing_from_frozen_file():
    """R2: the un-prefixed specific-factor columns are allowed in a
    frame (the exploratory specific-factors arm). The frame is
    rejected ONLY if it ALSO contains a block-listed name or an
    archive_* column without provenance."""
    df = pd.DataFrame({
        "subject_id": ["S1"],
        "internalizing": [0.5],
        "externalizing": [0.3],
        "adhd_attention": [0.4],
    })
    # No raise: un-prefixed specifics are not in L2_BLOCK_LIST_EXACT
    l2_assert_no_blocked_columns(df, frame_label="frozen_file")


def test_l2_raises_on_archive_internalizing_without_l3_provenance():
    """R2: archive_* columns are produced only by L3. A frame with
    archive_internalizing but no L3 provenance is a leak."""
    df = pd.DataFrame({"subject_id": ["S1"], "archive_internalizing": [0.5]})
    with pytest.raises(ValueError, match="L3 provenance"):
        l2_assert_no_blocked_columns(df, frame_label="feature_frame")


def test_l2_accepts_archive_columns_with_l3_provenance():
    """L2 accepts archive_* columns when the frame's _provenance
    attribute is set to L3 (set by l3_load_archive_scores)."""
    df = pd.DataFrame({
        "subject_id": ["S1"],
        "archive_p_factor": [0.5],
        "archive_internalizing": [0.3],
    })
    df._provenance = "L3_sensitivity_entry_point"
    l2_assert_no_blocked_columns(df, frame_label="l3_output_frame")


def test_l2_block_list_exact_does_not_contain_specifics():
    """R2: internalizing/externalizing/adhd_attention REMOVED from
    L2_BLOCK_LIST_EXACT (they are in target_allow_list)."""
    assert "internalizing" not in L2_BLOCK_LIST_EXACT
    assert "externalizing" not in L2_BLOCK_LIST_EXACT
    assert "adhd_attention" not in L2_BLOCK_LIST_EXACT
    # p_factor and attention stay
    assert "p_factor" in L2_BLOCK_LIST_EXACT
    assert "attention" in L2_BLOCK_LIST_EXACT


def test_l2_block_patterns_do_not_contain_archive():
    """S01: ^archive_.*$ REMOVED from L2 patterns (archive_* columns
    are produced by L3)."""
    pattern_strings = [p.pattern for p in L2_BLOCK_LIST_PATTERNS]
    for s in pattern_strings:
        assert "archive" not in s, f"archive pattern leaked: {s}"


# --------------------------------------------------------------- L3 tests

def test_l3_only_callable_from_pretrained_arms():
    """S01: L3 is callable only from archive_p / archive_6cbcl arms."""
    cfg = _arm_cfg(arm="primary")
    with pytest.raises(ValueError,
                        match="only 'archive_p' and 'archive_6cbcl'"):
        l3_load_archive_scores("archive_p", d_forbidden_check=pd.DataFrame(),
                                cfg=cfg)


def test_l3_renames_archive_columns_no_collision(tmp_path):
    """R2 / S23a: L3 renames archive columns with archive_ prefix; no
    unprefixed collision in the merged feature frame."""
    archive_path = tmp_path / "archive.parquet"
    archive_df = pd.DataFrame({
        "subject_id": ["S1", "S2"],
        "d": [0.5, 0.7],   # collides with frozen-score column
    })
    archive_df.to_parquet(archive_path)
    cfg = _arm_cfg(arm="archive_p",
                    archive_scores_path=str(archive_path))
    archive_renamed = l3_load_archive_scores(
        "archive_p", d_forbidden_check=pd.DataFrame(), cfg=cfg)
    # archive_d MUST be present; raw "d" must NOT be present
    assert "archive_d" in archive_renamed.columns
    assert "d" not in archive_renamed.columns
    # Provenance attribute set
    assert archive_renamed._provenance == "L3_sensitivity_entry_point"
    # Merged frame must NOT have _x/_y suffixes
    target = pd.DataFrame({"subject_id": ["S1"], "d": [0.5]})
    merged = target.merge(archive_renamed, on="subject_id")
    assert not any(c.endswith(("_x", "_y")) for c in merged.columns)


# --------------------------------------------------------------- helpers

def _arm_cfg(arm: str, archive_scores_path=None) -> object:
    """Build a minimal cfg with `arm` and optional `archive_scores_path`."""
    cfg = type("Cfg", (), {})()
    cfg.arm = arm
    cfg.caller_function = "test"
    if archive_scores_path is not None:
        cfg.archive_scores_path = archive_scores_path
    return cfg


# --------------------------------------------------------------- integration

def test_poison_column_participants_tsv_layered_pipeline(tmp_path):
    """S01: a poison-column participants.tsv produces the expected
    behavior at each layer: L1 drops+logs (no raise); L2 raises on
    the column reaching a feature frame; L3 is the only loader of
    archive_*."""
    tsv = tmp_path / "participants.tsv"
    tsv.write_text(
        "participant_id\tp_factor\n"
        "sub-01\t0.5\n"
    )
    log = tmp_path / "leakage.log"
    # L1: drops and logs
    df_l1 = l1_load_participants_tsv(
        path=tsv, allow_list=PARTICIPANTS_TSV_ALLOW_LIST, log_path=log)
    assert "p_factor" not in df_l1.columns
    # L2: would raise if a frame reintroduced p_factor
    bad_frame = pd.DataFrame({"subject_id": ["S1"], "p_factor": [0.5]})
    with pytest.raises(ValueError, match="\\[L2 BLOCK\\]"):
        l2_assert_no_blocked_columns(bad_frame, frame_label="bad_frame")
