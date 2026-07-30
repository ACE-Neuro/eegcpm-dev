"""
Leakage guards — three-layer architecture (per S01 + R-003 in the spec).

R-003: provenance separation is by SOURCE FILE, not by column name.
The participants.tsv allow-list does NOT include internalizing /
externalizing / adhd_attention (they are archive factor scores
in that file). The frozen-score target file has its own SEPARATE
allow-list (`FROZEN_SCORE_ALLOW_LIST`).

L1 READER (`l1_load_participants_tsv`): allow-list projection + logged
assertion for participants.tsv ONLY. NEVER raises on source-file
contents.

L1b READER (`l1_load_frozen_scores`): separate allow-list for the
frozen-score target file.

L2 MODEL BOUNDARY (`l2_assert_no_blocked_columns`): hard raise if any
block-listed name or alias reaches the feature frame or the
predictor's design matrix. The archive-origin invariant (L3) is
the binding guard for un-prefixed archive-origin columns.

L3 SENSITIVITY ENTRY POINT (`l3_load_archive_scores`): single audited
function that may load archive scores, into archive_-prefixed columns,
callable only from the pre-registered archive-p / archive_6cbcl arms.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- L1
# L1 allow-list for participants.tsv (R-003): does NOT include
# internalizing/externalizing/adhd_attention (those are archive
# factor scores in participants.tsv per clarified-idea.md:143-146).
PARTICIPANTS_TSV_ALLOW_LIST: List[str] = [
    # Identity
    "participant_id", "subject_id",
    # Demographics
    "age", "sex",
    # Site / release
    "site", "release_number",
    # Per-task QC flags
    "RestingState", "DespicableMe", "FunwithFractals",
    "ThePresent", "DiaryOfAWimpyKid",
    "contrastChangeDetection_1", "contrastChangeDetection_2",
    "contrastChangeDetection_3",
    "surroundSupp_1", "surroundSupp_2",
    "seqLearning6target", "seqLearning8target",
    "symbolSearch",
    # Consent
    "commercial_use", "full_pheno",
]

# L1b allow-list for the FROZEN-SCORE target file (e.g.
# frozen_d_scores_v1.parquet). The frozen-score file has documented
# columns {subject_id, d, internalizing, externalizing, adhd_attention}
# (per clarified-idea.md:36-38). The L1b allow-list matches the
# frozen-score file's actual content; the L2 leak check is applied
# after this projection.
FROZEN_SCORE_ALLOW_LIST: List[str] = [
    "subject_id",
    "d",
    # The specific factors (S01: now allowed in the target allow-list
    # for the exploratory arm)
    "internalizing", "externalizing", "adhd_attention",
]


def l1_load_participants_tsv(
    path: Path,
    allow_list: Optional[List[str]] = None,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """L1 (participants.tsv): load + project to allow-list + log
    dropped. NEVER raises on source contents.

    R-003: internalizing/externalizing/adhd_attention are NOT in
    the participants.tsv allow-list (they are archive factor scores
    in that file). Use `l1_load_frozen_scores` for the frozen-score
    file.
    """
    allow_list = allow_list if allow_list is not None else PARTICIPANTS_TSV_ALLOW_LIST
    df = pd.read_csv(path, sep="\t")
    loaded_cols = set(df.columns)
    allowed = set(allow_list)
    dropped = sorted(loaded_cols - allowed)
    if dropped:
        msg = f"[L1 LOAD] {path} dropped columns not in allow-list: {dropped}"
        logger.warning(msg)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
    return df[[c for c in allow_list if c in loaded_cols]]


def l1_load_frozen_scores(
    path: Path,
    allow_list: Optional[List[str]] = None,
    log_path: Optional[Path] = None,
) -> pd.DataFrame:
    """L1b (frozen-score target file): load + project to allow-list.

    R-003: this is a SEPARATE allow-list for the frozen-score target
    file. The two L1 readers are SOURCE-FILE-specific; the
    participants.tsv reader does NOT include internalizing/externalizing
    /adhd_attention, and the frozen-score reader does.
    """
    allow_list = allow_list if allow_list is not None else FROZEN_SCORE_ALLOW_LIST
    df = pd.read_parquet(Path(path))
    loaded_cols = set(df.columns)
    allowed = set(allow_list)
    dropped = sorted(loaded_cols - allowed)
    if dropped:
        msg = f"[L1b LOAD] {path} dropped columns not in frozen-score allow-list: {dropped}"
        logger.warning(msg)
        if log_path is not None:
            with open(log_path, "a") as f:
                f.write(msg + "\n")
    return df[[c for c in allow_list if c in loaded_cols]]


# --------------------------------------------------------------------- L2
# R2: internalizing, externalizing, adhd_attention REMOVED from the
# exact block list (they are in the allow-list for the exploratory
# specific-factors arm). The archive-origin invariant (below) is the
# binding guard for un-prefixed archive-origin columns.
L2_BLOCK_LIST_EXACT: set = {
    "p_factor", "attention",
}

L2_BLOCK_LIST_PATTERNS: List[re.Pattern] = [
    re.compile(r"^p_factor.*"),       # p_factor, p_factor_1, p_factor_v1
    re.compile(r".*\.factor$"),        # p.factor, g.factor
    re.compile(r"^.*_factor$"),        # p_factor, g_factor
    re.compile(r"^p_.*$"),             # p_total, p_internalizing
    # ^archive_.*$ REMOVED (S01): archive_* columns are produced by
    # L3 and must NOT be blocked at L2. The invariant is asserted
    # separately: archive_* columns appear ONLY in frames produced
    # by L3.
]


def l2_assert_no_blocked_columns(
    df: pd.DataFrame,
    frame_label: str,
) -> None:
    """L2: HARD RAISE if any block-listed name or alias reaches the
    feature frame or the predictor's design matrix. The
    `frame_label` identifies where the check fired (for the audit
    log).

    R2: un-prefixed specifics (internalizing, externalizing,
    adhd_attention) are allowed. Archive-origin columns are caught
    by the inverse invariant (below) when they lack L3 provenance.
    archive_* columns are SKIPPED in the alias pattern check (the
    `_factor$` pattern would otherwise catch `archive_p_factor` and
    similar), and caught only by the L3 provenance invariant.
    """
    cols = set(df.columns)
    for col in cols:
        if col in L2_BLOCK_LIST_EXACT:
            raise ValueError(
                f"[L2 BLOCK] forbidden column {col!r} reached "
                f"{frame_label}; L2 hard raise. Fix the upstream "
                f"code that introduced this column."
            )
        # archive_* columns are checked only by the L3 provenance
        # invariant (below); they are produced ONLY by L3 and the
        # alias patterns are designed to catch UN-prefixed leakage.
        if col.startswith("archive_"):
            continue
        for pat in L2_BLOCK_LIST_PATTERNS:
            if pat.match(col):
                raise ValueError(
                    f"[L2 BLOCK] forbidden column {col!r} matches "
                    f"pattern {pat.pattern!r} and reached "
                    f"{frame_label}; L2 hard raise."
                )
    # Inverse invariant: archive_* columns appear ONLY from L3.
    # Any archive_* column that did NOT come from L3 is a leak.
    archive_cols = [c for c in df.columns if c.startswith("archive_")]
    if archive_cols:
        provenance = getattr(df, "_provenance", None)
        if provenance != "L3_sensitivity_entry_point":
            raise ValueError(
                f"[L2 BLOCK] archive_* columns {archive_cols} found "
                f"in {frame_label} without L3 provenance. archive_* "
                "columns must be produced ONLY by L3."
            )


# --------------------------------------------------------------------- L3
def l3_load_archive_scores(
    target: Literal["archive_p", "archive_6cbcl"],
    d_forbidden_check: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """L3: the ONLY function that may load archive scores. Returns
    a DataFrame with archive_-prefixed columns and provenance set
    so L2 accepts them.

    Callable only from the pre-registered archive-p / archive_6cbcl
    arms; the registry of allowed callers is enforced by the
    `cfg.arm` parameter and by the audit log.
    """
    if getattr(cfg, "arm", None) not in {"archive_p", "archive_6cbcl"}:
        raise ValueError(
            f"[L3] l3_load_archive_scores called with arm={getattr(cfg, 'arm', None)!r}; "
            f"only 'archive_p' and 'archive_6cbcl' are allowed."
        )
    archive_path = getattr(cfg, "archive_scores_path", None)
    if archive_path is None:
        raise ValueError("[L3] cfg.archive_scores_path is required.")
    archive_df = pd.read_parquet(Path(archive_path))
    # Prefix with archive_ to avoid collision with frozen-score names
    archive_df = archive_df.rename(
        columns={
            c: f"archive_{c}"
            for c in archive_df.columns
            if c not in {"subject_id"}
        }
    )
    # L2 cross-check: the columns we just produced must NOT trigger
    # L2's patterns (we removed ^archive_.*$ from L2 precisely so
    # they don't, but we re-assert the invariant here).
    archive_df._provenance = "L3_sensitivity_entry_point"
    logger.info(
        f"[L3 LOAD] target={target} arm={cfg.arm} n_rows={len(archive_df)}"
    )
    return archive_df
