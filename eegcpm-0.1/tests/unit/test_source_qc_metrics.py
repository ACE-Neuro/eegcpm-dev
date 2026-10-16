"""Source-QC validity metrics must be able to fail the QC status (P2-F15, §B7 task 9).

The old QC was blind to under-populated ROIs, non-finite values and missing
time axes, and it *fabricated* a time axis (``arange(n)/500.0`` centred on the
middle sample) when a condition had no ``{cond}_times`` key. These tests pin:

- ``bad`` status for ``roi_vertex_counts < roi_min_vertices``;
- ``bad`` status for non-finite ROI values;
- ``bad`` status for a condition without its ``{cond}_times`` axis (and no
  fabricated axis: the ERP figure is omitted instead);
- ``ok`` for a well-formed archive (so the metric is not always bad).
"""

from pathlib import Path

import numpy as np
import pytest

from eegcpm.modules.qc.source_qc import SourceQC


def _roi_data(n_rois=3, n_times=50, conditions=("a",), finite=True, with_times=True):
    times = np.linspace(-0.2, 0.6, n_times)
    data = {
        "roi_names": [f"roi{i}" for i in range(n_rois)],
        "roi_vertex_counts": [10] * n_rois,
        "sfreq": 50.0,
    }
    for cond in conditions:
        arr = np.random.default_rng(0).standard_normal((n_rois, n_times))
        if not finite:
            arr[0, 0] = np.nan
        data[cond] = arr
        data[f"{cond}_nave"] = 10
        if with_times:
            data[f"{cond}_times"] = times
    return data


def _compute(tmp_path, roi_data, **kwargs):
    qc = SourceQC(Path(tmp_path))
    return qc.compute(data={"roi_data": roi_data, **kwargs}, subject_id="s1")


def _metric(result, name):
    return next(m for m in result.metrics if m.name == name)


def test_ok_archive_is_ok(tmp_path):
    result = _compute(tmp_path, _roi_data(), roi_min_vertices=3)
    assert result.status == "ok"
    assert _metric(result, "ROI Vertex Counts").status == "ok"
    assert _metric(result, "ROI Values Finite").status == "ok"
    assert _metric(result, "Condition Time Axes").status == "ok"


def test_underpopulated_roi_is_bad(tmp_path):
    roi_data = _roi_data()
    roi_data["roi_vertex_counts"] = [10, 2, 10]
    result = _compute(tmp_path, roi_data, roi_min_vertices=3)
    assert result.status == "bad"
    metric = _metric(result, "ROI Vertex Counts")
    assert metric.status == "bad"
    assert metric.value == 2
    assert any("fewer than 3 vertices" in note for note in result.notes)


def test_non_finite_roi_value_is_bad(tmp_path):
    result = _compute(tmp_path, _roi_data(finite=False), roi_min_vertices=3)
    assert result.status == "bad"
    metric = _metric(result, "ROI Values Finite")
    assert metric.status == "bad"
    assert metric.value == 1
    assert any("Non-finite ROI values" in note for note in result.notes)


def test_missing_time_axis_is_bad_and_no_fabricated_axis(tmp_path):
    """A condition without `{cond}_times` is `bad`; no arange/500 Hz axis."""
    result = _compute(tmp_path, _roi_data(with_times=False), roi_min_vertices=3)
    assert result.status == "bad"
    metric = _metric(result, "Condition Time Axes")
    assert metric.status == "bad"
    assert any("Missing time axis" in note for note in result.notes)
    # The ERP figure is skipped (the old code fabricated a time axis and plotted).
    assert "source_erp" not in result.figures


def test_roi_min_vertices_defaults_to_three(tmp_path):
    roi_data = _roi_data()
    roi_data["roi_vertex_counts"] = [3, 2, 10]
    result = _compute(tmp_path, roi_data)
    assert result.status == "bad"


def test_crosstalk_note_when_labels_without_matrix(tmp_path):
    """Labels without a matrix is reported, never recomputed in QC (P2-F14)."""
    result = _compute(tmp_path, _roi_data(), labels=[object()])
    assert any("Crosstalk matrix not provided" in note for note in result.notes)
    assert "crosstalk_matrix" not in result.figures


class _Label:
    def __init__(self, name):
        self.name = name


def test_crosstalk_pair_count_is_directed(tmp_path):
    """`n_directed_above_0.3` counts directed entries, not unique pairs (B7-A3).

    ``C[i, j]`` is the leakage from ROI j (source, column) into ROI i (target,
    row) — row = target, column = source (B7 rev2 fix 5; the earlier wording had
    the direction backwards). The matrix is asymmetric. With C[0, 1] = 0.4 and
    C[1, 0] = 0.35 (both above 0.3) the directed count is 2; the old `triu`
    count reported 1 (the reciprocal pair was silently halved). The mean stays
    the mean over *all* off-diagonal entries.
    """
    matrix = np.array([
        [1.0, 0.40, 0.10],
        [0.35, 1.0, 0.05],
        [0.20, 0.25, 1.0],
    ])
    labels = [_Label(n) for n in ("roi0", "roi1", "roi2")]
    result = _compute(
        tmp_path, _roi_data(), labels=labels, crosstalk_matrix=matrix,
        crosstalk_matrix_kind="crosstalk_flip_aligned",
    )
    assert result.metadata['n_directed_above_0.3'] == 2
    assert 'crosstalk_high_pairs' not in result.metadata
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert result.metadata['crosstalk_mean'] == pytest.approx(float(np.mean(off_diag)))
    # B7 rev2 fix 6: the statistics state which matrix they were computed on.
    assert result.metadata['crosstalk_matrix_kind'] == "crosstalk_flip_aligned"
    assert any("crosstalk_flip_aligned" in note for note in result.notes)
    # The figure (and its printed statistics) is still produced.
    assert "crosstalk_matrix" in result.figures


def test_crosstalk_pair_count_symmetric_matrix_counts_both_directions(tmp_path):
    """A symmetric above-threshold pair counts twice (directed), not once.

    This is the semantic difference B7-A3 is about: with C[0, 1] = C[1, 0] =
    0.4 the directed count is 2, while the old `triu` count reported 1.
    """
    matrix = np.array([
        [1.0, 0.40, 0.10],
        [0.40, 1.0, 0.05],
        [0.10, 0.05, 1.0],
    ])
    labels = [_Label(n) for n in ("roi0", "roi1", "roi2")]
    result = _compute(
        tmp_path, _roi_data(), labels=labels, crosstalk_matrix=matrix,
    )
    assert result.metadata['n_directed_above_0.3'] == 2
    assert int(np.sum(np.triu(matrix, k=1) > 0.3)) == 1  # the old count
