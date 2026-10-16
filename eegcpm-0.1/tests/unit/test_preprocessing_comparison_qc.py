"""F6: the preprocessing-comparison QC mark is appended, never set.

The comparison figure's ``raw_annotated`` is an original-duration *display*
copy of the recording: it must keep the recording's own annotations (stim
events, earlier ``BAD_*`` marks) and only add the ``BAD_eegprep_removed_*``
mark. The pre-F6 code called ``set_annotations(annot)``, which replaced the
whole timeline and silently dropped those annotations from the figure.
``append_eegprep_removed_mark`` wraps the shared append-only recipe
(``append_bad_annotations``); these tests pin the append semantics on a
synthetic recording without running the (slow) eegprep comparison pipeline.
"""

import mne
import numpy as np
import pytest

from eegcpm.modules.qc.preprocessing_comparison_qc import (
    append_eegprep_removed_mark,
)


def _raw_with_stim() -> mne.io.RawArray:
    info = mne.create_info([f"E{i}" for i in range(1, 4)], 250.0, "eeg")
    rng = np.random.RandomState(0)
    raw = mne.io.RawArray(rng.randn(3, 1000) * 1e-6, info, verbose=False)
    raw.set_annotations(mne.Annotations([1.0], [0.1], ["stim"]))
    return raw


def _annotation(raw: mne.io.BaseRaw, description: str):
    matches = [
        annot for annot in raw.annotations
        if annot["description"] == description
    ]
    assert len(matches) == 1, (
        f"expected exactly one {description!r} annotation, got {len(matches)}"
    )
    return matches[0]


def test_removed_mark_is_appended_and_stim_survives():
    """The stim annotation stays and the mark lands where it was asked for."""
    raw = _raw_with_stim()
    append_eegprep_removed_mark(raw, 3.0, 1.0, "BAD_eegprep_removed_25.0pct")

    assert set(raw.annotations.description) == {
        "stim", "BAD_eegprep_removed_25.0pct",
    }
    stim = _annotation(raw, "stim")
    assert stim["onset"] == pytest.approx(1.0)
    assert stim["duration"] == pytest.approx(0.1)
    mark = _annotation(raw, "BAD_eegprep_removed_25.0pct")
    assert mark["onset"] == pytest.approx(3.0)
    assert mark["duration"] == pytest.approx(1.0)


def test_removed_mark_does_not_replace_existing_bad_marks():
    """An earlier BAD_* mark survives the append (the pre-F6 bug)."""
    raw = _raw_with_stim()
    raw.set_annotations(
        raw.annotations + mne.Annotations([2.0], [0.5], ["BAD_existing"])
    )
    append_eegprep_removed_mark(raw, 3.0, 1.0, "BAD_eegprep_removed_10.0pct")

    assert set(raw.annotations.description) == {
        "stim", "BAD_existing", "BAD_eegprep_removed_10.0pct",
    }
    assert _annotation(raw, "BAD_existing")["onset"] == pytest.approx(2.0)


def test_removed_mark_keeps_the_timeline_on_the_same_samples():
    """Sample positions of pre-existing annotations are unchanged (append-only)."""
    raw = _raw_with_stim()
    sfreq = raw.info["sfreq"]
    before = [
        (round(annot["onset"] * sfreq), round(annot["duration"] * sfreq),
         annot["description"])
        for annot in raw.annotations
    ]
    append_eegprep_removed_mark(raw, 3.0, 1.0, "BAD_eegprep_removed_25.0pct")
    after = [
        (round(annot["onset"] * sfreq), round(annot["duration"] * sfreq),
         annot["description"])
        for annot in raw.annotations
        if annot["description"] != "BAD_eegprep_removed_25.0pct"
    ]
    assert after == before
