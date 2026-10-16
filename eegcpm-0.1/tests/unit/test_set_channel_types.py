"""Tests for the generic `set_channel_types` step (2026-10-17).

Added for the 25SusAttn dataset, whose BrainVision exports type the digital
trigger as EEG and whose auxiliary EOG/ECG/EMG electrodes are recorded but
unusable (flat or DC-offset dominated). The step applies an explicit
config-declared name -> type mapping instead of a name heuristic.
"""

from __future__ import annotations

import mne
import numpy as np
import pytest

from eegcpm.modules.preprocessing.steps import STEP_REGISTRY
from eegcpm.modules.preprocessing.steps.set_channel_types import (
    SetChannelTypesStep,
)


def _raw():
    ch_names = ["Fz", "Cz", "VEOG", "HEOG", "EKG", "Trigger"]
    ch_types = ["eeg", "eeg", "eog", "eog", "ecg", "eeg"]
    data = np.random.default_rng(0).standard_normal((len(ch_names), 500)) * 5e-6
    return mne.io.RawArray(
        data, mne.create_info(ch_names, 250.0, ch_types), verbose=False
    )


def test_registered_in_step_registry():
    assert STEP_REGISTRY["set_channel_types"] is SetChannelTypesStep


def test_sets_requested_types_and_records_provenance():
    raw = _raw()
    step = SetChannelTypesStep(types={"VEOG": "misc", "HEOG": "misc", "EKG": "misc"})

    out, meta = step.process(raw, {})

    types = dict(zip(out.ch_names, out.get_channel_types()))
    assert types["VEOG"] == "misc"
    assert types["HEOG"] == "misc"
    assert types["EKG"] == "misc"
    assert types["Fz"] == "eeg"
    assert meta["applied"] is True
    assert meta["n_channels"] == 3
    assert meta["previous_types"] == {"VEOG": "eog", "HEOG": "eog", "EKG": "ecg"}
    assert meta["missing"] == []


def test_missing_channels_are_recorded_not_silent():
    raw = _raw()
    step = SetChannelTypesStep(types={"VEOG": "misc", "EOGX": "misc"})

    _, meta = step.process(raw, {})

    assert meta["applied"] is True
    assert meta["types"] == {"VEOG": "misc"}
    assert meta["missing"] == ["EOGX"]


def test_all_missing_is_an_allowlisted_noop():
    raw = _raw()
    step = SetChannelTypesStep(types={"EOGX": "misc"})

    out, meta = step.process(raw, {})

    assert out is raw
    assert meta["skipped"] is True
    assert meta["reason"] == "no_matching_channels"
    assert meta["reason"] in SetChannelTypesStep.ALLOWED_NOOP_REASONS


def test_on_missing_raise_fails_loudly():
    raw = _raw()
    step = SetChannelTypesStep(types={"EOGX": "misc"}, on_missing="raise")

    with pytest.raises(ValueError, match="EOGX"):
        step.process(raw, {})
