"""Tests for task configuration system (trial filtering and binning).

Binning is covered through the consolidated ``TrialSorter`` API
(``bin_trials`` / ``split``); the dead ``modules/epochs/binning.py`` module
was deleted in pass 1 of the sorting/filtering remediation.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eegcpm.modules.epochs.trial_filter import (
    drop_missing_responses,
    drop_sequential_positions,
    filter_by_accuracy,
    filter_by_rt,
)


def _load_task_config():
    """Load a real task config for TrialSorter tests (else skip)."""
    from eegcpm.core.task_config import TaskConfig

    config_path = Path(__file__).parent.parent / "eegcpm" / "config" / "tasks" / "cptvis.yaml"
    if not config_path.exists():
        pytest.skip("cptvis task config not found")
    return TaskConfig.from_yaml(config_path)


def _epochs_from_events_df(events_df):
    """Wrap a behavioral events DataFrame in synthetic epochs + metadata."""
    n_trials = len(events_df)
    sfreq = 100.0
    info = mne.create_info(["EEG1", "EEG2"], sfreq, ch_types=["eeg", "eeg"])
    data = np.zeros((n_trials, 2, 11))
    events = np.column_stack([
        np.arange(1, n_trials + 1) * 10,
        np.zeros(n_trials, dtype=int),
        np.ones(n_trials, dtype=int),
    ])
    return mne.EpochsArray(
        data,
        info,
        events=events,
        tmin=-0.05,
        event_id={"stim": 1},
        metadata=events_df,
        verbose=False,
    )


@pytest.fixture
def mock_events_df():
    """Create mock events dataframe for testing."""
    np.random.seed(42)
    n_trials = 100

    df = pd.DataFrame({
        "onset": np.linspace(0, 200, n_trials),
        "duration": np.ones(n_trials) * 0.5,
        "trial_type": ["target_left" if i % 2 == 0 else "target_right" for i in range(n_trials)],
        "response_time": np.random.uniform(0.2, 1.5, n_trials),
        "accuracy": np.random.choice([0, 1], n_trials, p=[0.2, 0.8]),
    })

    # Add some missing responses
    df.loc[5:8, "response_time"] = np.nan
    df.loc[5:8, "accuracy"] = np.nan

    # Add some too-fast responses
    df.loc[10:12, "response_time"] = 0.08

    return df


def test_filter_by_rt_min(mock_events_df):
    """Test filtering by minimum RT."""
    filtered = filter_by_rt(mock_events_df, rt_min=0.15)

    assert len(filtered) < len(mock_events_df)
    assert (filtered["response_time"] >= 0.15).all() or filtered["response_time"].isna().all()


def test_filter_by_rt_max(mock_events_df):
    """Test filtering by maximum RT."""
    filtered = filter_by_rt(mock_events_df, rt_max=1.0)

    assert len(filtered) < len(mock_events_df)
    assert (filtered["response_time"] <= 1.0).all() or filtered["response_time"].isna().all()


def test_filter_by_rt_both(mock_events_df):
    """Test filtering by both RT bounds."""
    filtered = filter_by_rt(mock_events_df, rt_min=0.15, rt_max=1.0)

    valid_rts = filtered["response_time"].dropna()
    assert (valid_rts >= 0.15).all()
    assert (valid_rts <= 1.0).all()


def test_filter_by_accuracy_correct(mock_events_df):
    """Test filtering for correct trials only."""
    filtered = filter_by_accuracy(mock_events_df, keep="correct")

    assert (filtered["accuracy"] == 1).all()
    assert len(filtered) < len(mock_events_df)


def test_filter_by_accuracy_incorrect(mock_events_df):
    """Test filtering for incorrect trials only."""
    filtered = filter_by_accuracy(mock_events_df, keep="incorrect")

    assert (filtered["accuracy"] == 0).all()
    assert len(filtered) < len(mock_events_df)


def test_drop_missing_responses(mock_events_df):
    """Test dropping trials with missing responses."""
    filtered = drop_missing_responses(mock_events_df)

    assert filtered["response_time"].notna().all()
    assert len(filtered) < len(mock_events_df)


def test_drop_sequential_positions_first(mock_events_df):
    """Test dropping first N trials."""
    n_drop = 10
    filtered = drop_sequential_positions(mock_events_df, drop_first_n=n_drop)

    assert len(filtered) == len(mock_events_df) - n_drop
    # First trial in filtered should be trial #10 from original
    assert filtered.iloc[0]["onset"] == mock_events_df.iloc[n_drop]["onset"]


def test_drop_sequential_positions_last(mock_events_df):
    """Test dropping last N trials."""
    n_drop = 10
    filtered = drop_sequential_positions(mock_events_df, drop_last_n=n_drop)

    assert len(filtered) == len(mock_events_df) - n_drop
    # Last trial in filtered should be trial #-11 from original
    assert filtered.iloc[-1]["onset"] == mock_events_df.iloc[-(n_drop+1)]["onset"]


def test_drop_sequential_positions_both(mock_events_df):
    """Test dropping both first and last trials."""
    filtered = drop_sequential_positions(mock_events_df, drop_first_n=5, drop_last_n=5)

    assert len(filtered) == len(mock_events_df) - 10


def test_bin_rt_quartiles(mock_events_df):
    """Bin trials by RT quartiles through TrialSorter.bin_trials."""
    from eegcpm.core.task_config import BinSpec, TrialSorter

    task_config = _load_task_config()
    sorter = TrialSorter(task_config)

    bin_spec = BinSpec(
        name="rt_quartiles",
        method="quantile",
        n_bins=4,
        column="response_time",
        labels=["Q1", "Q2", "Q3", "Q4"],
    )
    epochs = _epochs_from_events_df(mock_events_df)
    bins = sorter.bin_trials(epochs, bin_spec=bin_spec)

    assert len(bins) == 4
    assert "Q1" in bins
    assert "Q4" in bins

    # Check all trials are included (NaNs in RT are excluded from the bins)
    total_trials = sum(info["n_trials"] for info in bins.values())
    n_valid = int(mock_events_df["response_time"].notna().sum())
    assert total_trials == n_valid
    # Provenance is attached
    assert "selection_record" in bins["Q1"]


def test_bin_sequential_position(mock_events_df):
    """Split trials by sequential position through TrialSorter.split."""
    from eegcpm.core.task_config import TrialSorter

    task_config = _load_task_config()
    sorter = TrialSorter(task_config)

    epochs = _epochs_from_events_df(mock_events_df)
    parts = sorter.split(epochs, by="position", parts=5)

    assert len(parts) == 5
    assert "half_1" in parts
    assert "half_5" in parts

    # Check all trials are included and the chunks are disjoint
    total_trials = sum(len(info[0]) for info in parts.values())
    assert total_trials == len(mock_events_df)
    for name, (part_epochs, record) in parts.items():
        assert record.n_before == len(mock_events_df)
        assert record.n_after == len(part_epochs)
        assert record.mask_hash  # boolean mask hash recorded


def test_combined_filtering_pipeline(mock_events_df):
    """Test combined filtering (sequential, RT, accuracy)."""
    # 1. Drop practice trials
    df = drop_sequential_positions(mock_events_df, drop_first_n=5)

    # 2. Drop missing responses
    df = drop_missing_responses(df)

    # 3. Filter by RT
    df = filter_by_rt(df, rt_min=0.15, rt_max=1.5)

    # 4. Keep only correct trials
    df = filter_by_accuracy(df, keep="correct")

    # Check result
    assert len(df) < len(mock_events_df)
    assert (df["accuracy"] == 1).all()
    assert (df["response_time"] >= 0.15).all()
    assert (df["response_time"] <= 1.5).all()


def test_task_config_loading():
    """Test loading SAIIT task configuration."""
    from eegcpm.core.task_config import TaskConfig

    # Load SAIIT config
    config_path = Path(__file__).parent.parent / "eegcpm" / "config" / "tasks" / "saiit.yaml"

    if config_path.exists():
        config = TaskConfig.from_yaml(config_path)

        assert config.task_name == "saiit"
        assert config.task_type == "event-related"
        assert config.tmin == -0.2
        assert config.tmax == 0.8
        assert len(config.conditions) == 2
        # These configs exist but use the existing schema
        assert config.response_mapping is not None
    else:
        pytest.skip("SAIIT config not found")


def test_trial_sorter():
    """Test TrialSorter class."""
    from eegcpm.core.task_config import TaskConfig, TrialSorter

    config_path = Path(__file__).parent.parent / "eegcpm" / "config" / "tasks" / "saiit.yaml"

    if not config_path.exists():
        pytest.skip("SAIIT config not found")

    config = TaskConfig.from_yaml(config_path)
    sorter = TrialSorter(config)

    # Create mock events
    np.random.seed(42)
    n_trials = 100

    events_df = pd.DataFrame({
        "onset": np.linspace(0, 200, n_trials),
        "trial_type": ["target_left" if i % 2 == 0 else "target_right" for i in range(n_trials)],
        "response_time": np.random.uniform(0.2, 1.5, n_trials),
        "accuracy": np.random.choice([0, 1], n_trials, p=[0.2, 0.8]),
    })

    # Test basic functionality
    assert sorter.config == config


def test_erp_component_spec():
    """Test ERP component specification."""
    from eegcpm.core.task_config import TaskConfig

    config_path = Path(__file__).parent.parent / "eegcpm" / "config" / "tasks" / "saiit.yaml"

    if not config_path.exists():
        pytest.skip("SAIIT config not found")

    config = TaskConfig.from_yaml(config_path)

    if config.erp_components:
        assert len(config.erp_components) > 0

        # Check P3 component
        p3 = [c for c in config.erp_components if c.name == "P3"]
        if p3:
            p3 = p3[0]
            assert p3.polarity == "positive"
            assert len(p3.search_window) == 2
            assert len(p3.channels) > 0
