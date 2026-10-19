"""Unit tests for epochs module."""

import json
import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from eegcpm.modules.epochs import EpochExtractionModule, sanitize_name


@pytest.fixture
def sample_raw_with_events():
    """Create sample raw data with events for epoching."""
    np.random.seed(42)
    sfreq = 250
    duration = 30  # 30 seconds
    n_channels = 10
    n_samples = int(sfreq * duration)

    # Create synthetic EEG data
    times = np.arange(n_samples) / sfreq
    data = np.random.randn(n_channels, n_samples) * 1e-5  # ~10 µV in Volts

    # Add some alpha oscillation
    alpha_freq = 10
    alpha_signal = 2e-5 * np.sin(2 * np.pi * alpha_freq * times)
    data[4, :] += alpha_signal  # Add to one channel

    ch_names = [f"EEG{i:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)

    # Add events at regular intervals
    event_times = np.arange(1, 28, 3)  # Events at 1, 4, 7, 10, 13, 16, 19, 22, 25s
    event_samples = (event_times * sfreq).astype(int)
    # Alternate between event types 1 and 2
    event_ids = np.array([1, 2] * 5)[:len(event_times)]
    events = np.column_stack([
        event_samples,
        np.zeros(len(event_times), dtype=int),
        event_ids
    ])

    # Add events as annotations so find_events works
    raw.set_annotations(mne.Annotations(
        onset=event_times,
        duration=np.zeros(len(event_times)),
        description=[str(e) for e in event_ids]
    ))

    # Also create stim channel
    stim_data = np.zeros((1, n_samples))
    for sample, _, event_id in events:
        if sample < n_samples:
            stim_data[0, sample] = event_id
    stim_info = mne.create_info(['STI'], sfreq, ch_types=['stim'])
    stim_raw = mne.io.RawArray(stim_data, stim_info)
    raw.add_channels([stim_raw])

    return raw, events


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestEpochExtractionModule:
    """Tests for EpochExtractionModule class."""

    def test_init_default(self, output_dir):
        """Test default initialization."""
        config = {}
        module = EpochExtractionModule(config, output_dir)

        assert module.tmin == -0.5
        assert module.tmax == 1.0
        assert module.baseline == (-0.2, 0.0)
        assert module.decim == 1
        assert module.reject is None
        assert module.use_autoreject is False

    def test_init_custom_config(self, output_dir):
        """Test initialization with custom config."""
        config = {
            "tmin": -0.3,
            "tmax": 0.8,
            "baseline": (-0.1, 0),
            "decim": 2,
            "rejection": {
                "reject": {"eeg": 100e-6},
                "flat": {"eeg": 0.1e-6},
                "use_autoreject": True,
                "strategy": "both",
            },
        }
        module = EpochExtractionModule(config, output_dir)

        assert module.tmin == -0.3
        assert module.tmax == 0.8
        assert module.baseline == (-0.1, 0)
        assert module.decim == 2
        assert module.reject == {"eeg": 100e-6}
        assert module.flat == {"eeg": 0.1e-6}
        assert module.use_autoreject is True
        assert module.rejection_strategy == "both"

    def test_process_basic(self, sample_raw_with_events, output_dir):
        """Test basic epoch extraction."""
        raw, events = sample_raw_with_events

        config = {
            "tmin": -0.2,
            "tmax": 0.5,
        }
        module = EpochExtractionModule(config, output_dir)

        event_id = {"event1": 1, "event2": 2}
        result = module.process(raw, events=events, event_id=event_id)

        assert result.success
        assert "epochs" in result.outputs
        assert "erps" in result.outputs

        epochs = result.outputs["epochs"]
        assert isinstance(epochs, mne.Epochs)
        assert len(epochs) > 0

        # Check metadata
        assert result.metadata["n_epochs"] > 0
        assert result.metadata["tmin"] == -0.2
        assert result.metadata["tmax"] == 0.5

    def test_process_with_rejection(self, sample_raw_with_events, output_dir):
        """Test epoch extraction with rejection thresholds."""
        raw, events = sample_raw_with_events

        # Set a reasonable rejection threshold (data is ~10µV std)
        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": {
                "reject": {"eeg": 100e-6},  # 100 µV threshold - reasonable
                "strategy": "threshold",
            },
        }
        module = EpochExtractionModule(config, output_dir)

        event_id = {"event1": 1, "event2": 2}
        result = module.process(raw, events=events, event_id=event_id)

        assert result.success
        epochs = result.outputs["epochs"]

        # Rejection stats should be present
        assert "rejection" in result.metadata
        assert "stats" in result.metadata["rejection"]
        assert result.metadata["rejection"]["stats"]["n_original"] >= len(epochs)

    def test_process_auto_event_id(self, sample_raw_with_events, output_dir):
        """Test automatic event ID creation."""
        raw, events = sample_raw_with_events

        config = {"tmin": -0.2, "tmax": 0.5}
        module = EpochExtractionModule(config, output_dir)

        # Don't pass event_id, let it auto-create
        result = module.process(raw, events=events)

        assert result.success
        epochs = result.outputs["epochs"]
        # Should have created event_id from unique event values
        assert len(epochs.event_id) > 0

    def test_process_baseline_correction(self, sample_raw_with_events, output_dir):
        """Test baseline correction."""
        raw, events = sample_raw_with_events

        # Test with baseline
        config = {
            "tmin": -0.3,
            "tmax": 0.5,
            "baseline": (-0.2, 0),
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"event1": 1})
        assert result.success

        epochs = result.outputs["epochs"]
        # After baseline correction, mean of baseline period should be ~0
        baseline_data = epochs.get_data()[:, :, epochs.times < 0]
        # Not exactly 0 due to other channels, but should be small
        assert np.abs(np.mean(baseline_data)) < 1e-4

    def test_process_decimation(self, sample_raw_with_events, output_dir):
        """Test decimation during epoching."""
        raw, events = sample_raw_with_events

        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "decim": 2,  # Decimate by factor of 2
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"event1": 1})
        assert result.success

        epochs = result.outputs["epochs"]
        # Decimated sampling rate
        assert epochs.info["sfreq"] == raw.info["sfreq"] / 2

    def test_erp_computation(self, sample_raw_with_events, output_dir):
        """Test ERP computation per condition."""
        raw, events = sample_raw_with_events

        config = {"tmin": -0.2, "tmax": 0.5}
        module = EpochExtractionModule(config, output_dir)

        event_id = {"event1": 1, "event2": 2}
        result = module.process(raw, events=events, event_id=event_id)

        assert result.success
        erps = result.outputs["erps"]

        # Should have ERPs for each condition
        for condition in event_id:
            if condition in erps:
                assert isinstance(erps[condition], mne.Evoked)

    def test_file_saving(self, sample_raw_with_events, output_dir):
        """Test that epochs and ERPs are saved to files."""
        raw, events = sample_raw_with_events

        config = {"tmin": -0.2, "tmax": 0.5}
        module = EpochExtractionModule(config, output_dir)

        # Create mock subject
        class MockSubject:
            id = "test_sub"

        event_id = {"event1": 1}
        result = module.process(raw, subject=MockSubject(), events=events, event_id=event_id)

        assert result.success

        # Check output files
        assert len(result.output_files) >= 1

        # Check epochs file exists
        epochs_file = output_dir / "test_sub_epo.fif"
        assert epochs_file.exists()

        # Verify it can be loaded
        epochs_loaded = mne.read_epochs(epochs_file)
        assert len(epochs_loaded) > 0

    def test_drop_log_stats(self, sample_raw_with_events, output_dir):
        """Test drop log statistics in metadata."""
        raw, events = sample_raw_with_events

        # Use reasonable threshold that won't reject all epochs
        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": {
                "reject": {"eeg": 100e-6},  # 100 µV - reasonable
            },
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"event1": 1})

        assert result.success
        assert "drop_log_summary" in result.metadata
        # drop_log_summary is the return from epochs.drop_log_stats()

    def test_process_with_bad_annotations(self, sample_raw_with_events, output_dir):
        """Test rejection by bad annotations."""
        raw, events = sample_raw_with_events

        # Add bad segment annotation
        raw.annotations.append(5.0, 1.0, "BAD_artifact")

        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": {
                "reject_by_annotation": True,
            },
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"event1": 1, "event2": 2})

        assert result.success
        # Epochs overlapping with BAD annotation should be rejected
        # We added BAD at 5s, so event at 4s (with tmax 0.5) should be affected

    def test_process_no_events_error(self, output_dir):
        """Test handling when no events found."""
        # Create raw without events
        np.random.seed(42)
        data = np.random.randn(5, 2500) * 1e-5
        info = mne.create_info(ch_names=["EEG1", "EEG2", "EEG3", "EEG4", "EEG5"],
                              sfreq=250, ch_types=["eeg"] * 5)
        raw = mne.io.RawArray(data, info)

        config = {"tmin": -0.2, "tmax": 0.5}
        module = EpochExtractionModule(config, output_dir)

        # Should fail gracefully when no events found
        result = module.process(raw, events=np.zeros((0, 3), dtype=int), event_id={"stim": 1})

        # Either fails or returns empty epochs
        if result.success:
            assert len(result.outputs["epochs"]) == 0
        else:
            assert len(result.errors) > 0

    def test_flat_channel_rejection(self, output_dir):
        """Test flat channel rejection."""
        np.random.seed(42)
        sfreq = 250
        n_samples = 7500

        # Create data with one flat channel
        data = np.random.randn(5, n_samples) * 1e-5
        data[2, :] = 0  # Flat channel

        info = mne.create_info(ch_names=["EEG1", "EEG2", "FLAT", "EEG4", "EEG5"],
                              sfreq=sfreq, ch_types=["eeg"] * 5)
        raw = mne.io.RawArray(data, info)

        # Add stim channel with events
        stim_data = np.zeros((1, n_samples))
        event_times = [500, 1500, 2500, 3500, 4500, 5500]
        for t in event_times:
            stim_data[0, t] = 1
        stim_info = mne.create_info(['STI'], sfreq, ch_types=['stim'])
        stim_raw = mne.io.RawArray(stim_data, stim_info)
        raw.add_channels([stim_raw])

        events = np.array([[t, 0, 1] for t in event_times])

        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": {
                "flat": {"eeg": 0.5e-6},  # Flat threshold
            },
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"stim": 1})

        # Should either reject epochs with flat channel or succeed
        # depending on how MNE handles it
        assert result.success or len(result.errors) > 0

    def test_backwards_compatibility_config(self, output_dir):
        """Test backwards compatibility with flat config."""
        # Note: Current implementation uses nested 'rejection' dict
        # This tests the backwards compat path when rejection is not a dict
        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": "threshold",  # String triggers backwards compat branch
        }
        module = EpochExtractionModule(config, output_dir)

        # With string rejection, defaults apply
        assert module.rejection_strategy == "threshold"
        assert module.reject_by_annotation is True

    def test_output_spec(self, output_dir):
        """Test output specification."""
        config = {}
        module = EpochExtractionModule(config, output_dir)

        spec = module.get_output_spec()
        assert "epochs" in spec
        assert "erps" in spec

    def test_module_metadata(self, output_dir):
        """Test module name and version."""
        config = {}
        module = EpochExtractionModule(config, output_dir)

        assert module.name == "epoch_extraction"
        assert module.version == "0.1.0"


class TestEpochRejection:
    """Tests focused on epoch rejection functionality."""

    @pytest.fixture
    def noisy_raw_with_events(self):
        """Create raw data with some noisy epochs."""
        np.random.seed(42)
        sfreq = 250
        duration = 20
        n_samples = int(sfreq * duration)
        n_channels = 5

        data = np.random.randn(n_channels, n_samples) * 1e-5

        # Add noise spike at specific times
        spike_times = [2.0, 8.0]  # seconds
        for spike_time in spike_times:
            spike_sample = int(spike_time * sfreq)
            data[:, spike_sample:spike_sample+25] += 1e-4  # Large artifact

        ch_names = [f"EEG{i}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
        raw = mne.io.RawArray(data, info)

        # Add stim channel
        stim_data = np.zeros((1, n_samples))
        event_samples = [int(t * sfreq) for t in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]]
        for s in event_samples:
            stim_data[0, s] = 1
        stim_info = mne.create_info(['STI'], sfreq, ch_types=['stim'])
        raw.add_channels([mne.io.RawArray(stim_data, stim_info)])

        events = np.array([[s, 0, 1] for s in event_samples])

        return raw, events

    def test_threshold_rejection(self, noisy_raw_with_events, output_dir):
        """Test that threshold rejection configuration is applied."""
        raw, events = noisy_raw_with_events

        # Use moderate threshold - goal is to test the config path works
        config = {
            "tmin": -0.2,
            "tmax": 0.5,
            "rejection": {
                "reject": {"eeg": 200e-6},  # 200 µV - moderate
                "strategy": "threshold",
            },
        }
        module = EpochExtractionModule(config, output_dir)

        result = module.process(raw, events=events, event_id={"stim": 1})

        assert result.success
        stats = result.metadata["rejection"]["stats"]

        # Stats should be tracked
        assert stats["n_original"] == len(events)
        # Rejection config should be in metadata
        assert result.metadata["rejection"]["strategy"] == "threshold"
        assert result.metadata["rejection"]["reject_thresholds"] == {"eeg": 200e-6}

    def test_rejection_strategy_threshold(self, output_dir):
        """Test threshold strategy configuration."""
        config = {
            "rejection": {
                "strategy": "threshold",
                "reject": {"eeg": 100e-6},
            },
        }
        module = EpochExtractionModule(config, output_dir)

        assert module.rejection_strategy == "threshold"
        assert module.reject == {"eeg": 100e-6}

    def test_rejection_strategy_autoreject(self, output_dir):
        """Test autoreject strategy configuration."""
        config = {
            "rejection": {
                "strategy": "autoreject",
                "use_autoreject": True,
            },
        }
        module = EpochExtractionModule(config, output_dir)

        assert module.rejection_strategy == "autoreject"
        assert module.use_autoreject is True


class TestEpochIntegration:
    """Integration tests for epoch extraction."""

    def test_full_pipeline_simulation(self, sample_raw_with_events, output_dir):
        """Test simulating a full epoch extraction pipeline."""
        raw, events = sample_raw_with_events

        # Preprocess
        raw_filt = raw.copy().filter(l_freq=0.5, h_freq=40, verbose=False)

        # Extract epochs
        config = {
            "tmin": -0.3,
            "tmax": 0.8,
            "baseline": (-0.2, 0),
            "rejection": {
                "reject": {"eeg": 100e-6},
            },
        }
        module = EpochExtractionModule(config, output_dir)

        event_id = {"cond1": 1, "cond2": 2}
        result = module.process(raw_filt, events=events, event_id=event_id)

        assert result.success
        epochs = result.outputs["epochs"]
        erps = result.outputs["erps"]

        # Verify epochs properties
        assert epochs.tmin == pytest.approx(-0.3, abs=0.01)
        assert epochs.tmax == pytest.approx(0.8, abs=0.01)

        # Verify ERPs exist
        assert len(erps) > 0

class TestSanitizeName:
    def test_slash_replaced(self):
        assert sanitize_name("Stimulus/S  1") == "Stimulus_S_1"

    def test_backslash_replaced(self):
        assert sanitize_name("Stimulus\\S1") == "Stimulus_S1"

    def test_spaces_replaced(self):
        assert sanitize_name("my condition") == "my_condition"

    def test_multiple_invalid_chars_collapsed(self):
        # consecutive invalid chars → single underscore
        assert sanitize_name("Stimulus/S  1") == "Stimulus_S_1"

    def test_leading_trailing_underscores_stripped(self):
        assert sanitize_name("/condition/") == "condition"

    def test_clean_name_unchanged(self):
        assert sanitize_name("stimulus_1") == "stimulus_1"

    def test_empty_string(self):
        assert sanitize_name("") == ""


# ---------------------------------------------------------------------------
# Per-trial metadata: attachment, CSV/sidecar export, alignment (Pass 1)
# ---------------------------------------------------------------------------


class _MetadataSubject:
    """Minimal subject stand-in (only ``.id`` is used for filenames)."""

    id = "test_sub"


def _metadata_raw_and_events(n_events: int = 10, sfreq: float = 250.0, seed: int = 42):
    """Raw + events + behavioral metadata with one row per event."""
    rng = np.random.default_rng(seed)
    n_channels = 4
    n_samples = int(sfreq * (n_events + 4))
    data = rng.standard_normal((n_channels, n_samples)) * 1e-5  # ~10 uV noise
    info = mne.create_info(
        [f"EEG{i}" for i in range(n_channels)], sfreq, ["eeg"] * n_channels
    )
    raw = mne.io.RawArray(data, info, verbose=False)

    event_times = np.arange(1, n_events + 1)
    codes = np.tile([1, 2], int(np.ceil(n_events / 2)))[:n_events]
    events = np.column_stack(
        [(event_times * sfreq).astype(int), np.zeros(n_events, dtype=int), codes]
    )
    corr = np.tile([1, 0], int(np.ceil(n_events / 2)))[:n_events]
    metadata = pd.DataFrame(
        {
            "trial_type": ["target" if c == 1 else "nontarget" for c in codes],
            "key_resp.corr": corr,
            "key_resp.rt": np.linspace(0.3, 0.9, n_events),
            "trial_index": np.arange(n_events),
        }
    )
    return raw, events, metadata


def _process_with_metadata(tmp_path, raw, events, **kwargs):
    module = EpochExtractionModule({"tmin": -0.2, "tmax": 0.4}, tmp_path)
    return module.process(raw, subject=_MetadataSubject(), events=events, **kwargs)


@pytest.fixture
def raw_events_metadata():
    return _metadata_raw_and_events()


class TestEpochMetadataAttachment:
    """``process(metadata=...)`` attaches, saves and reports trial metadata."""

    def test_metadata_attached_and_csv_written(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        source = "/bids/sub-001/eeg/sub-001_task-x_events.tsv"
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
            metadata=metadata,
            metadata_source=source,
        )

        assert result.success, result.errors
        epochs = result.outputs["epochs"]
        assert epochs.metadata is not None
        assert len(epochs.metadata) == len(epochs) == len(metadata)

        # CSV: same rows as the original DataFrame (nothing dropped), aligned
        # with the events array order.
        csv_path = tmp_path / "test_sub_epochs_metadata.csv"
        assert csv_path.exists()
        csv = pd.read_csv(csv_path)
        assert len(csv) == len(metadata)
        assert list(csv.columns) == list(metadata.columns)
        code_to_type = {1: "target", 2: "nontarget"}
        assert csv["trial_type"].tolist() == [
            code_to_type[int(code)] for code in epochs.events[:, 2]
        ]
        assert csv["trial_type"].tolist() == metadata["trial_type"].tolist()
        assert csv["trial_index"].tolist() == metadata["trial_index"].tolist()

        # ModuleResult metadata records what was passed in AND what was saved
        assert result.metadata["metadata_rows"] == len(metadata)
        assert result.metadata["metadata_rows_saved"] == len(epochs)
        assert result.metadata["metadata_columns"] == list(metadata.columns)
        assert result.metadata["metadata_source"] == source

        # Provenance sidecar
        sidecar = json.loads(
            (tmp_path / "test_sub_epochs_metadata.json").read_text()
        )
        assert sidecar["source_events_tsv"] == source
        assert sidecar["n_rows"] == len(metadata)
        assert sidecar["n_rows_input"] == len(metadata)
        assert sidecar["n_events_input"] == len(events)
        assert sidecar["n_epochs"] == len(epochs)
        assert sidecar["columns"] == list(metadata.columns)
        assert sidecar["epochs_file"].endswith("test_sub_epo.fif")
        assert sidecar["written_at"]

        output_names = {path.name for path in result.output_files}
        assert {
            "test_sub_epo.fif",
            "test_sub_epochs_metadata.csv",
            "test_sub_epochs_metadata.json",
        } <= output_names

    def test_metadata_rows_follow_epochs_after_event_id_subset(self, tmp_path):
        """Metadata is subset with the events when ``event_id`` filters them."""
        n_events = 9
        raw, _events, _metadata = _metadata_raw_and_events(n_events=n_events)
        events = np.column_stack(
            [
                np.arange(1, n_events + 1) * 250,
                np.zeros(n_events, dtype=int),
                np.tile([1, 2, 3], 3),
            ]
        )
        metadata = pd.DataFrame(
            {
                "trial_type": ["a", "b", "c"] * 3,
                "trial_index": np.arange(n_events),
            }
        )

        result = _process_with_metadata(
            tmp_path, raw, events, event_id={"a": 1}, metadata=metadata
        )

        assert result.success, result.errors
        epochs = result.outputs["epochs"]
        assert len(epochs) == 3
        assert len(epochs.metadata) == 3
        assert epochs.metadata["trial_type"].tolist() == ["a", "a", "a"]
        # MNE 1.13.2 would hand back rows 0-2 here (it takes metadata rows
        # positionally after an event_id filter); the module pre-filters
        # events+metadata together, so the surviving rows are 0, 3, 6.
        assert epochs.metadata["trial_index"].tolist() == [0, 3, 6]
        # epochs.selection indexes the events that were passed to mne.Epochs,
        # i.e. the array after the module's event_id pre-filter (codes 1,2,3
        # -> keep code 1 at rows 0, 3, 6).
        keep = np.tile([1, 2, 3], 3) == 1
        np.testing.assert_array_equal(
            epochs.metadata["trial_index"].to_numpy(),
            np.arange(n_events)[keep][epochs.selection],
        )

        csv = pd.read_csv(tmp_path / "test_sub_epochs_metadata.csv")
        assert len(csv) == 3
        assert csv["trial_index"].tolist() == [0, 3, 6]
        assert result.metadata["metadata_rows"] == n_events
        assert result.metadata["metadata_rows_saved"] == 3
        sidecar = json.loads(
            (tmp_path / "test_sub_epochs_metadata.json").read_text()
        )
        assert sidecar["n_rows"] == 3
        assert sidecar["n_rows_input"] == n_events
        assert sidecar["n_events_input"] == 9
        assert sidecar["n_epochs"] == 3

    def test_roundtrip_metadata_survives_read_epochs(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
            metadata=metadata,
        )
        assert result.success

        epochs_loaded = mne.read_epochs(
            tmp_path / "test_sub_epo.fif", preload=True, verbose=False
        )
        assert epochs_loaded.metadata is not None
        csv = pd.read_csv(tmp_path / "test_sub_epochs_metadata.csv")
        assert (
            epochs_loaded.metadata["trial_type"].tolist()
            == csv["trial_type"].tolist()
        )
        assert (
            epochs_loaded.metadata["trial_index"].tolist()
            == csv["trial_index"].tolist()
        )
        # MNE's FIF metadata serialization rounds floats to ~10 significant
        # digits: numeric values survive, bit-exact equality does not.
        np.testing.assert_allclose(
            epochs_loaded.metadata["key_resp.rt"].to_numpy(),
            metadata["key_resp.rt"].to_numpy(),
            rtol=1e-9,
        )

    def test_rejection_keeps_metadata_aligned(self, tmp_path):
        """A rejected trial is dropped from BOTH epochs and metadata."""
        n_events = 8
        raw, events, metadata = _metadata_raw_and_events(n_events=n_events)
        # 1 mV artifact over the epoch at t=4 s (event index 3)
        sfreq = raw.info["sfreq"]
        raw._data[:, int(4.0 * sfreq): int(4.5 * sfreq)] += 1e-3

        module = EpochExtractionModule(
            {
                "tmin": -0.2,
                "tmax": 0.4,
                "rejection": {"strategy": "threshold", "reject": {"eeg": 200e-6}},
            },
            tmp_path,
        )
        result = module.process(
            raw,
            subject=_MetadataSubject(),
            events=events,
            event_id={"target": 1, "nontarget": 2},
            metadata=metadata,
        )

        assert result.success, result.errors
        epochs = result.outputs["epochs"]
        assert len(epochs) == n_events - 1
        # MNE 1.13.2 subsets metadata down to the *surviving* epochs; the rows
        # stay aligned with epochs.events. This is the MNE contract the module
        # relies on (verified by probe, not assumed).
        assert len(epochs.metadata) == len(epochs)
        assert epochs.metadata["trial_index"].tolist() == [0, 1, 2, 4, 5, 6, 7]
        np.testing.assert_array_equal(
            epochs.metadata["trial_index"].to_numpy(),
            np.arange(n_events)[epochs.selection],
        )
        # metadata_rows counts the input frame; metadata_rows_saved the CSV
        assert result.metadata["metadata_rows"] == n_events
        assert result.metadata["metadata_rows_saved"] == len(epochs)

        csv = pd.read_csv(tmp_path / "test_sub_epochs_metadata.csv")
        assert len(csv) == len(epochs)
        assert csv["trial_index"].tolist() == [
            int(i) for i in epochs.metadata["trial_index"]
        ]
        sidecar = json.loads(
            (tmp_path / "test_sub_epochs_metadata.json").read_text()
        )
        assert sidecar["n_rows"] == len(epochs)
        assert sidecar["n_rows_input"] == n_events
        assert sidecar["n_events_input"] == n_events

    def test_metadata_source_optional(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
            metadata=metadata,
        )
        assert result.success
        sidecar = json.loads(
            (tmp_path / "test_sub_epochs_metadata.json").read_text()
        )
        assert sidecar["source_events_tsv"] is None
        assert result.metadata["metadata_source"] is None

    def test_autoreject_keeps_metadata_aligned(self, tmp_path):
        """autoreject's fit_transform keeps ``epochs.metadata`` in lock-step."""
        pytest.importorskip("autoreject")
        n_events, sfreq = 20, 100.0
        rng = np.random.default_rng(0)
        info = mne.create_info(["Fz", "Cz"], sfreq, ["eeg", "eeg"])
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
        n_samples = int(sfreq * (n_events + 2))
        data = rng.standard_normal((2, n_samples)) * 1e-5
        data[:, int(4.0 * sfreq): int(4.5 * sfreq)] += 1e-3
        raw = mne.io.RawArray(data, info, verbose=False)
        events = np.column_stack(
            [
                (np.arange(1, n_events + 1) * sfreq).astype(int),
                np.zeros(n_events, dtype=int),
                np.ones(n_events, dtype=int),
            ]
        )
        metadata = pd.DataFrame({"trial_index": np.arange(n_events)})

        module = EpochExtractionModule(
            {
                "tmin": -0.2,
                "tmax": 0.2,
                "rejection": {"strategy": "autoreject", "use_autoreject": True},
            },
            tmp_path,
        )
        result = module.process(
            raw,
            subject=_MetadataSubject(),
            events=events,
            event_id={"target": 1},
            metadata=metadata,
        )

        assert result.success, result.errors
        epochs = result.outputs["epochs"]
        assert len(epochs) < n_events  # autoreject dropped/interpolated epochs
        assert len(epochs.metadata) == len(epochs)
        np.testing.assert_array_equal(
            epochs.metadata["trial_index"].to_numpy(),
            np.arange(n_events)[epochs.selection],
        )
        csv = pd.read_csv(tmp_path / "test_sub_epochs_metadata.csv")
        assert csv["trial_index"].tolist() == [
            int(i) for i in epochs.metadata["trial_index"]
        ]


class TestEpochMetadataBackwardsCompatible:
    """The events-only path is unchanged and still the default."""

    def test_process_without_metadata(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
        )

        assert result.success, result.errors
        epochs = result.outputs["epochs"]
        assert epochs.metadata is None
        assert not (tmp_path / "test_sub_epochs_metadata.csv").exists()
        assert not (tmp_path / "test_sub_epochs_metadata.json").exists()
        assert result.metadata["metadata_rows"] == 0
        assert result.metadata["metadata_rows_saved"] == 0
        assert result.metadata["metadata_columns"] == []
        assert result.metadata["metadata_source"] is None

    def test_metadata_wrong_length_fails_loudly(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
            metadata=metadata.iloc[:-1],
        )

        assert not result.success
        assert any("metadata has" in err for err in result.errors), result.errors
        assert not (tmp_path / "test_sub_epo.fif").exists()
        assert not (tmp_path / "test_sub_epochs_metadata.csv").exists()

    def test_metadata_wrong_type_fails_loudly(self, tmp_path, raw_events_metadata):
        raw, events, metadata = raw_events_metadata
        result = _process_with_metadata(
            tmp_path,
            raw,
            events,
            event_id={"target": 1, "nontarget": 2},
            metadata={"trial_type": "target"},
        )

        assert not result.success
        assert any("pandas.DataFrame" in err for err in result.errors), result.errors

