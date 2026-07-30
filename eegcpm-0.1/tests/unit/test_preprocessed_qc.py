"""Tests for Preprocessed QC module."""

import pytest
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from eegcpm.modules.qc.preprocessed_qc import PreprocessedQC
from eegcpm.modules.qc.base import QCResult


class TestPreprocessedQCBasic:
    """Test PreprocessedQC basic functionality."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample preprocessed raw object for testing."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 60  # 60 seconds
        data = np.random.randn(len(ch_names), n_samples) * 15e-6  # ~15 µV (cleaner than raw)
        raw = mne.io.RawArray(data, info)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="warn")
        return raw

    @pytest.fixture
    def sample_ica(self, sample_raw):
        """Create a sample ICA object."""
        ica = mne.preprocessing.ICA(n_components=10, random_state=42, max_iter=100)
        # Filter before ICA to avoid warnings
        raw_filt = sample_raw.copy().filter(1.0, 40.0, verbose=False)
        ica.fit(raw_filt, verbose=False)
        # Exclude some components
        ica.exclude = [0, 2]
        return ica

    def test_preprocessed_qc_initialization(self):
        """Test PreprocessedQC initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir), dpi=72)

            assert qc.dpi == 72
            assert qc.output_dir.exists()

    def test_preprocessed_qc_with_config(self):
        """Test PreprocessedQC with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"custom_setting": True}
            qc = PreprocessedQC(Path(tmpdir), config=config)

            assert qc.config == config

    def test_compute_basic_metrics(self, sample_raw):
        """Test computing basic QC metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            assert result.subject_id == "sub-001"
            assert len(result.metrics) > 0

            metric_names = [m.name for m in result.metrics]
            assert "Duration" in metric_names
            assert "Sampling Rate" in metric_names
            assert "N Channels" in metric_names
            assert "N EEG Channels" in metric_names

    def test_compute_amplitude_metrics(self, sample_raw):
        """Test amplitude metrics computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            metric_names = [m.name for m in result.metrics]
            assert "Mean Amplitude" in metric_names
            assert "Max Amplitude" in metric_names
            assert "Std Amplitude" in metric_names

    def test_compute_generates_figures(self, sample_raw):
        """Test that figures are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            assert len(result.figures) > 0
            assert "psd" in result.figures
            assert "amplitude_dist" in result.figures
            assert "channel_variance" in result.figures

    def test_overall_status_ok(self, sample_raw):
        """Test overall status is OK for clean data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            # Clean preprocessed data should be OK
            assert result.status in ["ok", "warning"]


class TestPreprocessedQCWithICA:
    """Test PreprocessedQC with ICA information."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample preprocessed raw object."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3", "Cz", "C4"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 60
        data = np.random.randn(len(ch_names), n_samples) * 15e-6
        raw = mne.io.RawArray(data, info)
        return raw

    @pytest.fixture
    def sample_ica(self, sample_raw):
        """Create a sample ICA object."""
        ica = mne.preprocessing.ICA(n_components=6, random_state=42, max_iter=100)
        raw_filt = sample_raw.copy().filter(1.0, 40.0, verbose=False)
        ica.fit(raw_filt, verbose=False)
        ica.exclude = [0, 1]  # Exclude 2 components
        return ica

    def test_compute_with_ica(self, sample_raw, sample_ica):
        """Test QC with ICA object provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001", ica=sample_ica)

            metric_names = [m.name for m in result.metrics]
            assert "ICA Components" in metric_names
            assert "ICA Excluded" in metric_names
            assert "ICA Excluded %" in metric_names

            # Check ICA figure is generated
            assert "ica_components" in result.figures

    def test_ica_metrics_values(self, sample_raw, sample_ica):
        """Test ICA metric values are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001", ica=sample_ica)

            ica_comp = next(m for m in result.metrics if m.name == "ICA Components")
            ica_excl = next(m for m in result.metrics if m.name == "ICA Excluded")

            assert ica_comp.value == 6
            assert ica_excl.value == 2

    def test_ica_high_exclusion_warning(self, sample_raw):
        """Test warning when too many ICA components are excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create ICA with many excluded components
            ica = mne.preprocessing.ICA(n_components=6, random_state=42, max_iter=100)
            raw_filt = sample_raw.copy().filter(1.0, 40.0, verbose=False)
            ica.fit(raw_filt, verbose=False)
            ica.exclude = [0, 1, 2, 3, 4]  # Exclude 5 of 6 (>50%)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001", ica=ica)

            # Should have a warning note
            assert any("ICA components excluded" in note for note in result.notes)


class TestPreprocessedQCBeforeAfter:
    """Test before/after comparison functionality."""

    @pytest.fixture
    def raw_before(self):
        """Create a 'before preprocessing' raw object."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 30
        # More noise (larger amplitude)
        data = np.random.randn(len(ch_names), n_samples) * 50e-6
        return mne.io.RawArray(data, info)

    @pytest.fixture
    def raw_after(self):
        """Create an 'after preprocessing' raw object."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz", "F4", "C3"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 30
        # Less noise (smaller amplitude)
        data = np.random.randn(len(ch_names), n_samples) * 15e-6
        return mne.io.RawArray(data, info)

    def test_before_after_comparison(self, raw_before, raw_after):
        """Test before/after PSD comparison plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw_after, "sub-001", raw_before=raw_before)

            assert "before_after" in result.figures


class TestPreprocessedQCAnnotations:
    """Test handling of bad annotations/segments."""

    @pytest.fixture
    def raw_with_annotations(self):
        """Create raw with bad segment annotations."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 60
        data = np.random.randn(len(ch_names), n_samples) * 15e-6
        raw = mne.io.RawArray(data, info)

        # Add bad segment annotations
        onset = [5, 15, 30]
        duration = [2, 3, 5]  # Total 10s of 60s = 16.7%
        description = ["BAD_artifact", "BAD_blink", "BAD_muscle"]
        annotations = mne.Annotations(onset, duration, description)
        raw.set_annotations(annotations)

        return raw

    def test_bad_segment_metrics(self, raw_with_annotations):
        """Test bad segment metrics are computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw_with_annotations, "sub-001")

            metric_names = [m.name for m in result.metrics]
            assert "Bad Segments" in metric_names
            assert "Bad Segment %" in metric_names

    def test_bad_segment_values(self, raw_with_annotations):
        """Test bad segment metric values are correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw_with_annotations, "sub-001")

            bad_seg = next(m for m in result.metrics if m.name == "Bad Segments")
            bad_pct = next(m for m in result.metrics if m.name == "Bad Segment %")

            assert bad_seg.value == 3
            assert 16 < bad_pct.value < 18  # ~16.7%


class TestPreprocessedQCHTMLReport:
    """Test HTML report generation."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample preprocessed raw object."""
        ch_names = ["Fp1", "Fp2", "F3", "Fz"]
        info = mne.create_info(
            ch_names=ch_names,
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        n_samples = 256 * 30
        data = np.random.randn(len(ch_names), n_samples) * 15e-6
        return mne.io.RawArray(data, info)

    def test_generate_html_report(self, sample_raw):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            html = qc.generate_html_report(result)

            assert "<!DOCTYPE html>" in html
            assert "sub-001" in html
            assert "Preprocessing Metrics" in html

    def test_generate_html_report_save(self, sample_raw):
        """Test saving HTML report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            html_path = Path(tmpdir) / "test_report.html"
            qc.generate_html_report(result, save_path=html_path)

            assert html_path.exists()
            assert html_path.stat().st_size > 0


class TestPreprocessedQCEdgeCases:
    """Test edge cases and error handling."""

    def test_no_eeg_channels(self):
        """Test handling of data with no EEG channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EMG 001", "EMG 002"],
                sfreq=256.0,
                ch_types="emg",
            )
            data = np.random.randn(2, 256 * 10) * 20e-6
            raw = mne.io.RawArray(data, info)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            n_eeg = next(m for m in result.metrics if m.name == "N EEG Channels")
            assert n_eeg.value == 0

    def test_all_channels_bad(self):
        """Test handling when all channels are marked bad."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002"],
                sfreq=256.0,
                ch_types="eeg",
            )
            data = np.random.randn(2, 256 * 10) * 15e-6
            raw = mne.io.RawArray(data, info)
            raw.info["bads"] = ["EEG 001", "EEG 002"]

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            assert result is not None
            bad_pct = next(m for m in result.metrics if m.name == "Bad Channels")
            assert bad_pct.status == "bad"

    def test_short_recording(self):
        """Test with very short recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003"],
                sfreq=256.0,
                ch_types="eeg",
            )
            data = np.random.randn(3, 256 * 3) * 15e-6  # 3 seconds
            raw = mne.io.RawArray(data, info)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            assert result is not None
            assert result.subject_id == "sub-001"

    def test_minimal_channels(self):
        """Test with minimal number of channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001"],
                sfreq=256.0,
                ch_types="eeg",
            )
            data = np.random.randn(1, 256 * 10) * 15e-6
            raw = mne.io.RawArray(data, info)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            assert result is not None

    def test_no_annotations(self):
        """Test handling of data with no annotations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002"],
                sfreq=256.0,
                ch_types="eeg",
            )
            data = np.random.randn(2, 256 * 10) * 15e-6
            raw = mne.io.RawArray(data, info)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            # No annotations -> Bad Segments metric reports zero (schema is
            # stable: metric always emitted, value 0 when none)
            bad = [m for m in result.metrics if m.name == "Bad Segments"]
            assert len(bad) == 1 and bad[0].value == 0

    def test_ica_few_components(self):
        """Test with ICA having very few components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ch_names = ["EEG 001", "EEG 002", "EEG 003"]
            info = mne.create_info(
                ch_names=ch_names,
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            data = np.random.randn(3, 256 * 30) * 15e-6
            raw = mne.io.RawArray(data, info)

            ica = mne.preprocessing.ICA(n_components=2, random_state=42, max_iter=100)
            raw_filt = raw.copy().filter(1.0, 40.0, verbose=False)
            ica.fit(raw_filt, verbose=False)

            qc = PreprocessedQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001", ica=ica)

            assert "ica_components" in result.figures
