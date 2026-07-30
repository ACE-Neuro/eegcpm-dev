"""Tests for Quality Control module."""

import json
import pytest
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from eegcpm.modules.qc.base import BaseQC, QCMetric, QCResult
from eegcpm.modules.qc.html_report import HTMLReportBuilder, QCIndexBuilder
from eegcpm.modules.qc.raw_qc import RawQC, run_raw_qc_batch


class TestQCMetric:
    """Test QCMetric dataclass."""

    def test_qc_metric_basic(self):
        """Test basic QCMetric creation."""
        metric = QCMetric(
            name="Test Metric",
            value=42.5,
            unit="µV",
            status="ok",
        )

        assert metric.name == "Test Metric"
        assert metric.value == 42.5
        assert metric.unit == "µV"
        assert metric.status == "ok"

    def test_qc_metric_with_thresholds(self):
        """Test QCMetric with threshold values."""
        metric = QCMetric(
            name="Amplitude",
            value=150.0,
            unit="µV",
            status="warning",
            threshold_warning=100.0,
            threshold_bad=200.0,
        )

        assert metric.threshold_warning == 100.0
        assert metric.threshold_bad == 200.0

    def test_qc_metric_to_dict(self):
        """Test conversion to dictionary."""
        metric = QCMetric(
            name="Duration",
            value=120.5,
            unit="s",
            status="ok",
        )

        d = metric.to_dict()

        assert d["name"] == "Duration"
        assert d["value"] == 120.5
        assert d["unit"] == "s"
        assert d["status"] == "ok"


class TestQCResult:
    """Test QCResult dataclass."""

    def test_qc_result_basic(self):
        """Test basic QCResult creation."""
        result = QCResult(subject_id="sub-001")

        assert result.subject_id == "sub-001"
        assert len(result.metrics) == 0
        assert len(result.figures) == 0
        assert result.status == "ok"

    def test_add_metric_chaining(self):
        """Test method chaining for add_metric."""
        result = (
            QCResult(subject_id="sub-001")
            .add_metric(QCMetric("M1", 1.0, ""))
            .add_metric(QCMetric("M2", 2.0, ""))
        )

        assert len(result.metrics) == 2
        assert result.metrics[0].name == "M1"
        assert result.metrics[1].name == "M2"

    def test_add_figure(self):
        """Test adding figures."""
        result = QCResult(subject_id="sub-001")

        # Create dummy PNG bytes
        fig_bytes = b"PNG_DATA"
        result.add_figure("test_plot", fig_bytes)

        assert "test_plot" in result.figures
        assert result.figures["test_plot"] == fig_bytes

    def test_add_note_chaining(self):
        """Test method chaining for add_note."""
        result = (
            QCResult(subject_id="sub-001")
            .add_note("Note 1")
            .add_note("Note 2")
        )

        assert len(result.notes) == 2
        assert "Note 1" in result.notes

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QCResult(subject_id="sub-001")
        result.add_metric(QCMetric("Test", 1.0, ""))
        result.add_note("Test note")
        result.add_figure("plot", b"PNG")

        d = result.to_dict()

        assert d["subject_id"] == "sub-001"
        assert len(d["metrics"]) == 1
        assert "plot" in d["figure_names"]
        assert "Test note" in d["notes"]

    def test_save_json(self):
        """Test saving to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = QCResult(subject_id="sub-001")
            result.add_metric(QCMetric("Duration", 100.0, "s"))

            path = Path(tmpdir) / "test_qc.json"
            result.save_json(path)

            assert path.exists()

            with open(path) as f:
                data = json.load(f)

            assert data["subject_id"] == "sub-001"


class TestBaseQC:
    """Test BaseQC abstract class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseQC cannot be instantiated directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(TypeError):
                BaseQC(Path(tmpdir))

    def test_fig_to_base64(self):
        """Test figure to base64 conversion."""
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot([1, 2, 3], [1, 4, 9])

        png_str = BaseQC.fig_to_base64(fig)

        # Documented contract: base64-encoded PNG *string* (data-URI embeddable)
        assert isinstance(png_str, str)
        import base64
        assert base64.b64decode(png_str)[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG
        assert len(png_str) > 0

        plt.close(fig)

    def test_bytes_to_data_uri(self):
        """Test converting bytes to data URI."""
        png_bytes = b'\x89PNG\r\n\x1a\nTEST'

        uri = BaseQC.bytes_to_data_uri(png_bytes)

        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > len("data:image/png;base64,")


class TestHTMLReportBuilder:
    """Test HTMLReportBuilder class."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = HTMLReportBuilder(title="Test Report")

        assert builder.title == "Test Report"
        assert len(builder.sections) == 0

    def test_add_header(self):
        """Test adding header sections."""
        builder = HTMLReportBuilder()
        builder.add_header("Test Header", level=2)

        assert len(builder.sections) == 1
        assert "<h2>" in builder.sections[0]
        assert "Test Header" in builder.sections[0]

    def test_add_text(self):
        """Test adding text paragraphs."""
        builder = HTMLReportBuilder()
        builder.add_text("Test paragraph")

        assert len(builder.sections) == 1
        assert "<p>" in builder.sections[0]

    def test_add_metrics_table(self):
        """Test adding metrics table."""
        builder = HTMLReportBuilder()
        metrics = [
            QCMetric("M1", 1.0, "unit", "ok"),
            QCMetric("M2", 2.0, "unit", "warning"),
        ]

        builder.add_metrics_table(metrics)

        assert len(builder.sections) == 1
        assert "<table>" in builder.sections[0]
        assert "M1" in builder.sections[0]
        assert "M2" in builder.sections[0]

    def test_add_figure(self):
        """Test adding figure with base64 encoding."""
        builder = HTMLReportBuilder()
        fig_bytes = b'\x89PNG\r\n\x1a\nTEST'

        builder.add_figure("test", fig_bytes, "Test caption")

        assert len(builder.sections) == 1
        assert "data:image/png;base64," in builder.sections[0]
        assert "Test caption" in builder.sections[0]

    def test_add_notes(self):
        """Test adding notes section."""
        builder = HTMLReportBuilder()
        builder.add_notes(["Note 1", "Note 2"])

        assert len(builder.sections) == 1
        assert "Note 1" in builder.sections[0]
        assert "Note 2" in builder.sections[0]

    def test_method_chaining(self):
        """Test fluent interface."""
        builder = (
            HTMLReportBuilder(title="Test")
            .add_header("Header")
            .add_text("Text")
            .add_notes(["Note"])
        )

        assert len(builder.sections) == 3

    def test_build_html(self):
        """Test building complete HTML."""
        builder = HTMLReportBuilder(title="Test Report")
        builder.add_header("Section")
        builder.add_text("Content")

        html = builder.build()

        assert "<!DOCTYPE html>" in html
        assert "<title>Test Report</title>" in html
        assert "<style>" in html
        assert "Section" in html

    def test_save_html(self):
        """Test saving HTML to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = HTMLReportBuilder(title="Test")
            builder.add_text("Test content")

            path = Path(tmpdir) / "test.html"
            builder.save(path)

            assert path.exists()
            content = path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_html_escaping(self):
        """Test that special characters are escaped."""
        builder = HTMLReportBuilder()
        builder.add_text("<script>alert('xss')</script>")

        html = builder.build()

        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestQCIndexBuilder:
    """Test QCIndexBuilder class."""

    def test_index_initialization(self):
        """Test index builder initialization."""
        index = QCIndexBuilder(title="QC Index")

        assert index.title == "QC Index"
        assert len(index.subjects) == 0

    def test_add_subject(self):
        """Test adding subjects."""
        index = QCIndexBuilder()
        index.add_subject("sub-001", "ok", "sub-001_qc.html")
        index.add_subject("sub-002", "warning", "sub-002_qc.html")

        assert len(index.subjects) == 2
        assert index.subjects[0]["id"] == "sub-001"
        assert index.subjects[1]["status"] == "warning"

    def test_method_chaining(self):
        """Test fluent interface."""
        index = (
            QCIndexBuilder()
            .add_subject("sub-001", "ok", "sub-001.html")
            .add_subject("sub-002", "bad", "sub-002.html")
        )

        assert len(index.subjects) == 2

    def test_build_index(self):
        """Test building index HTML."""
        index = QCIndexBuilder(title="Test Index")
        index.add_subject("sub-001", "ok", "sub-001.html")

        html = index.build()

        assert "<!DOCTYPE html>" in html
        assert "Test Index" in html
        assert "sub-001" in html
        assert "iframe" in html

    def test_save_index(self):
        """Test saving index HTML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index = QCIndexBuilder()
            index.add_subject("sub-001", "ok", "sub-001.html")

            path = Path(tmpdir) / "index.html"
            index.save(path)

            assert path.exists()


class TestRawQC:
    """Test RawQC class."""

    @pytest.fixture
    def sample_raw(self):
        """Create a sample raw object for testing."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "EEG 003", "EEG 004"],
            sfreq=256.0,
            ch_types="eeg",
        )
        # 30 seconds of data with realistic EEG-like values
        np.random.seed(42)
        data = np.random.randn(4, 256 * 30) * 20e-6  # ~20 µV std
        return mne.io.RawArray(data, info)

    @pytest.fixture
    def sample_raw_with_artifacts(self):
        """Create raw with artifacts for testing."""
        info = mne.create_info(
            ch_names=["EEG 001", "EEG 002", "EEG 003", "EEG 004"],
            sfreq=256.0,
            ch_types="eeg",
        )
        np.random.seed(42)
        data = np.random.randn(4, 256 * 30) * 20e-6

        # Add flatline to channel 2
        data[1, :1280] = 0

        # Mark channel 3 as bad
        raw = mne.io.RawArray(data, info)
        raw.info["bads"] = ["EEG 003"]

        return raw

    def test_raw_qc_initialization(self):
        """Test RawQC initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir), line_freq=60.0)

            assert qc.line_freq == 60.0
            assert qc.dpi == 100
            assert qc.output_dir.exists()

    def test_raw_qc_with_config(self):
        """Test RawQC with custom config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "thresh_bad_chan_pct": 30.0,
                "thresh_max_amplitude_uv": 1000.0,
            }
            qc = RawQC(Path(tmpdir), config=config)

            assert qc.thresh_bad_chan_pct == 30.0
            assert qc.thresh_max_amplitude_uv == 1000.0

    def test_compute_basic_metrics(self, sample_raw):
        """Test computing basic QC metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            assert result.subject_id == "sub-001"
            assert len(result.metrics) > 0

            # Check expected metrics exist
            metric_names = [m.name for m in result.metrics]
            assert "Duration" in metric_names
            assert "Sampling Rate" in metric_names
            assert "N Channels" in metric_names
            assert "N EEG Channels" in metric_names

    def test_compute_with_bad_channels(self, sample_raw_with_artifacts):
        """Test QC with bad channels marked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw_with_artifacts, "sub-001")

            # Find bad channel metric
            bad_metric = next(m for m in result.metrics if m.name == "Bad Channels")
            assert bad_metric.value == 1

    def test_compute_generates_figures(self, sample_raw):
        """Test that figures are generated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            assert len(result.figures) > 0
            assert "psd" in result.figures
            assert "channel_variance" in result.figures
            assert "time_series" in result.figures
            assert "correlation" in result.figures

    def test_detect_flatlines(self, sample_raw_with_artifacts):
        """Test flatline detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw_with_artifacts, "sub-001")

            flatline_metric = next(m for m in result.metrics if m.name == "Flatline Channels")
            assert flatline_metric.value >= 1

    def test_overall_status_ok(self, sample_raw):
        """Test overall status is OK for clean data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            # Clean data should be OK
            assert result.status in ["ok", "warning"]  # warning possible due to low line noise

    def test_generate_html_report(self, sample_raw):
        """Test HTML report generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            html = qc.generate_html_report(result)

            assert "<!DOCTYPE html>" in html
            assert "sub-001" in html
            assert "Metrics" in html

    def test_generate_html_report_save(self, sample_raw):
        """Test saving HTML report to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            html_path = Path(tmpdir) / "test_report.html"
            qc.generate_html_report(result, save_path=html_path)

            assert html_path.exists()
            assert html_path.stat().st_size > 0

    def test_save_report_json(self, sample_raw):
        """Test saving QC report as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir))
            result = qc.compute(sample_raw, "sub-001")

            json_path = qc.save_report(result)

            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert data["subject_id"] == "sub-001"


class TestRunRawQCBatch:
    """Test run_raw_qc_batch function."""

    @pytest.fixture
    def sample_raw_files(self):
        """Create sample raw files for batch testing."""
        tmpdir = tempfile.mkdtemp()
        files = []

        for i in range(3):
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003"],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(i)
            data = np.random.randn(3, 256 * 10) * 20e-6
            raw = mne.io.RawArray(data, info)

            fif_path = Path(tmpdir) / f"sub-{i:03d}_task-test_eeg.fif"
            raw.save(fif_path, overwrite=True, verbose=False)
            files.append(fif_path)

        yield files, tmpdir

        # Cleanup
        import shutil
        shutil.rmtree(tmpdir)

    def test_batch_processing(self, sample_raw_files):
        """Test batch QC processing."""
        files, tmpdir = sample_raw_files
        output_dir = Path(tmpdir) / "qc_output"

        results, index_path = run_raw_qc_batch(files, output_dir, line_freq=50.0)

        assert len(results) == 3
        assert index_path.exists()
        assert (output_dir / "index.html").exists()

    def test_batch_generates_individual_reports(self, sample_raw_files):
        """Test that individual reports are generated."""
        files, tmpdir = sample_raw_files
        output_dir = Path(tmpdir) / "qc_output"

        results, _ = run_raw_qc_batch(files, output_dir)

        # Check HTML and JSON files exist for each subject
        for result in results:
            html_path = output_dir / f"{result.subject_id}_qc.html"
            json_path = output_dir / f"{result.subject_id}_qc.json"

            assert html_path.exists()
            assert json_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of very short data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001"],
                sfreq=256.0,
                ch_types="eeg",
            )
            # Very short data (less than 1 second)
            data = np.random.randn(1, 100) * 20e-6
            raw = mne.io.RawArray(data, info)

            qc = RawQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            # Should still produce a result
            assert result is not None
            assert result.subject_id == "sub-001"

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

            qc = RawQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            # Should handle gracefully
            n_eeg_metric = next(m for m in result.metrics if m.name == "N EEG Channels")
            assert n_eeg_metric.value == 0

    def test_all_channels_bad(self):
        """Test handling when all channels are marked bad."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002"],
                sfreq=256.0,
                ch_types="eeg",
            )
            data = np.random.randn(2, 256 * 10) * 20e-6
            raw = mne.io.RawArray(data, info)
            raw.info["bads"] = ["EEG 001", "EEG 002"]

            qc = RawQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            # Should handle gracefully with notes
            assert result is not None
            bad_pct = next(m for m in result.metrics if m.name == "Bad Channel %")
            assert bad_pct.value == 100.0

    def test_dead_channels_correlation(self):
        """Test correlation matrix handles dead channels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            info = mne.create_info(
                ch_names=["EEG 001", "EEG 002", "EEG 003", "EEG 004"],
                sfreq=256.0,
                ch_types="eeg",
            )
            np.random.seed(42)
            data = np.random.randn(4, 256 * 10) * 20e-6
            # Make channel 1 dead (zero variance)
            data[1, :] = 0
            raw = mne.io.RawArray(data, info)

            qc = RawQC(Path(tmpdir))
            result = qc.compute(raw, "sub-001")

            # Should generate correlation without errors
            assert "correlation" in result.figures
