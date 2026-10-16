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

    def test_line_noise_failure_is_nan_not_zero(self, monkeypatch):
        """B7 rev2 fix 9: a failed line-noise measurement is NaN with a note.

        The old handler returned 0.0 — indistinguishable from a clean
        recording — and the metric then reported status ``ok``.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            qc = RawQC(Path(tmpdir), line_freq=60.0)
            raw = mne.io.RawArray(
                np.random.default_rng(0).standard_normal((4, 1000)) * 10e-6,
                mne.create_info(["A", "B", "C", "D"], 250.0, "eeg"),
            )

            def _boom(*args, **kwargs):
                raise RuntimeError("injected PSD failure")

            monkeypatch.setattr(raw, "compute_psd", _boom)
            value = qc._compute_line_noise_ratio(raw, np.arange(4))
            assert np.isnan(value)

            result = qc.compute(raw, "test_subject")
            metric = next(m for m in result.metrics if m.name == "Line Noise Ratio")
            assert np.isnan(metric.value)
            assert metric.status == "warning"
            assert any("Line noise ratio could not be computed" in n
                       for n in result.notes)

    def test_quality_control_module_line_noise_failure_is_nan(self, monkeypatch):
        """B7 rev2 fix 9: QualityControlModule records NaN + a warning too."""
        from eegcpm.modules.qc import QualityControlModule

        with tempfile.TemporaryDirectory() as tmpdir:
            module = QualityControlModule({"line_freq": 60.0}, Path(tmpdir))
            raw = mne.io.RawArray(
                np.random.default_rng(0).standard_normal((4, 2500)) * 10e-6,
                mne.create_info(["A", "B", "C", "D"], 250.0, "eeg"),
            )

            def _boom(*args, **kwargs):
                raise RuntimeError("injected PSD failure")

            monkeypatch.setattr(raw, "compute_psd", _boom)
            value = module._compute_line_noise(raw, np.arange(4))
            assert np.isnan(value)

            class _Subject:
                id = "s1"

            result = module.process(raw, subject=_Subject())
            assert result.success
            assert np.isnan(result.metadata["line_noise_power"])
            assert any("Line noise power could not be computed" in w
                       for w in result.warnings)

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


# ---------------------------------------------------------------------------
# P1-F10 (plan §B7 task 3): per-recording chain QC (hbn_recording_qc)
# ---------------------------------------------------------------------------


def _eog_raw(blink_scale=1.0, leak=0.0, sfreq=250.0, seconds=60.0, seed=0,
             line_uv=0.0):
    """Raw with 9 EGI EOG channels, blinks, and optional frontal leakage.

    ``leak`` adds a fraction of the blink waveform to the frontal EEG channels
    (volume conduction); an EOG regression can then remove it. ``line_uv`` adds
    60 Hz line noise to the frontal channels (B7 rev2 fix 4 fixture: the bias
    probe).
    """
    from eegcpm.modules.qc.hbn_recording_qc import EOG_HYDROCEL_NAMES

    ch_names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Cz"] + list(EOG_HYDROCEL_NAMES)
    ch_types = ["eeg"] * 8 + ["eog"] * len(EOG_HYDROCEL_NAMES)
    rng = np.random.default_rng(seed)
    n_times = int(sfreq * seconds)
    data = rng.standard_normal((len(ch_names), n_times)) * 5e-6
    if line_uv:
        t = np.arange(n_times) / sfreq
        frontal_all = [ch_names.index(ch) for ch in
                       ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")]
        data[frontal_all] += line_uv * 1e-6 * np.sin(2 * np.pi * 60.0 * t)[None, :]
    blink = np.exp(-0.5 * ((np.arange(int(0.3 * sfreq)) / sfreq - 0.15) / 0.06) ** 2)
    blink_onsets = np.arange(5.0, seconds - 5.0, 4.0)
    eog_idx = [ch_names.index(ch) for ch in EOG_HYDROCEL_NAMES]
    frontal_idx = [ch_names.index(ch) for ch in ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")]
    for onset in blink_onsets:
        start = int(onset * sfreq)
        stop = start + blink.size
        data[eog_idx, start:stop] += (250e-6 * blink_scale * blink)[None, :]
        if leak:
            data[frontal_idx, start:stop] += (250e-6 * blink_scale * leak * blink)[None, :]
    info = mne.create_info(ch_names, sfreq, ch_types)
    return mne.io.RawArray(data, info, verbose=False)


class TestBlinkResidualGate:
    """`blink_residual_ratio` replaces the vacuous in-sample corr gate (P1-F10)."""

    def test_blinks_removed_pass(self):
        """A recording whose frontal blinks were removed passes the gate."""
        from eegcpm.modules.qc.hbn_recording_qc import (
            OCULAR_GATE, blink_residual_ratio,
        )

        raw_pre = _eog_raw(leak=1.0)
        # Simulate the EOG regression: subtract the EOG-derived blink from the
        # frontal channels (what a real OLS regression does).
        raw_post = raw_pre.copy()
        eog_data = raw_pre.get_data(picks=["E8"])
        frontal = raw_pre.get_data(picks=["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"])
        coeffs = np.linalg.lstsq(eog_data.T, frontal.T, rcond=None)[0]
        raw_post._data[[raw_post.ch_names.index(ch) for ch in
                        ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")]] = (
            frontal - (eog_data.T @ coeffs).T
        )

        result = blink_residual_ratio(raw_pre, raw_post)
        assert result["n_blinks"] > 0
        assert result["blink_channel"] == "E8"
        assert result["ratio"] is not None
        assert result["ratio"] < OCULAR_GATE
        assert result["pass"] is True

    def test_blinks_kept_fail(self):
        """Unremoved frontal blinks must fail the gate (no vacuous pass)."""
        from eegcpm.modules.qc.hbn_recording_qc import OCULAR_GATE, blink_residual_ratio

        raw = _eog_raw(leak=0.8)
        result = blink_residual_ratio(raw, raw)
        assert result["ratio"] is not None
        assert result["ratio"] >= OCULAR_GATE
        assert result["pass"] is False

    def test_missing_raw_pre_is_not_a_pass(self):
        """No pre-chain recording → ratio/pass are None, with a reason."""
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        raw = _eog_raw()
        result = blink_residual_ratio(None, raw)
        assert result["ratio"] is None
        assert result["pass"] is None
        assert result["reason"] == "raw_pre_missing"

    def test_synthetic_frontal_proxy_without_eog(self):
        """Without EGI EOG names the blinks are found on a frontal proxy channel."""
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        ch_names = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "Cz", "Pz", "Oz"]
        rng = np.random.default_rng(1)
        sfreq, seconds = 250.0, 60.0
        data = rng.standard_normal((len(ch_names), int(sfreq * seconds))) * 10e-6
        blink = np.exp(-0.5 * ((np.arange(int(0.3 * sfreq)) / sfreq - 0.15) / 0.06) ** 2)
        frontal = [ch_names.index(ch) for ch in ("Fp1", "Fp2", "F7", "F8")]
        for onset in np.arange(5.0, seconds - 5.0, 4.0):
            start = int(onset * sfreq)
            data[frontal, start:start + blink.size] += (200e-6 * blink)[None, :]
        raw = mne.io.RawArray(
            data, mne.create_info(ch_names, sfreq, "eeg"), verbose=False)

        result = blink_residual_ratio(raw, raw)
        assert result["blink_channel"] in ("Fp1", "Fp2")
        assert result["n_blinks"] > 0
        assert result["frontal_channels"]

    def test_dead_eog_falls_back_to_frontal_proxy(self):
        """A dead EOG channel must not drive the gate (25SusAttn sub-001 case).

        ``find_eog_events`` thresholds a flat channel's own noise (probed on
        real data: a VEOG with 1.4 µV of noise reported 92 blinks/min and the
        gate passed); the rate guard must reject it and retry the frontal
        proxy instead of reporting a meaningless ratio.
        """
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        ch_names = ["Fp1", "Fp2", "F7", "F8", "Fz", "Cz", "VEOG"]
        ch_types = ["eeg"] * 6 + ["eog"]
        rng = np.random.default_rng(0)
        sfreq, seconds = 250.0, 60.0
        n_times = int(sfreq * seconds)
        data = rng.standard_normal((len(ch_names), n_times)) * 5e-6
        data[ch_names.index("VEOG")] = rng.standard_normal(n_times) * 1.4e-6
        blink = np.exp(-0.5 * ((np.arange(int(0.3 * sfreq)) / sfreq - 0.15) / 0.06) ** 2)
        frontal = [ch_names.index(ch) for ch in ("Fp1", "Fp2", "F7", "F8")]
        for onset in np.arange(5.0, seconds - 5.0, 4.0):
            start = int(onset * sfreq)
            data[frontal, start:start + blink.size] += (200e-6 * blink)[None, :]
        raw = mne.io.RawArray(
            data, mne.create_info(ch_names, sfreq, ch_types), verbose=False)

        result = blink_residual_ratio(raw, raw)

        assert result["blink_channel"] in ("Fp1", "Fp2", "F7", "F8")
        assert result["blinks_per_minute"] is not None
        assert 1.0 <= result["blinks_per_minute"] <= 60.0
        assert any(
            attempt["channel"] == "VEOG"
            and attempt["reason"] == "implausible_blink_rate"
            for attempt in result["attempts"]
        )
        assert result["ratio"] is not None

    def test_retyped_eog_channels_are_not_used(self):
        """A chain that retyped the EOG channels away must not have them used.

        The gate reads ``raw_pre`` (pre-chain types) but the chain's own typing
        decides what is usable: 25SusAttn's auxiliary electrodes are unusable
        and the config retypes them to ``misc``, so the gate must fall back to
        the frontal proxy instead of consuming them.
        """
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        ch_names = ["Fp1", "Fp2", "F7", "F8", "Fz", "Cz", "VEOG"]
        ch_types = ["eeg"] * 6 + ["eog"]
        rng = np.random.default_rng(0)
        sfreq, seconds = 250.0, 60.0
        n_times = int(sfreq * seconds)
        data = rng.standard_normal((len(ch_names), n_times)) * 5e-6
        data[ch_names.index("VEOG")] = rng.standard_normal(n_times) * 1.4e-6
        blink = np.exp(-0.5 * ((np.arange(int(0.3 * sfreq)) / sfreq - 0.15) / 0.06) ** 2)
        frontal = [ch_names.index(ch) for ch in ("Fp1", "Fp2", "F7", "F8")]
        for onset in np.arange(5.0, seconds - 5.0, 4.0):
            start = int(onset * sfreq)
            data[frontal, start:start + blink.size] += (200e-6 * blink)[None, :]
        raw_pre = mne.io.RawArray(
            data, mne.create_info(ch_names, sfreq, ch_types), verbose=False)
        raw_post = raw_pre.copy()
        raw_post.set_channel_types({"VEOG": "misc"})

        result = blink_residual_ratio(raw_pre, raw_post)

        assert result["retyped_eog_channels"] == ["VEOG"]
        assert result["blink_channel"] in ("Fp1", "Fp2", "F7", "F8")
        assert result["blinks_per_minute"] is not None
        assert result["ratio"] is not None

    def test_no_usable_blink_channel_never_passes(self, monkeypatch):
        """Every candidate implausible → pass None with a reason (no false pass)."""
        from eegcpm.modules.qc import hbn_recording_qc
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        raw = _eog_raw()
        monkeypatch.setattr(
            hbn_recording_qc, "_blink_events", lambda raw_, name: np.arange(10_000)
        )

        result = blink_residual_ratio(raw, raw)

        assert result["pass"] is None
        assert result["ratio"] is None
        assert result["reason"] == "no_usable_blink_channel"
        assert result["attempts"]
        assert all(
            attempt["reason"] == "implausible_blink_rate"
            for attempt in result["attempts"]
        )

    def test_resampled_post_is_timeline_mismatch(self):
        """B7 rev2 fix 3: blink indices from raw_pre are not usable on a resampled post.

        Before the guard, a resampled post (different ``sfreq``/``n_times``) was
        indexed with the pre recording's sample numbers and still produced a
        pass/fail verdict (probed: ratio 0.50 on a 250→200 Hz resample; the
        audit measured 0.149/pass=True on its own probe). The gate must refuse
        to run instead: ratio/pass ``None``, reason ``timeline_mismatch``.
        """
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        raw_pre = _eog_raw(leak=0.8)
        raw_post = raw_pre.copy().resample(200.0, verbose=False)
        assert raw_post.info["sfreq"] != raw_pre.info["sfreq"]

        result = blink_residual_ratio(raw_pre, raw_post)
        assert result["ratio"] is None
        assert result["pass"] is None
        assert result["reason"] == "timeline_mismatch"
        # The mismatch is recorded, not just asserted.
        assert result["timeline"]["pre"]["sfreq"] == raw_pre.info["sfreq"]
        assert result["timeline"]["post"]["n_times"] == raw_post.n_times

    def test_timeline_guard_accepts_matching_frames(self):
        """A post recording with the identical frame still runs the gate."""
        from eegcpm.modules.qc.hbn_recording_qc import blink_residual_ratio

        raw = _eog_raw(leak=0.8)
        result = blink_residual_ratio(raw, raw)
        assert result["reason"] is None
        assert result["ratio"] is not None

    def test_line_noise_in_pre_does_not_bias_ratio_toward_pass(self):
        """B7 rev2 fix 4: identical band-pass on pre and post before the ptp.

        The pre amplitude used to be measured on unfiltered data while the post
        was filtered, so 60 Hz line noise in the *pre* recording inflated
        ``ptp_pre`` and pushed the ratio toward the 0.30 gate even with the
        blinks fully retained (probed: a false pass at 100 µV 60 Hz, leak 0.2).
        Both sides are now band-passed 1-15 Hz, so retained blinks fail the gate
        at any line-noise level.
        """
        from eegcpm.modules.qc.hbn_recording_qc import (
            OCULAR_GATE, blink_residual_ratio,
        )

        raw_pre = _eog_raw(leak=0.2, line_uv=100.0)
        # Simulate a real chain: the 0.1-45 Hz filter removed the line noise
        # from the post recording, but the blinks were NOT removed (regression
        # failed).
        raw_post = raw_pre.copy()
        frontal = [raw_post.ch_names.index(ch) for ch in
                   ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")]
        t = np.arange(raw_post.n_times) / raw_post.info["sfreq"]
        raw_post._data[frontal] -= (
            100e-6 * np.sin(2 * np.pi * 60.0 * t)[None, :])

        result = blink_residual_ratio(raw_pre, raw_post)
        assert result["ratio"] is not None
        # Blinks retained -> the gate must FAIL (ratio ~1), not pass on the
        # line-noise-inflated pre amplitude.
        assert result["ratio"] >= OCULAR_GATE
        assert result["pass"] is False
        assert result["ratio"] > 0.9, (
            f"band-pass did not remove the line-noise bias: ratio "
            f"{result['ratio']:.3f}"
        )

        # Non-vacuity: the OLD measurement (unfiltered ptp) on the same data is
        # a false pass — this is the bias the fix removes.
        from eegcpm.modules.qc.hbn_recording_qc import (
            _blink_events, _blink_locked_ptp, _frontal_channels,
        )
        events = _blink_events(raw_pre, "E8")
        names = _frontal_channels(raw_pre)
        unfiltered_ratio = (
            _blink_locked_ptp(raw_post, events, names)
            / _blink_locked_ptp(raw_pre, events, names)
        )
        assert unfiltered_ratio < OCULAR_GATE, (
            "fixture no longer reproduces the pre-fix false pass "
            f"(unfiltered ratio {unfiltered_ratio:.3f})"
        )

    def test_metrics_row_fields(self):
        """`recording_qc_metrics` reads the chain metadata (task 3)."""
        from eegcpm.modules.qc.hbn_recording_qc import recording_qc_metrics

        raw = _eog_raw()
        chain_metadata = {
            "asr": {"asr_burden": 0.25, "window_rejected_fraction": 0.1,
                    "repaired_fraction": 0.2},
            "bad_channel_detection": {"detected": ["E1", "E2"], "n_detected_precap": 5},
            "block_rejection": {"rejected_fraction": 0.12, "n_blocks_rejected": 4},
        }
        row = recording_qc_metrics(raw, "RestingState", "s1",
                                   chain_metadata=chain_metadata, raw_pre=raw)
        assert row["asr_burden"] == 0.25
        assert row["asr_rejected_fraction"] == 0.1
        assert row["asr_repaired_fraction"] == 0.2
        assert row["bad_channels_n"] == 2
        assert row["bad_channels_n_precap"] == 5
        assert row["bad_channels_detected"] == ["E1", "E2"]
        # The block keys are the ones the block-rejection step emits.
        assert row["block_rejected_fraction"] == 0.12
        assert row["block_n_rejected"] == 4
        # The old in-sample corr survives as an informational field only.
        assert "residual_ocular_maxcorr_insample" in row
        assert "residual_ocular_maxcorr" not in row
        assert "residual_ocular_pass" not in row
        assert row["blink_n"] > 0

    def test_metrics_row_without_metadata(self):
        """Without chain metadata the ASR/block fields are None, bads come from info."""
        from eegcpm.modules.qc.hbn_recording_qc import recording_qc_metrics

        raw = _eog_raw()
        raw.info["bads"] = ["Fp1"]
        row = recording_qc_metrics(raw, "RestingState", "s1")
        assert row["asr_burden"] is None
        assert row["bad_channels_n"] == 1
        assert row["blink_residual_pass"] is None
        assert row["blink_residual_reason"] == "raw_pre_missing"


class TestRecommendedActionGateVeto:
    """A failed QC gate must veto an 'accept' recommendation (2026-10-17).

    The action heuristic only looked at bad-channel percentage and ICA success,
    so 25SusAttn sub-007 stroopvis reported ``accept`` while its ocular gate
    failed (blink_residual_pass=false). ``gates_ok=False`` now downgrades an
    ``accept`` to ``review``; ``None`` keeps the previous behaviour.
    """

    def test_failed_gate_vetoes_accept(self):
        from eegcpm.modules.qc.quality_assessment import get_recommended_action

        assert get_recommended_action("excellent", True) == "accept"
        assert get_recommended_action("excellent", True, gates_ok=False) == "review"

    def test_passed_or_absent_gate_leaves_action_unchanged(self):
        from eegcpm.modules.qc.quality_assessment import get_recommended_action

        assert get_recommended_action("excellent", True, gates_ok=True) == "accept"
        assert get_recommended_action("excellent", True, gates_ok=None) == "accept"

    def test_veto_never_upgrades_a_stricter_action(self):
        from eegcpm.modules.qc.quality_assessment import get_recommended_action

        assert get_recommended_action("poor", True, gates_ok=False) == "reject"
        assert get_recommended_action("acceptable", True, gates_ok=False) == "review"

    def test_metrics_record_gates_ok(self):
        from eegcpm.modules.qc.base import QCResult
        from eegcpm.modules.qc.quality_assessment import extract_metrics_from_qc_result

        qc_result = QCResult(subject_id="s1", metrics=[], notes=[])
        data = extract_metrics_from_qc_result(
            qc_result, subject_id="s1", task="cptvis", gates_ok=False
        )
        assert data["quality_metrics"]["gates_ok"] is False
        assert data["quality_metrics"]["recommended_action"] == "review"


class TestRecordingQCLineFrequency:
    """The line-noise metric measures the recording's declared line frequency.

    2026-10-17 (25SusAttn): the metric was hardcoded to 60 Hz (HBN, US data),
    so a 50 Hz recording was measured at an irrelevant bin. The loader now
    records the BIDS ``PowerLineFrequency`` on ``info['line_freq']`` and the
    metric uses it (60 Hz default when the recording does not declare one).
    """

    def _raw_with_line_noise(self, line_freq=50.0, sfreq=250.0, seconds=20.0):
        rng = np.random.default_rng(0)
        n_times = int(sfreq * seconds)
        t = np.arange(n_times) / sfreq
        data = rng.standard_normal((4, n_times)) * 5e-6
        data += (50e-6 * np.sin(2 * np.pi * line_freq * t))[None, :]
        return mne.io.RawArray(
            data,
            mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, "eeg"),
            verbose=False,
        )

    def test_measures_declared_line_frequency_not_60(self):
        from eegcpm.modules.qc.hbn_recording_qc import recording_qc_metrics

        raw = self._raw_with_line_noise(50.0)

        row_default = recording_qc_metrics(raw, task="cptvis")
        assert row_default["line_noise_freq_hz"] == 60.0

        with raw.info._unlock():
            raw.info["line_freq"] = 50.0
        row_50 = recording_qc_metrics(raw, task="cptvis")

        assert row_50["line_noise_freq_hz"] == 50.0
        assert (row_50["line_noise_residual_ratio"]
                > row_default["line_noise_residual_ratio"])
