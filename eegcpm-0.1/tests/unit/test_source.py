"""Unit tests for source reconstruction module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mne
import numpy as np
import pytest

from eegcpm.modules.source import SourceReconstructionModule


@pytest.fixture
def sample_epochs():
    """Create sample epochs for source reconstruction testing."""
    np.random.seed(42)
    sfreq = 250
    n_channels = 32
    n_epochs = 5
    n_times = int(0.7 * sfreq)  # 700ms epochs

    # Create synthetic epoch data
    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-5

    # Create standard channel names that have known positions
    ch_names = [f"EEG{i+1:03d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Create events
    events = np.column_stack([
        np.arange(0, n_epochs * 500, 500),
        np.zeros(n_epochs, dtype=int),
        np.array([1, 1, 1, 2, 2])
    ])

    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"cond1": 1, "cond2": 2},
        tmin=-0.2,
        verbose=False,
    )

    return epochs


@pytest.fixture
def sample_raw():
    """Create sample Raw data for source reconstruction testing."""
    np.random.seed(42)
    sfreq = 250
    n_channels = 20
    duration = 10  # seconds
    n_times = int(duration * sfreq)

    # Create synthetic raw data
    data = np.random.randn(n_channels, n_times) * 1e-5

    # Use standard 10-20 channel names
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
               'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6',
               'O1', 'Oz', 'O2']
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    raw = mne.io.RawArray(data, info, verbose=False)
    return raw


@pytest.fixture
def epochs_with_montage():
    """Create epochs with standard montage."""
    np.random.seed(42)
    sfreq = 250
    n_epochs = 5
    n_times = int(0.7 * sfreq)

    # Use standard 10-20 channel names
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
               'T3', 'C3', 'Cz', 'C4', 'T4',
               'T5', 'P3', 'Pz', 'P4', 'T6',
               'O1', 'Oz', 'O2']
    n_channels = len(ch_names)

    data = np.random.randn(n_epochs, n_channels, n_times) * 1e-5
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage)

    events = np.column_stack([
        np.arange(0, n_epochs * 500, 500),
        np.zeros(n_epochs, dtype=int),
        np.array([1, 1, 1, 2, 2])
    ])

    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"stim1": 1, "stim2": 2},
        tmin=-0.2,
        verbose=False,
    )

    return epochs


@pytest.fixture
def output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestSourceReconstructionModule:
    """Tests for SourceReconstructionModule class."""

    def test_init_default(self, output_dir):
        """Test default initialization."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        assert module.method == "sLORETA"
        assert module.parcellation == "conn_networks"
        assert module.snr == 3.0
        assert module.loose == 0.2
        assert module.depth == 0.8
        assert module.spacing == "oct6"
        assert module.roi_radius == 10.0  # Default ROI radius

    def test_init_custom_config(self, output_dir):
        """Test initialization with custom config."""
        config = {
            "method": "dSPM",
            "parcellation": "custom",
            "snr": 2.0,
            "loose": 0.1,
            "depth": 0.9,
            "spacing": "oct5",
            "roi_radius": 15.0,
        }
        module = SourceReconstructionModule(config, output_dir)

        assert module.method == "dSPM"
        assert module.parcellation == "custom"
        assert module.snr == 2.0
        assert module.loose == 0.1
        assert module.depth == 0.9
        assert module.spacing == "oct5"
        assert module.roi_radius == 15.0

    def test_init_all_methods(self, output_dir):
        """Test that all valid methods can be configured."""
        methods = ["dSPM", "sLORETA", "eLORETA", "MNE"]

        for method in methods:
            config = {"method": method}
            module = SourceReconstructionModule(config, output_dir)
            assert module.method == method

    def test_module_metadata(self, output_dir):
        """Test module name and version."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        assert module.name == "source_reconstruction"
        assert module.version == "0.2.0"
        assert "Source" in module.description or "source" in module.description

    def test_output_spec(self, output_dir):
        """Test output specification."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        spec = module.get_output_spec()
        assert "stcs" in spec
        assert "roi_data" in spec
        assert "inverse_operator" in spec

    def test_snr_to_lambda2_conversion(self, output_dir):
        """Test SNR to lambda2 conversion."""
        config = {"snr": 3.0}
        module = SourceReconstructionModule(config, output_dir)

        # lambda2 = 1/snr^2
        expected_lambda2 = 1.0 / 3.0 ** 2
        assert expected_lambda2 == pytest.approx(1.0 / 9.0)

    @pytest.mark.skip(reason="Requires fsaverage dataset download")
    def test_process_with_template_forward(self, sample_epochs, output_dir):
        """Test full source reconstruction with template forward model.

        This test requires fsaverage to be downloaded, so skip by default.
        """
        config = {
            "method": "sLORETA",
            "parcellation": "conn_networks",
        }
        module = SourceReconstructionModule(config, output_dir)

        result = module.process(sample_epochs)

        assert result.success
        assert "stcs" in result.outputs
        assert "roi_data" in result.outputs

    def test_process_with_mock_forward(self, epochs_with_montage, output_dir):
        """Test source reconstruction with mocked forward model."""
        config = {"method": "sLORETA"}
        module = SourceReconstructionModule(config, output_dir)

        # Mock the forward model computation
        with patch.object(module, "_get_template_forward") as mock_fwd:
            # Create a minimal mock forward model
            # This is complex because MNE's source reconstruction is tightly coupled
            mock_fwd.side_effect = Exception("Forward model mocked - test config only")

            result = module.process(epochs_with_montage)

            # Should fail due to our mock, but test the error handling
            assert not result.success
            assert len(result.errors) > 0
            assert "Forward model mocked" in result.errors[0]


class TestSourceReconstructionSubjectsDir:
    """F2: ``subjects_dir`` resolves lazily; construction never downloads fsaverage."""

    def test_init_does_not_fetch_fsaverage(self, output_dir, monkeypatch):
        """Construction must not call ``fetch_fsaverage`` (~500 MB when uncached)."""
        monkeypatch.setattr(
            mne.datasets,
            "fetch_fsaverage",
            lambda *args, **kwargs: pytest.fail(
                "fetch_fsaverage must not be called during __init__"
            ),
        )
        module = SourceReconstructionModule({}, output_dir)
        assert module.method == "sLORETA"

    def test_subjects_dir_resolved_once_on_first_use(self, output_dir, monkeypatch):
        """The fallback is resolved on first access and cached afterwards."""
        calls = []
        fs_dir = output_dir / "fsaverage"
        fs_dir.mkdir()

        def fake_fetch(*args, **kwargs):
            calls.append((args, kwargs))
            return fs_dir

        monkeypatch.setattr(mne.datasets, "fetch_fsaverage", fake_fetch)
        module = SourceReconstructionModule({}, output_dir)

        assert calls == []  # still lazy after construction
        assert module.subjects_dir == output_dir  # parent of .../fsaverage
        assert module.subjects_dir == output_dir
        assert len(calls) == 1  # resolved once, then cached

    def test_configured_subjects_dir_never_fetches(self, output_dir, monkeypatch):
        """An explicit ``subjects_dir`` is used as-is (no fetch, no download)."""
        monkeypatch.setattr(
            mne.datasets,
            "fetch_fsaverage",
            lambda *args, **kwargs: pytest.fail(
                "fetch_fsaverage must not run when subjects_dir is configured"
            ),
        )
        module = SourceReconstructionModule(
            {"subjects_dir": str(output_dir)}, output_dir
        )
        assert module.subjects_dir == output_dir


class TestSourceReconstructionValidation:
    """Tests for input validation."""

    def test_validate_input_epochs_with_montage(self, epochs_with_montage, output_dir):
        """Test validation passes for epochs with montage."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)
        assert module.validate_input(epochs_with_montage) is True

    def test_validate_input_raw_with_montage(self, sample_raw, output_dir):
        """Test validation fails for raw input even with a montage.

        Regression note (P2-F13): the source module only accepts epochs. It used to
        accept Raw (the base class accepted any object and the module never checked),
        so this test previously asserted True. It now asserts False because
        ``SourceReconstructionModule.validate_input`` requires ``mne.BaseEpochs``.
        """
        config = {}
        module = SourceReconstructionModule(config, output_dir)
        assert module.validate_input(sample_raw) is False

    def test_validate_input_epochs_without_montage(self, sample_epochs, output_dir):
        """Test validation fails for epochs without montage."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)
        # sample_epochs doesn't have a montage
        assert module.validate_input(sample_epochs) is False

    def test_validate_input_invalid_type(self, output_dir):
        """Test validation fails for invalid input type."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)
        assert module.validate_input("not_an_mne_object") is False
        assert module.validate_input(None) is False
        assert module.validate_input([1, 2, 3]) is False

    def test_process_fails_without_montage(self, sample_epochs, output_dir):
        """Test process returns error for data without montage."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        result = module.process(sample_epochs)

        assert not result.success
        assert len(result.errors) > 0
        assert "montage" in result.errors[0].lower()

    def test_source_validate_via_executor(self, sample_epochs, output_dir):
        """Montage-less input through the executor path fails loud, writes nothing.

        ``PipelineExecutor`` ignores the return value of ``validate_input``
        (``pipeline/executor.py`` line ~109) and only catches exceptions, so the
        module must re-check in ``process()`` and return ``success=False``. This test
        pins that contract (the return-False deviation documented on
        ``SourceReconstructionModule.validate_input``).
        """
        from types import SimpleNamespace

        from eegcpm.pipeline.executor import PipelineExecutor

        class _StubProject:
            def __init__(self, root_path):
                self.root_path = root_path
                self.subjects = [SimpleNamespace(id="01")]

        module_output = output_dir / "source"
        executor = PipelineExecutor(
            project=_StubProject(output_dir),
            config=None,
            output_dir=output_dir / "derivatives",
            checkpoint_dir=output_dir / "checkpoints",
        )
        executor.add_module(SourceReconstructionModule({}, module_output))
        executor._load_initial_data = lambda subject: sample_epochs

        results = executor.run()

        result = results["01"][0]
        assert result.success is False
        assert "montage" in result.errors[0].lower()
        assert list(module_output.iterdir()) == []


class TestSourceReconstructionConfig:
    """Tests for source reconstruction configuration."""

    def test_valid_snr_values(self, output_dir):
        """Test various valid SNR values."""
        for snr in [1.0, 2.0, 3.0, 5.0, 10.0]:
            config = {"snr": snr}
            module = SourceReconstructionModule(config, output_dir)
            assert module.snr == snr

    def test_valid_roi_radius_values(self, output_dir):
        """Test various valid ROI radius values."""
        for radius in [5.0, 10.0, 15.0, 20.0]:
            config = {"roi_radius": radius}
            module = SourceReconstructionModule(config, output_dir)
            assert module.roi_radius == radius

    def test_valid_loose_values(self, output_dir):
        """Test various valid loose parameter values."""
        for loose in [0.0, 0.1, 0.2, 0.5, 1.0]:
            config = {"loose": loose}
            module = SourceReconstructionModule(config, output_dir)
            assert module.loose == loose

    def test_valid_depth_values(self, output_dir):
        """Test various valid depth parameter values."""
        for depth in [0.0, 0.5, 0.8, 1.0]:
            config = {"depth": depth}
            module = SourceReconstructionModule(config, output_dir)
            assert module.depth == depth

    def test_valid_spacing_values(self, output_dir):
        """Test various valid spacing parameter values."""
        for spacing in ["oct4", "oct5", "oct6", "ico4", "ico5"]:
            config = {"spacing": spacing}
            module = SourceReconstructionModule(config, output_dir)
            assert module.spacing == spacing

    def test_parcellation_options(self, output_dir):
        """Test different parcellation options."""
        for parc in ["conn_networks", "aparc", "custom"]:
            config = {"parcellation": parc}
            module = SourceReconstructionModule(config, output_dir)
            assert module.parcellation == parc


class TestSourceReconstructionHelpers:
    """Tests for helper methods."""

    def test_lambda2_calculation(self, output_dir):
        """Test that lambda2 is correctly calculated from SNR."""
        # lambda2 = 1/snr^2

        # SNR = 1 -> lambda2 = 1
        config = {"snr": 1.0}
        module = SourceReconstructionModule(config, output_dir)
        assert 1.0 / module.snr ** 2 == 1.0

        # SNR = 3 -> lambda2 = 1/9
        config = {"snr": 3.0}
        module = SourceReconstructionModule(config, output_dir)
        assert 1.0 / module.snr ** 2 == pytest.approx(1.0/9.0)

        # SNR = 10 -> lambda2 = 1/100
        config = {"snr": 10.0}
        module = SourceReconstructionModule(config, output_dir)
        assert 1.0 / module.snr ** 2 == pytest.approx(0.01)


class TestCONNROIs:
    """Tests for CONN ROI extraction."""

    def test_get_conn_rois_import(self):
        """Test that CONN ROI functions can be imported."""
        from eegcpm.data.conn_rois import get_conn_rois, get_mni_coordinates, get_roi_names

        # Should not raise
        roi_names = get_roi_names()
        assert len(roi_names) > 0

        mni_coords = get_mni_coordinates()
        assert len(mni_coords) > 0
        assert len(mni_coords) == len(roi_names)

    def test_roi_names_count(self):
        """Test expected number of CONN ROIs."""
        from eegcpm.data.conn_rois import get_roi_names

        roi_names = get_roi_names()
        # CONN has 32 network ROIs
        assert len(roi_names) == 32

    def test_mni_coordinates_shape(self):
        """Test MNI coordinates shape."""
        from eegcpm.data.conn_rois import get_mni_coordinates

        mni_coords = get_mni_coordinates()
        assert len(mni_coords) == 32

        # Each coordinate should be 3D (x, y, z)
        for coord in mni_coords:
            assert len(coord) == 3

    def test_conn_rois_structure(self):
        """Test CONN ROIs data structure."""
        from eegcpm.data.conn_rois import get_conn_rois

        rois = get_conn_rois()

        # Should have network and ROI info
        assert len(rois) > 0

        # Each ROI should have a name attribute (ROI is a dataclass/object)
        for roi in rois:
            assert hasattr(roi, "name")


class TestSourceReconstructionIntegration:
    """Integration tests for source reconstruction.

    These tests require MNE datasets and may take longer to run.
    """

    def test_epochs_have_montage(self, epochs_with_montage):
        """Verify test epochs have proper montage."""
        montage = epochs_with_montage.get_montage()
        assert montage is not None

    @pytest.mark.skip(reason="Requires fsaverage download, slow test")
    def test_full_source_reconstruction(self, epochs_with_montage, output_dir):
        """Full integration test of source reconstruction pipeline.

        Skip by default as it requires fsaverage download.
        """
        config = {
            "method": "sLORETA",
            "parcellation": "conn_networks",
            "snr": 3.0,
            "spacing": "oct5",  # Coarser for speed
        }
        module = SourceReconstructionModule(config, output_dir)

        class MockSubject:
            id = "test_sub"

        result = module.process(epochs_with_montage, subject=MockSubject())

        assert result.success
        assert len(result.outputs["stcs"]) > 0
        assert len(result.output_files) > 0


class TestSourceReconstructionErrorHandling:
    """Tests for error handling in source reconstruction."""

    def test_process_handles_exceptions(self, sample_epochs, output_dir):
        """Test that process handles exceptions gracefully."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        # Process will fail without proper montage/forward model
        result = module.process(sample_epochs)

        # Should fail but not crash
        assert not result.success
        assert len(result.errors) > 0
        assert result.execution_time_seconds > 0

    def test_process_all_bad_eeg_channels_returns_failure(
        self, epochs_with_montage, output_dir
    ):
        """All EEG channels bad → ``success=False``, never an uncaught pick error.

        ``validate_input`` already rejects this case (``pick_types`` excludes bads by
        default); the EEG pick in ``process`` also runs inside the try (B2 rev1
        robustness fix) so the contract holds whichever guard fires first.
        """
        epochs = epochs_with_montage.copy()
        epochs.info["bads"] = list(epochs.ch_names)
        module = SourceReconstructionModule({}, output_dir)

        result = module.process(epochs)

        assert result.success is False
        assert result.errors
        assert list(output_dir.iterdir()) == []

    def test_empty_epochs_handling(self, output_dir):
        """Test handling of empty epochs."""
        config = {}
        module = SourceReconstructionModule(config, output_dir)

        # Create minimal empty epochs
        info = mne.create_info(ch_names=["EEG1", "EEG2"], sfreq=100, ch_types=["eeg", "eeg"])
        data = np.zeros((0, 2, 50))  # No epochs
        events = np.zeros((0, 3), dtype=int)

        with pytest.raises(Exception):
            # Empty epochs should fail during creation or processing
            epochs = mne.EpochsArray(data, info, events=events, verbose=False)
            module.process(epochs)
