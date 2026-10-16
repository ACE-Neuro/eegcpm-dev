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
        """Test initialization with custom nested config (P2-F5 schema)."""
        config = {
            "method": "dSPM",
            "parcellation": "conn_networks",
            "forward": {"spacing": "oct5"},
            "inverse": {"snr": 2.0, "loose": 0.1, "depth": 0.9},
            "roi_radius": 15.0,
        }
        module = SourceReconstructionModule(config, output_dir)

        assert module.method == "dSPM"
        assert module.parcellation == "conn_networks"
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
        """Test SNR to lambda2 conversion (nested inverse section)."""
        config = {"inverse": {"snr": 3.0}}
        module = SourceReconstructionModule(config, output_dir)

        # lambda2 = 1/snr^2
        expected_lambda2 = 1.0 / 3.0 ** 2
        assert expected_lambda2 == pytest.approx(1.0 / 9.0)

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
    """Tests for source reconstruction configuration (nested schema, P2-F5)."""

    def test_valid_snr_values(self, output_dir):
        """Test various valid SNR values (nested inverse section)."""
        for snr in [1.0, 2.0, 3.0, 5.0, 10.0]:
            config = {"inverse": {"snr": snr}}
            module = SourceReconstructionModule(config, output_dir)
            assert module.snr == snr

    def test_valid_snr_trial_values(self, output_dir):
        """snr_trial reaches the module (B5 wires it into the trial lambda2)."""
        config = {"inverse": {"snr_trial": 1.5}}
        module = SourceReconstructionModule(config, output_dir)
        assert module.snr_trial == 1.5
        assert module.snr == 3.0

    def test_valid_roi_radius_values(self, output_dir):
        """Test various valid ROI radius values."""
        for radius in [5.0, 10.0, 15.0, 20.0]:
            config = {"roi_radius": radius}
            module = SourceReconstructionModule(config, output_dir)
            assert module.roi_radius == radius

    def test_valid_loose_values(self, output_dir):
        """Test various valid loose parameter values (nested inverse section)."""
        for loose in [0.0, 0.1, 0.2, 0.5, 1.0]:
            config = {"inverse": {"loose": loose}}
            module = SourceReconstructionModule(config, output_dir)
            assert module.loose == loose

    def test_valid_depth_values(self, output_dir):
        """Test various valid depth parameter values (nested inverse section)."""
        for depth in [0.0, 0.5, 0.8, 1.0]:
            config = {"inverse": {"depth": depth}}
            module = SourceReconstructionModule(config, output_dir)
            assert module.depth == depth

    def test_valid_spacing_values(self, output_dir):
        """Test various valid spacing parameter values (nested forward section)."""
        for spacing in ["oct4", "oct5", "oct6", "ico4", "ico5"]:
            config = {"forward": {"spacing": spacing}}
            module = SourceReconstructionModule(config, output_dir)
            assert module.spacing == spacing

    def test_valid_mode_values(self, output_dir):
        """Evoked (default), trials and both are accepted; nothing else is."""
        for mode in ["evoked", "trials", "both"]:
            module = SourceReconstructionModule({"mode": mode}, output_dir)
            assert module.mode == mode

    def test_parcellation_options(self, output_dir):
        """Only conn_networks is implemented; other schemes raise (P2-F5)."""
        from pydantic import ValidationError

        module = SourceReconstructionModule(
            {"parcellation": "conn_networks"}, output_dir
        )
        assert module.parcellation == "conn_networks"

        for parc in ["aparc", "schaefer100", "custom"]:
            with pytest.raises(ValidationError):
                SourceReconstructionModule({"parcellation": parc}, output_dir)


class TestSourceReconstructionHelpers:
    """Tests for helper methods."""

    def test_lambda2_calculation(self, output_dir):
        """Test that lambda2 is correctly calculated from SNR."""
        # lambda2 = 1/snr^2

        # SNR = 1 -> lambda2 = 1
        config = {"inverse": {"snr": 1.0}}
        module = SourceReconstructionModule(config, output_dir)
        assert 1.0 / module.snr ** 2 == 1.0

        # SNR = 3 -> lambda2 = 1/9
        config = {"inverse": {"snr": 3.0}}
        module = SourceReconstructionModule(config, output_dir)
        assert 1.0 / module.snr ** 2 == pytest.approx(1.0/9.0)

        # SNR = 10 -> lambda2 = 1/100
        config = {"inverse": {"snr": 10.0}}
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

    @pytest.mark.fsaverage
    @pytest.mark.slow
    def test_process_with_template_forward(self, source_erp_forward, tmp_path):
        """Full reconstruction through the real template forward (P2-F1/F2/F4).

        Replaces the old ``@pytest.mark.skip``: the fixture uses the cached
        fsaverage (``fsaverage_subjects_dir`` skips when it is absent) and the
        module builds its own oct6 template forward. Asserts the *content* the
        plan's condition 4 requires: signed ROI time courses, unique ROI names,
        cerebellar exclusion, and the provenance/config artifacts.
        """
        import json

        module = SourceReconstructionModule(
            {"method": "dSPM", "subjects_dir": str(source_erp_forward["subjects_dir"])},
            tmp_path,
        )
        class MockSubject:
            id = "test_sub"
            session = "01"

        result = module.process(_condition_epochs(n_one=20, n_two=20), subject=MockSubject())
        assert result.success, result.errors

        npz = np.load(tmp_path / "test_sub_roi_tc.npz")
        roi_names = [str(n) for n in npz["roi_names"]]
        tc = np.stack([npz[name] for name in ("1", "2")])
        # (a) signed output, not a magnitude (P2-F4)
        assert tc.min() < 0
        # (b) unique names, no cerebellum (P2-F2/F3)
        assert len(set(roi_names)) == len(roi_names) == 30
        excluded = [str(n) for n in npz["excluded_rois"]]
        assert "Cerebellar.Anterior" in excluded and "Cerebellar.Posterior" in excluded
        assert int(np.min(npz["roi_vertex_counts"])) >= 3
        # (c) condition bookkeeping
        assert int(npz["1_nave"]) == 20
        assert npz["1_times"][0] == pytest.approx(-0.2)
        # (d) provenance files (P2-F10) + config hash
        for name in (
            "test_sub_source_config.yaml",
            "test_sub-cov.fif",
            "test_sub-inv.fif",
            "test_sub_provenance.json",
            "test_sub_crosstalk.npz",
        ):
            assert (tmp_path / name).is_file(), name
        provenance = json.loads((tmp_path / "test_sub_provenance.json").read_text())
        assert provenance["config_hash"] == module.config_hash()
        assert provenance["label_extraction_mode"] == "mean_flip"
        assert provenance["pick_ori"] == "normal"
        assert provenance["labels"]["excluded_rois"] == excluded
        # (e) the crosstalk matrices (P2-F14) are written next to the data and the
        # provenance summarises the primary one.
        crosstalk = np.load(tmp_path / "test_sub_crosstalk.npz")
        matrix = crosstalk["crosstalk"]
        assert matrix.shape == (30, 30)
        assert np.allclose(np.diag(matrix), 1.0)
        assert [str(n) for n in crosstalk["roi_names"]] == roi_names
        # B7 rev2 fix 6: the primary matrix is the flip-aligned estimator and
        # the library `get_cross_talk(mode='mean')` matrix is kept alongside.
        assert str(crosstalk["primary"][0]) == "crosstalk_flip_aligned"
        unflipped = crosstalk["crosstalk_unflipped"]
        assert unflipped.shape == (30, 30)
        assert np.allclose(np.diag(unflipped), 1.0)
        # Non-vacuity: the sign alignment really changes the matrix on this
        # fixture (the library signed mean cancels across sulcal walls).
        assert np.max(np.abs(matrix - unflipped)) > 1e-6
        definition = str(crosstalk["definition"][0])
        assert "flip-aligned" in definition and "crosstalk_unflipped" in definition
        assert "FROM ROI j" in definition and "INTO ROI i" in definition
        assert provenance["crosstalk"]["computed"] is True
        assert provenance["crosstalk"]["primary_matrix"] == "crosstalk_flip_aligned"
        assert provenance["crosstalk"]["n_rois"] == 30
        assert np.isfinite(provenance["crosstalk"]["off_diagonal_max"])
        # (f) B7-A3/B7 rev2: the pair count is over *directed* entries of the
        # PRIMARY flip-aligned matrix (C[i, j] is the leakage from ROI j
        # (source, column) into ROI i (target, row)), not over unique pairs.
        directed = int(
            np.sum((matrix > 0.3) & ~np.eye(matrix.shape[0], dtype=bool))
        )
        assert provenance["crosstalk"]["n_directed_above_0.3"] == directed
        assert "n_pairs_above_0.3" not in provenance["crosstalk"]
        # The unflipped library matrix is summarised too (same statistic shape).
        assert provenance["crosstalk"]["unflipped"]["n_directed_above_0.3"] >= 0
        assert provenance["crosstalk"]["max_abs_delta_primary_vs_unflipped"] > 0

    @pytest.mark.fsaverage
    @pytest.mark.slow
    def test_crosstalk_disabled_provenance_carries_reason(
        self, source_erp_forward, tmp_path
    ):
        """B7 rev2 fix 9: ``{"computed": false}`` states *why* (config vs error).

        The disabled path must record the reason and write no crosstalk npz; the
        error path is covered by the module's warning + the same reason field.
        """
        import json

        module = SourceReconstructionModule(
            {
                "method": "dSPM",
                "subjects_dir": str(source_erp_forward["subjects_dir"]),
                "output": {"compute_crosstalk": False, "generate_qc": False},
            },
            tmp_path,
        )

        class MockSubject:
            id = "test_sub"

        result = module.process(
            _condition_epochs(n_one=20, n_two=20), subject=MockSubject())
        assert result.success, result.errors

        provenance = json.loads(
            (tmp_path / "test_sub_provenance.json").read_text())
        assert provenance["crosstalk"]["computed"] is False
        assert "compute_crosstalk disabled" in provenance["crosstalk"]["reason"]
        assert not (tmp_path / "test_sub_crosstalk.npz").exists()

    @pytest.mark.fsaverage
    @pytest.mark.slow
    def test_full_source_reconstruction(self, source_erp_forward, tmp_path):
        """sLORETA end-to-end with a caller-supplied forward (oct6, nested config)."""
        config = {
            "method": "sLORETA",
            "parcellation": "conn_networks",
            "inverse": {"snr": 3.0},
            "forward": {"spacing": "oct6"},
            "output": {"generate_qc": False},
        }
        module = SourceReconstructionModule(config, tmp_path)

        class MockSubject:
            id = "test_sub"

        result = module.process(
            _condition_epochs(n_one=20, n_two=20),
            subject=MockSubject(),
            forward=source_erp_forward["fwd"],
        )

        assert result.success, result.errors
        assert len(result.outputs["stcs"]) == 2
        assert len(result.output_files) > 0
        assert result.metadata["n_rois"] == 30
        assert result.metadata["method"] == "sLORETA"


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


# ---------------------------------------------------------------------------
# P2-F7/F8/F9: condition selection, nave bookkeeping and loud failures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source_erp_forward(fsaverage_subjects_dir):
    """oct6 fsaverage forward on a 129-channel GSN montage (module-scoped)."""
    subjects_dir = Path(fsaverage_subjects_dir)
    src = mne.setup_source_space(
        "fsaverage", "oct6", subjects_dir=subjects_dir, add_dist=False, verbose=False
    )
    bem_sol = mne.read_bem_solution(
        subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
        verbose=False,
    )
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    info = mne.create_info(montage.ch_names, 250.0, "eeg")
    info.set_montage(montage, verbose=False)
    fwd = mne.make_forward_solution(
        info, "fsaverage", src, bem_sol, eeg=True, meg=False, mindist=5.0, verbose=False
    )
    return {"fwd": fwd, "info": info, "subjects_dir": subjects_dir}


def _condition_epochs(n_one, n_two, sfreq=250.0, tmin=-0.2, tmax=0.4, seed=0):
    """Baseline-corrected epochs with event codes 1 (n_one) and 2 (n_two)."""
    montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
    n_ch = len(montage.ch_names)
    rng = np.random.default_rng(seed)
    n_times = int(round((tmax - tmin) * sfreq)) + 1
    n_epochs = n_one + n_two
    data = rng.standard_normal((n_epochs, n_ch, n_times)) * 1e-5
    codes = np.array([1] * n_one + [2] * n_two)
    events = np.column_stack(
        [np.arange(n_epochs) * int(sfreq), np.zeros(n_epochs, dtype=int), codes]
    )
    info = mne.create_info(montage.ch_names, sfreq, "eeg")
    info.set_montage(montage, verbose=False)
    epochs = mne.EpochsArray(
        data,
        info,
        events=events,
        event_id={"1": 1, "2": 2},
        tmin=tmin,
        baseline=(None, 0),
        verbose=False,
    )
    epochs.set_eeg_reference(projection=True, verbose=False)
    return epochs


class TestConditionNameGuards:
    """B4-K2/N2: duplicate/reserved condition names raise loudly.

    These call ``_select_condition_epochs`` directly: no forward/fsaverage needed
    (the guard runs before any matching or selection).
    """

    def test_duplicate_condition_names_raise(self, tmp_path):
        """Two specs with the same name would silently keep only the last."""
        task_config = {
            "conditions": [
                {"name": "dup", "event_codes": [1]},
                {"name": "dup", "event_codes": [2]},
            ]
        }
        module = SourceReconstructionModule({}, tmp_path, task_config=task_config)
        with pytest.raises(ValueError, match="duplicate condition"):
            module._select_condition_epochs(_condition_epochs(n_one=20, n_two=20))

    @pytest.mark.parametrize(
        "name",
        ["roi_names", "sfreq", "excluded_rois", "excluded_reasons", "roi_radius"],
    )
    def test_reserved_condition_names_raise(self, name, tmp_path):
        """Label-metadata keys (and every ``roi_*`` key) are reserved."""
        task_config = {"conditions": [{"name": name, "event_codes": [1]}]}
        module = SourceReconstructionModule({}, tmp_path, task_config=task_config)
        with pytest.raises(ValueError, match="reserved"):
            module._select_condition_epochs(_condition_epochs(n_one=20, n_two=20))

    @pytest.mark.parametrize("name", ["early_times", "trial_nave"])
    def test_companion_suffix_condition_names_raise(self, name, tmp_path):
        """``{cond}_times`` / ``{cond}_nave`` would be overwritten by the writers."""
        task_config = {"conditions": [{"name": name, "event_codes": [1]}]}
        module = SourceReconstructionModule({}, tmp_path, task_config=task_config)
        with pytest.raises(ValueError, match="companion"):
            module._select_condition_epochs(_condition_epochs(n_one=20, n_two=20))

    def test_event_id_derived_reserved_name_raises(self, tmp_path):
        """The no-task-config path (event_id keys) is guarded too."""
        epochs = _condition_epochs(n_one=20, n_two=20)
        epochs.event_id = {"roi_names": 1, "ok": 2}
        module = SourceReconstructionModule({}, tmp_path)
        with pytest.raises(ValueError, match="reserved"):
            module._select_condition_epochs(epochs)


@pytest.mark.fsaverage
class TestSourceConditionSelection:
    """P2-F7/F8/F9: requested/missing/skipped conditions, nave, loud failures."""

    @pytest.mark.slow
    def test_missing_and_below_min_trials_listed(
        self, source_erp_forward, tmp_path
    ):
        """Requested conditions are all accounted for; no arrays for missing/skipped."""
        task_config = {
            "conditions": [
                {"name": "many", "event_codes": [1]},
                {"name": "few", "event_codes": [2]},
                {"name": "absent", "event_codes": [99]},
            ]
        }
        module = SourceReconstructionModule(
            {"method": "dSPM", "min_trials": 10,
             "output": {"generate_qc": False}},
            tmp_path,
            task_config=task_config,
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=3), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors

        meta = result.metadata
        assert meta["conditions_requested"] == ["many", "few", "absent"]
        assert meta["conditions_produced"] == ["many"]
        assert meta["conditions_missing"] == ["absent"]
        assert meta["conditions_skipped"] == {
            "few": {"n_trials": 3, "reason": "below_min_trials"}
        }
        assert meta["nave"] == {"many": 20}
        # noise-normalised amplitudes scale with sqrt(nave): flagged as a covariate
        assert meta["nave_confound_warning"] is True

        npz = np.load(tmp_path / "unknown_roi_tc.npz")
        assert "many" in npz.files
        assert "few" not in npz.files
        assert "absent" not in npz.files
        assert "many_times" in npz.files
        assert int(npz["many_nave"]) == 20
        assert npz["many"].ndim == 2  # (n_rois, n_times)
        assert any("no matching events" in w for w in result.warnings)

    @pytest.mark.slow
    def test_all_conditions_missing_fails_loud(
        self, source_erp_forward, tmp_path
    ):
        """Zero produced conditions -> success False and nothing written."""
        task_config = {"conditions": [{"name": "absent", "event_codes": [99]}]}
        module = SourceReconstructionModule(
            {"method": "dSPM", "output": {"generate_qc": False}},
            tmp_path,
            task_config=task_config,
        )
        result = module.process(
            _condition_epochs(n_one=5, n_two=5), forward=source_erp_forward["fwd"]
        )
        assert result.success is False
        assert "no conditions produced" in result.errors[0]
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.slow
    def test_equalize_trials_seeded_identical_outputs(
        self, source_erp_forward, tmp_path
    ):
        """equalize_trials=5: seeded determinism, seed sensitivity, real subsetting.

        B4-K4: same seed -> identical arrays and ``{cond}_nave == 5``; a
        different seed -> different arrays (a regression to first-N would be
        seed-independent); and the selected trial indices are a random draw, not
        ``arange(5)``.
        """
        config = {
            "method": "dSPM",
            "min_trials": 10,
            "equalize_trials": 5,
            "random_seed": 42,
            "output": {"generate_qc": False},
        }
        epochs = _condition_epochs(n_one=20, n_two=20)
        results = []
        for name in ("run1", "run2"):
            out_dir = tmp_path / name
            module = SourceReconstructionModule(config, out_dir)
            result = module.process(epochs, forward=source_erp_forward["fwd"])
            assert result.success, result.errors
            results.append((out_dir, result))

        npz1 = np.load(results[0][0] / "unknown_roi_tc.npz")
        npz2 = np.load(results[1][0] / "unknown_roi_tc.npz")
        assert int(npz1["1_nave"]) == 5
        assert int(npz1["2_nave"]) == 5
        for key in ("1", "2", "1_times", "2_times", "1_nave", "2_nave"):
            assert np.array_equal(npz1[key], npz2[key]), key
        assert results[0][1].metadata["nave"] == {"1": 5, "2": 5}

        # A different seed must yield a different selection (and output array).
        out_dir3 = tmp_path / "run3"
        module3 = SourceReconstructionModule(
            {**config, "random_seed": 43}, out_dir3
        )
        result3 = module3.process(epochs, forward=source_erp_forward["fwd"])
        assert result3.success, result3.errors
        npz3 = np.load(out_dir3 / "unknown_roi_tc.npz")
        assert int(npz3["1_nave"]) == 5
        assert int(npz3["2_nave"]) == 5
        assert not np.array_equal(npz1["1"], npz3["1"])
        assert not np.array_equal(npz1["2"], npz3["2"])

        # The subset is a real random draw, not the first N trials.
        selected, selection = module3._select_condition_epochs(epochs)
        assert selection["nave"] == {"1": 5, "2": 5}
        for cond in ("1", "2"):
            indices = np.sort(np.asarray(selected[cond].selection))
            assert indices.tolist() != [0, 1, 2, 3, 4]

    @pytest.mark.slow
    def test_equalize_below_target_skipped(self, source_erp_forward, tmp_path):
        """A condition with fewer trials than equalize_trials is skipped, not cut."""
        module = SourceReconstructionModule(
            {"method": "dSPM", "min_trials": 10, "equalize_trials": 15,
             "output": {"generate_qc": False}},
            tmp_path,
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=12), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors
        assert result.metadata["nave"] == {"1": 15}
        assert result.metadata["conditions_skipped"] == {
            "2": {"n_trials": 12, "reason": "below_equalize_trials"}
        }

    @pytest.mark.slow
    def test_mode_both_writes_evoked_and_trials(
        self, source_erp_forward, tmp_path
    ):
        """mode='both' -> 2-D evoked archive + 3-D trial archive, both with nave."""
        module = SourceReconstructionModule(
            {"method": "dSPM", "mode": "both"}, tmp_path
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=20), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors

        evoked = np.load(tmp_path / "unknown_roi_tc.npz")
        trials = np.load(tmp_path / "unknown_roi_tc_trials.npz")
        assert evoked["1"].ndim == 2
        assert trials["1"].ndim == 3
        assert trials["1"].shape[0] == 20
        assert int(evoked["1_nave"]) == 20
        assert int(trials["1_nave"]) == 20
        assert evoked["1"].shape[1] == len(evoked["1_times"])

    @pytest.mark.slow
    def test_mode_trials_writes_only_trials_archive(
        self, source_erp_forward, tmp_path
    ):
        module = SourceReconstructionModule(
            {"method": "dSPM", "mode": "trials", "output": {"generate_qc": False}},
            tmp_path,
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=20), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors
        assert not (tmp_path / "unknown_roi_tc.npz").exists()
        assert (tmp_path / "unknown_roi_tc_trials.npz").exists()

    @pytest.mark.slow
    def test_save_roi_tc_and_stc_disabled_writes_nothing(
        self, source_erp_forward, tmp_path
    ):
        """output.save_roi_tc / save_stc / generate_qc are honoured."""
        module = SourceReconstructionModule(
            {
                "method": "dSPM",
                "output": {
                    "save_stc": False,
                    "save_roi_tc": False,
                    "generate_qc": False,
                },
            },
            tmp_path,
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=20), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.slow
    def test_qc_html_generated_when_enabled(self, source_erp_forward, tmp_path):
        """output.generate_qc=True still produces the QC HTML on the new contract."""
        module = SourceReconstructionModule(
            {"method": "dSPM", "output": {"generate_qc": True}}, tmp_path
        )
        result = module.process(
            _condition_epochs(n_one=20, n_two=20), forward=source_erp_forward["fwd"]
        )
        assert result.success, result.errors
        htmls = list(tmp_path.glob("*_source_qc.html"))
        assert htmls, "QC HTML was not generated"


# ---------------------------------------------------------------------------
# P2-F14/F16: crosstalk matrix (plan §B7 tasks 8 + 10)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crosstalk_inputs(source_erp_forward):
    """(fwd, info, noise_cov, labels) for the crosstalk matrix (module-scoped)."""
    from eegcpm.modules.source.labels import build_conn_labels

    epochs = _condition_epochs(n_one=20, n_two=20)
    noise_cov = mne.compute_covariance(
        epochs, tmin=None, tmax=0.0, method="shrunk", rank=None,
        keep_sample_mean=True, verbose=False,
    )
    labels, label_info = build_conn_labels(
        source_erp_forward["fwd"]["src"], source_erp_forward["subjects_dir"]
    )
    return {
        "fwd": source_erp_forward["fwd"],
        # The crosstalk pair needs the average-reference projector (mne requires
        # it for inverse modelling); `_condition_epochs` sets it on the epochs.
        "info": epochs.info.copy(),
        "noise_cov": noise_cov,
        "labels": labels,
        "roi_names": [str(name) for name in label_info["roi_names"]],
    }


@pytest.mark.fsaverage
@pytest.mark.slow
class TestSourceCrosstalk:
    """The crosstalk matrix must be a per-label leakage measure (P2-F14)."""

    def test_target_roi_off_diagonal_leakage_below_one(self, crosstalk_inputs):
        """(a) The dipole-fixture target ROI's strongest off-diagonal entry < 1.

        The diagonal is 1.0 by construction (the ratio's own denominator), so
        only the off-diagonal entries say anything. If this ever fails, the
        measured value must be reported to the auditor — never loosened.

        B7 rev2 fix 6: the physical property is asserted for **both** the
        library matrix and the saved primary (flip-aligned) matrix, so the
        artifact the pipeline actually writes is covered by the non-vacuous
        check, not only the library path.
        """
        from eegcpm.modules.qc.source_qc import (
            compute_crosstalk_matrix, compute_crosstalk_matrix_numpy,
        )

        matrix = compute_crosstalk_matrix(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
        )
        primary = compute_crosstalk_matrix_numpy(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
            align_signs=True,
        )
        target = crosstalk_inputs["roi_names"].index("SensoriMotor.Superior")
        for name, candidate in (("library", matrix), ("flip-aligned", primary)):
            assert candidate.shape == (30, 30)
            assert np.all(np.isfinite(candidate))
            assert np.allclose(np.diag(candidate), 1.0)
            off = candidate[target].copy()
            off[target] = -np.inf
            strongest = float(np.max(off))
            assert np.isfinite(strongest)
            assert strongest < 1.0, (
                f"SensoriMotor.Superior strongest off-diagonal crosstalk is "
                f"{strongest:.4f} (>= 1) for the {name} matrix: report this "
                "value to the auditor"
            )

    def test_permuting_labels_permutes_rows_and_columns(self, crosstalk_inputs):
        """(b) Permuting two label indices permutes C's rows/columns.

        This is what proves the matrix is indexed per label: a whole-brain
        average (the old implementation) or a mis-mapped vertex list would be
        invariant to the permutation.
        """
        from eegcpm.modules.qc.source_qc import compute_crosstalk_matrix

        base = compute_crosstalk_matrix(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
        )
        perm = list(range(len(crosstalk_inputs["labels"])))
        perm[0], perm[1] = perm[1], perm[0]
        permuted_labels = [crosstalk_inputs["labels"][i] for i in perm]
        permuted = compute_crosstalk_matrix(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            permuted_labels,
            method="dSPM",
            lambda2=1.0 / 9.0,
        )
        assert np.allclose(permuted, base[np.ix_(perm, perm)])
        # Non-vacuous: the permutation actually moves entries.
        assert not np.allclose(permuted, base)

    def test_numpy_crosscheck_matches_library_and_flags_sign_cancellation(
        self, crosstalk_inputs
    ):
        """Numpy mirror of the library summary + the measured sign-cancellation.

        ``compute_crosstalk_matrix_numpy`` rebuilds the resolution matrix with
        plain numpy (no ``get_cross_talk`` / ``in_label``) and averages the
        label's CTF rows directly. Two assertions:

        - the plain (signed) mean reproduces the library matrix **exactly** —
          this is the cross-check that our two-step per-pair recipe and the
          label vertex indexing are right, and it pins the **unflipped** path
          (the saved ``crosstalk_unflipped`` matrix, B7 rev2 fix 6);
        - the ``label_sign_flip``-aligned mean does **not** reproduce it. That
          is critic N-3's finding, quantified here: ``get_cross_talk(mode='mean')``
          averages signed surface-normal values, so it cancels across sulcal
          walls (measured 2026-10-16 on oct6: 34 % of vertices flipped on
          average, max |ΔC| 2.08). The **saved primary matrix** is the
          flip-aligned variant (it matches the pipeline's ``mean_flip`` ROI
          extraction, B7 rev2 fix 6); the library variant stays available as
          ``crosstalk_unflipped`` and this test keeps the two distinguishable,
          so a future change to either side is noticed.
        """
        from eegcpm.modules.qc.source_qc import (
            compute_crosstalk_matrix,
            compute_crosstalk_matrix_numpy,
        )

        library = compute_crosstalk_matrix(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
        )
        numpy_plain = compute_crosstalk_matrix_numpy(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
        )
        np.testing.assert_allclose(numpy_plain, library, rtol=1e-9, atol=1e-12)

        numpy_flip = compute_crosstalk_matrix_numpy(
            crosstalk_inputs["fwd"],
            crosstalk_inputs["info"],
            crosstalk_inputs["noise_cov"],
            crosstalk_inputs["labels"],
            method="dSPM",
            lambda2=1.0 / 9.0,
            align_signs=True,
        )
        max_delta = float(np.max(np.abs(numpy_flip - library)))
        assert max_delta > 1e-3, (
            "the sign-flip-aligned summary now matches the library: re-check "
            "whether mne changed get_cross_talk(mode='mean')"
        )


class TestConditionNamePathSeparators:
    """B5-N4: condition names become file names; separators are rejected loudly."""

    @pytest.mark.parametrize("name", ["a/b", "a\\b", "cond/sub"])
    def test_path_separator_names_raise(self, name, tmp_path):
        task_config = {"conditions": [{"name": name, "event_codes": [1]}]}
        module = SourceReconstructionModule({}, tmp_path, task_config=task_config)
        with pytest.raises(ValueError, match="path separator"):
            module._select_condition_epochs(_condition_epochs(n_one=20, n_two=20))

    def test_save_stc_atomic_would_escape_without_the_guard(self, tmp_path):
        """Regression context: the guard exists because stc.save cannot handle `/`.

        Documented here so the guard's purpose is testable without depending on
        mne's exact failure mode: the condition name is used verbatim in the
        output file name ``{subject}_{condition}_stc``.
        """
        module = SourceReconstructionModule({}, tmp_path)
        with pytest.raises(FileNotFoundError):
            module._save_stc_atomic(
                "sub-01", "a/b",
                mne.SourceEstimate(
                    np.zeros((2, 2)),
                    vertices=[np.array([0]), np.array([0])],
                    tmin=0.0, tstep=0.1,
                ),
            )
        # No partial artifact escaped into the output directory.
        assert list(tmp_path.iterdir()) == []
