"""P2-F6: the source-ERP feature consumer reads the B4 evoked contract.

Contract: ``{cond}: (n_rois, n_times)`` + ``{cond}_times`` + ``roi_names``
(+ optional ``{cond}_nave``). Conditions are the keys with a ``{key}_times``
partner; the time axis comes from ``{cond}_times`` (negative tmin preserved);
columns are per condition x ROI x component x metric; nothing is written when
nothing matched; 3-D (trial) arrays raise.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from eegcpm.core.provenance import provenance_path_for_archive, sha256_file
from eegcpm.modules.features.source_erp_features import SourceERPFeatureModule

SFREQ = 250.0
TIMES = np.arange(-0.2, 0.6 + 1e-9, 1.0 / SFREQ)
N_TIMES = len(TIMES)
COMPONENTS = {
    "N1": {"search_window": [0.05, 0.15], "polarity": "negative"},
    "N0": {"search_window": [-0.2, 0.0], "polarity": "negative"},
}
CONFIG_HASH = "c" * 64


def _write_provenance(npz_path, *, config_hash=CONFIG_HASH, artifacts="auto", raw=None,
                      run_identity_hash=None):
    """Write the source-stage provenance sidecar for an archive (B5 task 8).

    ``artifacts`` is ``'auto'`` (hash the archive), a mapping to write verbatim
    (e.g. a deliberately wrong hash), or a falsy value to omit the ``artifacts``
    block entirely.
    """
    payload = {"config_hash": config_hash, "epochs_sha256": "e" * 64}
    if run_identity_hash is not None:
        payload["run_identity_hash"] = run_identity_hash
    if artifacts == "auto":
        payload["artifacts"] = {npz_path.name: sha256_file(npz_path)}
    elif artifacts:
        payload["artifacts"] = dict(artifacts)
    provenance_path = provenance_path_for_archive(npz_path)
    provenance_path.write_text(
        raw if raw is not None else json.dumps(payload)
    )
    return provenance_path


def _synthetic_data(seed: int = 0, n_rois: int = 3) -> dict:
    """Two conditions, known negative N1 at 0.100 s (ROI_0) and N0 at -0.100 s (ROI_1)."""
    rng = np.random.default_rng(seed)
    roi_names = np.array([f"ROI_{i}" for i in range(n_rois)], dtype=str)
    data = {"roi_names": roi_names, "sfreq": SFREQ}
    for cond in ("condA", "condB"):
        tc = rng.standard_normal((n_rois, N_TIMES)) * 0.01
        tc[0] += -1.0 * np.exp(-0.5 * ((TIMES - 0.1) / 0.015) ** 2)
        tc[1] += -0.8 * np.exp(-0.5 * ((TIMES + 0.1) / 0.015) ** 2)
        data[cond] = tc
        data[f"{cond}_times"] = TIMES.copy()
        data[f"{cond}_nave"] = 20
    return data


def _save_npz(path, data=None):
    data = data if data is not None else _synthetic_data()
    np.savez(path, **data)
    return path


def _config(**overrides) -> dict:
    config = {
        "stage": "features",
        "variant": "source_erp_test",
        "components": COMPONENTS,
        "metrics": ["peak_amplitude", "peak_latency", "mean_amplitude", "area_under_curve"],
        "generate_qc": False,
    }
    config.update(overrides)
    return config


def _process(tmp_path, data=None, **config_overrides):
    module = SourceERPFeatureModule(_config(**config_overrides), tmp_path)
    return module.process(
        data if data is not None else _synthetic_data(),
        subject=SimpleNamespace(id="sub-01", session="01"),
    )


# --- happy path -------------------------------------------------------------

def test_peak_latency_column_per_condition(tmp_path):
    result = _process(tmp_path)
    assert result.success, result.errors
    features = result.outputs["features"]
    assert isinstance(features, pd.DataFrame)

    # Column contract: {cond}_{roi}_{comp}_{metric}
    lat_col = "condA_ROI_0_N1_peak_latency"
    amp_col = "condA_ROI_0_N1_peak_amplitude"
    assert lat_col in features.columns
    assert amp_col in features.columns
    assert float(features[lat_col].iloc[0]) == pytest.approx(0.100, abs=1.0 / SFREQ)
    assert float(features[amp_col].iloc[0]) < 0  # negative N1 (signed source output)

    # Negative tmin is preserved: the N0 component peaks at -0.100 s.
    n0_col = "condA_ROI_1_N0_peak_latency"
    assert float(features[n0_col].iloc[0]) == pytest.approx(-0.100, abs=1.0 / SFREQ)

    # np.trapezoid path produces the AUC metric
    assert "condA_ROI_0_N1_area_under_curve" in features.columns


def test_per_condition_columns_are_distinct(tmp_path):
    result = _process(tmp_path)
    features = result.outputs["features"]
    assert "condA_ROI_0_N1_peak_amplitude" in features.columns
    assert "condB_ROI_0_N1_peak_amplitude" in features.columns
    assert not np.isclose(
        float(features["condA_ROI_0_N1_peak_amplitude"].iloc[0]),
        float(features["condB_ROI_0_N1_peak_amplitude"].iloc[0]),
    )


def test_nave_columns_when_present(tmp_path):
    result = _process(tmp_path)
    features = result.outputs["features"]
    assert int(features["condA_nave"].iloc[0]) == 20
    assert int(features["condB_nave"].iloc[0]) == 20


def test_metadata_sfreq_derived_from_times(tmp_path):
    result = _process(tmp_path)
    assert result.metadata["sfreq"] == pytest.approx(SFREQ)
    assert result.metadata["conditions"] == ["condA", "condB"]


# --- input types ------------------------------------------------------------

def test_npzfile_input_works(tmp_path):
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path)
    with np.load(npz_path) as npz:
        module = SourceERPFeatureModule(_config(), tmp_path)
        result = module.process(npz, subject=SimpleNamespace(id="sub-01", session="01"))
    assert result.success, result.errors
    assert "condA_ROI_0_N1_peak_latency" in result.outputs["features"].columns


def test_path_input_works(tmp_path):
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, run_identity_hash="i" * 64)
    result = _process(tmp_path, data=npz_path)
    assert result.success, result.errors
    assert "condA_ROI_0_N1_peak_latency" in result.outputs["features"].columns
    # the consumed run's identity is carried into the feature metadata
    assert result.metadata["source_config_hash"] == CONFIG_HASH
    assert result.metadata["source_epochs_sha256"] == "e" * 64
    # B5-N3: the run identity (config hash + condition mapping) travels too.
    assert result.metadata["source_run_identity_hash"] == "i" * 64


def test_mapping_input_needs_no_provenance(tmp_path):
    """In-memory mappings have no sidecar; the caller owns their freshness."""
    result = _process(tmp_path)  # dict input
    assert result.success, result.errors
    assert result.metadata["source_config_hash"] is None


def test_validate_input_accepts_path_and_npzfile(tmp_path):
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    module = SourceERPFeatureModule(_config(), tmp_path)
    assert module.validate_input(_synthetic_data()) is True
    # A path without provenance is not a valid input (B5 task 8) ...
    assert module.validate_input(npz_path) is False
    with np.load(npz_path) as npz:
        assert module.validate_input(npz) is False
    _write_provenance(npz_path)
    assert module.validate_input(npz_path) is True
    with np.load(npz_path) as npz:
        assert module.validate_input(npz) is True
    assert module.validate_input("not an archive") is False


# --- provenance requirement (B5 task 8) ------------------------------------


def test_path_without_provenance_rejected(tmp_path):
    """A stale/partial archive must not be consumed silently."""
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        module.process(npz_path, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_provenance_without_config_hash_rejected(tmp_path):
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, config_hash="")
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="config_hash"):
        module.process(npz_path, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_provenance_hash_mismatch_rejected(tmp_path):
    """A recorded sha256 that does not match the file means a partial write."""
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, artifacts={"sub-01_roi_tc.npz": "0" * 64})
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        module.process(npz_path, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_unreadable_provenance_rejected(tmp_path):
    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, raw="{not json")
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="unreadable provenance"):
        module.process(npz_path, subject=SimpleNamespace(id="sub-01", session="01"))


# --- loud failures (nothing written) ---------------------------------------

def test_trial_level_input_raises_and_writes_no_csv(tmp_path):
    data = _synthetic_data()
    data["condA"] = np.zeros((2, 3, N_TIMES))
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="trial-level"):
        module.process(data, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_no_conditions_raises_and_writes_no_csv(tmp_path):
    data = {"roi_names": np.array(["ROI_0"]), "sfreq": SFREQ}
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="no conditions"):
        module.process(data, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_no_matching_rois_raises_and_writes_no_csv(tmp_path):
    module = SourceERPFeatureModule(_config(rois=["NOPE"]), tmp_path)
    with pytest.raises(ValueError, match="no ROIs matched"):
        module.process(_synthetic_data(), subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


def test_shape_mismatch_raises(tmp_path):
    data = _synthetic_data()
    data["condA"] = data["condA"][:, :-1]  # n_times mismatch
    module = SourceERPFeatureModule(_config(), tmp_path)
    with pytest.raises(ValueError, match="times"):
        module.process(data, subject=SimpleNamespace(id="sub-01", session="01"))
    assert not (tmp_path / "features.csv").exists()


# --- CLI wiring -------------------------------------------------------------

def test_features_cli_source_erp_from_input_dir(tmp_path):
    from eegcpm.cli.features import features_command

    npz_dir = tmp_path / "source"
    npz_dir.mkdir()
    npz_path = _save_npz(npz_dir / "sub-01_roi_tc.npz")
    _write_provenance(npz_path)
    config_path = tmp_path / "features.yaml"
    config_path.write_text(yaml.safe_dump(_config()))

    args = SimpleNamespace(
        config=config_path,
        project=tmp_path,
        feature_type="source_erp",
        input_dir=npz_dir,
        output_dir=tmp_path / "features_out",
        condition="resting_ec",
        sfreq=500.0,
        source=None,
        preprocessing=None,
        task=None,
        subjects=None,
        sessions=None,
    )
    result = features_command(args)
    out_path = tmp_path / "features_out" / "source_erp_resting_ec.parquet"
    assert out_path.exists()
    df = pd.read_parquet(out_path)
    assert "condA_ROI_0_N1_peak_latency" in df.columns
    assert result["n_rows"] == 1
    assert result["n_missing"] == 0


def test_features_cli_rejects_archive_without_provenance(tmp_path):
    """The CLI must not silently consume an unattributable archive (B5 task 8)."""
    from eegcpm.cli.features import features_command

    npz_dir = tmp_path / "source"
    npz_dir.mkdir()
    _save_npz(npz_dir / "sub-01_roi_tc.npz")  # no provenance sidecar
    config_path = tmp_path / "features.yaml"
    config_path.write_text(yaml.safe_dump(_config()))

    args = SimpleNamespace(
        config=config_path,
        project=tmp_path,
        feature_type="source_erp",
        input_dir=npz_dir,
        output_dir=tmp_path / "features_out",
        condition="resting_ec",
        sfreq=500.0,
        source=None,
        preprocessing=None,
        task=None,
        subjects=None,
        sessions=None,
    )
    with pytest.raises(ValueError, match="provenance"):
        features_command(args)
    assert not (tmp_path / "features_out" / "source_erp_resting_ec.parquet").exists()


# --- B5 rev1 (audit conditions N1/N2/A1) ------------------------------------

def test_provenance_without_artifacts_map_rejected(tmp_path):
    """B5-N2: the artifacts map is mandatory — omitting it must not skip the
    hash check."""
    from eegcpm.core.provenance import verify_archive_provenance

    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, artifacts=False)
    with pytest.raises(ValueError, match="artifacts"):
        verify_archive_provenance(npz_path)


def test_archive_not_listed_in_artifacts_rejected(tmp_path):
    """B5-N2: an archive absent from the artifacts map is rejected."""
    from eegcpm.core.provenance import verify_archive_provenance

    npz_path = _save_npz(tmp_path / "sub-01_roi_tc.npz")
    _write_provenance(npz_path, artifacts={"other.npz": "0" * 64})
    with pytest.raises(ValueError, match="not listed"):
        verify_archive_provenance(npz_path)


def test_trials_archive_pairs_with_subject_provenance(tmp_path):
    """B5-A1: a trials archive verifies against ``{subject}_provenance.json``
    (the file the source stage actually writes), not a never-written
    ``*_trials_provenance.json``."""
    from eegcpm.core.provenance import verify_archive_provenance

    trials = _save_npz(tmp_path / "sub-01_roi_tc_trials.npz")
    _write_provenance(trials)
    assert (tmp_path / "sub-01_provenance.json").exists()
    assert verify_archive_provenance(trials)["config_hash"] == CONFIG_HASH


def test_features_cli_refuses_mixed_source_configs(tmp_path):
    """B5-N1: a feature table mixing archives from different source configs is
    refused by default; the explicit override allows it and reports the hashes."""
    from eegcpm.cli.features import features_command

    npz_dir = tmp_path / "source"
    npz_dir.mkdir()
    a = _save_npz(npz_dir / "sub-01_roi_tc.npz")
    b = _save_npz(npz_dir / "sub-02_roi_tc.npz")
    _write_provenance(a, config_hash="a" * 64, run_identity_hash="1" * 64)
    _write_provenance(b, config_hash="b" * 64, run_identity_hash="2" * 64)
    config_path = tmp_path / "features.yaml"
    config_path.write_text(yaml.safe_dump(_config()))

    args = SimpleNamespace(
        config=config_path,
        project=tmp_path,
        feature_type="source_erp",
        input_dir=npz_dir,
        output_dir=tmp_path / "features_out",
        condition="resting_ec",
        sfreq=500.0,
        source=None,
        preprocessing=None,
        task=None,
        subjects=None,
        sessions=None,
    )
    with pytest.raises(RuntimeError, match="different source configs"):
        features_command(args)
    assert not (tmp_path / "features_out" / "source_erp_resting_ec.parquet").exists()

    args.allow_mixed_source_configs = True
    result = features_command(args)
    assert sorted(result["source_config_hashes"]) == ["a" * 64, "b" * 64]
    assert sorted(result["source_run_identity_hashes"]) == ["1" * 64, "2" * 64]
    parquet = tmp_path / "features_out" / "source_erp_resting_ec.parquet"
    assert parquet.exists()

    # B6 task 19: the golden-hash provenance sits next to the parquet.
    sidecar = tmp_path / "features_out" / "source_erp_resting_ec.provenance.json"
    payload = json.loads(sidecar.read_text())
    assert payload["feature_type"] == "source_erp"
    assert payload["condition"] == "resting_ec"
    assert payload["source_config_hashes"] == ["a" * 64, "b" * 64]
    assert payload["source_run_identity_hashes"] == ["1" * 64, "2" * 64]
    assert payload["parquet"] == parquet.name
    assert payload["parquet_sha256"] == sha256_file(parquet)
    assert payload["library_versions"]["mne"]
    assert payload["n_rows"] == int(result["n_rows"])


def test_qc_report_is_written_with_embedded_figure(tmp_path):
    """The source-ERP QC report must build through the real builder API.

    2026-10-17: the report path called a non-existent
    ``HTMLReportBuilder.add_section`` (and passed ``add_figure`` arguments in the
    wrong order), so every source-ERP QC report was skipped with a warning and
    no HTML was written.
    """
    module = SourceERPFeatureModule(_config(generate_qc=True), tmp_path)
    module.process(_synthetic_data(), subject=SimpleNamespace(id="sub-01", session="01"))

    qc_path = tmp_path / "sub-01_source_erp_qc.html"
    assert qc_path.exists(), "source-ERP QC report was not written"
    html = qc_path.read_text()
    assert "Overview" in html
    assert "ROI Event-Related Potentials" in html
    assert 'src="data:image/png;base64,' in html, "figure not embedded"
