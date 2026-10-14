"""Tests for anatomical CONN ROI label construction (P2-F1/F3).

Uses the cached fsaverage at ``oct6`` (the probe fact: minimum non-cerebellar vertex
count at oct6 is 12; at oct5 it is 3, so ``oct6`` is required — see the plan).
"""

import mne
import numpy as np
import pytest

from eegcpm.data.conn_rois import (
    CONN_ROIS,
    SOURCE_EXCLUDED_NETWORKS,
    get_roi_by_name,
)
from eegcpm.modules.source import SourceReconstructionModule
from eegcpm.modules.source.labels import build_conn_labels

pytestmark = pytest.mark.fsaverage


@pytest.fixture(scope="module")
def oct6_src(fsaverage_subjects_dir):
    """oct6 fsaverage source space (shared; ``add_dist=False``)."""
    return mne.setup_source_space(
        "fsaverage",
        "oct6",
        subjects_dir=fsaverage_subjects_dir,
        add_dist=False,
        verbose=False,
    )


@pytest.fixture(scope="module")
def oct5_src(fsaverage_subjects_dir):
    """oct5 fsaverage source space: different vertno, still >= 3 vertices per ROI."""
    return mne.setup_source_space(
        "fsaverage",
        "oct5",
        subjects_dir=fsaverage_subjects_dir,
        add_dist=False,
        verbose=False,
    )


@pytest.fixture(scope="module")
def conn_labels(oct6_src, fsaverage_subjects_dir):
    """(labels, info) built with the plan defaults (10 mm lateral / 15 mm midline)."""
    return build_conn_labels(oct6_src, fsaverage_subjects_dir)


def _non_excluded_names():
    return [
        roi.full_name
        for roi in CONN_ROIS
        if roi.network not in SOURCE_EXCLUDED_NETWORKS
    ]


class TestBuildConnLabels:
    def test_label_count_and_order(self, conn_labels):
        labels, info = conn_labels
        expected = _non_excluded_names()
        assert len(labels) == 30
        assert list(info["roi_names"]) == expected
        assert len(set(info["roi_names"])) == 30

    def test_excluded_cerebellar_rois(self, conn_labels):
        labels, info = conn_labels
        assert list(info["excluded_rois"]) == [
            "Cerebellar.Anterior",
            "Cerebellar.Posterior",
        ]
        assert set(info["excluded_reasons"]) == set(info["excluded_rois"])
        assert all(
            "vertices" in reason or "surface" in reason
            for reason in info["excluded_reasons"].values()
        )
        assert not any(label.name.startswith("Cerebellar") for label in labels)

    def test_vertex_counts(self, conn_labels):
        labels, info = conn_labels
        counts = np.asarray(info["roi_vertex_counts"])
        assert counts.shape == (30,)
        assert counts.min() >= 3
        # The plan probe fact: minimum non-cerebellar count at oct6 is 12.
        assert counts.min() == 12
        assert all(count == len(label) for count, label in zip(counts, labels))

    def test_midline_labels_bilateral(self, conn_labels):
        labels, info = conn_labels
        n_midline = 0
        for label, name, hemi in zip(
            labels, info["roi_names"], info["roi_hemi"]
        ):
            if get_roi_by_name(name).is_midline:
                n_midline += 1
                assert hemi == "midline"
                assert isinstance(label, mne.BiHemiLabel)
                assert label.name == name
                assert len(label.lh.vertices) > 0
                assert len(label.rh.vertices) > 0
                assert label.lh.hemi == "lh" and label.rh.hemi == "rh"
        # 6 of the 30 extracted ROIs are midline (MPFC, PCC, SensoriMotor.Superior,
        # Visual.Medial, Visual.Occipital, ACC).
        assert n_midline == 6

    def test_lateral_labels_single_hemi(self, conn_labels):
        labels, info = conn_labels
        for label, name, hemi in zip(
            labels, info["roi_names"], info["roi_hemi"]
        ):
            roi = get_roi_by_name(name)
            if not roi.is_midline:
                assert hemi == roi.hemisphere
                assert isinstance(label, mne.Label)
                assert label.hemi == ("lh" if roi.hemisphere == "L" else "rh")
                assert label.name == name

    def test_coords_and_radii(self, conn_labels):
        _, info = conn_labels
        coords = info["roi_coords_mni"]
        assert coords.shape == (30, 3)
        for name, center in zip(info["roi_names"], coords):
            assert np.allclose(center, get_roi_by_name(name).mni_coords)
        radii = info["roi_radius_mm"]
        for name, radius in zip(info["roi_names"], radii):
            expected = 15.0 if get_roi_by_name(name).is_midline else 10.0
            assert radius == pytest.approx(expected)

    def test_all_lateral_and_midline_rois_present_once(self, conn_labels):
        _, info = conn_labels
        names = list(info["roi_names"])
        # Companion hemisphere for every lateral pair must be present.
        for name in names:
            roi = get_roi_by_name(name)
            if roi.hemisphere == "L":
                assert f"{roi.network}.{roi.name}_R" in names
        for excluded in SOURCE_EXCLUDED_NETWORKS:
            assert not any(n.startswith(f"{excluded}.") for n in names)

    def test_min_vertices_raises(self, oct6_src, fsaverage_subjects_dir):
        with pytest.raises(ValueError, match="min_vertices"):
            build_conn_labels(
                oct6_src, fsaverage_subjects_dir, min_vertices=10 ** 6
            )

    def test_radius_override_increases_count(self, oct6_src, fsaverage_subjects_dir):
        _, base_info = build_conn_labels(oct6_src, fsaverage_subjects_dir)
        name = "DorsalAttention.FEF_L"
        base_count = int(
            base_info["roi_vertex_counts"][list(base_info["roi_names"]).index(name)]
        )
        _, override_info = build_conn_labels(
            oct6_src,
            fsaverage_subjects_dir,
            radius_overrides={name: 20.0},
        )
        idx = list(override_info["roi_names"]).index(name)
        assert override_info["roi_radius_mm"][idx] == pytest.approx(20.0)
        assert int(override_info["roi_vertex_counts"][idx]) > base_count
        # Other ROIs keep their default radii.
        other = "DefaultMode.MPFC"
        j = list(override_info["roi_names"]).index(other)
        assert override_info["roi_radius_mm"][j] == pytest.approx(15.0)

    def test_excluded_roi_not_in_names_even_with_override(
        self, oct6_src, fsaverage_subjects_dir
    ):
        _, info = build_conn_labels(
            oct6_src,
            fsaverage_subjects_dir,
            radius_overrides={"Cerebellar.Anterior": 40.0},
        )
        assert "Cerebellar.Anterior" not in list(info["roi_names"])
        assert "Cerebellar.Anterior" in list(info["excluded_rois"])

    def test_individual_subject_source_space_raises(
        self, oct6_src, fsaverage_subjects_dir
    ):
        """N2: an individual-subject source space must not map onto fsaverage.

        ``mne.vertex_to_mni``/``mne.Label`` only warn when the source space subject
        differs from the label subject, so the guard must be explicit.
        """
        src_individual = oct6_src.copy()
        src_individual[0]["subject_his_id"] = "sub-01"
        src_individual[1]["subject_his_id"] = "sub-01"
        with pytest.raises(ValueError, match="subject"):
            build_conn_labels(src_individual, fsaverage_subjects_dir)

        # The reverse direction (individual labels requested on an fsaverage source
        # space) is equally loud.
        with pytest.raises(ValueError, match="subject"):
            build_conn_labels(oct6_src, fsaverage_subjects_dir, subject="sub-01")

    def test_unknown_radius_override_key_raises(
        self, oct6_src, fsaverage_subjects_dir
    ):
        """N3: stale pre-B2 override names fail loudly instead of doing nothing."""
        with pytest.raises(ValueError, match="radius_overrides"):
            build_conn_labels(
                oct6_src,
                fsaverage_subjects_dir,
                radius_overrides={"DefaultMode.LP": 20.0},
            )


class TestLabelCacheKeyedBySourceSpace:
    """N1: the module label cache is keyed on the source space vertno arrays."""

    def test_same_vertno_cached_different_vertno_rebuilt(
        self, oct6_src, oct5_src, fsaverage_subjects_dir, tmp_path, monkeypatch
    ):
        calls = []
        real_build = build_conn_labels

        def counting_build(src, subjects_dir, **kwargs):
            calls.append(
                tuple(
                    tuple(np.asarray(src[h]["vertno"], dtype=int).tolist())
                    for h in (0, 1)
                )
            )
            return real_build(src, subjects_dir, **kwargs)

        monkeypatch.setattr("eegcpm.modules.source.build_conn_labels", counting_build)
        module = SourceReconstructionModule(
            {"subjects_dir": str(fsaverage_subjects_dir)}, tmp_path
        )

        labels_oct6, info_oct6 = module._get_conn_labels(oct6_src)
        # A distinct source-space object with the same vertno must hit the cache
        # (keyed on vertno, not object identity).
        labels_same, info_same = module._get_conn_labels(oct6_src.copy())
        assert labels_same is labels_oct6
        assert info_same is info_oct6
        assert len(calls) == 1

        labels_oct5, info_oct5 = module._get_conn_labels(oct5_src)
        assert len(calls) == 2  # different vertno -> rebuilt
        assert labels_oct5 is not labels_oct6
        assert not np.array_equal(
            info_oct6["roi_vertex_counts"], info_oct5["roi_vertex_counts"]
        )

    def test_cache_hit_still_enforces_subject_guard(
        self, oct6_src, fsaverage_subjects_dir, tmp_path
    ):
        """CF-1: a cached instance must not serve fsaverage labels for a
        same-vertno source space carrying a different subject name."""
        module = SourceReconstructionModule(
            {"subjects_dir": str(fsaverage_subjects_dir)}, tmp_path
        )
        module._get_conn_labels(oct6_src)  # populate the cache

        scaled = oct6_src.copy()
        for space in scaled:
            space["subject_his_id"] = "sub-01"
        with pytest.raises(ValueError, match="sub-01"):
            module._get_conn_labels(scaled)
