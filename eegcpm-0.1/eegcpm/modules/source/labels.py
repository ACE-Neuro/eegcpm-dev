"""Anatomical ROI labels for the CONN 32-network parcellation (P2-F1/F3).

Labels are built directly on the source space actually used by the inverse operator
(``inv['src']``), so that ``extract_label_time_course`` cannot silently mismatch the
source space (see the note in ``mne.extract_label_time_course``).

Geometry rules (approved plan §0.5):

- ROI centres are CONN MNI coordinates; source vertices are mapped to MNI with
  :func:`mne.vertex_to_mni`.
- Lateral ROIs (``ROI.hemisphere`` set) use a single-hemisphere :class:`mne.Label`
  with radius ``radius_mm`` (default 10 mm).
- Midline ROIs (``|x| <= MIDLINE_X_MM``) use a :class:`mne.BiHemiLabel`
  (both hemispheres must contain vertices, otherwise ``ValueError``) with radius
  ``midline_radius_mm`` (default 15 mm).
- ``radius_overrides`` maps a ROI full name to a per-ROI radius (highest precedence).
  Unknown keys raise ``ValueError`` (N3); an override for an excluded ROI is allowed.
- ``src._subject`` must match ``subject`` (default fsaverage): an individual-subject
  source space raises ``ValueError`` instead of being silently mapped onto fsaverage
  (N2).
- Networks in ``SOURCE_EXCLUDED_NETWORKS`` (Cerebellar) are skipped and reported in
  the returned metadata.
- A ROI with fewer than ``min_vertices`` vertices raises ``ValueError`` — never a
  silently empty or overwritten time course.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mne
import numpy as np

from eegcpm.data.conn_rois import CONN_ROIS, SOURCE_EXCLUDED_NETWORKS, get_roi_names

# Source-space hemisphere index for each anatomical hemisphere.
HEMI_TO_SRC_INDEX = {"L": 0, "R": 1}
_SRC_INDEX_TO_HEMI = {0: "lh", 1: "rh"}


def build_conn_labels(
    src: mne.SourceSpaces,
    subjects_dir: Union[str, Path],
    radius_mm: float = 10.0,
    midline_radius_mm: float = 15.0,
    radius_overrides: Optional[Dict[str, float]] = None,
    min_vertices: int = 3,
    subject: str = "fsaverage",
) -> Tuple[List[Any], Dict[str, Any]]:
    """Build one label per CONN ROI on ``src``.

    Parameters
    ----------
    src : mne.SourceSpaces
        Source space used by the inverse operator (``inv['src']`` or ``fwd['src']``);
        must contain both hemispheres.
    subjects_dir : str | Path
        Subjects directory containing ``<subject>`` (fsaverage by default).
    radius_mm : float
        Extraction radius (mm) for lateral ROIs.
    midline_radius_mm : float
        Extraction radius (mm) for midline ROIs.
    radius_overrides : dict | None
        ``{roi_full_name: radius_mm}`` overrides applied before the midline default.
    min_vertices : int
        Minimum number of vertices per label; smaller labels raise ``ValueError``.
    subject : str
        Subject name the labels belong to (default ``fsaverage``). Must match the
        source space subject (``src._subject``); an individual-subject source space
        is rejected instead of being silently mapped onto the fsaverage surface.

    Returns
    -------
    labels : list
        ``mne.Label`` / ``mne.BiHemiLabel`` objects, in ``CONN_ROIS`` order minus the
        excluded networks.
    info : dict
        ``roi_names``, ``roi_coords_mni`` (n, 3), ``roi_radius_mm`` (n,),
        ``roi_vertex_counts`` (n,), ``roi_hemi`` ('L'|'R'|'midline'),
        ``excluded_rois``, ``excluded_reasons``.

    Raises
    ------
    ValueError
        If ``src`` is not a 2-hemisphere surface source space, if its subject does not
        match ``subject`` (N2), if ``radius_overrides`` contains unknown ROI names
        (N3), if a midline ROI has no vertices in one hemisphere, or if a ROI has
        fewer than ``min_vertices`` vertices.
    """
    if len(src) != 2:
        raise ValueError(f"Expected a 2-hemisphere surface source space, got {len(src)} spaces")

    # N2: labels are built with ``subject`` but the vertices come from ``src``. MNE only
    # warns when the two disagree, so an individual-subject source space would silently
    # map individual vertex numbers onto the fsaverage surface.
    src_subject = getattr(src, "_subject", None)
    if src_subject != subject:
        raise ValueError(
            f"Source space subject {src_subject!r} does not match label subject "
            f"{subject!r}; individual-subject source spaces must not be mapped onto "
            f"the fsaverage surface"
        )

    radius_overrides = radius_overrides or {}
    # N3: unknown override keys (e.g. stale pre-B2 names like 'DefaultMode.LP') would
    # otherwise be silently ignored and the override would do nothing. Overrides for
    # excluded ROIs (e.g. 'Cerebellar.Anterior') remain valid keys.
    unknown_overrides = set(radius_overrides) - set(get_roi_names())
    if unknown_overrides:
        raise ValueError(
            f"radius_overrides contains unknown ROI names: {sorted(unknown_overrides)}; "
            "valid names are the get_roi_names() entries (lateral ROIs carry a _L/_R "
            "suffix, midline names have none)"
        )

    subjects_dir = Path(subjects_dir)
    mni = [
        mne.vertex_to_mni(
            src[h]["vertno"], h, subject, subjects_dir=subjects_dir, verbose=False
        )
        for h in (0, 1)
    ]

    labels: List[Any] = []
    roi_names: List[str] = []
    roi_coords: List[np.ndarray] = []
    roi_radii: List[float] = []
    roi_counts: List[int] = []
    roi_hemi: List[str] = []
    excluded_rois: List[str] = []
    excluded_reasons: Dict[str, str] = {}

    for roi in CONN_ROIS:
        name = roi.full_name
        if roi.network in SOURCE_EXCLUDED_NETWORKS:
            excluded_rois.append(name)
            excluded_reasons[name] = SOURCE_EXCLUDED_NETWORKS[roi.network]
            continue

        center = np.asarray(roi.mni_coords, dtype=float)
        radius = float(
            radius_overrides.get(
                name, midline_radius_mm if roi.is_midline else radius_mm
            )
        )

        if roi.is_midline:
            selected = []
            for h in (0, 1):
                dist = np.linalg.norm(mni[h] - center, axis=1)
                selected.append(src[h]["vertno"][dist <= radius])
            empty_hemis = [
                _SRC_INDEX_TO_HEMI[h] for h, sel in enumerate(selected) if len(sel) == 0
            ]
            if empty_hemis:
                raise ValueError(
                    f"Midline ROI {name}: no source-space vertices within {radius} mm "
                    f"in hemisphere(s) {empty_hemis}; bilateral extraction is impossible"
                )
            label = mne.BiHemiLabel(
                mne.Label(
                    selected[0], hemi="lh", name=f"{name}-lh", subject=subject
                ),
                mne.Label(
                    selected[1], hemi="rh", name=f"{name}-rh", subject=subject
                ),
                name=name,
            )
            hemi = "midline"
        else:
            if roi.hemisphere not in HEMI_TO_SRC_INDEX:
                raise ValueError(
                    f"ROI {name}: hemisphere must be one of "
                    f"{sorted(HEMI_TO_SRC_INDEX)}, got {roi.hemisphere!r}"
                )
            h = HEMI_TO_SRC_INDEX[roi.hemisphere]
            dist = np.linalg.norm(mni[h] - center, axis=1)
            label = mne.Label(
                src[h]["vertno"][dist <= radius],
                hemi=_SRC_INDEX_TO_HEMI[h],
                name=name,
                subject=subject,
            )
            hemi = roi.hemisphere

        count = len(label)  # BiHemiLabel has __len__ but no .vertices (mne 1.13)
        if count < min_vertices:
            raise ValueError(f"ROI {name}: {count} vertices < min_vertices={min_vertices}")

        labels.append(label)
        roi_names.append(name)
        roi_coords.append(center)
        roi_radii.append(radius)
        roi_counts.append(count)
        roi_hemi.append(hemi)

    info: Dict[str, Any] = {
        "roi_names": roi_names,
        "roi_coords_mni": np.asarray(roi_coords, dtype=float),
        "roi_radius_mm": np.asarray(roi_radii, dtype=float),
        "roi_vertex_counts": np.asarray(roi_counts, dtype=int),
        "roi_hemi": roi_hemi,
        "excluded_rois": excluded_rois,
        "excluded_reasons": excluded_reasons,
    }
    return labels, info
