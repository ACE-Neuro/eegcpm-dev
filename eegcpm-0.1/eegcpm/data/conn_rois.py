"""
CONN Toolbox Network ROIs (32 nodes, 8 networks).

Source: CONN's ICA analyses of HCP dataset (497 subjects)
Reference: https://web.conn-toolbox.org/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Midline ROIs (|x| <= MIDLINE_X_MM in MNI space) are extracted bilaterally with
# a larger radius because their centres fall between the two hemispheres (P2-F3).
MIDLINE_X_MM = 5.0

# Networks that cannot be sampled from a cortical surface source space. Cerebellar
# ROIs have no fsaverage surface vertices within the ROI radius (P2-F3); they are
# excluded from extraction and reported in the per-run ROI metadata.
SOURCE_EXCLUDED_NETWORKS: Dict[str, str] = {
    "Cerebellar": "no cortical surface source-space vertices within radius (fsaverage)",
}


@dataclass
class ROI:
    """Region of Interest definition."""
    name: str
    network: str
    mni_coords: Tuple[float, float, float]
    hemisphere: Optional[str] = None  # "L", "R", or None for midline

    @property
    def full_name(self) -> str:
        """Unique ROI key: hemisphere suffix only for lateral ROIs (P2-F2)."""
        if self.hemisphere:
            return f"{self.network}.{self.name}_{self.hemisphere}"
        return f"{self.network}.{self.name}"

    @property
    def is_midline(self) -> bool:
        """True when the ROI centre is within ``MIDLINE_X_MM`` of the midline."""
        return abs(self.mni_coords[0]) <= MIDLINE_X_MM


# CONN Network ROIs - 32 nodes across 8 networks
CONN_ROIS: List[ROI] = [
    # Default Mode Network (4 ROIs)
    ROI("MPFC", "DefaultMode", (1, 55, -3), None),
    ROI("LP", "DefaultMode", (-39, -77, 33), "L"),
    ROI("LP", "DefaultMode", (47, -67, 29), "R"),
    ROI("PCC", "DefaultMode", (1, -61, 38), None),

    # SensoriMotor (3 ROIs)
    ROI("Lateral", "SensoriMotor", (-55, -12, 29), "L"),
    ROI("Lateral", "SensoriMotor", (56, -10, 29), "R"),
    ROI("Superior", "SensoriMotor", (0, -31, 67), None),

    # Visual (4 ROIs)
    ROI("Medial", "Visual", (2, -79, 12), None),
    ROI("Occipital", "Visual", (0, -93, -4), None),
    ROI("Lateral", "Visual", (-37, -79, 10), "L"),
    ROI("Lateral", "Visual", (38, -72, 13), "R"),

    # Salience / Cingulo-Opercular (7 ROIs)
    ROI("ACC", "Salience", (0, 22, 35), None),
    ROI("AInsula", "Salience", (-44, 13, 1), "L"),
    ROI("AInsula", "Salience", (47, 14, 0), "R"),
    ROI("RPFC", "Salience", (-32, 45, 27), "L"),
    ROI("RPFC", "Salience", (32, 46, 27), "R"),
    ROI("SMG", "Salience", (-60, -39, 31), "L"),
    ROI("SMG", "Salience", (62, -35, 32), "R"),

    # Dorsal Attention (4 ROIs)
    ROI("FEF", "DorsalAttention", (-27, -9, 64), "L"),
    ROI("FEF", "DorsalAttention", (30, -6, 64), "R"),
    ROI("IPS", "DorsalAttention", (-39, -43, 52), "L"),
    ROI("IPS", "DorsalAttention", (39, -42, 54), "R"),

    # FrontoParietal / Central Executive (4 ROIs)
    ROI("LPFC", "FrontoParietal", (-43, 33, 28), "L"),
    ROI("PPC", "FrontoParietal", (-46, -58, 49), "L"),
    ROI("LPFC", "FrontoParietal", (41, 38, 30), "R"),
    ROI("PPC", "FrontoParietal", (52, -52, 45), "R"),

    # Language (4 ROIs)
    ROI("IFG", "Language", (-51, 26, 2), "L"),
    ROI("IFG", "Language", (54, 28, 1), "R"),
    ROI("pSTG", "Language", (-57, -47, 15), "L"),
    ROI("pSTG", "Language", (59, -42, 13), "R"),

    # Cerebellar (2 ROIs)
    ROI("Anterior", "Cerebellar", (0, -63, -30), None),
    ROI("Posterior", "Cerebellar", (0, -79, -32), None),
]

# Network names and ROI counts
CONN_NETWORKS: Dict[str, int] = {
    "DefaultMode": 4,
    "SensoriMotor": 3,
    "Visual": 4,
    "Salience": 7,
    "DorsalAttention": 4,
    "FrontoParietal": 4,
    "Language": 4,
    "Cerebellar": 2,
}


def get_conn_rois() -> List[ROI]:
    """Get all CONN ROIs."""
    return CONN_ROIS.copy()


def get_roi_names() -> List[str]:
    """Get list of ROI full names in order."""
    return [roi.full_name for roi in CONN_ROIS]


# P2-F2: 32 ROI definitions collapse to 20 unique names without a hemisphere suffix,
# which silently overwrote left/right time courses. Enforce uniqueness at import time.
_ROI_NAMES = get_roi_names()
assert len(_ROI_NAMES) == len(set(_ROI_NAMES)), (
    "CONN ROI full names must be unique; duplicates: "
    f"{sorted({n for n in _ROI_NAMES if _ROI_NAMES.count(n) > 1})}"
)


def get_roi_by_name(name: str) -> Optional[ROI]:
    """Get ROI by full name (e.g., 'DefaultMode.MPFC')."""
    for roi in CONN_ROIS:
        if roi.full_name == name:
            return roi
    return None


def get_rois_by_network(network: str) -> List[ROI]:
    """Get all ROIs in a network."""
    return [roi for roi in CONN_ROIS if roi.network == network]


def get_mni_coordinates() -> np.ndarray:
    """
    Get MNI coordinates as numpy array.

    Returns:
        Array of shape (32, 3) with MNI coordinates
    """
    return np.array([roi.mni_coords for roi in CONN_ROIS])


def get_network_indices() -> Dict[str, List[int]]:
    """
    Get ROI indices grouped by network.

    Returns:
        Dict mapping network name to list of ROI indices
    """
    indices = {}
    for i, roi in enumerate(CONN_ROIS):
        if roi.network not in indices:
            indices[roi.network] = []
        indices[roi.network].append(i)
    return indices


def get_template_path() -> Path:
    """Get path to CONN networks template directory."""
    return Path(__file__).parent / "templates" / "conn_networks"


def load_nifti_template():
    """
    Load CONN networks NIfTI template.

    Returns:
        nibabel Nifti1Image or None if nibabel not available
    """
    try:
        import nibabel as nib
        template_path = get_template_path() / "networks.nii"
        if template_path.exists():
            return nib.load(template_path)
    except ImportError:
        pass
    return None
