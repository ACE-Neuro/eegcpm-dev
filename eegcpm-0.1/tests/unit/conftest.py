"""Shared fixtures for unit tests.

``fsaverage_subjects_dir`` resolves the *cached* fsaverage only. Tests must never
download data: if the 3-layer BEM solution is missing the dependent tests are
skipped (the CI workflow prefetches fsaverage before running the suite).
"""

from pathlib import Path

import mne
import pytest


@pytest.fixture(scope="session")
def fsaverage_subjects_dir():
    """Subjects dir containing a cached fsaverage, else skip.

    The path is resolved the same way ``mne.datasets.fetch_fsaverage`` resolves it
    (``SUBJECTS_DIR`` config variable, falling back to ``~/mne_data/MNE-fsaverage-data``)
    but without calling the fetch function, which would download.
    """
    subjects_dir = mne.get_config("SUBJECTS_DIR")
    if not subjects_dir:
        subjects_dir = str(Path.home() / "mne_data" / "MNE-fsaverage-data")
    subjects_dir = Path(subjects_dir).expanduser()

    bem_sol = (
        subjects_dir / "fsaverage" / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
    )
    if not bem_sol.exists():
        pytest.skip(f"fsaverage not cached under {subjects_dir}; no download inside tests")
    return subjects_dir
