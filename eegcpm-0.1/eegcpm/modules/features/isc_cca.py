"""
ISC (Inter-Subject Correlation) module (per spec §3.m + S13 + R-011).

ISC is an OUTCOME, not a CPM feature. Drop fold-purity language.
Inference is by PERMUTATION of d labels holding the ISC vector
fixed (this preserves the LOO dependency structure of the ISC
values).

Band pinned to broadband 2-45 Hz (per S13c — the only band with
effective ratio > 1). n_components=1 (per S13d — the only stably
estimable component at broadband ratio ~2.5).

R-011: the implementation is a regularized one-component CorrCA
(per ENG-014 + R-011). fit_template assembles the cross-subject
covariance with LW shrinkage, solves the eigenproblem, and
stores weights + template time course. transform projects held-out
subjects onto the template and computes ISC = correlation of
projected time course with template time course.

This module re-exports the new implementation (isc_corrca) for
backward compatibility with the existing tests in test_isc.py.
"""

from .isc_corrca import (
    ISC_BAND_HZ,
    ISC_N_COMPONENTS,
    ISC_REGULARIZATION,
    ISCTemplate,
    fit_template,
    isc_effective_sample_ratio,
    ledoit_wolf_shrinkage,
    loo_transform,
    process,
    regularized_covariance,
    transform,
)
from .isc_regression import (
    isc_regression_freedman_lane,
    _partial_corr,
)
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# Legacy aliases (for backward compatibility with existing tests
# that call `ledoit_wolf_shrinkage`, `regularized_covariance`, etc.
# with the old direct-import style)
__all__ = [
    "ISC_BAND_HZ",
    "ISC_N_COMPONENTS",
    "ISC_REGULARIZATION",
    "ISCTemplate",
    "fit_template",
    "isc_effective_sample_ratio",
    "ledoit_wolf_shrinkage",
    "loo_transform",
    "process",
    "regularized_covariance",
    "transform",
    "isc_regression_freedman_lane",
]
