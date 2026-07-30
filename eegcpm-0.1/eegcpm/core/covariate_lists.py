"""
Covariate lists — single source of truth (per S05 in the spec).

PRIMARY_UNPENALIZED: always-on columns required in the primary config.
PRIMARY_UNPENALIZED_PROBED: PRIMARY_UNPENALIZED + availability-probed
columns (impedance_avg may be all-NaN at runtime; impedance_available
is the per-subject bool).
POST_CLEANING_QC_SENSITIVITY: post-cleaning burden metrics that are
FORBIDDEN in the primary config.

The validator (validate_primary_config) is in `eegcpm.core.config_validator`.
"""

from __future__ import annotations

from typing import List


# Demographics: required, always in primary
DEMOGRAPHICS: List[str] = [
    "age_spline_1", "age_spline_2", "age_spline_3", "age_spline_4",
    "sex_M",
    "site_CBIC", "site_CUNY", "site_NA",     # explicit NA category
    "release_number_2", "release_number_3",  # ... (one per release)
]

# Acquisition-side QC (PRIMARY adjustment set).
# impedance_avg is REQUIRED-as-field (the column MUST exist in the
# feature frame) but its VALUE may be NaN when the recording lacks
# impedance metadata. The implementation probes availability at
# feature-extraction time; if all values are NaN, the column is
# silently dropped from the ridge model (with a logged audit entry
# and a per-subject `impedance_available: bool` covariate so the
# analyst can see the missingness pattern).
ACQUISITION_SIDE_QC: List[str] = [
    "raw_bad_channel_count",        # channels flagged at detection
    "duration_attempted_seconds",   # pre-cleaning duration
]
ACQUISITION_SIDE_QC_PROBED: List[str] = ACQUISITION_SIDE_QC + [
    "impedance_avg",                # availability-probed; may be all-NaN
    "impedance_available",          # per-subject bool; False iff all-NaN
]

# Drowsiness / vigilance (UNCONDITIONAL)
DROWSINESS: List[str] = [
    "alpha_theta_trajectory",
    "alpha_dropout_count",
    "theta_intrusion_index",
]

# Channel availability
CHANNEL_AVAILABILITY: List[str] = ["frac_channels_passing"]

# ISC-side (when ISC is the outcome)
ISC_COVARIATES: List[str] = [
    "n_zeroed_channels",
    "zeroed_topography_diversity",
]

# Combined PRIMARY set (always-on; not availability-probed)
PRIMARY_UNPENALIZED: List[str] = (
    DEMOGRAPHICS + ACQUISITION_SIDE_QC + DROWSINESS + CHANNEL_AVAILABILITY
)

# Combined PRIMARY set + availability-probed columns (validator
# checks both, but impedance_avg may be all-NaN at runtime).
PRIMARY_UNPENALIZED_PROBED: List[str] = (
    DEMOGRAPHICS + ACQUISITION_SIDE_QC_PROBED + DROWSINESS
    + CHANNEL_AVAILABILITY
)

# Post-cleaning burden — SENSITIVITY ONLY (forbidden in primary config)
POST_CLEANING_QC_SENSITIVITY: List[str] = [
    "asr_burst_fraction",
    "ica_components_removed",
    "emg_proxy",
    "line_noise_residual_db",
]
