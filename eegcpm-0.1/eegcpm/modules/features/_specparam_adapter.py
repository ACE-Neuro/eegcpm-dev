"""
specparam / fooof adapter (per spec §3.j + ENG-007 + S20c).

The ACTUAL specparam 2.0.0rc7 API is:
    from specparam import SpectralModel, SpectralGroupModel

NOT SpectrumModel (which code.md imports and which does not exist).
This adapter isolates the third-party API from our module code so
that a fooof↔specparam swap is a single-file change.

SPECPARAM_BACKEND is read from the env var. Allowed values:
    - "specparam"  (default; primary backend, specparam==2.0.0rc7)
    - "fooof"      (fallback; fooof==1.1.0)

"auto" is REMOVED (METH-033): it would create an environment-dependent
divergence that the parity gate would surface only as an unexplained
mismatch.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Optional, Tuple

import numpy as np


# Allowed backends
SPECPARAM_BACKENDS = ("specparam", "fooof")

# Read the backend from the env var (or default to "specparam")
SPECPARAM_BACKEND: str = os.environ.get("SPECPARAM_BACKEND", "specparam")
if SPECPARAM_BACKEND not in SPECPARAM_BACKENDS:
    raise RuntimeError(
        f"SPECPARAM_BACKEND={SPECPARAM_BACKEND!r} not in "
        f"{SPECPARAM_BACKENDS}; set SPECPARAM_BACKEND to one of these."
    )

# Import the appropriate SpectralModel. Fail loudly if unavailable.
if SPECPARAM_BACKEND == "specparam":
    try:
        from specparam import SpectralModel, SpectralGroupModel
    except ImportError as e:
        raise RuntimeError(
            f"specparam is the requested backend (SPECPARAM_BACKEND="
            f"{SPECPARAM_BACKEND!r}) but failed to import: {e}. "
            f"Set SPECPARAM_BACKEND=fooof to use the fooof fallback."
        )
elif SPECPARAM_BACKEND == "fooof":
    try:
        from fooof import FOOOF as SpectralModel
        from fooof import FOOOFGroup as SpectralGroupModel
    except ImportError as e:
        raise RuntimeError(
            f"fooof is the requested backend (SPECPARAM_BACKEND="
            f"{SPECPARAM_BACKEND!r}) but failed to import: {e}. "
            f"Set SPECPARAM_BACKEND=specparam to use the specparam fallback."
        )


# BINDING RMSE definition (S20c).
# RMSE := sqrt(mean((log10(power) - log10(modeled))^2))
# Units: log10 power. This is the definition the specparam adapter
# is REQUIRED to match. The library's MAE attribute is NOT used.
def rmse_in_house(freqs: np.ndarray, power: np.ndarray,
                  modeled: np.ndarray) -> float:
    """Binding in-house RMSE definition per S20c.

    RMSE := sqrt(mean((log10(power) - log10(modeled))^2))

    Units: log10 power.

    NOTE: specparam 2.0.0rc7's `data.power_spectrum` is already in
    log10 scale (the library takes log10 internally before fitting).
    The binding RMSE in the spec is in log10-power units; the
    function therefore computes the RMSE on the values the library
    returns, which are already log10. Callers that pass raw linear
    power MUST convert to log10 before calling.
    """
    p = np.asarray(power, dtype=float)
    m = np.asarray(modeled, dtype=float)
    # The library's power_spectrum is already in log10 units; we
    # take the diff directly. If the caller passes linear power,
    # we convert to log10 first.
    if p.min() > 0:
        p = np.log10(p)
        m = np.log10(m)
    return float(np.sqrt(np.mean((p - m) ** 2)))


def get_modeled_and_power(model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (modeled_spectrum, power_spectrum) from a fitted
    SpectralModel. Both must be available; the specparam 2.0.0rc7
    API exposes them via:
      - model.data.power_spectrum  (input, in log10 scale)
      - model.results.model.modeled_spectrum  (modeled, in log10 scale)
    """
    try:
        power = model.data.power_spectrum
        modeled = model.results.model.modeled_spectrum
    except AttributeError as e:
        raise RuntimeError(
            f"specparam 2.0.0rc7 API does not expose the expected "
            f"attributes (data.power_spectrum / results.model."
            f"modeled_spectrum); the binding RMSE cannot be "
            f"computed. STOP. Original error: {e}"
        )
    return modeled, power


def specparam_rmse(model: Any) -> float:
    """Get the binding RMSE for a fitted SpectralModel.

    S20c: the in-house definition is BINDING. We do NOT silently
    fall back to MAE if the library attribute is absent. If the
    library exposes a `metrics['rmse']` attribute, we assert it
    equals the in-house value. If the library does NOT expose an
    RMSE attribute, we fall back to in-house ONLY (we never use MAE).
    """
    modeled, power = get_modeled_and_power(model)
    freqs = model.data.freqs
    in_house = rmse_in_house(freqs, power, modeled)
    # If the library exposes a metrics dict, try to read RMSE for
    # cross-validation. The specparam 2.0.0rc7 API does NOT expose
    # RMSE directly; it exposes MAE via metrics.results['error_mae'].
    # We use the in-house value and ignore the library's MAE.
    if hasattr(model, "results") and hasattr(model.results, "metrics"):
        lib_rmse = None
        for key in ("rmse", "error_rmse", "gof_rmse"):
            try:
                v = model.results.get_metrics(key)
                if v is not None:
                    lib_rmse = float(v)
                    break
            except Exception:
                continue
        if lib_rmse is not None:
            if abs(lib_rmse - in_house) > 1e-6:
                raise RuntimeError(
                    f"specparam library RMSE ({lib_rmse}) does not "
                    f"match in-house RMSE ({in_house}); library "
                    f"attribute may not be RMSE. STOP — do not "
                    f"silently fall back to MAE."
                )
            return lib_rmse
    # Library has no RMSE attribute: use in-house (NEVER MAE)
    return in_house


def fit_spectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    freq_range: Tuple[float, float] = (2.0, 40.0),
    aperiodic_mode: Literal["fixed", "knee"] = "fixed",
    peak_width_limits: Tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.0,
    verbose: bool = False,
) -> Any:
    """Fit a single spectrum using the configured backend.

    Returns a fitted SpectralModel with both power_spectrum and
    modeled_spectrum available for the binding RMSE computation.
    """
    if SPECPARAM_BACKEND == "specparam":
        model = SpectralModel(
            aperiodic_mode=aperiodic_mode,
            peak_width_limits=list(peak_width_limits),
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            verbose=verbose,
        )
    else:
        # fooof: FOOOF does not have a min_peak_height argument
        kwargs = {
            "peak_width_limits": list(peak_width_limits),
            "max_n_peaks": max_n_peaks,
            "verbose": verbose,
        }
        model = SpectralModel(**kwargs)
    model.fit(freqs, power, freq_range=list(freq_range))
    return model


def get_aperiodic_params(model: Any) -> np.ndarray:
    """Get the aperiodic parameters [offset, exponent] (or
    [offset, knee, exponent] for knee mode) from a fitted model."""
    results = model.results.get_results()
    return np.asarray(results.aperiodic_fit, dtype=float)


def get_n_peaks(model: Any) -> int:
    """Number of detected peaks."""
    return int(model.results.n_peaks)


def get_peak_params(model: Any) -> np.ndarray:
    """Get peak parameters (CF, power, BW) for each detected peak;
    returns shape (n_peaks, 3)."""
    results = model.results.get_results()
    if len(results.peak_fit) == 0:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(results.peak_fit, dtype=float)


def get_r_squared(model: Any) -> float:
    """Get R² from the specparam results."""
    return float(model.results.get_results().metrics["gof_rsquared"])


def get_mae(model: Any) -> float:
    """Get the library's MAE (we never use this as the binding QC
    metric; included for completeness / cross-validation)."""
    return float(model.results.get_results().metrics["error_mae"])
