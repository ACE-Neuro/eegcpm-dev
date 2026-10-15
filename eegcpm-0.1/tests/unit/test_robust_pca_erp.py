"""P1-F2 (Checkpoint A, decision (a)): RPCA destroys the ERP — why it was removed.

The synthetic fixture injects a fixed-topography N1/P2 waveform at 25 onsets
into rank-10 mixed AR(1) noise (109 channels, 250 Hz) and measures the
**retention** of the epoch-averaged ERP after ``RobustPCAStep()``:

    e_w  = epoch-averaged ERP after RPCA (channels x 0-600 ms)
    e_wo = epoch-averaged ERP without RPCA
    retention = <e_w, e_wo> / <e_wo, e_wo>          (plan section B6, task 3)

Decision (a) (plan section 12 item 1, resolved 2026-10-15) removed
``RobustPCAStep`` from ``build_hbn_langer_chain()`` because the pinned IALM
solve (lam = 0.003162) puts most of the ERP into the *sparse* component S,
which the step discards. The retention assertions below are therefore
``xfail(strict=True, raises=AssertionError)``: they document the measured
destruction (they must fail with an ``AssertionError`` — B6 rev2 NEW-8, so an
error in the fixture or the step can no longer masquerade as the expected
failure), and they turn into hard failures if a future change ever makes RPCA
ERP-safe — at which point the removal decision must be re-opened explicitly
instead of silently.

The instrumentation/metadata assertions and the "retention is below the
rejected acceptance criterion" facts live in **normal** tests (NEW-8): the
xfail tests only carry the failing acceptance assertion.

Two fixtures are measured (each computed once per module):

- the plan's fixture (60 s / 25 onsets): retention ~0.67 — the ERP shape
  survives but a third of its amplitude is removed, a systematic ERP bias;
- the audit's destruction regime (30 s / 15 onsets): retention ~0.19 with
  topography corr ~0.83, i.e. most of the ERP is gone (the audit probe
  measured 2-28% retention; nnz_S_frac is ~1.0 in both regimes — the
  "dense S" signature).

The chain assertion is a normal (non-xfail) test: the removal itself must
hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from eegcpm.modules.preprocessing.hbn_langer_chain import (
    build_hbn_langer_chain,
)
from eegcpm.modules.preprocessing.pipeline_builder import (
    PreprocessingPipeline,
)
from eegcpm.modules.preprocessing.steps.robust_pca import RobustPCAStep


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _build_fixture(
    seed: int = 2026,
    duration: float = 60.0,
    n_onsets: int = 25,
    onset_spacing: float = 2.0,
    first_onset: float = 2.0,
    sfreq: float = 250.0,
    n_channels: int = 109,
):
    """Rank-10 mixed AR(1) noise + fixed-topography N1/P2 at fixed onsets.

    Returns ``(data, sfreq, onsets)`` with ``data`` in Volts and ``onsets``
    as sample indices.
    """
    rng = np.random.default_rng(seed)
    n_times = int(sfreq * duration)
    n_latent = 10
    ar = np.zeros((n_latent, n_times))
    ar[:, 0] = rng.standard_normal(n_latent)
    for t in range(1, n_times):
        ar[:, t] = 0.95 * ar[:, t - 1] + rng.standard_normal(n_latent)
    mixing = rng.standard_normal((n_channels, n_latent))
    data = (mixing @ ar) * 1e-6  # rank-10 background, Volts

    # Fixed-topography N1 (negative, 100 ms) / P2 (positive, 200 ms) waveform.
    t_erp = np.arange(int(-0.2 * sfreq), int(0.6 * sfreq)) / sfreq
    waveform = -1.0 * _gauss(t_erp, 0.100, 0.015) + 0.6 * _gauss(t_erp, 0.200, 0.030)
    topography = rng.standard_normal(n_channels)
    topography /= np.linalg.norm(topography)
    onsets = np.arange(
        int(sfreq * first_onset), n_times - int(sfreq), int(sfreq * onset_spacing)
    )[:n_onsets]
    for onset in onsets:
        data[:, onset:onset + len(waveform)] += np.outer(topography, waveform) * 1e-6
    return data, sfreq, onsets


def _epoch_average(data: np.ndarray, sfreq: float, onsets: np.ndarray) -> np.ndarray:
    """Baseline-corrected (-200..0 ms) epoch average over 0-600 ms."""
    pre, post = int(0.2 * sfreq), int(0.6 * sfreq)
    epochs = np.stack([data[:, onset - pre:onset + post] for onset in onsets])
    epochs -= epochs[:, :, :pre].mean(axis=2, keepdims=True)
    return epochs.mean(axis=0)


def _retention_and_metadata(data, sfreq, onsets):
    """Run the default ``RobustPCAStep`` and measure the ERP retention."""
    import mne

    e_wo = _epoch_average(data, sfreq, onsets)
    info = mne.create_info(
        [f"E{i}" for i in range(1, data.shape[0] + 1)], sfreq, "eeg"
    )
    raw = mne.io.RawArray(data.copy(), info)
    _, step_meta = RobustPCAStep().process(raw, {})
    e_w = _epoch_average(raw.get_data(), sfreq, onsets)
    retention = float(np.sum(e_w * e_wo) / np.sum(e_wo * e_wo))
    topography_corr = float(np.corrcoef(e_w.ravel(), e_wo.ravel())[0, 1])
    return retention, topography_corr, step_meta


@pytest.fixture(scope="module")
def plan_fixture_results():
    """Plan fixture (109 ch x 60 s, 25 onsets): (retention, corr, metadata)."""
    data, sfreq, onsets = _build_fixture()
    return _retention_and_metadata(data, sfreq, onsets)


@pytest.fixture(scope="module")
def audit_regime_results():
    """Audit regime (30 s, 15 onsets): (retention, corr, metadata)."""
    data, sfreq, onsets = _build_fixture(
        duration=30.0, n_onsets=15, onset_spacing=1.5, first_onset=1.0
    )
    return _retention_and_metadata(data, sfreq, onsets)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "P1-F2 / Checkpoint A (a): the pinned IALM solve attenuates the "
        "epoch-averaged ERP (measured retention ~0.67 < 0.95); the "
        "assertion documents the failure that caused the removal"
    ),
)
def test_rpca_erp_retention_plan_fixture(plan_fixture_results):
    """Plan fixture: the rejected acceptance criterion ``retention >= 0.95``.

    Must fail with an ``AssertionError`` (NEW-8): any other failure (fixture
    bug, crash) is an error, not the documented destruction.
    """
    retention, topography_corr, _ = plan_fixture_results
    assert retention >= 0.95, (
        f"retention {retention:.3f} < 0.95 (topography corr "
        f"{topography_corr:.3f}); RPCA does not preserve the ERP"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "P1-F2 / Checkpoint A (a): in the audit's destruction regime "
        "(30 s, 15 onsets) most of the ERP is removed (measured retention "
        "~0.19, topography corr ~0.83; audit probe: 2-28%)"
    ),
)
def test_rpca_erp_retention_audit_regime(audit_regime_results):
    """Audit-regime fixture: the rejected acceptance criterion ``>= 0.95``."""
    retention, topography_corr, _ = audit_regime_results
    assert retention >= 0.95, (
        f"retention {retention:.3f} < 0.95 (topography corr "
        f"{topography_corr:.3f}); most of the ERP is gone"
    )


def test_rpca_erp_destruction_measured_plan_fixture(plan_fixture_results):
    """Normal test (NEW-8): the measured retention and instrumentation.

    The measured value (0.675) is below the rejected acceptance criterion
    ``retention >= 0.95``, and the sparse component is dense
    (``nnz_S_frac > 0.9``): the signature of the ERP being absorbed into S
    and discarded — the reason ``robust_pca`` is not in the chain.
    """
    retention, topography_corr, step_meta = plan_fixture_results

    assert set(step_meta) >= {
        "nnz_S_frac", "energy_S_frac", "n_iter", "converged",
    }
    assert retention < 0.95, retention
    assert retention == pytest.approx(0.675, abs=0.05), retention
    assert topography_corr > 0.95  # shape survives, amplitude does not
    assert step_meta["nnz_S_frac"] > 0.9, step_meta
    assert step_meta["energy_S_frac"] > 0.0
    assert step_meta["n_iter"] >= 1
    assert step_meta["converged"] is True  # B6-K8 re-check: 31 iters
    print(
        f"\n[RPCA ERP] 60 s fixture: retention={retention:+.3f} "
        f"topography_corr={topography_corr:+.3f} "
        f"nnz_S_frac={step_meta['nnz_S_frac']:.3f} "
        f"energy_S_frac={step_meta['energy_S_frac']:.3f} "
        f"n_iter={step_meta['n_iter']} converged={step_meta['converged']}"
    )


def test_rpca_erp_destruction_measured_audit_regime(audit_regime_results):
    """Normal test (NEW-8): the audit-regime destruction is measured."""
    retention, topography_corr, step_meta = audit_regime_results

    assert retention < 0.95, retention
    assert retention == pytest.approx(0.188, abs=0.05), retention
    assert topography_corr < 0.95
    assert step_meta["nnz_S_frac"] > 0.9, step_meta
    assert step_meta["converged"] is True  # B6-K8 re-check: 27 iters
    print(
        f"\n[RPCA ERP] 30 s audit regime: retention={retention:+.3f} "
        f"topography_corr={topography_corr:+.3f} "
        f"nnz_S_frac={step_meta['nnz_S_frac']:.3f} "
        f"n_iter={step_meta['n_iter']} converged={step_meta['converged']}"
    )


def test_robust_pca_removed_from_hbn_chain():
    """Checkpoint A decision (a): robust_pca is not part of the ERP chain.

    The step class and its registry entry stay for standalone use; only the
    canonical chain composition changed (9 steps), recorded as an explicit
    spec deviation in the resolved config.
    """
    chain = build_hbn_langer_chain()
    assert "robust_pca" not in [step.name for step in chain]
    assert len(chain) == 9

    pipeline = PreprocessingPipeline("hbn_langer")
    config = pipeline.get_config()
    assert config["spec_deviations"] == [
        "P1-F2: robust_pca removed (destroys ERP; synthetic retention 2-28%)"
    ]
    # The step class stays importable/usable standalone. (There is no
    # STEP_REGISTRY entry to keep: robust_pca was only ever instantiated by
    # the named chain, never registered for YAML step lists.)
    from eegcpm.modules.preprocessing.steps.robust_pca import RobustPCAStep

    step = RobustPCAStep()
    assert step.name == "robust_pca"
    assert step.get_config()["lam"] == pytest.approx(0.0031622776601683794)
