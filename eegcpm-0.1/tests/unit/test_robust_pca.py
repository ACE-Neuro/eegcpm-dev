
from pathlib import Path


def test_ialm_against_published_reference_fixture():
    """R-014: the port must recover the published-regime decomposition,
    not just self-consistency. Fixture: tests/fixtures/robust_pca/
    lin2011_exact_recovery.npz (see tests/fixtures/INDEX.md)."""
    import numpy as np
    from eegcpm.modules.preprocessing.steps.robust_pca import ialm_decompose

    # Resolve from __file__ (B6 task 2): the test must pass regardless of
    # the working directory pytest is invoked from.
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures" / "robust_pca" / "lin2011_exact_recovery.npz"
    )
    assert fixture.is_file(), f"fixture missing: {fixture}"
    fx = np.load(fixture)
    M, L0, S0, lam = fx["M"], fx["L0"], fx["S0"], float(fx["lam"])

    L, S, solve_info = ialm_decompose(M, lam=lam, max_iter=500, return_info=True)

    # Instrumentation (P1-F2, B6 task 1): the info dict is reported.
    assert set(solve_info) == {"n_iter", "converged"}
    assert solve_info["n_iter"] >= 1

    # Exact-recovery assertions (ground truth, not self-consistency)
    rel_L = np.linalg.norm(L - L0, "fro") / np.linalg.norm(L0, "fro")
    assert rel_L < 0.05, f"L recovery error {rel_L:.4f} >= 0.05"

    # S support recovery F1
    true_supp = np.abs(S0) > 1e-8
    est_supp = np.abs(S) > 0.1 * np.abs(S0[true_supp]).min()
    tp = np.sum(true_supp & est_supp)
    prec = tp / max(est_supp.sum(), 1)
    rec = tp / true_supp.sum()
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    assert f1 > 0.9, f"S support F1 {f1:.3f} <= 0.9"

    # Reconstruction
    rel_M = np.linalg.norm(M - L - S, "fro") / np.linalg.norm(M, "fro")
    assert rel_M < 1e-3, f"reconstruction residual {rel_M:.2e}"


def test_ialm_return_info_default_unchanged():
    """`return_info=False` keeps the historical 2-tuple return."""
    import numpy as np
    from eegcpm.modules.preprocessing.steps.robust_pca import ialm_decompose

    rng = np.random.RandomState(0)
    M = rng.randn(20, 10)
    result = ialm_decompose(M, lam=0.1, mu=10.0)
    assert isinstance(result, tuple) and len(result) == 2
    L, S = result
    assert L.shape == M.shape and S.shape == M.shape


def test_step_not_applied_when_ialm_hits_max_iter():
    """B6-K8/NEW-10: a non-converged solve is neither applied nor written back.

    `converged=False` at `max_iter` was metadata-only; the standalone step
    still returned `applied: True`. The step now reports `applied: False` with
    `reason='ialm_not_converged'` (the executor then fails a required run)
    **and** leaves the input data untouched (NEW-10: a partial `L` must not
    reach a caller that ignores `applied`).
    """
    import mne
    import numpy as np
    from eegcpm.modules.preprocessing.steps.robust_pca import RobustPCAStep

    rng = np.random.RandomState(3)
    data = rng.randn(20, 400) * 1e-6

    raw = mne.io.RawArray(data.copy(), mne.create_info(20, 250.0, "eeg"))
    _, meta = RobustPCAStep(max_iter=1).process(raw, {})
    assert meta["applied"] is False
    assert meta["reason"] == "ialm_not_converged"
    assert meta["converged"] is False
    assert meta["n_iter"] == 1
    # NEW-10: no write-back for a non-converged solve.
    assert np.array_equal(raw.get_data(), data)

    # A converged solve stays applied and writes the low-rank component back
    # (the healthy path is unchanged).
    raw2 = mne.io.RawArray(data.copy(), mne.create_info(20, 250.0, "eeg"))
    _, meta2 = RobustPCAStep(max_iter=500).process(raw2, {})
    assert meta2["converged"] is True
    assert meta2["applied"] is True
    assert "reason" not in meta2
    assert not np.allclose(raw2.get_data(), data)
