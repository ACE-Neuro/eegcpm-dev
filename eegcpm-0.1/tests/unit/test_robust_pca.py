

def test_ialm_against_published_reference_fixture():
    """R-014: the port must recover the published-regime decomposition,
    not just self-consistency. Fixture: tests/fixtures/robust_pca/
    lin2011_exact_recovery.npz (see tests/fixtures/INDEX.md)."""
    import numpy as np
    from eegcpm.modules.preprocessing.steps.robust_pca import ialm_decompose

    fx = np.load("tests/fixtures/robust_pca/lin2011_exact_recovery.npz")
    M, L0, S0, lam = fx["M"], fx["L0"], fx["S0"], float(fx["lam"])

    L, S = ialm_decompose(M, lam=lam, max_iter=500)

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
