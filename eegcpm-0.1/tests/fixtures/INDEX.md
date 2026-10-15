# Test fixtures index

| Fixture | Source / provenance | Ground truth asserted |
|---|---|---|
| `robust_pca/lin2011_exact_recovery.npz` | Constructed 2026-07-30 in the exact-recovery regime of Lin, Chen & Ma (2011), "The Augmented Lagrange Multiplier Method for Exact Recovery of Corrupted Low-Rank Matrices": L0 = A(200×5)·B(5×120) rank 5; S0 with 5% ±uniform(0.5,1.0) entries; M = L0 + S0; lam = 1/sqrt(max(m,n)) (the paper's canonical choice). Fixed seed 2011, immutable. Restored 2026-10-16 (B6 task 2) from the dev tree; `sha256 = 2ec821810142bcfae3dba35c0f85fcbac3cb8fae1c016caa27dff12ed8210e50`. Not git-ignored (negation `!tests/fixtures/**/*.npz` in `.gitignore`). | IALM recovers L with ‖L−L0‖_F/‖L0‖_F < 0.05 and S support with F1 > 0.9; M = L + S residual < 1e-6 relative |
