# Draft: eegprep ASR pseudo-inverse tolerance bug report (NOT FILED — user-gated)

**Status:** draft only. Do **not** file this issue without the user's explicit go-ahead.
Prepared 2026-10-16 during the EEGCPM ASR scoped audit (fix R1).

**Target repo:** eegprep (Python port of EEGLAB `clean_rawdata`) — the version installed
in `eegcpm-0.1/.venv` is **0.4.0** (wheel inspected under `/tmp/opencode/epkg/`).

---

## Title (suggested)

`clean_asr` returns garbage (~1e13–1e16 µV) on real recordings: `asr_process` calls
`np.linalg.pinv` with numpy's default `rcond=1e-15` instead of MATLAB's `pinv` tolerance

## Summary

`eegprep.plugins.clean_rawdata.asr_process` computes the ASR reconstruction matrix with a
pseudo-inverse of `Q` (`asr_process.py` line ~214: `Z = finite_pinv(Q)`), which resolves to
`np.linalg.pinv(Q)` with numpy's default cutoff `rcond=1e-15` (a **relative** cutoff:
singular values `<= rcond * s_max` are zeroed).

The original MATLAB `clean_asr` (`asr_process.m`) uses MATLAB's `pinv`, whose default
tolerance is **absolute**: `tol = max(size(A)) * eps(norm(A))` (`eps(x)` = distance to the
next float, `norm(A)` = largest singular value). For a ~100-channel matrix that is
≈ `100 * 2.2e-16 = 2.2e-14` — about **20x larger** than numpy's default cutoff for a ~100-channel matrix. The decisive difference is the `max(size)` factor (~22x here): both cutoffs scale with the matrix norm, but MATLAB's is `max(size) * eps(norm(A))` while numpy's is `rcond * s_max` with `rcond=1e-15`.

Consequence: directions that MATLAB treats as null (and therefore reconstructs with the
"identity/keep" branch) are inverted by the Python port, and the ASR output blows up to
~1e13–1e16 µV instead of repaired data. On the HBN pilot5 dataset, **4 of 6 recordings that
reach ASR fail this way** (see "Observed" below); with the MATLAB-equivalent tolerance all
6 process cleanly.

## Where

- `eegprep/plugins/clean_rawdata/asr_process.py`, line ~214: `Z = finite_pinv(Q)`
- `eegprep/functions/miscfunc/misc.py`, `finite_pinv` → default solver `np.linalg.pinv`
- MATLAB reference: `clean_rawdata/asr_process.m` → `pinv(Q)` (MATLAB default tolerance);
  MATLAB `pinv.m` default: `tol = max(size(A)) * eps(norm(A))`, `s(s <= tol) = 0`.

## Minimal reproducer

Deterministic synthetic case (32 channels, 250 Hz, 120 s; cumulative-sum 1/f noise at
10 µV plus a 20 µV common 10 Hz sinusoid, 0.1–45 Hz high-passed; calibration covariance
condition number ~1.2e4):

```python
import numpy as np, mne, eegprep
import importlib
asr_mod = importlib.import_module("eegprep.plugins.clean_rawdata.asr_process")

rng = np.random.default_rng(11)
n = 120 * 250
data = np.cumsum(rng.standard_normal((32, n)), axis=1)
data -= data.mean(axis=1, keepdims=True)
data *= 10e-6 / data.std(axis=1, keepdims=True)
t = np.arange(n) / 250
data += 20e-6 * np.sin(2 * np.pi * 10.0 * t)[None, :]
raw = mne.io.RawArray(data, mne.create_info([f"E{i+1}" for i in range(32)], 250.0, "eeg"))
raw.filter(0.1, 45.0, verbose=False)

eeg = eegprep.eeg_mne2eeg(raw)
pre = np.asarray(eeg["data"], dtype=float)
out = eegprep.clean_asr(eeg, cutoff=20.0, max_dims=0.66,
                        ref_maxbadchannels=pre[:, :15000])
post = np.asarray(out["data"], dtype=float)
print("max ratio:", np.max(np.abs(post)) / np.max(np.abs(pre)))   # ~1e13 (garbage)

# MATLAB-equivalent tolerance:
def matlab_pinv(matrix):
    u, s, vt = np.linalg.svd(np.asarray(matrix), full_matrices=False)
    tol = max(np.asarray(matrix).shape) * np.spacing(float(s[0]))
    s_inv = np.where(s > tol, 1.0 / np.maximum(s, np.finfo(float).tiny), 0.0)
    return (vt.T * s_inv) @ u.T

asr_mod.finite_pinv = matlab_pinv
out2 = eegprep.clean_asr(eeg, cutoff=20.0, max_dims=0.66,
                         ref_maxbadchannels=pre[:, :15000])
post2 = np.asarray(out2["data"], dtype=float)
print("max ratio with MATLAB tol:", np.max(np.abs(post2)) / np.max(np.abs(pre)))  # ~1.0
```

## Observed vs expected

Real data (HBN pilot5, chain steps 1–6 then `clean_asr` at cutoff 20, `max_dims=0.66`,
fixed first-60 s calibration, ASR input high-passed with `clean_drifts(0.25, 0.75)` — the
EEGLAB-standard path):

| recording | task | default `np.linalg.pinv` (ratio max\|out\|/max\|in\|) | MATLAB tolerance |
|---|---|---|---|
| sub-NDARAC853CR6 | RestingState | 0.379 (ok) | 0.379 |
| sub-NDARAC853CR6 | DespicableMe | **4.26e12 (garbage)** | 0.283 |
| sub-NDARAC904DMU | RestingState | **3.25e13 (garbage)** | 0.489 |
| sub-NDARAM873GAC | RestingState | **6.87e13 (garbage)** | 0.570 |
| sub-NDARCD661GL0 | RestingState | 1.267 (ok) | 1.267 |
| sub-NDARCD661GL0 | DespicableMe | **4.23e13 (garbage)** | 0.743 |

Expected: `clean_asr` returns repaired data at the input's scale (ratio ~0.3–1.3), as
MATLAB does. Observed: 4 of 6 recordings return ~1e13 µV (no NaN, no warning — the values
are finite, so downstream code sees plausible-looking garbage).

## Suggested fix (upstream)

In `asr_process` (or inside `finite_pinv` for this call site), use MATLAB's pinv semantics:

```python
u, s, vt = np.linalg.svd(Q, full_matrices=False)
tol = max(Q.shape) * np.spacing(float(s[0]))          # MATLAB: max(size)*eps(norm(A))
s_inv = np.where(s > tol, 1.0 / s, 0.0)
Z = (vt.T * s_inv) @ u.T
```

(or `np.linalg.pinv(Q, rcond=tol / s[0])`). Note this is a behavioural change: some
recordings that currently "succeed" produce different repairs, and the ones that blow up
change completely. The EEGCPM workaround (scoped monkey-patch of the module-global
`finite_pinv` around `clean_asr`, with a source-based guard that fails loudly if the call
site changes) is in `eegcpm/modules/preprocessing/steps/asr.py`
(`_pinv_with_matlab_tolerance` / `_MatlabPinvTolerancePatch`), with the reproducer at
`tests/unit/test_asr_timeline.py::test_pinv_tolerance_is_the_blowup_fix`.

## Evidence / provenance

- Auditor reproduction and EEGCPM fix record: `planning/VERIFICATION_TODO.md` (§ASR R1–R6).
- Probes: `/tmp/opencode/pilot5_pinv_probe.py`, `/tmp/opencode/pilot5_step_probe.py`,
  `/tmp/opencode/asr_pinv_probe.py` (2026-10-16, eegprep 0.4.0, mne 1.13.2, numpy 2.5.4).
- eegprep wheel inspected at `/tmp/opencode/epkg/wheel/eegprep/...`.
