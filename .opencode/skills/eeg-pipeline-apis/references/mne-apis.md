# MNE APIs — verified reference

Verified against **mne 1.13.2** in `eegcpm-0.1/.venv` on 2026-10-13.
Regenerate with `scripts/dump_signatures.py`. Entries tagged **[audit]** come from the
2026-10-13 deep audits (probed on mne 1.12.1) — re-verify on 1.13.2 before relying.

## Critical namespace change (mne 1.13.2)

`make_inverse_operator`, `apply_inverse`, `apply_inverse_epochs` are **not top-level
`mne.*` attributes in 1.13.2**. Import from `mne.minimum_norm`:

```python
from mne.minimum_norm import make_inverse_operator, apply_inverse, apply_inverse_epochs
```

## Signatures (verified 2026-10-13 unless tagged [audit])

```python
# -- source space / forward --
mne.setup_source_space(subject, spacing='oct6', surface='white', subjects_dir=None,
                       add_dist=True, n_jobs=None, *, verbose=None)
mne.make_forward_solution(info, trans, src, bem, meg=True, eeg=True, *,
                          mindist=0.0, ignore_ref=False, n_jobs=None, on_inside='raise', verbose=None)
mne.read_bem_solution(fname, *, verbose=None)
mne.write_source_spaces(fname, src, *, overwrite=False, verbose=None)
mne.read_source_spaces(fname, patch_stats=False, verbose=None)

# -- covariance / inverse --
mne.compute_covariance(inst=None, keep_sample_mean=True, tmin=None, tmax=None, projs=None, *,
                       epochs=None, on_few_samples='warn', method='empirical', method_params=None,
                       cv=3, scalings=None, n_jobs=None, return_estimators=False,
                       on_mismatch='raise', rank=None, verbose=None)
mne.minimum_norm.make_inverse_operator(info, forward, noise_cov, loose='auto', depth=0.8,
                                       fixed='auto', rank=None, use_cps=True, verbose=None)
mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=0.111..., method='dSPM',
                               pick_ori=None, prepared=False, label=None, method_params=None,
                               return_residual=False, use_cps=True, verbose=None)
mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2, method='dSPM',
                                      label=None, nave=1, pick_ori=None, return_generator=False,
                                      prepared=False, method_params=None, use_cps=True, verbose=None)

# -- labels / ROI extraction --
mne.Label(vertices=(), pos=None, values=None, hemi=None, comment='', name=None,
          filename=None, subject=None, color=None, *, verbose=None)
mne.BiHemiLabel(lh, rh, name=None, color=None)
mne.vertex_to_mni(vertices, hemis, subject, subjects_dir=None, verbose=None)
mne.extract_label_time_course(stcs, labels, src, mode='auto', allow_empty=False,
                              return_generator=False, *, mri_resolution=True, verbose=None)

# -- resolution / crosstalk QC --
mne.minimum_norm.make_inverse_resolution_matrix(forward, inverse_operator, method='dSPM',
                                                lambda2=0.111..., verbose=None)
mne.minimum_norm.get_cross_talk(resmat, src, idx, mode=None, *, n_comp=1, norm=False,
                                return_pca_vars=False, vector=False, verbose=None)
mne.minimum_norm.get_point_spread(resmat, src, idx, mode=None, *, n_comp=1, norm=False,
                                  return_pca_vars=False, vector=False, verbose=None)

# -- rank / reference / annotations / epochs --
mne.compute_rank(inst, rank=None, scalings=None, info=None, tol='auto', *, proj=True,
                 tol_kind='absolute', on_rank_mismatch='ignore', on_few_samples=None, verbose=None)
mne.set_eeg_reference(inst, ref_channels='average', copy=True, projection=False, ch_type='auto',
                      forward=None, *, joint=False, verbose=None)
mne.events_from_annotations(raw, event_id='auto',
                            regexp='^(?![Bb][Aa][Dd]|[Ee][Dd][Gg][Ee]).*$',
                            use_rounding=True, chunk_duration=None, tol=1e-08, verbose=None)
mne.Annotations(onset, duration, description, orig_time=None, ch_names=None, *, extras=None)
mne.read_epochs(fname, proj=True, preload=True, ...)  # returns EpochsFIF
```

## Approved usage (from the 2026-10-13 P2 deep audit)

- **Forward** [audit]: fsaverage, `setup_source_space(spacing=cfg.forward.spacing, add_dist=False)`,
  3-layer BEM only (`fsaverage-5120-5120-5120-bem-sol.fif`) — a 1-layer/single-shell BEM
  *raises* for EEG; do not implement a single-shell fallback. `make_forward_solution(..., eeg=True,
  meg=False, mindist=5.0)`. Cache src + forward keyed by (ch_names, positions, spacing, BEM, mne version).
- **Covariance** [audit]: `compute_covariance(epochs, tmin=None, tmax=0.0, method='shrunk', rank=None,
  keep_sample_mean=True)`; guard: baseline present, data finite, warn if `n_epochs * n_prestim_samples < 10*n_eeg`.
  Baseline-corrected epochs are *correct* for epoch covariance (H4 was refuted). Log `compute_rank`.
- **Inverse** [audit]: `make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8, rank=None)`.
  Evoked: `lambda2 = 1/9`. Single trials: `lambda2 = 1/snr_trial**2` (default snr_trial=1.0).
- **Polarity** [audit, verified on 1.13.2]: with `loose>0` the inverse keeps 3 orientations
  (`inv['source_ori']` FREE) and `pick_ori=None` returns the **unsigned norm** — destroys ERP
  polarity. Always `pick_ori='normal'` and extract with `mode='mean_flip'`.
- **eLORETA** [verified 1.13.2]: `mne.defaults.eloreta_options` has `force_equal=False` and
  `_eloreta.py` treats `None` as False. For loose orientation pass
  `method_params={'force_equal': True}` explicitly.
- **Trials** [audit]: use `apply_inverse_epochs(..., return_generator=True)` — per-trial
  `apply_inverse` re-prepares the operator each call (~10× slower).
- **ROI labels** [audit]: get vertices with `vertex_to_mni(src[h]['vertno'], h, 'fsaverage', subjects_dir)`;
  select within `roi_radius` mm (default 10); midline ROIs (|x| ≤ 5 mm) → `BiHemiLabel`;
  raise if a label has < `min_vertices` (default 3); cerebellar ROIs excluded (0 vertices within 10 mm).
- **Crosstalk QC** [audit]: `make_inverse_resolution_matrix(fwd, inv, method, lambda2)` +
  `get_cross_talk`/`get_point_spread` aggregated over label vertices.
- **Annotations** [audit]: append-only. Offset recipe that works for all `meas_date`/`first_samp`
  combinations:
  ```python
  off = raw.first_time if raw.annotations.orig_time is not None else 0.0
  new = mne.Annotations(rel_onsets + off, durs, descs, orig_time=raw.annotations.orig_time)
  raw.set_annotations(raw.annotations + new)
  ```
- **Epochs identity** [audit]: `read_epochs` returns `EpochsFIF`; `isinstance(x, mne.Epochs)` is
  False. Use `mne.BaseEpochs`.
- **Reference** [audit]: `set_eeg_reference(projection=True)` resets `custom_ref_applied`
  (needed for inverse modelling). Saved epochs carry the projector (`proj=False` at creation);
  the source module re-adds it.
- **events_from_annotations** [verified]: default `regexp` excludes `BAD*`/`EDGE` descriptions —
  ASR/block-rejection `BAD_*` annotations will not leak into events, but verify your descriptions
  start with `BAD`.
