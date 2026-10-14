# eegprep / pyzaplineplus / eeglabio / autoreject — verified reference

Versions in `eegcpm-0.1/.venv` (2026-10-13): **eegprep 0.4.0**, **pyzaplineplus 0.1.0**,
**eeglabio 0.1.4**, **autoreject 0.5.1**. Regenerate signatures with `scripts/dump_signatures.py`.

## eegprep 0.4.0

```python
eegprep.clean_artifacts(EEG, ChannelCriterion=0.8, LineNoiseCriterion=4.0, BurstCriterion=5.0,
                        WindowCriterion=0.25, Highpass=(0.25, 0.75), ChannelCriterionMaxBadTime=0.5,
                        BurstCriterionRefMaxBadChns=0.075, BurstCriterionRefTolerances=(-inf, 5.5),
                        BurstRejection=False, WindowCriterionTolerances=(-inf, 7),
                        FlatlineCriterion=5.0, NumSamples=50, SubsetSize=0.25, ...)
    -> (EEG, HP, BUR, removed_channels)          # 4-tuple, verified from source

eegprep.clean_windows(EEG, max_bad_channels=0.2, zthresholds=(-3.5, 5), window_len=1.0,
                      window_overlap=0.66, max_dropout_fraction=0.1, min_clean_fraction=0.25, ...)
    -> (EEG, sample_mask)                        # 2-tuple, verified

eegprep.clean_asr(EEG, cutoff=5.0, window_len=None, step_size=None, max_dims=0.66,
                  ref_maxbadchannels=0.075, ref_tolerances=(-3.5, 5.5), ref_wndlen=1.0, ...)
    -> EEG                                       # dict

eegprep.eeg_eegrej(EEG, regions)
eegprep.pop_loadset(file_path=None, *args, loadmode='all', memmap=None, **kwargs)
eegprep.pop_saveset(EEG, file_name=None, *args, **kwargs)
```

Verified semantics:

- `clean_artifacts` returns **(EEG, HP, BUR, removed_channels)**: `HP` = high-passed snapshot,
  `BUR` = ASR-repaired dataset (full length), `removed_channels` = ndarray. The old assumption
  that `result[0]` is everything, or that a *cleaned EEG struct* is all you get, is wrong.
- `clean_windows` returns **(EEG, sample_mask)**. Use it to *mark* rejected windows without
  cutting samples (F3 recipe: convert `~sample_mask` runs into `BAD_asr_window` annotations).
- `clean_asr` `ref_maxbadchannels` semantics [verified from source]:
  - `float` → automatic calibration window selection via `clean_windows`;
  - `None` / `'off'` → use the entire data for calibration;
  - **`np.ndarray` → the array itself is the calibration reference data** — pass the first 60 s
    of data here to implement a fixed-first-60s calibration (audit H10/F8).
- eegprep works on EEGLAB-style dicts (`EEG['data']`, `EEG['srate']`, `EEG['event']`, `EEG['etc']`).
  Probe units/dims before writing back into MNE (`*1e-6` if data are µV).
- [audit] The step wrapper must merge `etc`/event output back and must not silently discard it
  (current ASR wrapper loses `clean_sample_mask`, `skipped`, window statistics).

## pyzaplineplus 0.1.0

```python
from pyzaplineplus import zapline_plus     # NOT pyzaplineplus.run
# class equivalent: pyzaplineplus.PyZaplinePlus(data, sampling_rate, **kwargs).run()
```

- `zapline_plus(data, sampling_rate, **kwargs)` — `data` is **(n_times, n_channels)** (transpose).
- `.run()` returns **4-tuple `(clean_data, zapline_config, analytics_results, plot_handles)`**
  (verified `core.py:2533`).
- Accepted kwargs [verified from source]: `adaptiveNremove`, `adaptiveSigma`, `chunkLength`,
  `coarseFreqDetectLowerPowerDiff`, `coarseFreqDetectPowerDiff`, `detailedFreqBoundsLower`,
  `detailedFreqBoundsUpper`, `detectionWinsize`, `figBase`, `figPos`, `fixedNremove`,
  `freqDetectMultFine`, `maxProportionAboveUpper`, `maxProportionBelowLower`, `maxfreq`,
  `maxsigma`, `minChunkLength`, `minfreq`, `minsigma`, `nkeep`, `noiseCompDetectSigma`,
  `noisefreqs`, `overwritePlot`, `plotResults`, `prominenceQuantile`, `saveSpectra`,
  `searchIndividualNoise`, `segmentLength`, `winSizeCompleteSpectrum`.
- [audit] **`n_removed_components` does not exist** — the old code read it from `result[1]`
  (the config) and always reported 1. Removed-component counts live in `analytics_results`
  keyed per detected noise frequency (pattern `noise_freq_<f>`, count key `n_remove_final`) —
  verify the exact key at runtime before trusting.
- [audit] Pass `noisefreqs=[fline]` when a fixed line frequency is required; empty list enables
  auto-detection (17–99 Hz search).

## eeglabio 0.1.4

```python
from eeglabio.raw import export_set        # note: eeglabio.raw is a MODULE
export_set(fname, data, sfreq, ch_names, ch_locs=None, annotations=None,
           ref_channels='common', ch_types=None, precision='single', *, fmt='v5')

from eeglabio.epochs import export_set
export_set(fname, data, sfreq, events, tmin, tmax, ch_names, event_id=None,
           ch_locs=None, annotations=None, ref_channels='common', precision='single',
           *, epoch_indices=None, fmt='v5')
```

## autoreject 0.5.1

```python
from autoreject import Ransac
Ransac(n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4,
       n_jobs=1, random_state=435656, picks=None, verbose=True)
```

- Used by `modules/preprocessing/bad_channels.py:240` inside a `try/except ImportError`
  [audit F17]: when the dependency is missing, detection silently yields 0 bad channels.
  Per the approved fail-loud design, missing deps must fail the dependency preflight instead.
