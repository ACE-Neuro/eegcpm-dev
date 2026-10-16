"""
Independent Component Analysis (ICA) step.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import mne
import numpy as np

from .base import ProcessingStep


class ICAStep(ProcessingStep):
    """
    Independent Component Analysis (ICA) step.

    Performs ICA decomposition to identify and remove artifact components
    (eye blinks, muscle artifacts, heartbeat, etc.).

    Supports two implementations:
    - eegprep-picard: EEGLAB's proven PICARD (default, recommended)
    - MNE methods: fastica, infomax (fallback options)

    Parameters
    ----------
    method : str
        ICA method:
        - 'eegprep-picard': EEGLAB PICARD via eegprep (default, most robust)
        - 'fastica': MNE FastICA (numerically stable fallback)
        - 'infomax': MNE Infomax
        Note: MNE's 'picard' has known stability issues and is not supported
    n_components : int, float, or 'rank-1'
        Number of ICA components:
        - int: exact number
        - float (0-1): variance explained
        - 'rank-1': data rank minus 1 (recommended)
    random_state : int or None
        Random seed for reproducibility
    max_iter : int or 'auto'
        Maximum iterations for ICA convergence
    fit_params : dict or None
        Additional parameters for ICA fitting
        For extended-infomax: {'extended': True}
    auto_detect_artifacts : bool
        Auto-detect artifact components (EOG, ECG)
    manual_exclude : list of int
        Manually specified components to exclude
    reject_by_annotation : bool
        Exclude annotated bad segments during ICA fitting
    l_freq_fit : float or None
        If set, fit ICA on highpassed copy for numerical stability
        (applies to original data). Recommended: 1.0 Hz

    Examples
    --------
    EEGLAB Picard ICA (recommended):
    >>> step = ICAStep(method='eegprep-picard', n_components='rank-1')

    MNE FastICA (fallback):
    >>> step = ICAStep(method='fastica', n_components='rank-1', l_freq_fit=1.0)

    With manual exclusions:
    >>> step = ICAStep(
    ...     method='eegprep-picard',
    ...     manual_exclude=[0, 1, 5]
    ... )
    """

    name = "ica"
    version = "1.0"

    def __init__(
        self,
        method: str = 'fastica',
        n_components: Union[int, float, str] = 'rank-1',
        random_state: int = 42,
        max_iter: Union[int, str] = 'auto',
        fit_params: Optional[Dict] = None,
        auto_detect_artifacts: bool = True,
        manual_exclude: Optional[List[int]] = None,
        reject_by_annotation: bool = True,
        l_freq_fit: Optional[float] = None,
        enabled: bool = True,
    ):
        """Initialize ICA step."""
        super().__init__(enabled=enabled)

        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.fit_params = fit_params or {}
        self.auto_detect_artifacts = auto_detect_artifacts
        self.manual_exclude = manual_exclude or []
        self.reject_by_annotation = reject_by_annotation
        self.l_freq_fit = l_freq_fit

        # Validate method
        # NOTE: picard and eegprep-picard disabled due to numpy 2.x incompatibility
        # See: https://github.com/pierreablin/picard/issues/XXX
        valid_methods = ['fastica', 'infomax']
        if self.method not in valid_methods:
            if self.method in ['picard', 'eegprep-picard']:
                raise ValueError(
                    "PICARD is currently incompatible with NumPy 2.x "
                    "(FloatingPointError in scipy.linalg.expm). "
                    "Use 'fastica' (recommended) or 'infomax' instead. "
                    "Track progress: https://github.com/pierreablin/picard/issues/XXX"
                )
            raise ValueError(
                f"Unknown ICA method: {method}. "
                f"Valid options: {valid_methods}"
            )

        # Handle extended-infomax notation
        if self.method in ['extended-infomax', 'extended_infomax']:
            self.method = 'infomax'
            self.fit_params['extended'] = True

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply ICA.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data
        metadata : dict
            Metadata from previous steps

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with ICA applied (artifact components removed)
        step_metadata : dict
            ICA metadata including component info
        """
        # Get EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) == 0:
            return raw, {'skipped': True, 'reason': 'no_eeg_channels'}

        # Check for NaN/Inf
        data = raw.get_data(picks=eeg_picks)
        if np.isnan(data).any() or np.isinf(data).any():
            return raw, {
                'skipped': True,
                'reason': 'data_contains_nan_inf',
                'error': 'Data contains NaN/Inf values'
            }

        # Route to appropriate implementation
        if self.method == 'eegprep-picard':
            return self._apply_eegprep_picard(raw, metadata)
        else:
            return self._apply_mne_ica(raw, metadata)

    def _apply_eegprep_picard(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """Apply ICA using eegprep's PICARD (EEGLAB implementation)."""
        try:
            import eegprep
        except ImportError:
            raise ImportError(
                "eegprep not installed. Install with: pip install eegprep"
            )

        # Convert MNE → EEGPrep
        eeg = eegprep.eeg_mne2eeg(raw)

        # Apply ICA using EEGLAB's PICARD
        try:
            eeg = eegprep.eeg_picard(eeg)
        except Exception as e:
            return raw, {
                'skipped': True,
                'reason': 'eegprep_picard_failed',
                'error': str(e),
                'implementation': 'eegprep'
            }

        # Convert back to MNE
        raw_with_ica = eegprep.eeg_eeg2mne(eeg)

        # Create MNE ICA object from eegprep results for compatibility
        # This allows ICLabel and other steps to work
        ica = mne.preprocessing.ICA(
            n_components=eeg.icaweights.shape[0] if hasattr(eeg, 'icaweights') else None,
            method='picard',  # For metadata only
            random_state=self.random_state
        )

        # Populate ICA object with eegprep data
        if hasattr(eeg, 'icaweights'):
            ica.unmixing_matrix_ = eeg.icaweights
            ica.mixing_matrix_ = np.linalg.pinv(eeg.icaweights)
            ica.n_components_ = eeg.icaweights.shape[0]

        # Handle component exclusions
        exclude_indices = list(self.manual_exclude)
        auto_detected = {'eog': [], 'ecg': []}

        # Auto-detect artifacts using MNE on the ICA object
        if self.auto_detect_artifacts and hasattr(eeg, 'icaweights'):
            # EOG
            eog_picks = mne.pick_types(raw.info, eog=True)
            if len(eog_picks) > 0:
                try:
                    eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
                    auto_detected['eog'] = eog_indices
                    exclude_indices.extend(eog_indices)
                except (ValueError, RuntimeError) as exc:
                    # mne raises ValueError/RuntimeError when the detection
                    # cannot run (no usable EOG channel / no significant
                    # events). Record it: a silently empty auto-detection
                    # would look like "no ocular components found" (P1-F11).
                    auto_detected['eog_error'] = str(exc)

            # ECG
            ecg_picks = mne.pick_types(raw.info, ecg=True)
            if len(ecg_picks) > 0:
                try:
                    ecg_indices, _ = ica.find_bads_ecg(raw, verbose=False)
                    auto_detected['ecg'] = ecg_indices
                    exclude_indices.extend(ecg_indices)
                except (ValueError, RuntimeError) as exc:
                    auto_detected['ecg_error'] = str(exc)

        # Apply exclusions
        ica.exclude = sorted(list(set(exclude_indices)))
        if len(ica.exclude) > 0:
            raw_clean = ica.apply(raw_with_ica.copy(), verbose=False)
        else:
            raw_clean = raw_with_ica

        step_metadata = {
            'applied': True,
            'method': 'eegprep-picard',
            'implementation': 'eegprep',
            'n_components_fitted': ica.n_components_ if hasattr(ica, 'n_components_') else None,
            'n_excluded': len(ica.exclude),
            'excluded_indices': ica.exclude,
            'auto_detected': auto_detected,
            'manual_excluded': list(self.manual_exclude),
            'ica_object': ica,
        }

        return raw_clean, step_metadata

    def _apply_mne_ica(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """Apply ICA using MNE's implementation (fastica or infomax)."""
        # Determine n_components
        n_components = self._resolve_n_components(raw)

        # Create ICA object
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=self.method,
            fit_params=self.fit_params if self.fit_params else None,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )

        # Fit ICA
        try:
            # If l_freq_fit is set, apply ADDITIONAL highpass for numerical stability
            # IMPORTANT: This filters the already-processed data (after Zapline, ASR, etc.)
            # NOT the original raw data, preserving all cleaning done by previous steps
            if self.l_freq_fit is not None:
                current_highpass = raw.info.get('highpass', 0)

                if self.l_freq_fit > current_highpass:
                    # Apply additional highpass to processed data
                    # This preserves Zapline cleaning, ASR, etc.
                    current_lowpass = raw.info.get('lowpass', None)

                    # Check if lowpass is at/above Nyquist (can't filter at Nyquist)
                    nyquist = raw.info['sfreq'] / 2
                    if current_lowpass is not None and current_lowpass >= nyquist:
                        current_lowpass = None  # No lowpass needed at Nyquist

                    raw_for_fit = raw.copy().filter(
                        l_freq=self.l_freq_fit,
                        h_freq=current_lowpass,  # Preserve any existing lowpass (if < Nyquist)
                        picks='eeg',
                        verbose=False
                    )
                else:
                    # Already filtered at >= l_freq_fit, use processed data as-is
                    raw_for_fit = raw
            else:
                # No l_freq_fit specified, use processed data as-is
                raw_for_fit = raw

            ica.fit(
                raw_for_fit,
                picks='eeg',
                reject_by_annotation=self.reject_by_annotation,
                verbose=False
            )
        except Exception as e:
            return raw, {
                'skipped': True,
                'reason': 'ica_fit_failed',
                'error': str(e),
                'implementation': 'mne'
            }

        # Collect components to exclude
        exclude_indices = list(self.manual_exclude)
        auto_detected = {'eog': [], 'ecg': []}

        # Auto-detect artifacts
        if self.auto_detect_artifacts:
            # EOG artifacts
            eog_picks = mne.pick_types(raw.info, eog=True)
            if len(eog_picks) > 0:
                try:
                    eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
                    auto_detected['eog'] = eog_indices
                    exclude_indices.extend(eog_indices)
                except (ValueError, RuntimeError) as exc:
                    # Record a failed auto-detection (P1-F11) instead of
                    # reporting an empty (and therefore "clean") result.
                    auto_detected['eog_error'] = str(exc)

            # ECG artifacts
            ecg_picks = mne.pick_types(raw.info, ecg=True)
            if len(ecg_picks) > 0:
                try:
                    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, verbose=False)
                    auto_detected['ecg'] = ecg_indices
                    exclude_indices.extend(ecg_indices)
                except (ValueError, RuntimeError) as exc:
                    auto_detected['ecg_error'] = str(exc)

        # Remove duplicates
        ica.exclude = sorted(list(set(exclude_indices)))

        # Apply ICA
        raw_clean = ica.apply(raw.copy(), verbose=False)

        # Build metadata
        step_metadata = {
            'applied': True,
            'method': self.method,
            'implementation': 'mne',
            'n_components_fitted': ica.n_components_,
            'n_excluded': len(ica.exclude),
            'excluded_indices': ica.exclude,
            'auto_detected': auto_detected,
            'manual_excluded': list(self.manual_exclude),
            'ica_object': ica,  # Store for ICLabel or saving
        }

        return raw_clean, step_metadata

    def _resolve_n_components(self, raw: mne.io.BaseRaw) -> Union[int, float]:
        """Resolve n_components from string or return as-is.

        ``'rank-1'``: one fewer component than the data rank (never more —
        requesting more than rank-1 makes the decomposition rank-deficient).
        The old ``max(5, rank - 1)`` could ask for 5 components from a
        rank-4 recording; the clamp is now ``min(max(5, r - 1), r - 1)``
        (P1-F20) and ``r - 1 < 1`` raises instead of silently fitting an
        impossible decomposition.
        """
        if self.n_components == 'rank-1':
            # Compute data rank and use rank - 1
            rank_dict = mne.compute_rank(raw, rank='info')
            eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
            data_rank = rank_dict.get('eeg', len(eeg_picks))
            n_components = min(max(5, data_rank - 1), data_rank - 1)
            if n_components < 1:
                raise ValueError(
                    "ICA n_components='rank-1' needs a data rank >= 2 "
                    f"(got eeg rank {data_rank}); cannot fit any component"
                )
            return n_components
        else:
            return self.n_components

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs."""
        # Check we have enough EEG channels
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        if len(eeg_picks) < 10:
            raise ValueError(
                f"ICA requires at least 10 good EEG channels, found {len(eeg_picks)}"
            )

        return True

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'method': self.method,
            'n_components': self.n_components,
            'random_state': self.random_state,
            'auto_detect_artifacts': self.auto_detect_artifacts,
        })
        return config
