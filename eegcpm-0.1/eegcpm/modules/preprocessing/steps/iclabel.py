"""
ICLabel component classification step using mne-icalabel.

Uses MNE-Python's native ICLabel implementation.

Author: EEGCPM Development Team
Created: 2025-12
"""

from typing import Any, Dict, List, Tuple
import mne
import numpy as np

from .base import ProcessingStep


class ICLabelStep(ProcessingStep):
    """
    ICLabel automatic ICA component classification.

    Uses the ICLabel algorithm to classify ICA components as:
    - Brain
    - Muscle
    - Eye
    - Heart
    - Line Noise
    - Channel Noise
    - Other

    Components classified as artifacts above a threshold are removed.

    Parameters
    ----------
    threshold : float
        Probability threshold for artifact classification (default: 0.8)
    reject_labels : list of str
        Component types to reject. Options:
        - 'eye': Eye blinks/movements
        - 'muscle': Muscle artifacts
        - 'heart': Heartbeat/ECG
        - 'line_noise': Line noise (50/60 Hz)
        - 'channel_noise': Channel artifacts
        - 'other': Other artifacts
    remove_components : bool
        Whether to remove classified components (default: True)
        If False, only classifies without removing

    Examples
    --------
    Standard artifact removal (eye + muscle):
    >>> step = ICLabelStep(
    ...     threshold=0.8,
    ...     reject_labels=['eye', 'muscle']
    ... )

    Conservative (only eye blinks):
    >>> step = ICLabelStep(
    ...     threshold=0.9,
    ...     reject_labels=['eye']
    ... )

    Aggressive (all artifacts):
    >>> step = ICLabelStep(
    ...     threshold=0.7,
    ...     reject_labels=['eye', 'muscle', 'heart', 'line_noise', 'channel_noise']
    ... )

    Classification only (no removal):
    >>> step = ICLabelStep(
    ...     threshold=0.8,
    ...     reject_labels=['eye', 'muscle'],
    ...     remove_components=False
    ... )
    """

    name = "iclabel"
    version = "1.0"

    # ICLabel class indices
    LABEL_NAMES = [
        'brain',
        'muscle',
        'eye',
        'heart',
        'line_noise',
        'channel_noise',
        'other'
    ]

    LABEL_INDICES = {
        'brain': 0,
        'muscle': 1,
        'eye': 2,
        'heart': 3,
        'line_noise': 4,
        'channel_noise': 5,
        'other': 6,
    }

    def __init__(
        self,
        threshold: float = 0.8,
        reject_labels: List[str] = None,
        remove_components: bool = True,
        enabled: bool = True,
    ):
        """Initialize ICLabel step."""
        super().__init__(enabled=enabled)

        self.threshold = threshold
        self.reject_labels = reject_labels or ['eye', 'muscle']
        self.remove_components = remove_components

        # Validate labels
        for label in self.reject_labels:
            if label not in self.LABEL_INDICES:
                raise ValueError(
                    f"Unknown label '{label}'. "
                    f"Valid labels: {list(self.LABEL_INDICES.keys())}"
                )

    def process(
        self,
        raw: mne.io.BaseRaw,
        metadata: Dict[str, Any]
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply ICLabel classification.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Input raw data (should already have ICA applied from previous step)
        metadata : dict
            Metadata from previous steps (must contain 'ica' with 'ica_object')

        Returns
        -------
        raw : mne.io.BaseRaw
            Data with artifact components removed (if remove_components=True)
        step_metadata : dict
            Classification results
        """
        # Get ICA object from previous ICA step
        ica_step_name = self._find_ica_step(metadata)
        if not ica_step_name:
            return raw, {
                'skipped': True,
                'reason': 'no_ica_found',
                'error': 'ICLabel requires ICA step to run first'
            }

        ica = metadata[ica_step_name].get('ica_object')
        if ica is None:
            return raw, {
                'skipped': True,
                'reason': 'no_ica_object',
                'error': 'ICA object not found in metadata'
            }

        # Check for mne-icalabel
        try:
            from mne_icalabel import label_components
        except ImportError:
            return raw, {
                'skipped': True,
                'reason': 'mne_icalabel_not_installed',
                'error': 'Install mne-icalabel: pip install mne-icalabel'
            }

        # Run mne-icalabel
        try:
            # label_components returns IC_LABELS dict with 'labels' and 'y_pred_proba'
            ic_labels = label_components(raw, ica, method='iclabel')
        except Exception as e:
            return raw, {
                'skipped': True,
                'reason': 'iclabel_failed',
                'error': f'mne-icalabel failed: {str(e)}'
            }

        # Extract classifications
        # ic_labels['labels'] is list of label names (one per component)
        # ic_labels['y_pred_proba'] is 1D array of predicted class probabilities
        try:
            max_probabilities = ic_labels['y_pred_proba']  # Shape: (n_components,)
            predicted_labels = ic_labels['labels']  # List of strings
        except KeyError as e:
            return raw, {
                'skipped': True,
                'reason': 'iclabel_results_missing',
                'error': f'ICLabel results not found: {str(e)}'
            }

        # Normalize label names to match our conventions
        # mne-icalabel returns: 'brain', 'eye blink', 'muscle artifact', 'heart beat',
        #                       'line noise', 'channel noise', 'other'
        # We want: 'brain', 'eye', 'muscle', 'heart', 'line_noise', 'channel_noise', 'other'
        component_labels = []
        for label in predicted_labels:
            label_lower = label.lower()
            if 'eye' in label_lower:
                component_labels.append('eye')
            elif 'muscle' in label_lower:
                component_labels.append('muscle')
            elif 'heart' in label_lower:
                component_labels.append('heart')
            elif 'line' in label_lower and 'noise' in label_lower:
                component_labels.append('line_noise')
            elif 'channel' in label_lower and 'noise' in label_lower:
                component_labels.append('channel_noise')
            elif 'brain' in label_lower:
                component_labels.append('brain')
            else:
                component_labels.append('other')

        # Count components by type
        label_counts = {}
        for label_name in self.LABEL_NAMES:
            label_counts[label_name] = component_labels.count(label_name)

        # Identify components to reject
        components_to_reject = []
        for comp_idx in range(len(component_labels)):
            label = component_labels[comp_idx]
            prob = float(max_probabilities[comp_idx])

            if label in self.reject_labels and prob > self.threshold:
                components_to_reject.append(comp_idx)

        # Get removed labels for reporting
        removed_label_set = set()
        for idx in components_to_reject:
            removed_label_set.add(component_labels[idx])
        removed_labels = sorted(list(removed_label_set))

        # Remove components if requested
        if self.remove_components and len(components_to_reject) > 0:
            # Add to ICA exclusion list
            current_exclude = set(ica.exclude)
            current_exclude.update(components_to_reject)
            ica.exclude = sorted(list(current_exclude))

            # Apply ICA with updated exclusions.
            #
            # F19 (comment only): this can re-apply the exclusions an
            # earlier ICAStep already applied. ICA application is an
            # idempotent linear projection (applying the same unmixing/
            # mixing pair twice removes the component once), and the
            # audit refuted the double-application concern — the exclusion
            # set here is the union, so no component is removed twice
            # "harder". Kept as is; not part of any golden chain.
            raw_clean = ica.apply(raw.copy(), verbose=False)
        else:
            raw_clean = raw

        # Build metadata
        step_metadata = {
            'applied': True,
            'threshold': self.threshold,
            'reject_labels': self.reject_labels,
            'n_components_total': len(component_labels),
            'n_components_rejected': len(components_to_reject),
            'n_components_removed': len(components_to_reject),  # Alias for compatibility
            'rejected_components': components_to_reject,
            'removed_labels': removed_labels,  # For logging
            'label_counts': label_counts,
            'removed': self.remove_components,
        }

        # Add detailed component info with variance explained (if available in ICA)
        step_metadata['components'] = []

        # Compute variance explained ratio for each component (as percentage)
        # MNE's pca_explained_variance_ contains actual variance values, not ratios
        # We need to convert to ratios: variance / total_variance * 100
        variance_ratios = None
        if hasattr(ica, 'pca_explained_variance_') and ica.pca_explained_variance_ is not None:
            total_variance = ica.pca_explained_variance_.sum()
            if total_variance > 0:
                # Compute ratio for each component as percentage
                variance_ratios = (ica.pca_explained_variance_ / total_variance * 100)

        for i in range(len(component_labels)):
            label = component_labels[i]
            prob = float(max_probabilities[i])

            # Get variance explained ratio for this component (as percentage)
            variance_explained = None
            if variance_ratios is not None and i < len(variance_ratios):
                variance_explained = float(variance_ratios[i])

            # Determine rejection reason
            reject_reason = '-'
            if i in components_to_reject:
                reject_reason = f'{label} (p={prob:.2f} > {self.threshold})'

            step_metadata['components'].append({
                'index': i,
                'label': label,
                'probability': prob,
                'variance_explained': variance_explained,
                'rejected': i in components_to_reject,
                'reject_reason': reject_reason,
            })

        return raw_clean, step_metadata

    def _find_ica_step(self, metadata: Dict[str, Any]) -> str:
        """Find the ICA step in metadata."""
        # Look for 'ica' or 'ica_0', 'ica_1', etc.
        for key in metadata.keys():
            if key.startswith('ica'):
                return key
        return None

    def check_dependencies(self) -> None:
        """Fail at construction when mne-icalabel is missing."""
        try:
            import mne_icalabel  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"ICLabelStep requires mne-icalabel, which is not importable ({e}). "
                "Install with: pip install mne-icalabel"
            ) from e

    def validate_inputs(self, raw: mne.io.BaseRaw) -> bool:
        """Validate inputs."""
        # ICLabel validation happens in process() since we need metadata
        return True

    def skip_step(self, raw: mne.io.BaseRaw, metadata: Dict[str, Any]) -> bool:
        """Skip if ICA step not found."""
        if not self.enabled:
            return True

        # Check if ICA step exists
        ica_step_name = self._find_ica_step(metadata)
        return ica_step_name is None

    def get_config(self) -> Dict[str, Any]:
        """Get configuration."""
        config = super().get_config()
        config.update({
            'threshold': self.threshold,
            'reject_labels': self.reject_labels,
            'remove_components': self.remove_components,
        })
        return config
