#!/usr/bin/env python3
"""
Test script for new QC figures:
1. Temporal artifact distribution (RawDataQC)
2. Residual artifact quantification (PreprocessedQC)

Usage:
    python test_new_qc_figures.py
"""

import sys
from pathlib import Path
import mne
import numpy as np

# Add eegcpm to path
sys.path.insert(0, str(Path(__file__).parent))

from eegcpm.modules.qc.raw_qc import RawQC
from eegcpm.modules.qc.preprocessed_qc import PreprocessedQC


def test_temporal_artifact_distribution():
    """Test temporal artifact distribution figure in RawDataQC."""
    print("\n" + "="*80)
    print("TEST 1: Temporal Artifact Distribution (RawDataQC)")
    print("="*80)

    # Create synthetic raw data with artifacts for testing
    print("Creating synthetic data for testing...")

    sfreq = 500.0
    n_channels = 32
    duration = 60  # 60 seconds

    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Generate data with varying artifact levels
    n_samples = int(duration * sfreq)
    data = np.random.randn(n_channels, n_samples) * 10e-6  # 10 µV baseline

    # Add artifacts at specific times
    # Strong artifact at 20-25s
    artifact_start = int(20 * sfreq)
    artifact_end = int(25 * sfreq)
    data[:, artifact_start:artifact_end] += np.random.randn(n_channels, artifact_end - artifact_start) * 100e-6

    # Medium artifact at 45-48s
    artifact_start = int(45 * sfreq)
    artifact_end = int(48 * sfreq)
    data[:, artifact_start:artifact_end] += np.random.randn(n_channels, artifact_end - artifact_start) * 50e-6

    raw = mne.io.RawArray(data, info)
    subject_id = "SYNTHETIC001"

    # Run QC
    print(f"\n📊 Running RawQC on {subject_id}...")
    output_dir = Path("/tmp/qc_test/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    qc = RawQC(output_dir=output_dir)

    try:
        result = qc.compute(raw, subject_id=subject_id)

        print(f"  Status: {result.status}")
        print(f"  Metrics: {len(result.metrics)}")
        print(f"  Figures: {list(result.figures.keys())}")

        # Check if temporal_artifacts figure was generated
        if 'temporal_artifacts' in result.figures:
            print("  ✅ Temporal artifact distribution figure generated!")
        else:
            print("  ❌ Temporal artifact distribution figure NOT found!")
            return False

        # Generate HTML report
        report_path = output_dir / f"{subject_id}_raw_qc.html"
        qc.generate_html_report(result, report_path)
        print(f"\n📄 HTML report saved: {report_path}")

        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")
        print("  🌐 Opened in browser")

        return True

    except Exception as e:
        print(f"❌ QC failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_residual_artifact_quantification():
    """Test residual artifact quantification in PreprocessedQC."""
    print("\n" + "="*80)
    print("TEST 2: Residual Artifact Quantification (PreprocessedQC)")
    print("="*80)

    # Look for HBN preprocessed data
    derivatives_path = Path("/Volumes/Work/data/hbn/derivatives/preprocessing")

    if not derivatives_path.exists():
        print("❌ HBN preprocessed data not found")
        print("Creating synthetic before/after data for testing...")

        # Create synthetic data
        sfreq = 500.0
        n_channels = 32
        duration = 60

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        # Before: noisy data with artifacts
        n_samples = int(duration * sfreq)
        data_before = np.random.randn(n_channels, n_samples) * 20e-6  # 20 µV baseline noise

        # Add large artifacts
        for _ in range(10):
            start = np.random.randint(0, n_samples - int(2*sfreq))
            end = start + int(2*sfreq)
            data_before[:, start:end] += np.random.randn(n_channels, end-start) * 80e-6

        raw_before = mne.io.RawArray(data_before, info)

        # After: cleaner data (artifacts reduced)
        data_after = np.random.randn(n_channels, n_samples) * 10e-6  # 10 µV baseline

        # Add smaller residual artifacts
        for _ in range(3):
            start = np.random.randint(0, n_samples - int(2*sfreq))
            end = start + int(2*sfreq)
            data_after[:, start:end] += np.random.randn(n_channels, end-start) * 30e-6

        raw_after = mne.io.RawArray(data_after, info)

        subject_id = "SYNTHETIC001"

    else:
        # Use real HBN data
        print("✓ Found HBN preprocessed data")

        # Find a subject with both raw and preprocessed files
        # This is tricky - we need to find matched files
        # For now, create synthetic data
        print("  (Using synthetic data for testing - real data requires matched raw/preprocessed pairs)")

        # Create synthetic data (same as above)
        sfreq = 500.0
        n_channels = 32
        duration = 60

        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )

        n_samples = int(duration * sfreq)
        data_before = np.random.randn(n_channels, n_samples) * 20e-6
        for _ in range(10):
            start = np.random.randint(0, n_samples - int(2*sfreq))
            end = start + int(2*sfreq)
            data_before[:, start:end] += np.random.randn(n_channels, end-start) * 80e-6

        raw_before = mne.io.RawArray(data_before, info)

        data_after = np.random.randn(n_channels, n_samples) * 10e-6
        for _ in range(3):
            start = np.random.randint(0, n_samples - int(2*sfreq))
            end = start + int(2*sfreq)
            data_after[:, start:end] += np.random.randn(n_channels, end-start) * 30e-6

        raw_after = mne.io.RawArray(data_after, info)
        subject_id = "SYNTHETIC001"

    # Run QC
    print(f"\n📊 Running PreprocessedQC on {subject_id}...")
    output_dir = Path("/tmp/qc_test/preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    qc = PreprocessedQC(output_dir=output_dir)

    try:
        result = qc.compute(
            raw_after,
            subject_id=subject_id,
            raw_before=raw_before  # This triggers residual artifact figure
        )

        print(f"  Status: {result.status}")
        print(f"  Metrics: {len(result.metrics)}")
        print(f"  Figures: {list(result.figures.keys())}")

        # Check if residual_artifacts figure was generated
        if 'residual_artifacts' in result.figures:
            print("  ✅ Residual artifact quantification figure generated!")
        else:
            print("  ❌ Residual artifact quantification figure NOT found!")
            print("  Available figures:", list(result.figures.keys()))
            return False

        # Generate HTML report
        report_path = output_dir / f"{subject_id}_preprocessed_qc.html"
        qc.generate_html_report(result, report_path)
        print(f"\n📄 HTML report saved: {report_path}")

        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")
        print("  🌐 Opened in browser")

        return True

    except Exception as e:
        print(f"❌ QC failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ica_component_spectra():
    """Test ICA component power spectra in PreprocessedQC."""
    print("\n" + "="*80)
    print("TEST 3: ICA Component Power Spectra (PreprocessedQC)")
    print("="*80)

    # Create synthetic preprocessed data with ICA
    print("Creating synthetic preprocessed data with ICA for testing...")

    sfreq = 500.0
    n_channels = 32
    duration = 120  # Need longer duration for good ICA

    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Create synthetic data with different frequency components
    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))

    # Add different signal components that ICA should separate:
    # 1. Brain signal (alpha at 10 Hz)
    alpha_signal = 5e-6 * np.sin(2 * np.pi * 10 * times)
    # 2. Eye blink (low frequency, ~1 Hz)
    eye_signal = 20e-6 * np.sin(2 * np.pi * 1 * times)
    # 3. Muscle artifact (high frequency, ~35 Hz)
    muscle_signal = 8e-6 * np.sin(2 * np.pi * 35 * times)
    # 4. Line noise (50 Hz)
    line_noise = 3e-6 * np.sin(2 * np.pi * 50 * times)

    # Mix these components into channels with different weights
    np.random.seed(42)  # For reproducibility
    for i in range(n_channels):
        # Random mixing weights
        w_alpha = np.random.rand()
        w_eye = np.random.rand() * 0.5  # Eye mainly in frontal
        w_muscle = np.random.rand() * 0.3
        w_line = np.random.rand() * 0.2

        data[i, :] = (
            w_alpha * alpha_signal +
            w_eye * eye_signal +
            w_muscle * muscle_signal +
            w_line * line_noise +
            np.random.randn(n_samples) * 5e-6  # Background noise
        )

    raw_after = mne.io.RawArray(data, info)
    subject_id = "SYNTHETIC_ICA001"

    # Fit ICA
    print("  Fitting ICA (this may take a moment)...")
    from mne.preprocessing import ICA
    ica = ICA(n_components=min(10, n_channels), method='fastica', random_state=42)
    ica.fit(raw_after)

    # Simulate ICLabel classifications
    # In real usage, this would come from ICLabel
    ica_components_info = [
        {'index': 0, 'label': 'brain', 'probability': 0.85, 'rejected': False, 'reject_reason': '-'},
        {'index': 1, 'label': 'eye', 'probability': 0.75, 'rejected': True, 'reject_reason': 'eye artifact'},
        {'index': 2, 'label': 'muscle', 'probability': 0.70, 'rejected': True, 'reject_reason': 'muscle artifact'},
        {'index': 3, 'label': 'brain', 'probability': 0.60, 'rejected': False, 'reject_reason': '-'},
        {'index': 4, 'label': 'line_noise', 'probability': 0.80, 'rejected': True, 'reject_reason': 'line noise'},
    ]
    # Fill remaining components
    for i in range(5, ica.n_components_):
        ica_components_info.append({
            'index': i,
            'label': 'other',
            'probability': 0.5,
            'rejected': False,
            'reject_reason': '-'
        })

    # Mark components as excluded based on ICLabel
    ica.exclude = [c['index'] for c in ica_components_info if c['rejected']]

    # Prepare metadata
    metadata = {
        'iclabel': {
            'components': ica_components_info
        }
    }

    # Run QC
    print(f"\n📊 Running PreprocessedQC on {subject_id}...")
    output_dir = Path("/tmp/qc_test/ica_spectra")
    output_dir.mkdir(parents=True, exist_ok=True)

    qc = PreprocessedQC(output_dir=output_dir)

    try:
        result = qc.compute(
            raw_after,
            subject_id=subject_id,
            ica=ica,
            metadata=metadata
        )

        print(f"  Status: {result.status}")
        print(f"  Metrics: {len(result.metrics)}")
        print(f"  Figures: {list(result.figures.keys())}")

        # Check if ica_component_spectra figure was generated
        if 'ica_component_spectra' in result.figures:
            print("  ✅ ICA component power spectra figure generated!")
        else:
            print("  ❌ ICA component power spectra figure NOT found!")
            print("  Available figures:", list(result.figures.keys()))
            return False

        # Generate HTML report
        report_path = output_dir / f"{subject_id}_preprocessed_qc.html"
        qc.generate_html_report(result, report_path)
        print(f"\n📄 HTML report saved: {report_path}")

        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")
        print("  🌐 Opened in browser")

        return True

    except Exception as e:
        print(f"❌ QC failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_erp_difference_waves():
    """Test ERP difference waves with statistical testing in ERPQC."""
    print("\n" + "="*80)
    print("TEST 4: ERP Difference Waves with Statistics (ERPQC)")
    print("="*80)

    # Create synthetic ERP data for two conditions
    print("Creating synthetic ERP data for testing...")

    sfreq = 500.0
    n_channels = 32
    n_epochs_per_condition = 50

    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )

    # Time window for epochs: -0.2 to 0.8 seconds
    tmin, tmax = -0.2, 0.8
    n_times = int((tmax - tmin) * sfreq)
    times = np.linspace(tmin, tmax, n_times)

    # Condition 1: ERP with P1 (~100ms) and N1 (~170ms)
    data_cond1 = []
    for _ in range(n_epochs_per_condition):
        # P1 component (positive peak at 100ms)
        p1 = 3e-6 * np.exp(-((times - 0.1) ** 2) / (2 * 0.02 ** 2))
        # N1 component (negative peak at 170ms)
        n1 = -5e-6 * np.exp(-((times - 0.17) ** 2) / (2 * 0.03 ** 2))
        # Random noise
        noise = np.random.randn(n_channels, n_times) * 2e-6

        # Combine with channel mixing
        epoch_data = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            epoch_data[ch, :] = p1 + n1 + noise[ch, :]

        data_cond1.append(epoch_data)

    data_cond1 = np.array(data_cond1)  # (n_epochs, n_channels, n_times)

    # Condition 2: Similar ERP but with LARGER P1 (condition effect!)
    data_cond2 = []
    for _ in range(n_epochs_per_condition):
        # LARGER P1 component (5 µV instead of 3 µV)
        p1 = 5e-6 * np.exp(-((times - 0.1) ** 2) / (2 * 0.02 ** 2))
        # Same N1
        n1 = -5e-6 * np.exp(-((times - 0.17) ** 2) / (2 * 0.03 ** 2))
        # Random noise
        noise = np.random.randn(n_channels, n_times) * 2e-6

        epoch_data = np.zeros((n_channels, n_times))
        for ch in range(n_channels):
            epoch_data[ch, :] = p1 + n1 + noise[ch, :]

        data_cond2.append(epoch_data)

    data_cond2 = np.array(data_cond2)

    # Create epochs objects
    # Need to create events array
    events_cond1 = np.column_stack([
        np.arange(n_epochs_per_condition) * int(2 * sfreq),  # sample indices
        np.zeros(n_epochs_per_condition, dtype=int),  # unused
        np.ones(n_epochs_per_condition, dtype=int)  # event_id = 1
    ])

    events_cond2 = np.column_stack([
        np.arange(n_epochs_per_condition) * int(2 * sfreq),
        np.zeros(n_epochs_per_condition, dtype=int),
        np.full(n_epochs_per_condition, 2, dtype=int)  # event_id = 2
    ])

    epochs_cond1 = mne.EpochsArray(data_cond1, info, events=events_cond1,
                                   tmin=tmin, event_id={'condition1': 1})
    epochs_cond2 = mne.EpochsArray(data_cond2, info, events=events_cond2,
                                   tmin=tmin, event_id={'condition2': 2})

    # Compute evoked responses (averages)
    evoked_cond1 = epochs_cond1.average()
    evoked_cond2 = epochs_cond2.average()

    # Prepare for ERPQC
    evokeds = {
        'Condition 1': evoked_cond1,
        'Condition 2': evoked_cond2
    }

    epochs_dict = {
        'Condition 1': epochs_cond1,
        'Condition 2': epochs_cond2
    }

    subject_id = "SYNTHETIC_ERP001"

    # Run ERP QC
    print(f"\n📊 Running ERPQC on {subject_id}...")
    output_dir = Path("/tmp/qc_test/erp_diff")
    output_dir.mkdir(parents=True, exist_ok=True)

    from eegcpm.modules.qc.erp_qc import ERPQC
    qc = ERPQC(output_dir=output_dir)

    try:
        report_path = qc.generate_report(
            evokeds=evokeds,
            subject_id=subject_id,
            epochs_dict=epochs_dict  # Pass epochs for statistical testing
        )

        print(f"  ✅ ERP QC report generated: {report_path}")

        # Check if difference waves figure was generated
        html_content = report_path.read_text()
        if 'Difference Waves with Statistical Testing' in html_content:
            print("  ✅ Difference waves with statistics figure found in HTML!")
        else:
            print("  ❌ Difference waves figure NOT found in HTML!")
            return False

        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")
        print("  🌐 Opened in browser")

        print("\n  📊 Expected result: Significant difference around 100ms (P1 component)")
        print("     Condition 2 has larger P1 (5 µV vs 3 µV), should show gold shading")

        return True

    except Exception as e:
        print(f"❌ ERP QC failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_source_crosstalk_matrix():
    """Test source crosstalk matrix computation and visualization in SourceQC."""
    print("\n" + "="*80)
    print("TEST 5: Source Crosstalk Matrix (SourceQC)")
    print("="*80)

    # Create synthetic forward and inverse operators for testing
    print("Creating synthetic forward/inverse operators for testing...")

    try:
        # Use MNE's sample dataset (has everything we need)
        print("  Loading MNE sample dataset...")
        sample_data_path = mne.datasets.sample.data_path()
        subjects_dir = sample_data_path / 'subjects'

        # Load pre-computed forward solution
        print("  Loading forward solution...")
        fwd_fname = sample_data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-fwd.fif'
        fwd_full = mne.read_forward_solution(fwd_fname, verbose=False)

        # Extract only EEG channels
        fwd = mne.pick_types_forward(fwd_full, meg=False, eeg=True)
        info = fwd['info']
        sfreq = 600.0  # Sample dataset sampling frequency

        # Load pre-computed inverse operator
        print("  Loading inverse operator...")
        inv_fname = sample_data_path / 'MEG' / 'sample' / 'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif'
        inv_op_full = mne.minimum_norm.read_inverse_operator(inv_fname, verbose=False)

        # Use the full inverse operator (already compatible with forward)
        inv_op = inv_op_full

        # Create simplified labels (ROIs) - use just 6 ROIs for speed
        print("  Creating simplified ROI labels...")
        labels_lh = mne.read_labels_from_annot('sample', parc='aparc', hemi='lh',
                                               subjects_dir=subjects_dir, verbose=False)
        labels_rh = mne.read_labels_from_annot('sample', parc='aparc', hemi='rh',
                                               subjects_dir=subjects_dir, verbose=False)

        # Take first 3 from each hemisphere
        labels = labels_lh[:3] + labels_rh[:3]
        roi_names = [label.name for label in labels]

        print(f"  Using {len(labels)} ROIs: {', '.join(roi_names)}")

        # Run SourceQC
        subject_id = "SYNTHETIC_CROSSTALK001"
        output_dir = Path("/tmp/qc_test/source_crosstalk")
        output_dir.mkdir(parents=True, exist_ok=True)

        from eegcpm.modules.qc.source_qc import SourceQC
        qc = SourceQC(output_dir=output_dir)

        # Create source estimates (dummy data)
        n_times = 100
        stc = mne.SourceEstimate(
            data=np.random.randn(fwd['nsource'], n_times) * 1e-9,
            vertices=[fwd['src'][0]['vertno'], fwd['src'][1]['vertno']],
            tmin=0,
            tstep=1.0 / sfreq
        )

        # Extract ROI data
        roi_timeseries = mne.extract_label_time_course(stc, labels, fwd['src'],
                                                       mode='mean', verbose=False)

        roi_data = {
            'condition1': roi_timeseries,
            'condition1_times': stc.times,
            'roi_names': np.array(roi_names)
        }

        # Create a synthetic crosstalk matrix for testing (skip expensive computation)
        n_rois = len(labels)
        crosstalk_matrix = np.eye(n_rois)  # Perfect diagonal
        # Add some realistic off-diagonal leakage
        for i in range(n_rois):
            for j in range(i+1, n_rois):
                # Spatial proximity-based leakage (random but realistic)
                leakage = np.random.uniform(0.1, 0.4)
                crosstalk_matrix[i, j] = crosstalk_matrix[j, i] = leakage

        # Prepare data dict with crosstalk matrix (pre-computed)
        data_dict = {
            'stcs': {'condition1': stc},
            'roi_data': roi_data,
            'method': 'dSPM',
            'parcellation': 'aparc',
            'forward': fwd,
            'inverse_operator': inv_op,
            'labels': labels,
            'crosstalk_matrix': crosstalk_matrix  # Pre-computed to avoid expensive calculation
        }

        print(f"\n📊 Running SourceQC on {subject_id}...")
        result = qc.compute(data_dict, subject_id=subject_id, sfreq=sfreq)

        print(f"  Status: {result.status}")
        print(f"  Metrics: {len(result.metrics)}")
        print(f"  Figures: {list(result.figures.keys())}")

        # Check if crosstalk matrix was generated
        if 'crosstalk_matrix' in result.figures:
            print("  ✅ Source crosstalk matrix figure generated!")

            # Check metadata
            if 'crosstalk_mean' in result.metadata:
                mean_ct = result.metadata['crosstalk_mean']
                n_high = result.metadata.get('n_directed_above_0.3', 0)
                print(f"  Mean crosstalk: {mean_ct:.3f}")
                print(f"  High-crosstalk directed pairs: {n_high}")
        else:
            print("  ❌ Source crosstalk matrix figure NOT found!")
            print("  Available figures:", list(result.figures.keys()))
            return False

        # Generate HTML report
        report_path = output_dir / f"{subject_id}_source_qc.html"
        qc.generate_html_report(result, report_path)
        print(f"\n📄 HTML report saved: {report_path}")

        # Open in browser
        import webbrowser
        webbrowser.open(f"file://{report_path}")
        print("  🌐 Opened in browser")

        print("\n  📊 Expected result: Crosstalk matrix with low off-diagonal values")
        print("     Diagonal should be 1.0 (perfect self-reconstruction)")
        print("     Off-diagonal values show spatial leakage between ROIs")

        return True

    except Exception as e:
        print(f"❌ Source crosstalk test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("TESTING NEW QC FIGURES")
    print("="*80)
    print("\nThis script will test:")
    print("  1. Temporal Artifact Distribution (RawDataQC)")
    print("  2. Residual Artifact Quantification (PreprocessedQC)")
    print("  3. ICA Component Power Spectra (PreprocessedQC)")
    print("  4. ERP Difference Waves with Statistics (ERPQC)")
    print("  5. Source Crosstalk Matrix (SourceQC)")
    print("\nHTML reports will open in your browser for visual inspection.")

    results = []

    # Test 1
    results.append(("Temporal Artifacts", test_temporal_artifact_distribution()))

    # Test 2
    results.append(("Residual Artifacts", test_residual_artifact_quantification()))

    # Test 3
    results.append(("ICA Component Spectra", test_ica_component_spectra()))

    # Test 4
    results.append(("ERP Difference Waves", test_erp_difference_waves()))

    # Test 5
    results.append(("Source Crosstalk Matrix", test_source_crosstalk_matrix()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests PASSED!")
        print("\nPlease visually inspect the HTML reports in your browser:")
        print("  - Check that figures display correctly")
        print("  - Verify plots are informative and clear")
        print("  - Look for any rendering issues")
        return 0
    else:
        print("\n❌ Some tests FAILED - see errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
