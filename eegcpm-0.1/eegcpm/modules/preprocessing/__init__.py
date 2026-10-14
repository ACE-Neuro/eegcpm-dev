"""
Preprocessing Module - Modular EEG Preprocessing Pipeline

This module provides a flexible, step-based preprocessing pipeline for EEG data.

Architecture:
- PreprocessingPipeline: Main pipeline builder and executor
- ProcessingStep subclasses: Individual preprocessing steps
- Helper utilities: Reusable components for complex algorithms

Author: EEGCPM Development Team
Created: 2025-12
"""

# Pipeline builder
from .pipeline_builder import PreprocessingPipeline

# Base class
from .steps.base import ProcessingStep, RecordingExcluded

# Individual steps
from .steps.filter import FilterStep
from .steps.montage import MontageStep
from .steps.resample import ResampleStep
from .steps.bad_channels import BadChannelDetectionStep
from .steps.interpolate import InterpolateBadChannelsStep
from .steps.artifacts import ArtifactAnnotationStep
from .steps.reference import ReferenceStep
from .steps.asr import ASRStep
from .steps.ica import ICAStep
from .steps.iclabel import ICLabelStep
from .steps.zapline import ZaplineStep

# Standalone utilities (can be used outside pipeline)
from .bad_channels import BadChannelDetector
from .artifacts import ArtifactAnnotator
from .channel_clustering import compute_bad_channel_clustering
from .data_quality import detect_all_quality_issues

__all__ = [
    # Main pipeline
    'PreprocessingPipeline',

    # Base class
    'ProcessingStep',
    'RecordingExcluded',

    # Steps
    'FilterStep',
    'MontageStep',
    'ResampleStep',
    'BadChannelDetectionStep',
    'InterpolateBadChannelsStep',
    'ArtifactAnnotationStep',
    'ReferenceStep',
    'ASRStep',
    'ICAStep',
    'ICLabelStep',
    'ZaplineStep',

    # Standalone utilities
    'BadChannelDetector',
    'ArtifactAnnotator',
    'compute_bad_channel_clustering',
    'detect_all_quality_issues',
]
