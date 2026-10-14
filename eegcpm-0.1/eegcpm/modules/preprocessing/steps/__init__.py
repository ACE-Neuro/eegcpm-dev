"""
Modular preprocessing steps.

Each step is an independent, configurable processing unit that can be
composed into flexible preprocessing pipelines.

Author: EEGCPM Development Team
Created: 2025-12
"""

from .base import ProcessingStep, RecordingExcluded
from .filter import FilterStep
from .lowpass import LowpassStep
from .notch import NotchStep
from .asr import ASRStep
from .bad_channels import BadChannelDetectionStep
from .ica import ICAStep
from .iclabel import ICLabelStep
from .interpolate import InterpolateBadChannelsStep
from .reference import ReferenceStep
from .resample import ResampleStep
from .montage import MontageStep
from .artifacts import ArtifactAnnotationStep
from .drop_flat import DropFlatStep
from .zapline import ZaplineStep

# Step registry - maps config names to step classes
STEP_REGISTRY = {
    'filter': FilterStep,
    'lowpass': LowpassStep,
    'notch': NotchStep,
    'asr': ASRStep,
    'bad_channels': BadChannelDetectionStep,
    'ica': ICAStep,
    'iclabel': ICLabelStep,
    'interpolate': InterpolateBadChannelsStep,
    'reference': ReferenceStep,
    'resample': ResampleStep,
    'montage': MontageStep,
    'artifacts': ArtifactAnnotationStep,
    'drop_flat': DropFlatStep,
    'zapline': ZaplineStep,
}

__all__ = [
    'ProcessingStep',
    'RecordingExcluded',
    'FilterStep',
    'LowpassStep',
    'NotchStep',
    'ASRStep',
    'BadChannelDetectionStep',
    'ICAStep',
    'ICLabelStep',
    'InterpolateBadChannelsStep',
    'ReferenceStep',
    'ResampleStep',
    'MontageStep',
    'ArtifactAnnotationStep',
    'DropFlatStep',
    'ZaplineStep',
    'STEP_REGISTRY',
]
