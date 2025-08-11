"""
BCI Real-Time Pipeline Package

This package contains the real-time processing pipeline for BCI
(Brain-Computer Interface) applications.

Components:
- BCIPipeline: Main pipeline orchestrator
- EEGProcessor: EEG data processing
- SpeechDecoder: Speech intent decoding
- DataCollector: Multi-modal data collection
- Dashboard: Real-time visualization
- Utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "BCI Research Team"

# Re-export main components
from .bci_pipeline import BCIPipeline, BCIConfig, SpeechIntent
from .eeg_processor import EEGProcessor, EEGConfig, EEGQuality
from .speech_decoder import SpeechDecoder, SpeechConfig, DecoderConfig
from .data_collector import DataCollector, DataConfig, DataSample, DataQuality
from .dashboard import Dashboard, DashboardConfig
from .utils import validate_eeg_data, normalize_eeg_data, calculate_snr

__all__ = [
    # Core classes
    'BCIPipeline', 'EEGProcessor', 'SpeechDecoder', 'DataCollector', 'Dashboard',

    # Configuration classes
    'BCIConfig', 'EEGConfig', 'SpeechConfig', 'DecoderConfig', 'DataConfig', 'DashboardConfig',

    # Data classes
    'SpeechIntent', 'EEGQuality', 'DataSample', 'DataQuality',

    # Utility functions
    'validate_eeg_data', 'normalize_eeg_data', 'calculate_snr',
]

