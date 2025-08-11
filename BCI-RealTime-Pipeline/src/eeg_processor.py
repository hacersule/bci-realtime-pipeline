#!/usr/bin/env python3
"""
EEG Processor Module
Real-time EEG signal processing for BCI applications

This module provides:
- Real-time EEG data acquisition
- Signal preprocessing and filtering
- Artifact detection and removal
- Feature extraction for neural decoding
- Quality monitoring and validation
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy import signal
from scipy.stats import zscore
# import mne  # MNE library not available, using alternative approach

logger = logging.getLogger(__name__)


@dataclass
class EEGConfig:
    """Configuration for EEG processing"""
    # Hardware parameters
    sampling_rate: int = 1000  # Hz
    num_channels: int = 19
    channel_names: List[str] = None
    reference_channels: List[str] = None
    
    # Filtering parameters
    filter_low: float = 1.0    # Hz
    filter_high: float = 40.0  # Hz
    notch_freq: float = 60.0   # Hz
    filter_order: int = 4
    
    # Processing parameters
    buffer_size: int = 1000    # samples
    overlap: float = 0.5       # 50% overlap
    window_size: int = 500     # ms
    
    # Quality thresholds
    snr_threshold: float = 10.0  # dB
    artifact_threshold: float = 100.0  # microvolts
    max_artifact_ratio: float = 0.3
    
    def __post_init__(self):
        if self.channel_names is None:
            self.channel_names = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T3', 'C3', 'Cz', 'C4', 'T4',
                'T5', 'P3', 'Pz', 'P4', 'T6',
                'O1', 'O2'
            ]
        
        if self.reference_channels is None:
            self.reference_channels = ['A1', 'A2']  # Mastoid references


@dataclass
class EEGQuality:
    """EEG data quality metrics"""
    timestamp: float
    snr: float
    artifact_ratio: float
    channel_quality: Dict[str, float]
    overall_quality: str  # 'excellent', 'good', 'fair', 'poor'
    recommendations: List[str]


class EEGProcessor:
    """
    Real-time EEG signal processor for BCI applications
    
    This class handles:
    1. Continuous EEG data acquisition
    2. Real-time signal preprocessing
    3. Artifact detection and removal
    4. Feature extraction for neural decoding
    5. Quality monitoring and validation
    """
    
    def __init__(self, config: Optional[EEGConfig] = None, 
                 data_callback: Optional[Callable] = None):
        """
        Initialize EEG processor
        
        Args:
            config: EEG processing configuration
            data_callback: Callback function for processed data
        """
        self.config = config or EEGConfig()
        self.data_callback = data_callback
        
        # Initialize buffers
        self.raw_buffer = np.zeros((self.config.num_channels, self.config.buffer_size))
        self.processed_buffer = np.zeros((self.config.num_channels, self.config.buffer_size))
        self.feature_buffer = []
        
        # Processing state
        self.is_processing = False
        self.current_sample = 0
        self.last_quality_check = 0
        
        # Filter coefficients
        self._init_filters()
        
        # Quality metrics
        self.quality_history = []
        self.artifact_history = []
        
        logger.info("EEG Processor initialized successfully")
    
    def _init_filters(self) -> None:
        """Initialize digital filters for EEG processing"""
        try:
            # Bandpass filter
            nyquist = self.config.sampling_rate / 2
            low = self.config.filter_low / nyquist
            high = self.config.filter_high / nyquist
            
            self.bandpass_b, self.bandpass_a = signal.butter(
                self.config.filter_order, [low, high], btype='band'
            )
            
            # Notch filter for power line interference
            notch_freq = self.config.notch_freq / nyquist
            self.notch_b, self.notch_a = signal.iirnotch(
                notch_freq, 30, self.config.sampling_rate
            )
            
            # High-pass filter for DC removal
            self.highpass_b, self.highpass_a = signal.butter(
                2, 0.5 / nyquist, btype='high'
            )
            
            logger.info("Digital filters initialized successfully")
            
        except Exception as e:
            logger.error(f"Filter initialization failed: {e}")
            raise
    
    def start_processing(self) -> bool:
        """Start EEG processing"""
        try:
            self.is_processing = True
            logger.info("EEG processing started")
            return True
        except Exception as e:
            logger.error(f"Failed to start EEG processing: {e}")
            return False
    
    def stop_processing(self) -> None:
        """Stop EEG processing"""
        self.is_processing = False
        logger.info("EEG processing stopped")
    
    def process_sample(self, eeg_sample: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single EEG sample
        
        Args:
            eeg_sample: Raw EEG sample [channels x 1]
            
        Returns:
            Optional[np.ndarray]: Processed EEG sample
        """
        try:
            if not self.is_processing:
                return None
            
            # Add to buffer
            self.raw_buffer = np.roll(self.raw_buffer, -1, axis=1)
            self.raw_buffer[:, -1] = eeg_sample.flatten()
            
            # Process buffer when full
            if self.current_sample >= self.config.buffer_size:
                processed_data = self._process_buffer()
                
                if processed_data is not None:
                    # Extract features
                    features = self._extract_features(processed_data)
                    self.feature_buffer.append(features)
                    
                    # Quality check
                    if time.time() - self.last_quality_check > 1.0:  # Every second
                        quality = self._assess_quality(processed_data)
                        self.quality_history.append(quality)
                        self.last_quality_check = time.time()
                    
                    # Callback if provided
                    if self.data_callback:
                        self.data_callback({
                            'raw': self.raw_buffer.copy(),
                            'processed': processed_data,
                            'features': features,
                            'quality': self.quality_history[-1] if self.quality_history else None
                        })
                    
                    return processed_data
            
            self.current_sample += 1
            return None
            
        except Exception as e:
            logger.error(f"Sample processing failed: {e}")
            return None
    
    def _process_buffer(self) -> Optional[np.ndarray]:
        """
        Process the current EEG buffer
        
        Returns:
            Optional[np.ndarray]: Processed EEG data
        """
        try:
            # Copy buffer for processing
            eeg_data = self.raw_buffer.copy()
            
            # 1. Remove DC offset
            eeg_centered = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
            
            # 2. Apply high-pass filter for DC removal
            eeg_highpass = signal.filtfilt(self.highpass_b, self.highpass_a, eeg_centered, axis=1)
            
            # 3. Apply bandpass filter
            eeg_bandpass = signal.filtfilt(self.bandpass_b, self.bandpass_a, eeg_highpass, axis=1)
            
            # 4. Apply notch filter
            eeg_notch = signal.filtfilt(self.notch_b, self.notch_a, eeg_bandpass, axis=1)
            
            # 5. Artifact removal
            eeg_clean = self._remove_artifacts(eeg_notch)
            
            # Update processed buffer
            self.processed_buffer = eeg_clean
            
            return eeg_clean
            
        except Exception as e:
            logger.error(f"Buffer processing failed: {e}")
            return None
    
    def _remove_artifacts(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Remove artifacts from EEG data
        
        Args:
            eeg_data: Preprocessed EEG data
            
        Returns:
            np.ndarray: Clean EEG data
        """
        try:
            eeg_clean = eeg_data.copy()
            
            # 1. Amplitude thresholding
            artifact_mask = np.abs(eeg_data) > self.config.artifact_threshold
            eeg_clean[artifact_mask] = np.nan
            
            # 2. Statistical outlier detection
            for ch in range(eeg_data.shape[0]):
                channel_data = eeg_data[ch, :]
                z_scores = np.abs(zscore(channel_data))
                outlier_mask = z_scores > 3.0
                eeg_clean[ch, outlier_mask] = np.nan
            
            # 3. Interpolate removed segments
            for ch in range(eeg_clean.shape[0]):
                channel_data = eeg_clean[ch, :]
                if np.any(np.isnan(channel_data)):
                    # Simple linear interpolation
                    valid_indices = ~np.isnan(channel_data)
                    if np.sum(valid_indices) > 1:
                        valid_data = channel_data[valid_indices]
                        valid_positions = np.where(valid_indices)[0]
                        all_positions = np.arange(len(channel_data))
                        
                        channel_data = np.interp(all_positions, valid_positions, valid_data)
                        eeg_clean[ch, :] = channel_data
            
            return eeg_clean
            
        except Exception as e:
            logger.error(f"Artifact removal failed: {e}")
            return eeg_data
    
    def _extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features from processed EEG data
        
        Args:
            eeg_data: Processed EEG data
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            features = []
            
            for ch in range(eeg_data.shape[0]):
                channel_data = eeg_data[ch, :]
                
                # Time domain features
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                variance = np.var(channel_data)
                rms = np.sqrt(np.mean(channel_data**2))
                
                # Frequency domain features
                freqs, psd = signal.welch(channel_data, 
                                        fs=self.config.sampling_rate,
                                        nperseg=min(256, len(channel_data)))
                
                # Band powers
                delta_power = np.mean(psd[(freqs >= 0.5) & (freqs < 4)])
                theta_power = np.mean(psd[(freqs >= 4) & (freqs < 8)])
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs < 13)])
                beta_power = np.mean(psd[(freqs >= 13) & (freqs < 30)])
                gamma_power = np.mean(psd[(freqs >= 30) & (freqs < 100)])
                
                # Spectral features
                peak_freq = freqs[np.argmax(psd)]
                spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))
                spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
                
                # Hjorth parameters
                hjorth_activity = variance
                hjorth_mobility = np.sqrt(np.var(np.diff(channel_data)) / variance)
                hjorth_complexity = np.sqrt(np.var(np.diff(np.diff(channel_data))) / 
                                          np.var(np.diff(channel_data)))
                
                channel_features = [
                    mean_val, std_val, variance, rms,
                    delta_power, theta_power, alpha_power, beta_power, gamma_power,
                    peak_freq, spectral_entropy, spectral_centroid,
                    hjorth_activity, hjorth_mobility, hjorth_complexity
                ]
                
                features.extend(channel_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self.config.num_channels * 15)  # 15 features per channel
    
    def _assess_quality(self, eeg_data: np.ndarray) -> EEGQuality:
        """
        Assess EEG data quality
        
        Args:
            eeg_data: Processed EEG data
            
        Returns:
            EEGQuality: Quality assessment
        """
        try:
            # Calculate SNR
            signal_power = np.var(eeg_data, axis=1)
            noise_power = np.var(eeg_data - np.mean(eeg_data, axis=1, keepdims=True), axis=1)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            avg_snr = np.mean(snr)
            
            # Calculate artifact ratio
            artifact_mask = np.abs(eeg_data) > self.config.artifact_threshold
            artifact_ratio = np.sum(artifact_mask) / eeg_data.size
            
            # Channel quality assessment
            channel_quality = {}
            for i, ch_name in enumerate(self.config.channel_names):
                ch_snr = snr[i] if i < len(snr) else 0
                ch_artifacts = np.sum(artifact_mask[i, :]) / eeg_data.shape[1]
                
                if ch_snr > 20 and ch_artifacts < 0.1:
                    quality_score = 1.0
                elif ch_snr > 15 and ch_artifacts < 0.2:
                    quality_score = 0.8
                elif ch_snr > 10 and ch_artifacts < 0.3:
                    quality_score = 0.6
                else:
                    quality_score = 0.4
                
                channel_quality[ch_name] = quality_score
            
            # Overall quality assessment
            if avg_snr > 20 and artifact_ratio < 0.1:
                overall_quality = 'excellent'
            elif avg_snr > 15 and artifact_ratio < 0.2:
                overall_quality = 'good'
            elif avg_snr > 10 and artifact_ratio < 0.3:
                overall_quality = 'fair'
            else:
                overall_quality = 'poor'
            
            # Recommendations
            recommendations = []
            if avg_snr < 15:
                recommendations.append("Check electrode connections and impedance")
            if artifact_ratio > 0.2:
                recommendations.append("Reduce movement and muscle artifacts")
            if np.mean(list(channel_quality.values())) < 0.7:
                recommendations.append("Reapply electrodes for better contact")
            
            return EEGQuality(
                timestamp=time.time(),
                snr=avg_snr,
                artifact_ratio=artifact_ratio,
                channel_quality=channel_quality,
                overall_quality=overall_quality,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return EEGQuality(
                timestamp=time.time(),
                snr=0.0,
                artifact_ratio=1.0,
                channel_quality={},
                overall_quality='poor',
                recommendations=["Quality assessment failed"]
            )
    
    def get_quality_summary(self) -> Dict:
        """
        Get summary of EEG quality metrics
        
        Returns:
            Dict: Quality summary
        """
        if not self.quality_history:
            return {}
        
        recent_quality = self.quality_history[-10:]  # Last 10 assessments
        
        return {
            'current_quality': self.quality_history[-1].overall_quality,
            'avg_snr': np.mean([q.snr for q in recent_quality]),
            'avg_artifact_ratio': np.mean([q.artifact_ratio for q in recent_quality]),
            'quality_trend': 'improving' if len(recent_quality) > 1 and 
                            recent_quality[-1].snr > recent_quality[0].snr else 'stable',
            'recommendations': self.quality_history[-1].recommendations
        }
    
    def get_feature_buffer(self) -> List[np.ndarray]:
        """Get the feature buffer"""
        return self.feature_buffer.copy()
    
    def clear_buffers(self) -> None:
        """Clear all data buffers"""
        self.raw_buffer.fill(0)
        self.processed_buffer.fill(0)
        self.feature_buffer.clear()
        self.current_sample = 0
        logger.info("All buffers cleared")


def create_mne_info(config: EEGConfig) -> dict:
    """
    Create EEG info dictionary (MNE alternative)
    
    Args:
        config: EEG configuration
        
    Returns:
        dict: EEG info dictionary
    """
    try:
        info = {
            'ch_names': config.channel_names,
            'sfreq': config.sampling_rate,
            'ch_types': ['eeg'] * len(config.channel_names),
            'n_channels': len(config.channel_names)
        }
        return info
    except Exception as e:
        logger.error(f"Failed to create EEG Info: {e}")
        return None


def simulate_eeg_data(duration: float, config: EEGConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate EEG data for testing
    
    Args:
        duration: Duration in seconds
        config: EEG configuration
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Time array and simulated EEG data
    """
    try:
        # Time array
        t = np.arange(0, duration, 1/config.sampling_rate)
        
        # Simulate different frequency components
        alpha_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta_signal = np.sin(2 * np.pi * 20 * t)   # 20 Hz beta
        theta_signal = np.sin(2 * np.pi * 6 * t)   # 6 Hz theta
        
        # Combine signals with different weights for each channel
        eeg_data = np.zeros((config.num_channels, len(t)))
        
        for ch in range(config.num_channels):
            # Add different frequency components with random weights
            weights = np.random.rand(3)
            eeg_data[ch, :] = (weights[0] * alpha_signal + 
                               weights[1] * beta_signal + 
                               weights[2] * theta_signal)
            
            # Add noise
            noise = np.random.normal(0, 0.1, len(t))
            eeg_data[ch, :] += noise
            
            # Add occasional artifacts
            if np.random.rand() < 0.01:  # 1% chance of artifact
                artifact_start = np.random.randint(0, len(t) - 100)
                artifact_duration = np.random.randint(50, 100)
                eeg_data[ch, artifact_start:artifact_start + artifact_duration] += \
                    np.random.normal(0, 2, artifact_duration)
        
        return t, eeg_data
        
    except Exception as e:
        logger.error(f"EEG data simulation failed: {e}")
        return np.array([]), np.array([])


if __name__ == "__main__":
    # Test the EEG processor
    config = EEGConfig()
    processor = EEGProcessor(config)
    
    # Simulate data
    duration = 5.0  # 5 seconds
    t, eeg_data = simulate_eeg_data(duration, config)
    
    if len(t) > 0:
        print(f"Simulated {len(t)} samples of EEG data")
        
        # Process samples
        processor.start_processing()
        
        for i in range(0, len(t), 10):  # Process every 10th sample
            sample = eeg_data[:, i:i+10]
            result = processor.process_sample(sample)
            
            if result is not None:
                print(f"Processed buffer at sample {i}")
        
        processor.stop_processing()
        
        # Get quality summary
        quality_summary = processor.get_quality_summary()
        print(f"Quality Summary: {quality_summary}")
    else:
        print("Failed to simulate EEG data")
