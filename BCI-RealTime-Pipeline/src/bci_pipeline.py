#!/usr/bin/env python3
"""
BCI Real-Time Pipeline
Main pipeline for real-time brain-computer interface processing

This module implements the core BCI pipeline for:
- Real-time EEG signal processing
- Speech intent decoding
- Multi-modal data collection
- Clinical trial compliance
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BCIConfig:
    """Configuration for BCI pipeline"""
    # EEG parameters
    sampling_rate: int = 1000  # Hz
    buffer_size: int = 1000    # samples
    channels: List[str] = None
    
    # Processing parameters
    filter_low: float = 1.0    # Hz
    filter_high: float = 40.0  # Hz
    notch_freq: float = 60.0   # Hz (power line)
    
    # Speech decoding parameters
    window_size: int = 500     # ms
    overlap: float = 0.5       # 50% overlap
    
    # Clinical trial settings
    fda_compliant: bool = True
    data_backup: bool = True
    session_logging: bool = True
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T3', 'C3', 'Cz', 'C4', 'T4',
                'T5', 'P3', 'Pz', 'P4', 'T6',
                'O1', 'O2'
            ]


@dataclass
class SpeechIntent:
    """Decoded speech intent from BCI"""
    timestamp: float
    confidence: float
    intent_type: str  # 'speak', 'silent', 'thinking'
    decoded_text: Optional[str] = None
    neural_features: Optional[np.ndarray] = None


class BCIPipeline:
    """
    Main BCI pipeline for real-time speech decoding
    
    This class orchestrates the entire BCI processing pipeline:
    1. EEG data acquisition and preprocessing
    2. Real-time signal processing and artifact removal
    3. Speech intent decoding using neural patterns
    4. Multi-modal data collection and synchronization
    5. Clinical trial compliance and data validation
    """
    
    def __init__(self, config: Optional[BCIConfig] = None):
        """
        Initialize BCI pipeline
        
        Args:
            config: BCI configuration parameters
        """
        self.config = config or BCIConfig()
        self.is_running = False
        self.session_start_time = None
        
        # Data buffers
        self.eeg_buffer = np.zeros((len(self.config.channels), self.config.buffer_size))
        self.audio_buffer = []
        self.video_buffer = []
        
        # Processing state
        self.current_sample = 0
        self.last_decoded_intent = None
        
        # Clinical trial data
        self.session_data = []
        self.quality_metrics = {}
        
        # Threading
        self.processing_thread = None
        self.data_collection_thread = None
        
        logger.info("BCI Pipeline initialized successfully")
    
    def start(self) -> bool:
        """
        Start the BCI pipeline
        
        Returns:
            bool: True if started successfully
        """
        try:
            self.is_running = True
            self.session_start_time = time.time()
            
            # Start processing threads
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.data_collection_thread = threading.Thread(target=self._data_collection_loop)
            
            self.processing_thread.start()
            self.data_collection_thread.start()
            
            logger.info(f"BCI Pipeline started at {self.session_start_time}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start BCI pipeline: {e}")
            self.is_running = False
            return False
    
    def stop(self) -> None:
        """Stop the BCI pipeline"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join()
        if self.data_collection_thread:
            self.data_collection_thread.join()
        
        # Save session data
        self._save_session_data()
        
        logger.info("BCI Pipeline stopped")
    
    def _processing_loop(self) -> None:
        """Main processing loop for real-time EEG analysis"""
        while self.is_running:
            try:
                # Process current EEG buffer
                processed_eeg = self._preprocess_eeg(self.eeg_buffer)
                
                # Extract neural features
                features = self._extract_features(processed_eeg)
                
                # Decode speech intent
                speech_intent = self._decode_speech_intent(features)
                
                if speech_intent and speech_intent.confidence > 0.7:
                    self.last_decoded_intent = speech_intent
                    logger.info(f"Speech intent detected: {speech_intent.intent_type} "
                              f"(confidence: {speech_intent.confidence:.2f})")
                
                # Update quality metrics
                self._update_quality_metrics(processed_eeg)
                
                time.sleep(0.01)  # 10ms processing interval
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
    def _data_collection_loop(self) -> None:
        """Data collection loop for multi-modal data"""
        while self.is_running:
            try:
                # Collect EEG data (simulated for now)
                self._collect_eeg_data()
                
                # Collect audio data
                self._collect_audio_data()
                
                # Collect video data
                self._collect_video_data()
                
                # Synchronize data streams
                self._synchronize_data()
                
                time.sleep(0.05)  # 50ms collection interval
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(0.1)
    
    def _preprocess_eeg(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Preprocess EEG data for analysis
        
        Args:
            eeg_data: Raw EEG data [channels x samples]
            
        Returns:
            np.ndarray: Preprocessed EEG data
        """
        try:
            # Remove DC offset
            eeg_centered = eeg_data - np.mean(eeg_data, axis=1, keepdims=True)
            
            # Apply bandpass filter
            nyquist = self.config.sampling_rate / 2
            low = self.config.filter_low / nyquist
            high = self.config.filter_high / nyquist
            
            b, a = signal.butter(4, [low, high], btype='band')
            eeg_filtered = signal.filtfilt(b, a, eeg_centered, axis=1)
            
            # Apply notch filter for power line interference
            notch_freq = self.config.notch_freq / nyquist
            b_notch, a_notch = signal.iirnotch(notch_freq, 30)
            eeg_clean = signal.filtfilt(b_notch, a_notch, eeg_filtered, axis=1)
            
            return eeg_clean
            
        except Exception as e:
            logger.error(f"EEG preprocessing failed: {e}")
            return eeg_data
    
    def _extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract neural features from preprocessed EEG
        
        Args:
            eeg_data: Preprocessed EEG data
            
        Returns:
            np.ndarray: Extracted features
        """
        try:
            features = []
            
            for channel in range(eeg_data.shape[0]):
                # Power spectral density
                freqs, psd = signal.welch(eeg_data[channel], 
                                        fs=self.config.sampling_rate,
                                        nperseg=256)
                
                # Alpha, beta, gamma power
                alpha_power = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
                beta_power = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
                gamma_power = np.mean(psd[(freqs >= 30) & (freqs <= 100)])
                
                # Entropy
                entropy = -np.sum(psd * np.log(psd + 1e-10))
                
                features.extend([alpha_power, beta_power, gamma_power, entropy])
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(len(self.config.channels) * 4)
    
    def _decode_speech_intent(self, features: np.ndarray) -> Optional[SpeechIntent]:
        """
        Decode speech intent from neural features
        
        Args:
            features: Extracted neural features
            
        Returns:
            Optional[SpeechIntent]: Decoded speech intent
        """
        try:
            # Simple threshold-based decoding (replace with ML model)
            alpha_power = features[::4]  # Alpha power for each channel
            beta_power = features[1::4]  # Beta power for each channel
            
            # Speech intent detection based on motor cortex activity
            motor_channels = ['C3', 'Cz', 'C4']  # Motor cortex
            motor_indices = [self.config.channels.index(ch) for ch in motor_channels 
                           if ch in self.config.channels]
            
            if motor_indices:
                motor_activity = np.mean([alpha_power[i] + beta_power[i] 
                                        for i in motor_indices])
                
                # Threshold-based classification
                if motor_activity > 0.5:
                    confidence = min(motor_activity, 0.95)
                    intent_type = 'speak' if motor_activity > 0.7 else 'thinking'
                    
                    return SpeechIntent(
                        timestamp=time.time(),
                        confidence=confidence,
                        intent_type=intent_type,
                        neural_features=features
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Speech intent decoding failed: {e}")
            return None
    
    def _collect_eeg_data(self) -> None:
        """Collect EEG data (simulated for development)"""
        # Simulate EEG data collection
        # In real implementation, this would interface with EEG hardware
        pass
    
    def _collect_audio_data(self) -> None:
        """Collect audio data from microphone"""
        # Simulate audio data collection
        # In real implementation, this would interface with audio hardware
        pass
    
    def _collect_video_data(self) -> None:
        """Collect video data from camera"""
        # Simulate video data collection
        # In real implementation, this would interface with camera hardware
        pass
    
    def _synchronize_data(self) -> None:
        """Synchronize multi-modal data streams"""
        # Ensure temporal alignment of EEG, audio, and video data
        pass
    
    def _update_quality_metrics(self, eeg_data: np.ndarray) -> None:
        """
        Update data quality metrics
        
        Args:
            eeg_data: Processed EEG data
        """
        try:
            # Signal-to-noise ratio
            signal_power = np.var(eeg_data, axis=1)
            noise_power = np.var(eeg_data - np.mean(eeg_data, axis=1, keepdims=True), axis=1)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            # Artifact detection (simple threshold)
            artifact_threshold = 100  # microvolts
            artifacts = np.sum(np.abs(eeg_data) > artifact_threshold, axis=1)
            artifact_ratio = artifacts / eeg_data.shape[1]
            
            self.quality_metrics.update({
                'timestamp': time.time(),
                'snr_mean': np.mean(snr),
                'snr_std': np.std(snr),
                'artifact_ratio_mean': np.mean(artifact_ratio),
                'channels_with_artifacts': np.sum(artifact_ratio > 0.1)
            })
            
        except Exception as e:
            logger.error(f"Quality metrics update failed: {e}")
    
    def _save_session_data(self) -> None:
        """Save session data for clinical trial compliance"""
        try:
            if not self.session_data:
                return
            
            # Create data directory
            data_dir = Path("data") / f"session_{int(self.session_start_time)}"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save session summary
            session_summary = {
                'session_start': self.session_start_time,
                'session_duration': time.time() - self.session_start_time,
                'total_samples': self.current_sample,
                'quality_metrics': self.quality_metrics,
                'decoded_intents': len([d for d in self.session_data if d.get('intent')])
            }
            
            summary_file = data_dir / "session_summary.json"
            import json
            with open(summary_file, 'w') as f:
                json.dump(session_summary, f, indent=2)
            
            logger.info(f"Session data saved to {data_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def get_speech_intent(self) -> Optional[SpeechIntent]:
        """
        Get the most recently decoded speech intent
        
        Returns:
            Optional[SpeechIntent]: Latest speech intent
        """
        return self.last_decoded_intent
    
    def get_quality_metrics(self) -> Dict:
        """
        Get current data quality metrics
        
        Returns:
            Dict: Quality metrics
        """
        return self.quality_metrics.copy()
    
    def update_dashboard(self) -> Dict:
        """
        Update dashboard with current pipeline status
        
        Returns:
            Dict: Dashboard data
        """
        return {
            'is_running': self.is_running,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'current_sample': self.current_sample,
            'last_intent': self.last_decoded_intent,
            'quality_metrics': self.quality_metrics,
            'buffer_status': {
                'eeg_buffer_size': self.eeg_buffer.shape,
                'audio_buffer_size': len(self.audio_buffer),
                'video_buffer_size': len(self.video_buffer)
            }
        }


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BCI Real-Time Pipeline')
    parser.add_argument('--mode', choices=['realtime', 'clinical', 'demo'], 
                       default='demo', help='Pipeline mode')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Session duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config = BCIConfig()
    pipeline = BCIPipeline(config)
    
    try:
        # Start pipeline
        if pipeline.start():
            logger.info(f"BCI Pipeline running in {args.mode} mode for {args.duration} seconds")
            
            # Run for specified duration
            time.sleep(args.duration)
            
            # Stop pipeline
            pipeline.stop()
            logger.info("BCI Pipeline completed successfully")
        else:
            logger.error("Failed to start BCI Pipeline")
            
    except KeyboardInterrupt:
        logger.info("BCI Pipeline interrupted by user")
        pipeline.stop()
    except Exception as e:
        logger.error(f"BCI Pipeline error: {e}")
        pipeline.stop()


if __name__ == "__main__":
    main()
