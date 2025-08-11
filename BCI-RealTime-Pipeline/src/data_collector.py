#!/usr/bin/env python3
"""
Data Collector Module
Multi-modal data collection for BCI applications

This module provides:
- Real-time EEG data acquisition
- Audio recording and processing
- Video capture and analysis
- Data synchronization and storage
- Quality monitoring and validation
"""

import time
import logging
import numpy as np
import json
import threading
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import queue

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data collection"""
    # EEG settings
    eeg_sampling_rate: int = 1000  # Hz
    eeg_channels: List[str] = None
    eeg_buffer_size: int = 1000
    
    # Audio settings
    audio_enabled: bool = True
    audio_sampling_rate: int = 16000  # Hz
    audio_channels: int = 1
    audio_buffer_size: int = 1600  # 100ms at 16kHz
    
    # Video settings
    video_enabled: bool = True
    video_fps: int = 30
    video_resolution: Tuple[int, int] = (640, 480)
    video_buffer_size: int = 30  # 1 second at 30fps
    
    # Storage settings
    data_directory: str = "data"
    session_name: str = None
    auto_save: bool = True
    save_interval: int = 10  # seconds
    
    # Quality settings
    min_snr: float = 5.0
    max_artifact_ratio: float = 0.3
    data_validation: bool = True
    
    def __post_init__(self):
        if self.eeg_channels is None:
            self.eeg_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        
        if self.session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_name = f"bci_session_{timestamp}"


@dataclass
class DataSample:
    """Single data sample from all modalities"""
    timestamp: float
    eeg_data: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None
    video_frame: Optional[np.ndarray] = None
    quality_metrics: Optional[Dict] = None
    metadata: Optional[Dict] = None


@dataclass
class DataQuality:
    """Data quality metrics"""
    timestamp: float
    eeg_snr: float = 0.0
    eeg_artifact_ratio: float = 0.0
    audio_level: float = 0.0
    video_brightness: float = 0.0
    overall_quality: str = "unknown"
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class DataCollector:
    """
    Multi-modal data collector for BCI applications
    
    This class handles:
    1. Real-time EEG data acquisition
    2. Audio recording and processing
    3. Video capture and analysis
    4. Data synchronization and storage
    5. Quality monitoring and validation
    """
    
    def __init__(self, config: Optional[DataConfig] = None, 
                 data_callback: Optional[Callable] = None):
        """
        Initialize data collector
        
        Args:
            config: Data collection configuration
            data_callback: Callback function for processed data
        """
        self.config = config or DataConfig()
        self.data_callback = data_callback
        
        # Data buffers
        self.eeg_buffer = queue.Queue(maxsize=self.config.eeg_buffer_size)
        self.audio_buffer = queue.Queue(maxsize=self.config.audio_buffer_size)
        self.video_buffer = queue.Queue(maxsize=self.config.video_buffer_size)
        
        # Collection threads
        self.eeg_thread = None
        self.audio_thread = None
        self.video_thread = None
        self.processing_thread = None
        
        # Control flags
        self.is_collecting = False
        self.is_processing = False
        
        # Data storage
        self.session_data = []
        self.session_path = None
        self.last_save_time = time.time()
        
        # Quality monitoring
        self.quality_history = []
        self.current_quality = DataQuality(timestamp=time.time())
        
        # Initialize session directory
        self._init_session_directory()
        
        logger.info("Data Collector initialized successfully")
    
    def _init_session_directory(self) -> None:
        """Initialize session directory for data storage"""
        try:
            session_dir = Path(self.config.data_directory) / self.config.session_name
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (session_dir / "eeg").mkdir(exist_ok=True)
            (session_dir / "audio").mkdir(exist_ok=True)
            (session_dir / "video").mkdir(exist_ok=True)
            (session_dir / "metadata").mkdir(exist_ok=True)
            
            self.session_path = session_dir
            
            # Save configuration
            config_file = session_dir / "config.json"
            with open(config_file, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            
            logger.info(f"Session directory initialized: {session_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize session directory: {e}")
            raise
    
    def start_collection(self) -> bool:
        """
        Start data collection from all modalities
        
        Returns:
            bool: True if started successfully
        """
        try:
            if self.is_collecting:
                logger.warning("Data collection already running")
                return True
            
            self.is_collecting = True
            
            # Start EEG collection thread
            self.eeg_thread = threading.Thread(target=self._eeg_collection_loop, daemon=True)
            self.eeg_thread.start()
            
            # Start audio collection thread
            if self.config.audio_enabled:
                self.audio_thread = threading.Thread(target=self._audio_collection_loop, daemon=True)
                self.audio_thread.start()
            
            # Start video collection thread
            if self.config.video_enabled:
                self.video_thread = threading.Thread(target=self._video_collection_loop, daemon=True)
                self.video_thread.start()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            logger.info("Data collection started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start data collection: {e}")
            self.is_collecting = False
            return False
    
    def stop_collection(self) -> None:
        """Stop data collection and save remaining data"""
        try:
            self.is_collecting = False
            self.is_processing = False
            
            # Wait for threads to finish
            if self.eeg_thread and self.eeg_thread.is_alive():
                self.eeg_thread.join(timeout=2.0)
            
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=2.0)
            
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Save remaining data
            self._save_session_data()
            
            logger.info("Data collection stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping data collection: {e}")
    
    def _eeg_collection_loop(self) -> None:
        """EEG data collection loop"""
        try:
            logger.info("EEG collection loop started")
            
            while self.is_collecting:
                # Simulate EEG data collection
                eeg_sample = self._simulate_eeg_data()
                
                if eeg_sample is not None:
                    # Add timestamp
                    eeg_data = {
                        'timestamp': time.time(),
                        'data': eeg_sample,
                        'channels': self.config.eeg_channels
                    }
                    
                    # Add to buffer
                    try:
                        self.eeg_buffer.put_nowait(eeg_data)
                    except queue.Full:
                        # Remove oldest sample if buffer is full
                        try:
                            self.eeg_buffer.get_nowait()
                            self.eeg_buffer.put_nowait(eeg_data)
                        except queue.Empty:
                            pass
                
                # Sleep for sampling rate
                time.sleep(1.0 / self.config.eeg_sampling_rate)
            
            logger.info("EEG collection loop stopped")
            
        except Exception as e:
            logger.error(f"EEG collection loop error: {e}")
    
    def _audio_collection_loop(self) -> None:
        """Audio data collection loop"""
        try:
            logger.info("Audio collection loop started")
            
            while self.is_collecting:
                # Simulate audio data collection
                audio_sample = self._simulate_audio_data()
                
                if audio_sample is not None:
                    # Add timestamp
                    audio_data = {
                        'timestamp': time.time(),
                        'data': audio_sample,
                        'sample_rate': self.config.audio_sampling_rate
                    }
                    
                    # Add to buffer
                    try:
                        self.audio_buffer.put_nowait(audio_data)
                    except queue.Full:
                        # Remove oldest sample if buffer is full
                        try:
                            self.audio_buffer.get_nowait()
                            self.audio_buffer.put_nowait(audio_data)
                        except queue.Empty:
                            pass
                
                # Sleep for sampling rate
                time.sleep(1.0 / self.config.audio_sampling_rate)
            
            logger.info("Audio collection loop stopped")
            
        except Exception as e:
            logger.error(f"Audio collection loop error: {e}")
    
    def _video_collection_loop(self) -> None:
        """Video data collection loop"""
        try:
            logger.info("Video collection loop started")
            
            while self.is_collecting:
                # Simulate video frame capture
                video_frame = self._simulate_video_frame()
                
                if video_frame is not None:
                    # Add timestamp
                    video_data = {
                        'timestamp': time.time(),
                        'frame': video_frame,
                        'resolution': self.config.video_resolution
                    }
                    
                    # Add to buffer
                    try:
                        self.video_buffer.put_nowait(video_data)
                    except queue.Full:
                        # Remove oldest frame if buffer is full
                        try:
                            self.video_buffer.get_nowait()
                            self.video_buffer.put_nowait(video_data)
                        except queue.Empty:
                            pass
                
                # Sleep for frame rate
                time.sleep(1.0 / self.config.video_fps)
            
            logger.info("Video collection loop stopped")
            
        except Exception as e:
            logger.error(f"Video collection loop error: {e}")
    
    def _processing_loop(self) -> None:
        """Data processing and synchronization loop"""
        try:
            logger.info("Data processing loop started")
            
            while self.is_collecting:
                # Collect data from all modalities
                data_sample = self._collect_synchronized_data()
                
                if data_sample is not None:
                    # Assess data quality
                    quality = self._assess_data_quality(data_sample)
                    
                    # Update current quality
                    self.current_quality = quality
                    self.quality_history.append(quality)
                    
                    # Store sample with quality metrics
                    data_sample.quality_metrics = asdict(quality)
                    self.session_data.append(data_sample)
                    
                    # Call callback if provided
                    if self.data_callback:
                        try:
                            self.data_callback(data_sample)
                        except Exception as e:
                            logger.error(f"Data callback error: {e}")
                    
                    # Auto-save if enabled
                    if (self.config.auto_save and 
                        time.time() - self.last_save_time > self.config.save_interval):
                        self._save_session_data()
                        self.last_save_time = time.time()
                
                # Processing interval
                time.sleep(0.1)  # 100ms
            
            logger.info("Data processing loop stopped")
            
        except Exception as e:
            logger.error(f"Data processing loop error: {e}")
    
    def _collect_synchronized_data(self) -> Optional[DataSample]:
        """Collect synchronized data from all modalities"""
        try:
            # Get latest data from each buffer
            eeg_data = None
            audio_data = None
            video_data = None
            
            # EEG data
            try:
                eeg_data = self.eeg_buffer.get_nowait()
            except queue.Empty:
                pass
            
            # Audio data
            if self.config.audio_enabled:
                try:
                    audio_data = self.audio_buffer.get_nowait()
                except queue.Empty:
                    pass
            
            # Video data
            if self.config.video_enabled:
                try:
                    video_data = self.video_buffer.get_nowait()
                except queue.Empty:
                    pass
            
            # Create data sample if we have at least EEG data
            if eeg_data is not None:
                return DataSample(
                    timestamp=eeg_data['timestamp'],
                    eeg_data=eeg_data['data'],
                    audio_data=audio_data['data'] if audio_data else None,
                    video_frame=video_data['frame'] if video_data else None
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Data synchronization failed: {e}")
            return None
    
    def _assess_data_quality(self, data_sample: DataSample) -> DataQuality:
        """Assess quality of collected data"""
        try:
            quality = DataQuality(timestamp=data_sample.timestamp)
            
            # EEG quality assessment
            if data_sample.eeg_data is not None:
                eeg_quality = self._assess_eeg_quality(data_sample.eeg_data)
                quality.eeg_snr = eeg_quality['snr']
                quality.eeg_artifact_ratio = eeg_quality['artifact_ratio']
            
            # Audio quality assessment
            if data_sample.audio_data is not None:
                audio_quality = self._assess_audio_quality(data_sample.audio_data)
                quality.audio_level = audio_quality['level']
            
            # Video quality assessment
            if data_sample.video_frame is not None:
                video_quality = self._assess_video_quality(data_sample.video_frame)
                quality.video_brightness = video_quality['brightness']
            
            # Overall quality assessment
            quality.overall_quality = self._calculate_overall_quality(quality)
            quality.recommendations = self._generate_recommendations(quality)
            
            return quality
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return DataQuality(timestamp=data_sample.timestamp)
    
    def _assess_eeg_quality(self, eeg_data: np.ndarray) -> Dict:
        """Assess EEG data quality"""
        try:
            # Calculate SNR (Signal-to-Noise Ratio)
            signal_power = np.var(eeg_data)
            noise_power = np.var(np.diff(eeg_data, axis=0)) / 2
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
            
            # Calculate artifact ratio
            artifact_threshold = 100  # microvolts
            artifacts = np.sum(np.abs(eeg_data) > artifact_threshold)
            artifact_ratio = artifacts / eeg_data.size
            
            return {
                'snr': float(snr),
                'artifact_ratio': float(artifact_ratio)
            }
            
        except Exception as e:
            logger.error(f"EEG quality assessment failed: {e}")
            return {'snr': 0.0, 'artifact_ratio': 1.0}
    
    def _assess_audio_quality(self, audio_data: np.ndarray) -> Dict:
        """Assess audio data quality"""
        try:
            # Calculate audio level (RMS)
            audio_level = np.sqrt(np.mean(audio_data**2))
            
            return {
                'level': float(audio_level)
            }
            
        except Exception as e:
            logger.error(f"Audio quality assessment failed: {e}")
            return {'level': 0.0}
    
    def _assess_video_quality(self, video_frame: np.ndarray) -> Dict:
        """Assess video frame quality"""
        try:
            # Calculate brightness (mean pixel value)
            if len(video_frame.shape) == 3:
                # Color image
                brightness = np.mean(video_frame)
            else:
                # Grayscale image
                brightness = np.mean(video_frame)
            
            return {
                'brightness': float(brightness)
            }
            
        except Exception as e:
            logger.error(f"Video quality assessment failed: {e}")
            return {'brightness': 0.0}
    
    def _calculate_overall_quality(self, quality: DataQuality) -> str:
        """Calculate overall data quality rating"""
        try:
            # Check EEG quality
            eeg_good = (quality.eeg_snr >= self.config.min_snr and 
                       quality.eeg_artifact_ratio <= self.config.max_artifact_ratio)
            
            # Check audio quality (if enabled)
            audio_good = True
            if self.config.audio_enabled:
                audio_good = quality.audio_level > 0.01  # Minimum audio level
            
            # Check video quality (if enabled)
            video_good = True
            if self.config.video_enabled:
                video_good = quality.video_brightness > 10  # Minimum brightness
            
            # Determine overall quality
            if eeg_good and audio_good and video_good:
                return "excellent"
            elif eeg_good and (audio_good or video_good):
                return "good"
            elif eeg_good:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Overall quality calculation failed: {e}")
            return "unknown"
    
    def _generate_recommendations(self, quality: DataQuality) -> List[str]:
        """Generate recommendations for improving data quality"""
        recommendations = []
        
        try:
            # EEG recommendations
            if quality.eeg_snr < self.config.min_snr:
                recommendations.append("Increase EEG signal strength or reduce noise")
            
            if quality.eeg_artifact_ratio > self.config.max_artifact_ratio:
                recommendations.append("Check electrode connections and reduce movement")
            
            # Audio recommendations
            if self.config.audio_enabled and quality.audio_level < 0.01:
                recommendations.append("Check microphone connection and positioning")
            
            # Video recommendations
            if self.config.video_enabled and quality.video_brightness < 10:
                recommendations.append("Improve lighting conditions")
            
            # General recommendations
            if not recommendations:
                recommendations.append("Data quality is good, continue collection")
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations = ["Error generating recommendations"]
        
        return recommendations
    
    def _save_session_data(self) -> None:
        """Save collected session data to disk"""
        try:
            if not self.session_path or not self.session_data:
                return
            
            # Save EEG data
            eeg_file = self.session_path / "eeg" / f"eeg_{int(time.time())}.npy"
            eeg_data = [sample.eeg_data for sample in self.session_data if sample.eeg_data is not None]
            if eeg_data:
                np.save(eeg_file, np.array(eeg_data))
            
            # Save metadata
            metadata_file = self.session_path / "metadata" / f"metadata_{int(time.time())}.json"
            metadata = []
            for sample in self.session_data:
                sample_meta = {
                    'timestamp': sample.timestamp,
                    'quality_metrics': sample.quality_metrics,
                    'has_eeg': sample.eeg_data is not None,
                    'has_audio': sample.audio_data is not None,
                    'has_video': sample.video_frame is not None
                }
                metadata.append(sample_meta)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save quality history
            quality_file = self.session_path / "metadata" / f"quality_{int(time.time())}.json"
            quality_data = [asdict(q) for q in self.quality_history]
            with open(quality_file, 'w') as f:
                json.dump(quality_data, f, indent=2)
            
            logger.info(f"Session data saved: {len(self.session_data)} samples")
            
            # Clear session data after saving
            self.session_data.clear()
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def get_current_quality(self) -> DataQuality:
        """Get current data quality metrics"""
        return self.current_quality
    
    def get_quality_history(self) -> List[DataQuality]:
        """Get data quality history"""
        return self.quality_history.copy()
    
    def get_session_info(self) -> Dict:
        """Get session information and statistics"""
        try:
            info = {
                'session_name': self.config.session_name,
                'session_path': str(self.session_path) if self.session_path else None,
                'is_collecting': self.is_collecting,
                'total_samples': len(self.session_data),
                'quality_history_length': len(self.quality_history),
                'current_quality': asdict(self.current_quality),
                'config': asdict(self.config)
            }
            return info
            
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {}
    
    def _simulate_eeg_data(self) -> Optional[np.ndarray]:
        """Simulate EEG data for testing"""
        try:
            # Generate random EEG data (19 channels)
            num_channels = len(self.config.eeg_channels)
            eeg_data = np.random.normal(0, 10, (num_channels, 1))  # 1 sample per channel
            
            # Add some realistic structure
            for i in range(num_channels):
                # Add alpha rhythm (8-13 Hz) to some channels
                if i % 3 == 0:
                    alpha_freq = 10 + np.random.normal(0, 1)  # Hz
                    alpha_phase = 2 * np.pi * np.random.random()
                    eeg_data[i, 0] += 5 * np.sin(alpha_phase)
            
            return eeg_data
            
        except Exception as e:
            logger.error(f"EEG simulation failed: {e}")
            return None
    
    def _simulate_audio_data(self) -> Optional[np.ndarray]:
        """Simulate audio data for testing"""
        try:
            # Generate random audio data
            audio_data = np.random.normal(0, 0.1, self.config.audio_buffer_size)
            
            # Add some realistic structure (speech-like)
            t = np.linspace(0, 1, self.config.audio_buffer_size)
            speech_component = 0.05 * np.sin(2 * np.pi * 200 * t)  # 200 Hz tone
            audio_data += speech_component
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio simulation failed: {e}")
            return None
    
    def _simulate_video_frame(self) -> Optional[np.ndarray]:
        """Simulate video frame for testing"""
        try:
            # Generate random video frame (grayscale)
            height, width = self.config.video_resolution
            video_frame = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            
            # Add some realistic structure
            # Create a simple face-like pattern
            center_y, center_x = height // 2, width // 2
            
            # Eyes
            eye_y = center_y - 20
            video_frame[eye_y-5:eye_y+5, center_x-15:center_x-5] = 100
            video_frame[eye_y-5:eye_y+5, center_x+5:center_x+15] = 100
            
            # Mouth
            mouth_y = center_y + 20
            video_frame[mouth_y-3:mouth_y+3, center_x-10:center_x+10] = 80
            
            return video_frame
            
        except Exception as e:
            logger.error(f"Video simulation failed: {e}")
            return None


if __name__ == "__main__":
    # Test the data collector
    config = DataConfig()
    collector = DataCollector(config)
    
    print("Starting data collection test...")
    
    # Start collection
    if collector.start_collection():
        print("Data collection started successfully")
        
        # Run for a few seconds
        time.sleep(5)
        
        # Stop collection
        collector.stop_collection()
        print("Data collection stopped")
        
        # Get session info
        info = collector.get_session_info()
        print(f"Session info: {info}")
        
        # Get quality metrics
        quality = collector.get_current_quality()
        print(f"Current quality: {quality.overall_quality}")
        print(f"EEG SNR: {quality.eeg_snr:.2f}")
        print(f"EEG artifacts: {quality.eeg_artifact_ratio:.3f}")
        
        if quality.recommendations:
            print("Recommendations:")
            for rec in quality.recommendations:
                print(f"  - {rec}")
    else:
        print("Failed to start data collection")
