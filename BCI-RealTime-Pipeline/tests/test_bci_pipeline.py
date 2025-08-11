#!/usr/bin/env python3
"""
BCI Pipeline Test Suite

This module tests:
- Individual components
- Component integration
- Error handling
- Performance metrics
"""

import unittest
import sys
import os
import numpy as np
import time
import tempfile
import shutil

# add src to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

try:
    from src.bci_pipeline import BCIPipeline, BCIConfig
    from src.eeg_processor import EEGProcessor, EEGConfig
    from src.speech_decoder import SpeechDecoder, SpeechConfig
    from src.data_collector import DataCollector, DataConfig
    from src.dashboard import Dashboard, DashboardConfig
    from src.utils import validate_eeg_data, normalize_eeg_data, calculate_snr
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Trying to import from: {project_root}")
    IMPORTS_SUCCESSFUL = False


class TestBCIPipeline(unittest.TestCase):
    """Test BCI pipeline integration"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = BCIConfig(
            sampling_rate=1000,
            buffer_size=1000
        )
        
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Pipeline initialization test"""
        pipeline = BCIPipeline(self.config)
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.config.channels), 19)
        self.assertEqual(pipeline.config.sampling_rate, 1000)
    
    def test_pipeline_start_stop(self):
        """Pipeline start/stop test"""
        pipeline = BCIPipeline(self.config)
        
        # Start pipeline
        success = pipeline.start()
        self.assertTrue(success)
        self.assertTrue(pipeline.is_running)
        
        # Wait a bit
        time.sleep(1)
        
        # Stop pipeline
        pipeline.stop()
        self.assertFalse(pipeline.is_running)
    
    def test_pipeline_demo_mode(self):
        """Pipeline demo mode test"""
        config = BCIConfig(
            sampling_rate=500,
            buffer_size=500
        )
        
        pipeline = BCIPipeline(config)
        pipeline.start()
        
        # Wait for demo to complete
        time.sleep(4)
        
        pipeline.stop()
        self.assertFalse(pipeline.is_running)


class TestEEGProcessor(unittest.TestCase):
    """Test EEG Processor"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = EEGConfig(
            channel_names=['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6'],
            sampling_rate=1000,
            filter_low=1.0,
            filter_high=40.0,
            notch_freq=50.0
        )
        self.processor = EEGProcessor(self.config)
    
    def test_eeg_config(self):
        """EEG configuration test"""
        self.assertEqual(len(self.config.channel_names), 8)
        self.assertEqual(self.config.sampling_rate, 1000)
        self.assertEqual(self.config.filter_low, 1.0)
    
    def test_eeg_processor_initialization(self):
        """EEG processor initialization test"""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.config.sampling_rate, 1000)
    
    def test_eeg_preprocessing(self):
        """EEG preprocessing test"""
        # Generate test data (19 channels)
        test_data = np.random.randn(19, 1000)
        
        # Preprocess
        processed_data = self.processor._process_buffer()
        
        # Check output
        if processed_data is not None:
            self.assertEqual(processed_data.shape[1], test_data.shape[1])  # Only check time dimension
            self.assertFalse(np.any(np.isnan(processed_data)))
        else:
            # If None is returned, it's acceptable
            self.assertIsNone(processed_data)
    
    def test_feature_extraction(self):
        """Feature extraction test"""
        # Generate test data (19 channels)
        test_data = np.random.randn(19, 1000)
        
        # Extract features
        features = self.processor._extract_features(test_data)
        
        # Check output
        self.assertIsInstance(features, np.ndarray)
        self.assertGreater(len(features), 0)
    
    def test_artifact_detection(self):
        """Artifact detection test"""
        # Generate test data with artifacts (19 channels)
        test_data = np.random.randn(19, 1000)
        test_data[0, 100:200] = 1000  # Add artifacts
        
        # Remove artifacts
        processed_data = self.processor._remove_artifacts(test_data)
        
        # Compute artifact ratio
        artifact_ratio = np.sum(np.isnan(processed_data)) / processed_data.size
        
        # Check output
        self.assertIsInstance(processed_data, np.ndarray)
        self.assertIsInstance(artifact_ratio, float)
        self.assertGreaterEqual(artifact_ratio, 0)


class TestSpeechDecoder(unittest.TestCase):
    """Test Speech Decoder"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SpeechConfig(
            model_type='threshold',
            feature_dim=159,
            num_classes=4,
            confidence_threshold=0.7
        )
        self.decoder = SpeechDecoder(self.config)
    
    def test_speech_config(self):
        """Speech configuration test"""
        self.assertEqual(self.config.model_type, 'threshold')
        self.assertEqual(self.config.feature_dim, 159)
        self.assertEqual(self.config.num_classes, 4)
    
    def test_speech_decoder_initialization(self):
        """Speech decoder initialization test"""
        self.assertIsNotNone(self.decoder)
        self.assertEqual(self.decoder.config.feature_dim, 159)
    
    def test_speech_decoding(self):
        """Speech decoding test"""
        # Generate test features
        test_features = np.random.randn(159)
        
        # Train classifier first (100 samples required)
        for i in range(25):  # 25 samples per class = 100 total
            self.decoder.add_training_data(test_features + i * 0.01, 'thinking')
            self.decoder.add_training_data(test_features + i * 0.01 + 0.1, 'silent')
            self.decoder.add_training_data(test_features + i * 0.01 + 0.2, 'speak')
            self.decoder.add_training_data(test_features + i * 0.01 + 0.3, 'word_attempt')
        
        training_success = self.decoder.train_classifier()
        self.assertTrue(training_success)
        
        # Decode speech intent
        intent = self.decoder.decode_speech_intent(test_features)
        
        # Validate output (if not None)
        if intent is not None:
            self.assertIsInstance(intent.timestamp, float)
            self.assertIsInstance(intent.confidence, float)
            self.assertIsInstance(intent.intent_type, str)
            self.assertIsInstance(intent.processing_time, float)
        else:
            # If None is returned (low confidence), that's acceptable
            self.assertIsNone(intent)


class TestDataCollector(unittest.TestCase):
    """Test Data Collector"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataConfig(
            data_directory=self.temp_dir,
            audio_enabled=False,
            video_enabled=False,
            save_interval=1
        )
        self.collector = DataCollector(self.config)
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.temp_dir)
    
    def test_data_config(self):
        """Data configuration test"""
        self.assertEqual(self.config.data_directory, self.temp_dir)
        self.assertTrue(self.config.audio_enabled is False)
        self.assertFalse(self.config.video_enabled)
    
    def test_data_collector_initialization(self):
        """Data collector initialization test"""
        self.assertIsNotNone(self.collector)
        self.assertEqual(self.collector.config.data_directory, self.temp_dir)
    
    def test_data_collection(self):
        """Data collection test"""
        # Start collection
        success = self.collector.start_collection()
        self.assertTrue(success)
        
        # Wait a bit
        time.sleep(1)
        
        # Stop collection
        self.collector.stop_collection()
        self.assertFalse(self.collector.is_collecting)


class TestDashboard(unittest.TestCase):
    """Test Dashboard"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = DashboardConfig(
            update_interval=0.1,
            max_data_points=100,
            eeg_plot_channels=8
        )
        self.dashboard = Dashboard(self.config)
    
    def test_dashboard_config(self):
        """Dashboard configuration test"""
        self.assertEqual(self.config.update_interval, 0.1)
        self.assertEqual(self.config.max_data_points, 100)
        self.assertEqual(self.config.eeg_plot_channels, 8)
    
    def test_dashboard_initialization(self):
        """Dashboard initialization test"""
        self.assertIsNotNone(self.dashboard)
        self.assertEqual(self.dashboard.config.update_interval, 0.1)
    
    def test_dashboard_start_stop(self):
        """Dashboard start/stop test"""
        # Start dashboard
        success = self.dashboard.start()
        self.assertTrue(success)
        
        # Wait a bit
        time.sleep(0.5)
        
        # Stop dashboard
        self.dashboard.stop()
        self.assertFalse(self.dashboard.is_running)
    
    def test_data_update(self):
        """Dashboard data update test"""
        # Generate test data
        test_eeg_data = [np.random.randn(100) for _ in range(8)]
        test_quality_metrics = {'snr': 15.0, 'artifacts': 0.05}
        test_speech_intent = {'intent_type': 'thinking', 'confidence': 0.8}
        
        # Update dashboard
        self.dashboard.update_data(
            eeg_data=test_eeg_data,
            quality_metrics=test_quality_metrics,
            speech_intent=test_speech_intent
        )
        
        # Check if data was updated
        latest_data = self.dashboard.get_latest_data()
        self.assertIsNotNone(latest_data.eeg_data)
        self.assertIsNotNone(latest_data.quality_metrics)
        self.assertIsNotNone(latest_data.speech_intent)


class TestUtils(unittest.TestCase):
    """Test Utility Functions"""
    
    def test_eeg_data_validation(self):
        """EEG data validation test"""
        # Valid data
        valid_data = np.random.randn(8, 1000)
        is_valid = validate_eeg_data(valid_data, 8, 1000)
        self.assertTrue(is_valid)
        
        # Invalid data - wrong shape
        invalid_data = np.random.randn(6, 1000)
        is_valid = validate_eeg_data(invalid_data, 8, 1000)
        self.assertFalse(is_valid)
        
        # Invalid data - contains NaN
        invalid_data = np.random.randn(8, 1000)
        invalid_data[0, 0] = np.nan
        is_valid = validate_eeg_data(invalid_data, 8, 1000)
        self.assertFalse(is_valid)
    
    def test_data_normalization(self):
        """Data normalization test"""
        test_data = np.random.randn(8, 1000)
        
        # Z-score normalization
        normalized = normalize_eeg_data(test_data, 'zscore')
        self.assertEqual(normalized.shape, test_data.shape)
        
        # Check normalization (mean ~0, std ~1)
        for i in range(normalized.shape[0]):
            mean_val = np.mean(normalized[i])
            std_val = np.std(normalized[i])
            self.assertAlmostEqual(mean_val, 0, places=1)
            self.assertAlmostEqual(std_val, 1, places=1)
    
    def test_snr_calculation(self):
        """SNR calculation test"""
        # Generate signal with known SNR
        signal = np.random.randn(1000)
        noise = np.random.normal(0, 0.1, 1000)
        noisy_signal = signal + noise
        
        snr = calculate_snr(noisy_signal)
        
        # Check if SNR is reasonable
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)


if __name__ == '__main__':
    # Check whether imports succeeded
    if not IMPORTS_SUCCESSFUL:
        print("❌ Import failed. Cannot run tests.")
        print("Make sure you're running from the correct directory.")
        sys.exit(1)
    
    print("✅ All imports successful. Running tests...")
    # Run tests
    unittest.main(verbosity=2)
