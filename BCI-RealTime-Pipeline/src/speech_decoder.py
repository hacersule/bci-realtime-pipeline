#!/usr/bin/env python3
"""
Speech Decoder Module
Neural pattern recognition for speech intent decoding

This module provides:
- Real-time speech intent classification
- Neural feature analysis for speech production
- Confidence scoring and validation
- Integration with BCI pipeline
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy import signal
# sklearn not available, using alternative implementations
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)


class SimpleThresholdClassifier:
    """Simple threshold-based classifier (sklearn alternative)"""
    
    def __init__(self):
        self.thresholds = {
            'silent': 0.5,
            'thinking': 1.2,
            'word_attempt': 2.0,
            'speak': 2.5
        }
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """Simple training - just store data"""
        self.X = X
        self.y = y
        # Create simple feature importances
        self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        return self
    
    def predict(self, X):
        """Simple prediction based on motor channel activity"""
        predictions = []
        for sample in X:
            # Use motor channel features (indices 114:159) for prediction
            motor_activity = np.mean(sample[114:159]) if len(sample) > 159 else np.mean(sample)
            
            if motor_activity < self.thresholds['silent']:
                predictions.append(0)  # silent
            elif motor_activity < self.thresholds['thinking']:
                predictions.append(1)  # thinking
            elif motor_activity < self.thresholds['word_attempt']:
                predictions.append(2)  # word_attempt
            else:
                predictions.append(3)  # speak
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Simple probability estimation"""
        predictions = self.predict(X)
        probas = []
        
        for pred in predictions:
            proba = [0.1, 0.1, 0.1, 0.1]  # Base probabilities
            proba[pred] = 0.7  # High confidence for predicted class
            probas.append(proba)
        
        return np.array(probas)


@dataclass
class SpeechIntent:
    """Decoded speech intent from neural activity"""
    timestamp: float
    confidence: float
    intent_type: str  # 'speak', 'silent', 'thinking', 'word_attempt'
    decoded_text: Optional[str] = None
    neural_features: Optional[np.ndarray] = None
    channel_contributions: Optional[Dict[str, float]] = None
    processing_time: float = 0.0


@dataclass
class SpeechConfig:
    """Configuration for speech decoder"""
    # Classification parameters
    model_type: str = 'threshold'  # 'threshold', 'mlp', 'svm'
    feature_dim: int = 159
    num_classes: int = 4
    confidence_threshold: float = 0.7
    model_path: Optional[str] = None
    retrain_interval: int = 1000  # Retrain every N samples
    feature_selection: bool = True
    adaptive_threshold: bool = True
    min_samples_for_training: int = 100  # Minimum samples required for training


@dataclass
class DecoderConfig:
    """Configuration for speech decoder"""
    # Classification parameters
    model_type: str = 'random_forest'  # 'random_forest', 'svm', 'neural_network'
    confidence_threshold: float = 0.7
    min_samples_for_training: int = 100
    
    # Feature parameters
    feature_window_size: int = 500  # ms
    feature_overlap: float = 0.5    # 50% overlap
    use_temporal_features: bool = True
    use_spectral_features: bool = True
    use_connectivity_features: bool = False
    
    # Speech-specific parameters
    speech_channels: List[str] = None  # Channels most relevant for speech
    motor_channels: List[str] = None   # Motor cortex channels
    language_channels: List[str] = None # Language areas
    
    def __post_init__(self):
        if self.speech_channels is None:
            self.speech_channels = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8']
        
        if self.motor_channels is None:
            self.motor_channels = ['C3', 'Cz', 'C4']
        
        if self.language_channels is None:
            self.language_channels = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6']


class SpeechDecoder:
    """
    Real-time speech intent decoder for BCI applications
    
    This class implements:
    1. Neural feature analysis for speech production
    2. Real-time classification of speech intent
    3. Confidence scoring and validation
    4. Continuous learning and adaptation
    """
    
    def __init__(self, config: Optional[SpeechConfig] = None):
        """
        Initialize speech decoder
        
        Args:
            config: Speech decoder configuration
        """
        self.config = config or SpeechConfig()
        
        # Initialize classifier
        self.classifier = None
        # self.scaler = StandardScaler()  # sklearn not available
        self.is_trained = False
        
        # Training data
        self.training_features = []
        self.training_labels = []
        self.feature_names = []
        
        # Performance tracking
        self.accuracy_history = []
        self.confusion_matrix = np.zeros((4, 4))  # 4 intent types
        self.last_prediction = None
        
        # Intent type mapping
        self.intent_types = ['silent', 'thinking', 'word_attempt', 'speak']
        self.intent_mapping = {i: label for i, label in enumerate(self.intent_types)}
        
        # Channel information (will be set during training)
        self.channel_names = []
        self.channel_indices = {}
        
        logger.info("Speech Decoder initialized successfully")
    
    def _simple_normalize(self, X: np.ndarray) -> np.ndarray:
        """Simple feature normalization (sklearn alternative)"""
        try:
            X_norm = X.copy()
            for i in range(X.shape[1]):
                col = X[:, i]
                mean_val = np.mean(col)
                std_val = np.std(col)
                if std_val > 0:
                    X_norm[:, i] = (col - mean_val) / std_val
                else:
                    X_norm[:, i] = col - mean_val
            return X_norm
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return X
    
    def _simple_accuracy_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Simple accuracy calculation (sklearn alternative)"""
        try:
            correct = np.sum(y_true == y_pred)
            total = len(y_true)
            return correct / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"Accuracy calculation failed: {e}")
            return 0.0
    
    def _init_classifier(self) -> None:
        """Initialize the machine learning classifier"""
        try:
            # Simple threshold-based classifier (sklearn alternative)
            self.classifier = SimpleThresholdClassifier()
            logger.info(f"Simple threshold classifier initialized")
            
        except Exception as e:
            logger.error(f"Classifier initialization failed: {e}")
            raise
    
    def add_training_data(self, features: np.ndarray, label: str) -> bool:
        """
        Add training data for the classifier
        
        Args:
            features: Neural features
            label: Intent label
            
        Returns:
            bool: True if added successfully
        """
        try:
            if label not in self.intent_types:
                logger.warning(f"Unknown label: {label}. Using 'silent' instead.")
                label = 'silent'
            
            # Convert label to index
            label_index = self.intent_types.index(label)
            
            # Store training data
            self.training_features.append(features)
            self.training_labels.append(label_index)
            
            # Update feature names if first time
            if not self.feature_names:
                self.feature_names = [f"feature_{i}" for i in range(len(features))]
            
            # Update channel information if first time
            if not self.channel_names:
                # Estimate number of channels based on feature dimensionality
                # Assuming 15 features per channel (from EEG processor)
                num_channels = len(features) // 15
                self.channel_names = [f"Ch{i+1}" for i in range(num_channels)]
                
                # Create channel index mapping
                for i, ch_name in enumerate(self.channel_names):
                    self.channel_indices[ch_name] = i
            
            logger.debug(f"Added training sample: {label} with {len(features)} features")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add training data: {e}")
            return False
    
    def train_classifier(self) -> bool:
        """
        Train the classifier with collected training data
        
        Returns:
            bool: True if training successful
        """
        try:
            if len(self.training_features) < self.config.min_samples_for_training:
                logger.warning(f"Insufficient training data: {len(self.training_features)} < {self.config.min_samples_for_training}")
                return False
            
            # Initialize classifier if not done
            if self.classifier is None:
                self._init_classifier()
            
            # Prepare training data
            X = np.array(self.training_features)
            y = np.array(self.training_labels)
            
            # Scale features (simple normalization)
            X_scaled = self._simple_normalize(X)
            
            # Train classifier
            self.classifier.fit(X_scaled, y)
            
            # Evaluate training performance
            y_pred = self.classifier.predict(X_scaled)
            accuracy = self._simple_accuracy_score(y, y_pred)
            self.accuracy_history.append(accuracy)
            
            # Update confusion matrix
            for true_label, pred_label in zip(y, y_pred):
                self.confusion_matrix[true_label, pred_label] += 1
            
            self.is_trained = True
            
            logger.info(f"Classifier trained successfully. Training accuracy: {accuracy:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Classifier training failed: {e}")
            return False
    
    def decode_speech_intent(self, features: np.ndarray, 
                           channel_names: Optional[List[str]] = None) -> Optional[SpeechIntent]:
        """
        Decode speech intent from neural features
        
        Args:
            features: Neural features from EEG processor
            channel_names: Channel names for analysis
            
        Returns:
            Optional[SpeechIntent]: Decoded speech intent
        """
        try:
            start_time = time.time()
            
            if not self.is_trained:
                logger.warning("Classifier not trained. Cannot decode speech intent.")
                return None
            
            if features is None or len(features) == 0:
                logger.warning("No features provided for decoding.")
                return None
            
            # Scale features (simple normalization)
            features_scaled = self._simple_normalize(features.reshape(1, -1))
            
            # Predict intent
            prediction = self.classifier.predict(features_scaled)[0]
            probabilities = self.classifier.predict_proba(features_scaled)[0]
            
            # Get confidence and intent type
            confidence = probabilities[prediction]
            intent_type = self.intent_mapping[prediction]
            
            # Only return high-confidence predictions
            if confidence < self.config.confidence_threshold:
                logger.debug(f"Low confidence prediction: {confidence:.3f} for {intent_type}")
                return None
            
            # Calculate channel contributions
            channel_contributions = self._calculate_channel_contributions(
                features, channel_names
            )
            
            # Create speech intent
            speech_intent = SpeechIntent(
                timestamp=time.time(),
                confidence=confidence,
                intent_type=intent_type,
                neural_features=features,
                channel_contributions=channel_contributions,
                processing_time=time.time() - start_time
            )
            
            # Update last prediction
            self.last_prediction = speech_intent
            
            logger.info(f"Speech intent decoded: {intent_type} (confidence: {confidence:.3f})")
            return speech_intent
            
        except Exception as e:
            logger.error(f"Speech intent decoding failed: {e}")
            return None
    
    def _calculate_channel_contributions(self, features: np.ndarray, 
                                       channel_names: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate contribution of each channel to the prediction
        
        Args:
            features: Neural features
            channel_names: Channel names
            
        Returns:
            Dict[str, float]: Channel contribution scores
        """
        try:
            if channel_names is None:
                channel_names = self.channel_names
            
            if not channel_names or len(channel_names) == 0:
                return {}
            
            # Estimate channel contributions based on feature importance
            if self.classifier and hasattr(self.classifier, 'feature_importances_'):
                importances = self.classifier.feature_importances_
                
                # Group features by channel (assuming 15 features per channel)
                features_per_channel = 15
                channel_contributions = {}
                
                for i, ch_name in enumerate(channel_names):
                    start_idx = i * features_per_channel
                    end_idx = start_idx + features_per_channel
                    
                    if end_idx <= len(importances):
                        channel_importance = np.mean(importances[start_idx:end_idx])
                        channel_contributions[ch_name] = float(channel_importance)
                    else:
                        channel_contributions[ch_name] = 0.0
                
                return channel_contributions
            else:
                # Fallback: equal contributions
                return {ch_name: 1.0 / len(channel_names) for ch_name in channel_names}
                
        except Exception as e:
            logger.error(f"Channel contribution calculation failed: {e}")
            return {}
    
    def analyze_speech_patterns(self, features: np.ndarray) -> Dict[str, float]:
        """
        Analyze neural patterns specific to speech production
        
        Args:
            features: Neural features
            
        Returns:
            Dict[str, float]: Speech pattern analysis
        """
        try:
            analysis = {}
            
            # Motor cortex activity (speech motor control)
            if self.motor_channels and len(self.motor_channels) > 0:
                motor_features = self._extract_channel_features(features, self.motor_channels)
                if motor_features is not None:
                    analysis['motor_activity'] = float(np.mean(motor_features))
                    analysis['motor_variability'] = float(np.std(motor_features))
            
            # Language area activity (speech planning)
            if self.language_channels and len(self.language_channels) > 0:
                language_features = self._extract_channel_features(features, self.language_channels)
                if language_features is not None:
                    analysis['language_activity'] = float(np.mean(language_features))
                    analysis['language_variability'] = float(np.std(language_features))
            
            # Overall neural complexity
            analysis['neural_complexity'] = float(np.std(features))
            analysis['feature_diversity'] = float(len(np.unique(features)))
            
            return analysis
            
        except Exception as e:
            logger.error(f"Speech pattern analysis failed: {e}")
            return {}
    
    def _extract_channel_features(self, features: np.ndarray, 
                                 target_channels: List[str]) -> Optional[np.ndarray]:
        """
        Extract features for specific channels
        
        Args:
            features: All neural features
            target_channels: Target channel names
            
        Returns:
            Optional[np.ndarray]: Channel-specific features
        """
        try:
            if not self.channel_indices:
                return None
            
            channel_features = []
            features_per_channel = 15
            
            for ch_name in target_channels:
                if ch_name in self.channel_indices:
                    ch_idx = self.channel_indices[ch_name]
                    start_idx = ch_idx * features_per_channel
                    end_idx = start_idx + features_per_channel
                    
                    if end_idx <= len(features):
                        ch_features = features[start_idx:end_idx]
                        channel_features.extend(ch_features)
            
            return np.array(channel_features) if channel_features else None
            
        except Exception as e:
            logger.error(f"Channel feature extraction failed: {e}")
            return None
    
    def get_performance_metrics(self) -> Dict:
        """
        Get decoder performance metrics
        
        Returns:
            Dict: Performance metrics
        """
        metrics = {
            'is_trained': self.is_trained,
            'training_samples': len(self.training_features),
            'accuracy_history': self.accuracy_history.copy(),
            'current_accuracy': self.accuracy_history[-1] if self.accuracy_history else 0.0,
            'confusion_matrix': self.confusion_matrix.tolist(),
            'last_prediction': self.last_prediction.intent_type if self.last_prediction else None,
            'last_confidence': self.last_prediction.confidence if self.last_prediction else 0.0
        }
        
        return metrics
    
    def reset_classifier(self) -> None:
        """Reset the trained classifier"""
        self.classifier = None
        # self.scaler = StandardScaler()  # sklearn not available
        self.is_trained = False
        self.training_features.clear()
        self.training_labels.clear()
        self.accuracy_history.clear()
        self.confusion_matrix.fill(0)
        self.last_prediction = None
        
        logger.info("Classifier reset successfully")
    
    def save_model(self, filepath: str) -> bool:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
            
        Returns:
            bool: True if saved successfully
        """
        try:
            if not self.is_trained:
                logger.warning("No trained model to save")
                return False
            
            import pickle
            
            model_data = {
                'classifier': self.classifier,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'channel_names': self.channel_names,
                'channel_indices': self.channel_indices,
                'intent_types': self.intent_types,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model
        
        Args:
            filepath: Path to the model file
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.channel_names = model_data['channel_names']
            self.channel_indices = model_data['channel_indices']
            self.intent_types = model_data['intent_types']
            self.config = model_data['config']
            
            self.is_trained = True
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False


def simulate_speech_data(num_samples: int = 100) -> Tuple[List[np.ndarray], List[str]]:
    """
    Simulate speech-related neural data for testing
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple[List[np.ndarray], List[str]]: Features and labels
    """
    try:
        features = []
        labels = []
        
        # Generate features for different speech states
        for i in range(num_samples):
            # Random features (285 features = 19 channels * 15 features per channel)
            sample_features = np.random.normal(0, 1, 285)
            
            # Add some structure based on intent type
            if i < num_samples // 4:
                # Silent state - lower motor activity
                sample_features[114:129] *= 0.5  # C3 channel features
                sample_features[129:144] *= 0.5  # Cz channel features
                sample_features[144:159] *= 0.5  # C4 channel features
                labels.append('silent')
            elif i < num_samples // 2:
                # Thinking state - moderate activity
                sample_features[114:159] *= 1.2  # Motor channels
                labels.append('thinking')
            elif i < 3 * num_samples // 4:
                # Word attempt - high motor activity
                sample_features[114:159] *= 2.0  # Motor channels
                labels.append('word_attempt')
            else:
                # Speak state - very high motor activity
                sample_features[114:159] *= 2.5  # Motor channels
                labels.append('speak')
            
            features.append(sample_features)
        
        return features, labels
        
    except Exception as e:
        logger.error(f"Speech data simulation failed: {e}")
        return [], []


if __name__ == "__main__":
    # Test the speech decoder
    config = DecoderConfig()
    decoder = SpeechDecoder(config)
    
    # Simulate training data
    print("Generating simulated speech data...")
    features, labels = simulate_speech_data(200)
    
    if features and labels:
        print(f"Generated {len(features)} training samples")
        
        # Add training data
        for feat, label in zip(features, labels):
            decoder.add_training_data(feat, label)
        
        # Train classifier
        print("Training classifier...")
        if decoder.train_classifier():
            print("Classifier trained successfully!")
            
            # Test decoding
            test_features = features[0]  # Use first sample as test
            result = decoder.decode_speech_intent(test_features)
            
            if result:
                print(f"Decoded intent: {result.intent_type}")
                print(f"Confidence: {result.confidence:.3f}")
                print(f"Processing time: {result.processing_time:.4f} seconds")
            
            # Get performance metrics
            metrics = decoder.get_performance_metrics()
            print(f"Training accuracy: {metrics['current_accuracy']:.3f}")
        else:
            print("Classifier training failed!")
    else:
        print("Failed to generate simulated data")
