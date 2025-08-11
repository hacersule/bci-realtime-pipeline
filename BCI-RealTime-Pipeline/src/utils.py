#!/usr/bin/env python3
"""
Utility Functions Module
Common utilities for BCI applications

This module provides:
- Data validation and conversion
- File I/O operations
- Time utilities
- Mathematical helpers
- Configuration management
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def validate_eeg_data(data: np.ndarray, expected_channels: int = 8, 
                     expected_samples: Optional[int] = None) -> bool:
    """
    Validate EEG data format and quality
    
    Args:
        data: EEG data array
        expected_channels: Expected number of channels
        expected_samples: Expected number of samples per channel
        
    Returns:
        bool: True if data is valid
    """
    try:
        if not isinstance(data, np.ndarray):
            logger.error("Data must be numpy array")
            return False
        
        if data.ndim != 2:
            logger.error(f"Data must be 2D, got {data.ndim}D")
            return False
        
        if data.shape[0] != expected_channels:
            logger.error(f"Expected {expected_channels} channels, got {data.shape[0]}")
            return False
        
        if expected_samples and data.shape[1] != expected_samples:
            logger.error(f"Expected {expected_samples} samples, got {data.shape[1]}")
            return False
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.error("Data contains NaN or infinite values")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return False


def normalize_eeg_data(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize EEG data using specified method
    
    Args:
        data: Input EEG data
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        np.ndarray: Normalized data
    """
    try:
        if method == 'zscore':
            # Z-score normalization
            mean_val = np.mean(data, axis=1, keepdims=True)
            std_val = np.std(data, axis=1, keepdims=True)
            std_val[std_val == 0] = 1  # Avoid division by zero
            return (data - mean_val) / std_val
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  # Avoid division by zero
            return (data - min_val) / range_val
            
        elif method == 'robust':
            # Robust normalization using median and MAD
            median_val = np.median(data, axis=1, keepdims=True)
            mad_val = np.median(np.abs(data - median_val), axis=1, keepdims=True)
            mad_val[mad_val == 0] = 1  # Avoid division by zero
            return (data - median_val) / mad_val
            
        else:
            logger.warning(f"Unknown normalization method: {method}, using zscore")
            return normalize_eeg_data(data, 'zscore')
            
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return data


def calculate_snr(signal: np.ndarray, noise_floor: Optional[float] = None) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        signal: Input signal
        noise_floor: Noise floor level (if None, estimated from data)
        
    Returns:
        float: SNR in dB
    """
    try:
        if noise_floor is None:
            # Estimate noise floor from lower percentiles
            noise_floor = np.percentile(np.abs(signal), 10)
        
        signal_power = np.mean(signal ** 2)
        noise_power = noise_floor ** 2
        
        if noise_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return float(snr_db)
        
    except Exception as e:
        logger.error(f"SNR calculation error: {e}")
        return 0.0


def detect_artifacts(data: np.ndarray, threshold: float = 3.0, 
                    method: str = 'amplitude') -> Tuple[np.ndarray, float]:
    """
    Detect artifacts in EEG data
    
    Args:
        data: EEG data array
        threshold: Detection threshold
        method: Detection method ('amplitude', 'statistical', 'variance')
        
    Returns:
        Tuple[np.ndarray, float]: Artifact mask and artifact ratio
    """
    try:
        if method == 'amplitude':
            # Amplitude-based artifact detection
            artifact_mask = np.abs(data) > threshold
            
        elif method == 'statistical':
            # Statistical outlier detection
            z_scores = np.abs((data - np.mean(data, axis=1, keepdims=True)) / 
                             np.std(data, axis=1, keepdims=True))
            artifact_mask = z_scores > threshold
            
        elif method == 'variance':
            # Variance-based detection
            rolling_var = np.var(data, axis=1, keepdims=True)
            var_threshold = np.mean(rolling_var) + threshold * np.std(rolling_var)
            artifact_mask = rolling_var > var_threshold
            
        else:
            logger.warning(f"Unknown artifact detection method: {method}")
            artifact_mask = np.zeros_like(data, dtype=bool)
        
        artifact_ratio = np.mean(artifact_mask)
        return artifact_mask, artifact_ratio
        
    except Exception as e:
        logger.error(f"Artifact detection error: {e}")
        return np.zeros_like(data, dtype=bool), 0.0


def interpolate_artifacts(data: np.ndarray, artifact_mask: np.ndarray, 
                         method: str = 'linear') -> np.ndarray:
    """
    Interpolate artifact-contaminated data
    
    Args:
        data: EEG data array
        artifact_mask: Boolean mask of artifacts
        method: Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        np.ndarray: Cleaned data
    """
    try:
        cleaned_data = data.copy()
        
        for channel_idx in range(data.shape[0]):
            channel_data = data[channel_idx]
            channel_mask = artifact_mask[channel_idx]
            
            if np.any(channel_mask):
                # Find artifact indices
                artifact_indices = np.where(channel_mask)[0]
                
                # Find clean indices
                clean_indices = np.where(~channel_mask)[0]
                
                if len(clean_indices) > 0:
                    # Interpolate artifacts
                    if method == 'linear':
                        cleaned_data[channel_idx, artifact_indices] = np.interp(
                            artifact_indices, clean_indices, channel_data[clean_indices]
                        )
                    elif method == 'cubic':
                        # Simple cubic interpolation (for small gaps)
                        for idx in artifact_indices:
                            # Find nearest clean neighbors
                            left_idx = clean_indices[clean_indices < idx]
                            right_idx = clean_indices[clean_indices > idx]
                            
                            if len(left_idx) > 0 and len(right_idx) > 0:
                                left_val = channel_data[left_idx[-1]]
                                right_val = channel_data[right_idx[0]]
                                # Simple linear interpolation
                                cleaned_data[channel_idx, idx] = (left_val + right_val) / 2
                            elif len(left_idx) > 0:
                                cleaned_data[channel_idx, idx] = channel_data[left_idx[-1]]
                            elif len(right_idx) > 0:
                                cleaned_data[channel_idx, idx] = channel_data[right_idx[0]]
                    else:  # nearest
                        for idx in artifact_indices:
                            # Find nearest clean neighbor
                            distances = np.abs(clean_indices - idx)
                            nearest_idx = clean_indices[np.argmin(distances)]
                            cleaned_data[channel_idx, idx] = channel_data[nearest_idx]
        
        return cleaned_data
        
    except Exception as e:
        logger.error(f"Artifact interpolation error: {e}")
        return data


def save_session_data(data: Dict[str, Any], filepath: str, 
                     format: str = 'pickle') -> bool:
    """
    Save session data to file
    
    Args:
        data: Data to save
        filepath: Output file path
        format: File format ('pickle', 'json', 'numpy')
        
    Returns:
        bool: True if successful
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
        elif format == 'json':
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_data[key] = int(value)
                elif isinstance(value, np.floating):
                    json_data[key] = float(value)
                else:
                    json_data[key] = value
            
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        elif format == 'numpy':
            np.savez_compressed(filepath, **data)
            
        else:
            logger.error(f"Unknown format: {format}")
            return False
        
        logger.info(f"Session data saved to {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save session data: {e}")
        return False


def load_session_data(filepath: str, format: str = 'auto') -> Optional[Dict[str, Any]]:
    """
    Load session data from file
    
    Args:
        filepath: Input file path
        format: File format ('auto', 'pickle', 'json', 'numpy')
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data or None
    """
    try:
        if format == 'auto':
            # Auto-detect format from file extension
            ext = Path(filepath).suffix.lower()
            if ext == '.pkl':
                format = 'pickle'
            elif ext == '.json':
                format = 'json'
            elif ext in ['.npz', '.npy']:
                format = 'numpy'
            else:
                format = 'pickle'  # Default
        
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
        elif format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                
        elif format == 'numpy':
            data = dict(np.load(filepath))
            
        else:
            logger.error(f"Unknown format: {format}")
            return None
        
        logger.info(f"Session data loaded from {filepath}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to load session data: {e}")
        return None


def create_session_directory(base_path: str, session_name: str = None) -> str:
    """
    Create session directory with timestamp
    
    Args:
        base_path: Base directory path
        session_name: Optional session name
        
    Returns:
        str: Created session directory path
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if session_name:
            session_dir = f"{session_name}_{timestamp}"
        else:
            session_dir = f"session_{timestamp}"
        
        full_path = os.path.join(base_path, session_dir)
        os.makedirs(full_path, exist_ok=True)
        
        # Create subdirectories
        os.makedirs(os.path.join(full_path, "eeg"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "audio"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "video"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "logs"), exist_ok=True)
        os.makedirs(os.path.join(full_path, "models"), exist_ok=True)
        
        logger.info(f"Session directory created: {full_path}")
        return full_path
        
    except Exception as e:
        logger.error(f"Failed to create session directory: {e}")
        return base_path


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    try:
        duration = timedelta(seconds=seconds)
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        elif duration.seconds >= 3600:
            return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        elif duration.seconds >= 60:
            return f"{duration.seconds//60}m {duration.seconds%60}s"
        else:
            return f"{seconds:.3f}s"
            
    except Exception as e:
        logger.error(f"Duration formatting error: {e}")
        return f"{seconds:.3f}s"


def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        required_keys: List of required keys
        
    Returns:
        bool: True if valid
    """
    try:
        for key in required_keys:
            if key not in config:
                logger.error(f"Missing required config key: {key}")
                return False
            
            if config[key] is None:
                logger.error(f"Config key {key} cannot be None")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return False


def merge_configs(default_config: Dict[str, Any], 
                 user_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user configuration with defaults
    
    Args:
        default_config: Default configuration
        user_config: User configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    try:
        merged = default_config.copy()
        
        for key, value in user_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
        
    except Exception as e:
        logger.error(f"Config merge error: {e}")
        return default_config


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test EEG data validation
    test_data = np.random.randn(8, 100)
    is_valid = validate_eeg_data(test_data, 8, 100)
    print(f"EEG data validation: {is_valid}")
    
    # Test normalization
    normalized_data = normalize_eeg_data(test_data, 'zscore')
    print(f"Normalization shape: {normalized_data.shape}")
    
    # Test SNR calculation
    snr = calculate_snr(test_data)
    print(f"SNR: {snr:.2f} dB")
    
    # Test artifact detection
    artifact_mask, artifact_ratio = detect_artifacts(test_data)
    print(f"Artifact ratio: {artifact_ratio:.3f}")
    
    # Test session directory creation
    session_dir = create_session_directory("./test_sessions", "test_session")
    print(f"Session directory: {session_dir}")
    
    print("Utility functions test completed!")
