#!/usr/bin/env python3
"""
Dashboard Module
Real-time visualization and monitoring for BCI applications

This module provides:
- Real-time EEG signal visualization (console-based)
- Speech intent display
- Data quality monitoring
- Performance metrics
- User interaction controls
"""

import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import threading
import queue

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for dashboard"""
    # Display settings
    update_interval: float = 0.1  # seconds
    max_data_points: int = 1000
    eeg_plot_channels: int = 8  # Number of channels to display
    
    # Colors and styling (for console)
    primary_color: str = "BLUE"
    secondary_color: str = "MAGENTA"
    success_color: str = "GREEN"
    warning_color: str = "YELLOW"
    error_color: str = "RED"
    
    # Features
    enable_eeg_plot: bool = True
    enable_quality_display: bool = True
    enable_intent_display: bool = True
    enable_controls: bool = True


class DashboardData:
    """Data structure for dashboard updates"""
    def __init__(self):
        self.eeg_data = []
        self.quality_metrics = {}
        self.speech_intent = None
        self.performance_metrics = {}
        self.system_status = "idle"
        self.timestamp = time.time()


class ConsoleDashboard:
    """Console-based dashboard for BCI monitoring"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.is_running = False
        self.update_thread = None
        self.data_queue = queue.Queue()
        
        logger.info("Console Dashboard initialized")
    
    def start(self) -> bool:
        """Start the console dashboard"""
        try:
            self.is_running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            logger.info("Console Dashboard started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start console dashboard: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the console dashboard"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        logger.info("Console Dashboard stopped")
    
    def update_data(self, data: DashboardData) -> None:
        """Update dashboard data"""
        try:
            self.data_queue.put_nowait(data)
        except queue.Full:
            # Remove oldest data if queue is full
            try:
                self.data_queue.get_nowait()
                self.data_queue.put_nowait(data)
            except queue.Empty:
                pass
    
    def _update_loop(self) -> None:
        """Main update loop for console dashboard"""
        while self.is_running:
            try:
                # Get latest data
                data = self.data_queue.get(timeout=1.0)
                self._display_data(data)
                
            except queue.Empty:
                # No data available, continue
                pass
            except Exception as e:
                logger.error(f"Console dashboard update error: {e}")
            
            time.sleep(self.config.update_interval)
    
    def _display_data(self, data: DashboardData) -> None:
        """Display data in console format"""
        try:
            # Clear console (simple approach)
            print("\n" * 50)
            
            # Header
            print("=" * 80)
            print(f"BCI REAL-TIME DASHBOARD - {time.strftime('%H:%M:%S')}")
            print("=" * 80)
            
            # System status
            status_color = self._get_status_color(data.system_status)
            print(f"System Status: {status_color}{data.system_status.upper()}{self._reset_color()}")
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(data.timestamp))}")
            print()
            
            # EEG Data Summary
            if data.eeg_data:
                print("EEG DATA:")
                print(f"  Channels: {len(data.eeg_data)}")
                print(f"  Data points: {len(data.eeg_data[0]) if data.eeg_data else 0}")
                if data.eeg_data:
                    # Show first few values from first channel
                    first_channel = data.eeg_data[0]
                    if len(first_channel) > 0:
                        print(f"  Sample values: {first_channel[:5]}")
                print()
            
            # Quality Metrics
            if data.quality_metrics:
                print("QUALITY METRICS:")
                for key, value in data.quality_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
                print()
            
            # Speech Intent
            if data.speech_intent:
                print("SPEECH INTENT:")
                print(f"  Type: {data.speech_intent.get('intent_type', 'unknown')}")
                print(f"  Confidence: {data.speech_intent.get('confidence', 0.0):.3f}")
                print(f"  Processing time: {data.speech_intent.get('processing_time', 0.0):.4f}s")
                print()
            
            # Performance Metrics
            if data.performance_metrics:
                print("PERFORMANCE METRICS:")
                for key, value in data.performance_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
                print()
            
            # Footer
            print("=" * 80)
            print("Press Ctrl+C to stop")
            
        except Exception as e:
            logger.error(f"Console display error: {e}")
    
    def _get_status_color(self, status: str) -> str:
        """Get color code for status"""
        if status == "running":
            return "\033[92m"  # Green
        elif status == "idle":
            return "\033[94m"  # Blue
        elif status == "error":
            return "\033[91m"  # Red
        elif status == "warning":
            return "\033[93m"  # Yellow
        else:
            return "\033[0m"   # Default
    
    def _reset_color(self) -> str:
        """Reset color to default"""
        return "\033[0m"


class Dashboard:
    """
    Main dashboard class for BCI applications
    
    This class provides:
    1. Console-based real-time monitoring
    2. Data buffering and display management
    3. Integration with BCI pipeline
    """
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize dashboard
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self.is_running = False
        
        # Initialize console dashboard
        self.console_dashboard = ConsoleDashboard(self.config)
        
        # Data management
        self.latest_data = DashboardData()
        self.data_history = []
        
        logger.info("Dashboard initialized (console mode)")
    
    def start(self) -> bool:
        """Start the dashboard"""
        try:
            self.is_running = True
            return self.console_dashboard.start()
                
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return False
    
    def stop(self) -> None:
        """Stop the dashboard"""
        self.is_running = False
        self.console_dashboard.stop()
        logger.info("Dashboard stopped")
    
    def update_data(self, eeg_data: Optional[List[np.ndarray]] = None,
                   quality_metrics: Optional[Dict] = None,
                   speech_intent: Optional[Dict] = None,
                   performance_metrics: Optional[Dict] = None,
                   system_status: str = "idle") -> None:
        """
        Update dashboard with new data
        
        Args:
            eeg_data: Latest EEG data
            quality_metrics: Data quality metrics
            speech_intent: Speech decoding results
            performance_metrics: System performance metrics
            system_status: Current system status
        """
        try:
            # Update latest data
            if eeg_data is not None:
                self.latest_data.eeg_data = eeg_data
            
            if quality_metrics is not None:
                self.latest_data.quality_metrics = quality_metrics
            
            if speech_intent is not None:
                self.latest_data.speech_intent = speech_intent
            
            if performance_metrics is not None:
                self.latest_data.performance_metrics = performance_metrics
            
            self.latest_data.system_status = system_status
            self.latest_data.timestamp = time.time()
            
            # Store in history
            self.data_history.append(self.latest_data)
            
            # Limit history size
            if len(self.data_history) > self.config.max_data_points:
                self.data_history.pop(0)
            
            # Update console dashboard
            self.console_dashboard.update_data(self.latest_data)
            
        except Exception as e:
            logger.error(f"Dashboard update error: {e}")
    
    def get_latest_data(self) -> DashboardData:
        """Get the latest dashboard data"""
        return self.latest_data
    
    def get_data_history(self) -> List[DashboardData]:
        """Get dashboard data history"""
        return self.data_history.copy()


def simulate_dashboard_data() -> DashboardData:
    """Simulate dashboard data for testing"""
    data = DashboardData()
    
    # Simulate EEG data
    num_channels = 8
    num_samples = 100
    data.eeg_data = []
    for i in range(num_channels):
        # Generate realistic EEG-like data
        t = np.linspace(0, 1, num_samples)
        alpha_wave = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        noise = np.random.normal(0, 2, num_samples)
        channel_data = alpha_wave + noise
        data.eeg_data.append(channel_data)
    
    # Simulate quality metrics
    data.quality_metrics = {
        'eeg_snr': 15.2,
        'eeg_artifact_ratio': 0.05,
        'overall_quality': 'good'
    }
    
    # Simulate speech intent
    data.speech_intent = {
        'intent_type': 'thinking',
        'confidence': 0.85,
        'processing_time': 0.023
    }
    
    # Simulate performance metrics
    data.performance_metrics = {
        'latency': 0.045,
        'throughput': 22.2,
        'accuracy': 0.92
    }
    
    data.system_status = "running"
    data.timestamp = time.time()
    
    return data


if __name__ == "__main__":
    # Test the dashboard
    config = DashboardConfig()
    dashboard = Dashboard(config)
    
    print("Starting dashboard test...")
    
    # Start dashboard
    if dashboard.start():
        print("Dashboard started successfully")
        
        # Simulate data updates
        for i in range(10):
            data = simulate_dashboard_data()
            dashboard.update_data(
                eeg_data=data.eeg_data,
                quality_metrics=data.quality_metrics,
                speech_intent=data.speech_intent,
                performance_metrics=data.performance_metrics,
                system_status=data.system_status
            )
            
            time.sleep(0.5)
        
        print("Console dashboard running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        # Stop dashboard
        dashboard.stop()
        print("Dashboard stopped")
    else:
        print("Failed to start dashboard")
