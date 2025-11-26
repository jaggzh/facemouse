#!/usr/bin/env python3
"""
Landmark filtering system for gaze tracking.

Provides filters that can be applied to individual landmarks to smooth
jitter without introducing excessive lag.
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple


# Default filter parameters (can be overridden)
DEF_MEDIAN_WINDOW = 3  # Number of samples for median filter
DEF_EMA_ALPHA = 0.3  # Exponential moving average smoothing factor (0=smooth, 1=responsive)
DEF_FILTER_ENABLED = True  # Enable filtering by default


class LandmarkFilter:
    """Filter for a single landmark point (x, y, z)."""
    
    def __init__(self, median_window=DEF_MEDIAN_WINDOW, ema_alpha=DEF_EMA_ALPHA):
        self.median_window = median_window
        self.ema_alpha = ema_alpha
        
        # Median filter buffer (stores recent raw samples)
        self.median_buffer = deque(maxlen=median_window)
        
        # EMA state (filtered value)
        self.ema_value = None
        
    def reset(self):
        """Reset filter state."""
        self.median_buffer.clear()
        self.ema_value = None
        
    def update(self, point: np.ndarray) -> np.ndarray:
        """
        Filter a new landmark point.
        
        Args:
            point: numpy array [x, y, z]
            
        Returns:
            Filtered point as numpy array [x, y, z]
        """
        # Add to median buffer
        self.median_buffer.append(point.copy())
        
        # Apply median filter (element-wise median across buffer)
        if len(self.median_buffer) >= 1:
            buffer_array = np.array(list(self.median_buffer))
            median_filtered = np.median(buffer_array, axis=0)
        else:
            median_filtered = point.copy()
        
        # Apply exponential moving average
        if self.ema_value is None:
            # First sample - initialize
            self.ema_value = median_filtered.copy()
        else:
            # EMA: new_value = alpha * new_sample + (1 - alpha) * old_value
            self.ema_value = (self.ema_alpha * median_filtered + 
                            (1.0 - self.ema_alpha) * self.ema_value)
        
        return self.ema_value.copy()


class LandmarkFilterSet:
    """Manages filters for multiple landmarks."""
    
    def __init__(self, median_window=DEF_MEDIAN_WINDOW, ema_alpha=DEF_EMA_ALPHA, 
                 enabled=DEF_FILTER_ENABLED):
        self.median_window = median_window
        self.ema_alpha = ema_alpha
        self.enabled = enabled
        
        # Dictionary of filters keyed by landmark index
        self.filters = {}
        
    def set_parameters(self, median_window: Optional[int] = None, 
                      ema_alpha: Optional[float] = None,
                      enabled: Optional[bool] = None):
        """Update filter parameters and reset all filters."""
        if median_window is not None:
            self.median_window = median_window
        if ema_alpha is not None:
            self.ema_alpha = ema_alpha
        if enabled is not None:
            self.enabled = enabled
            
        # Reset all filters to apply new parameters
        self.reset_all()
        
    def reset_all(self):
        """Reset all filters."""
        self.filters.clear()
        
    def update(self, landmark_idx: int, point: np.ndarray) -> np.ndarray:
        """
        Filter a landmark point.
        
        Args:
            landmark_idx: Index of the landmark
            point: numpy array [x, y, z]
            
        Returns:
            Filtered point (or original if filtering disabled)
        """
        if not self.enabled:
            return point.copy()
            
        # Create filter for this landmark if it doesn't exist
        if landmark_idx not in self.filters:
            self.filters[landmark_idx] = LandmarkFilter(
                median_window=self.median_window,
                ema_alpha=self.ema_alpha
            )
        
        return self.filters[landmark_idx].update(point)
    
    def update_batch(self, landmark_indices: list, points: list) -> list:
        """
        Filter multiple landmarks.
        
        Args:
            landmark_indices: List of landmark indices
            points: List of numpy arrays [x, y, z]
            
        Returns:
            List of filtered points
        """
        return [self.update(idx, pt) for idx, pt in zip(landmark_indices, points)]


def get_default_config() -> dict:
    """Get default filter configuration."""
    return {
        'median_window': DEF_MEDIAN_WINDOW,
        'ema_alpha': DEF_EMA_ALPHA,
        'enabled': DEF_FILTER_ENABLED,
    }