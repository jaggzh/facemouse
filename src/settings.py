#!/usr/bin/env python3
"""
Global settings and defaults for the gaze tracker.
"""

from pathlib import Path

# ============================================================================
# Calibration defaults
# ============================================================================

# Visual settings
DEF_CAL_VID_DARKEN = 0.75  # Darken video to 25% brightness during calibration
DEF_CAL_TARGET_SIZE = 40  # Outer circle size in pixels
DEF_CAL_TARGET_CENTER_SIZE = 8  # Center dot size in pixels

# Colors (B, G, R)
CAL_TARGET_INACTIVE = (100, 100, 100)  # Gray
CAL_TARGET_ACTIVE = (0, 255, 255)  # Yellow
CAL_TARGET_ACCEPTED = (0, 255, 0)  # Green
CAL_TARGET_COUNT_COLOR = (255, 255, 255)  # White text

# Data collection
DEF_CAL_SAMPLES = 3  # Number of samples to collect per point
DEF_CAL_AUTO_ACCEPT_TIME = 3.0  # Seconds to wait for auto-accept (when enabled)
DEF_CAL_UNDO_TIMEOUT = 5.0  # Seconds before considering point uncomitted

# Point selection algorithm
DEF_CAL_DISTANT_CHOICES = 3  # Number of distant points to randomly choose from
DEF_CAL_DISTANT_LARGER_CHOICES = 6  # Larger set for occasional variation
DEF_CAL_DISTANT_LARGE_SET_PROB = 0.3  # Probability of using larger choice set

# Grid configurations
CAL_GRID_CONFIGS = {
    9: (3, 3),
    12: (4, 3),
    20: (5, 4),
    30: (6, 5),
}
DEF_CAL_GRID_SIZE = 9  # Default grid point count

# TTS
DEF_CAL_NO_TTS = False  # Enable TTS by default
CAL_TTS_MESSAGES = {
    'start': 'Starting calibration',
    'point': 'Look at the {direction}',
    'accepted': 'Point accepted',
    'undo': 'Point removed',
    'complete': 'Calibration complete',
}

# Paths
CAL_PRESETS_DIR = Path.home() / ".config" / "eye_gaze_tracker" / "presets"
CAL_VIDEOS_DIR = Path.home() / ".config" / "eye_gaze_tracker" / "vids-cal"

# Ensure directories exist
CAL_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
CAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)