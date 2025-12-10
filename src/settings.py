#!/usr/bin/env python3
"""
Global settings and defaults for the gaze tracker.
"""

from pathlib import Path

DEF_CAL_GRID_SIZE = 9  # Number of calibration points

# Preset management defaults
DEF_PRESETS_ORDER = []  # Display order of preset filenames
DEF_PRESETS_RANK = []   # Default priority (first = current default)

# ============================================================================
# UI Scaling
# ============================================================================
# Hierarchical: global -> main_window -> { nav_buttons, control_panel }
DEF_UI_SCALE_GLOBAL = 1.0  # Applied to all UI elements
DEF_UI_SCALE_MAIN_WINDOW = 1.0  # Additional scale for main window
DEF_UI_SCALE_NAV_BUTTONS = 1.0  # Toolbar/navigation buttons
DEF_UI_SCALE_CONTROL_PANEL = 0.9  # Right-side control panel (slightly smaller)
DEF_UI_SCALE_CAL = 1.0  # Calibration/visualization window
DEF_UI_SCALE_CAL_BUTTONS = 1.2  # Calibration Accept/Undo buttons (larger for clicking)

def ui_scale(*factors) -> float:
    """Compute effective scale from hierarchical factors."""
    result = DEF_UI_SCALE_GLOBAL
    for f in factors:
        result *= f
    return result

def scaled_font_size(base_size: int, *factors) -> int:
    """Get scaled font size (minimum 6pt)."""
    return max(6, int(base_size * ui_scale(*factors)))

def scaled_size(base_size: int, *factors) -> int:
    """Get scaled widget/element size (minimum 1)."""
    return max(1, int(base_size * ui_scale(*factors)))

# ============================================================================
# Mouse Control
# ============================================================================
DEF_MOUSE_CONTROL_ENABLED = False  # Default off, --mouse enables

# Visualization cursor (when mouse control is off, or in calibration view)
VIZ_CURSOR_COLOR = (0xb0, 0x30, 0x90)  # BGR for 0x9030b0 (purple/magenta)
VIZ_CURSOR_CROSSHAIR_COLOR = (0xb0, 0x30, 0x90)  # Same as cursor
VIZ_CURSOR_LINE_THICKNESS = 1

# ============================================================================
# Audio / TTS Settings
# ============================================================================
DEF_AUDIO_ENABLED = True   # Master audio toggle (False = complete silence)
DEF_USE_TTS = True         # True = TTS, False = tones

# TTS messages for calibration events
# Keys: cal_start, point_dir, point_accepted, point_undo, cal_complete
CAL_TTS_MESSAGES = {
    'cal_start': 'Starting calibration',
    'point_dir': '{direction}',
    'point_accepted': 'Accepted',
    'point_undo': 'Removed',
    'cal_complete': 'Complete',
}

# Tone settings: (frequency_hz, duration_seconds)
# Volume is controlled separately by TONE_DEFAULT_VOLUME
TONE_DEFAULT_VOLUME = 1.0  # 0.0 to 2.0 (1.0 = normal)
TONE_FREQS = {
    'cal_start': (370, 0.15),
    'point_accepted': (340, 0.15),
    'cal_complete': (260, 0.35),
    # point_dir and point_undo have no tones (TTS only)
}

# ============================================================================
# Calibration defaults
# ============================================================================

# Visual settings
DEF_CAL_VID_BRIGHTNESS = 0.55  # Darken video to a fraction (eg. .6 is 60%)
DEF_CAL_TARGET_SIZE = 60  # Outer circle size in pixels
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

# Paths
CAL_PRESETS_DIR = Path.home() / ".config" / "eye_gaze_tracker" / "presets"
CAL_VIDEOS_DIR = Path.home() / ".config" / "eye_gaze_tracker" / "vids-cal"

# Ensure directories exist
CAL_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
CAL_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
