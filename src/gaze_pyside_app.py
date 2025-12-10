#!/usr/bin/env python3
# ============================================================================
# Suppress verbose library output BEFORE imports
# ============================================================================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['GLOG_minloglevel'] = '2'  # Suppress glog messages

# Suppress protobuf warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
sys.stdout.reconfigure(line_buffering=True)

import time
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import yaml  # For settings save/load

# Suppress TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import av  # pip install av
import numpy as np
import cv2
import mediapipe as mp
from PySide6 import QtCore, QtGui, QtWidgets

import camsettingsh264 as camsettings
import filters
import calibration
import settings as app_settings
import mouse_control

# ============================================================================
# Configuration defaults
# ============================================================================
DEF_VFILE_FRAME_DELAY = 1.0  # Delay between frames in seconds (for video files)
DEF_VFILE_FPS = None  # FPS for video file playback (None = use frame delay)
DEF_VFILE_LOOP = True  # Loop video files by default

# Landmark filtering defaults
DEF_FILTER_ENABLED = True
DEF_MEDIAN_WINDOW = 7  # Number of samples for median filter
DEF_EMA_ALPHA = 0.09  # Exponential moving average (0=very smooth, 1=no smoothing)

# Image adjustment defaults
DEF_BRIGHTNESS = 0  # -100 to 100
DEF_CONTRAST = 1.0  # 0.5 to 2.0

# ============================================================================
# Visualization and processing constants
# ============================================================================
# Landmark sizes and scaling
DEF_LM_SIZE = 1  # Size for "all landmarks"
DEF_LM_SPECIAL_SCALE = 3  # Multiplier for special landmarks
DEF_VECTOR_END_SIZE = 5  # Size of dot at end of vector lines
DEF_LM_SPECIAL_BRIGHTNESS = 0.7  # 70% brightness for special landmarks

# Landmark colors (B, G, R)
COLOR_LM_ALL = (64, 64, 255)

# Gaze vectors
COLOR_GAZE_LEFT = (0, 255, 0)  # Green
COLOR_GAZE_RIGHT = (0, 200, 255)  # Yellow
COLOR_GAZE_AVG = (255, 255, 0)
COLOR_FACE_AIM = (255, 0, 255)

# Eyelid colors (swapped from iris for distinction)
COLOR_EYELID_LEFT = COLOR_GAZE_RIGHT  # Left eye lids = yellow (right iris color)
COLOR_EYELID_RIGHT = COLOR_GAZE_LEFT  # Right eye lids = green (left iris color)

# Opacity for landmark overlay (0..1)
ALPHA_LANDMARKS = 0.5

# Scale for vectors in pixels (debug visualization)
GAZE_VECTOR_SCALE_PX = 120
FACE_VECTOR_SCALE_PX = 140

# Mediapipe face / eye landmark indices (canonical face mesh + iris)
LEFT_IRIS = [474, 475, 476, 477] # outer, top, inner, bottom (clockwise)
LEFT_PUPIL = 473
RIGHT_IRIS = [469, 470, 471, 472] # inner, top, outer, bottom (counter-clockwise)
RIGHT_PUPIL = 468

# Expanded eye corners with adjacent points
RIGHT_EYE_CORNERS = [133, 243, 189, 244, 33, 130] # Inner corners + adjacent, outer + adjacent
LEFT_EYE_CORNERS = [362, 463, 413, 464, 263, 359] # Inner corners + adjacent, outer + adjacent

LEFT_EYE_LIDS = [386, 374] # Top, bottom
RIGHT_EYE_LIDS = [159, 145] # Top, bottom

NOSE_TIP = 1

# Split for separate coloring
LEFT_EYE_CORNERS_INNER = [362, 463, 413, 464]
LEFT_EYE_CORNERS_OUTER = [263, 359]
RIGHT_EYE_CORNERS_INNER = [133, 243, 189, 244]
RIGHT_EYE_CORNERS_OUTER = [33, 130]

# Face orientation landmarks
NOSE_TIP_POINTS = [1, 2, 4]
CHEEKBONE_R = [34, 143, 116]
CHEEKBONE_L = [254, 372, 345]
FOREHEAD_TOP_R = 103
FOREHEAD_TOP_L = 332
CHIN_TIP_POINTS = [208, 428, 175]  # Corrected chin tip points

# ============================================================================
# Plot data output helper
# ============================================================================
# Global print_data level (set from args in main())
_print_data_level = 0

def set_print_data_level(level: int):
    """Set the global print data verbosity level."""
    global _print_data_level
    _print_data_level = level

def termplot(level: int, line: str):
    """Print plot data if verbosity level is high enough.
    
    Usage:
        termplot(1, "PLOT:EYES.x:{:.4f} EYES.y:{:.4f} GAZE.x:{:.4f} GAZE.y:{:.4f}".format(...))
    
    Output can be filtered with:
        ./gaze.py -P 1 | grep ^PLOT: | sed 's/^PLOT://' | splotty --stdin
    
    Fields: EYES.x/y (eye-only), FACE.x/y (face aim), GAZE.x/y (combined)
    """
    if _print_data_level >= level:
        print(line)


class VideoWorker(QtCore.QObject):
    """Background worker: decodes H.264 or video file, applies pre-filter, runs Mediapipe,
    draws debug overlays, and emits annotated frames as NumPy arrays (BGR).
    """
    frameReady = QtCore.Signal(np.ndarray, int, int)  # frame, current_frame, total_frames
    gazeData = QtCore.Signal(np.ndarray, dict)  # frame, gaze_data dict
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, video_file=None, usb_device=None, dev_device=None,
                 frame_delay=DEF_VFILE_FRAME_DELAY, 
                 vfile_fps=DEF_VFILE_FPS, loop=DEF_VFILE_LOOP, parent=None):
        super().__init__(parent)
        self._running = False
        self._video_file = video_file
        self._usb_device = usb_device
        self._dev_device = dev_device
        self._frame_delay = frame_delay
        self._vfile_fps = vfile_fps
        self._loop = loop
        
        # Video playback controls
        self._paused = False
        self._step_forward = False
        self._step_backward = False
        self._jump_to_start = False
        self._jump_to_end = False
        self._playback_lock = QtCore.QMutex()

        # Image pre-filter controls (alpha, beta for convertScaleAbs)
        self._contrast = DEF_CONTRAST  # alpha
        self._brightness = DEF_BRIGHTNESS  # beta
        self._contrast_lock = QtCore.QMutex()
        self._brightness_lock = QtCore.QMutex()

        # Landmark filtering - separate filters for different groups
        self._used_landmark_indices = set(
            LEFT_IRIS + [LEFT_PUPIL] + LEFT_EYE_CORNERS + LEFT_EYE_LIDS +
            RIGHT_IRIS + [RIGHT_PUPIL] + RIGHT_EYE_CORNERS + RIGHT_EYE_LIDS +
            NOSE_TIP_POINTS + CHEEKBONE_R + CHEEKBONE_L + 
            [FOREHEAD_TOP_R, FOREHEAD_TOP_L] + CHIN_TIP_POINTS
        )
        
        # Separate filter groups for different landmark types
        self._filter_groups = {
            'iris_pupil': filters.LandmarkFilterSet(DEF_MEDIAN_WINDOW, DEF_EMA_ALPHA, DEF_FILTER_ENABLED),
            'eye_corners': filters.LandmarkFilterSet(DEF_MEDIAN_WINDOW, DEF_EMA_ALPHA, DEF_FILTER_ENABLED),
            'eye_lids': filters.LandmarkFilterSet(DEF_MEDIAN_WINDOW, DEF_EMA_ALPHA, DEF_FILTER_ENABLED),
            'face_orientation': filters.LandmarkFilterSet(DEF_MEDIAN_WINDOW, DEF_EMA_ALPHA, DEF_FILTER_ENABLED),
            'single_eye': filters.LandmarkFilterSet(DEF_MEDIAN_WINDOW, DEF_EMA_ALPHA * 0.5, DEF_FILTER_ENABLED),
        }
        
        # Map landmark indices to filter groups
        self._landmark_to_group = {}
        for idx in LEFT_IRIS + RIGHT_IRIS + [LEFT_PUPIL, RIGHT_PUPIL]:
            self._landmark_to_group[idx] = 'iris_pupil'
        for idx in LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS:
            self._landmark_to_group[idx] = 'eye_corners'
        for idx in LEFT_EYE_LIDS + RIGHT_EYE_LIDS:
            self._landmark_to_group[idx] = 'eye_lids'
        for idx in NOSE_TIP_POINTS + CHEEKBONE_R + CHEEKBONE_L + [FOREHEAD_TOP_R, FOREHEAD_TOP_L] + CHIN_TIP_POINTS:
            self._landmark_to_group[idx] = 'face_orientation'
        
        self._filter_lock = QtCore.QMutex()
        
        # Eye open ratio threshold (height / width) for blink / eye-closed detection
        self._eye_open_thresh = 0.20

        # Per-eye bias tracking (deviation from binocular average)
        # Updated every frame when both eyes visible
        self._left_bias = np.array([0.0, 0.0], dtype=np.float32)
        self._right_bias = np.array([0.0, 0.0], dtype=np.float32)
        self._bias_alpha = 0.1  # EMA alpha for bias updates
        self._bias_lock = QtCore.QMutex()
        
        # Calibration corner data for plotting (set from MainWindow when preset selected)
        self._cal_corners = None  # {'tl': {'gaze': [x,y]}, 'tr': ..., 'bl': ..., 'br': ..., 'center': ...}
        self._cal_corners_lock = QtCore.QMutex()
        
        # Landmark visualization toggles
        self._show_landmarks_raw = False  # Raw unfiltered landmarks
        self._show_landmarks_all = True  # Filtered landmarks (all)
        self._show_landmarks_used = True  # Special highlighting of used landmarks
        
        # Eye enable/disable toggles
        self._left_eye_enabled = True
        self._right_eye_enabled = True
        self._eye_enable_lock = QtCore.QMutex()
        
        # Gaze bounds tracking (for visualization and debugging)
        self._gaze_min_x = None
        self._gaze_max_x = None
        self._gaze_min_y = None
        self._gaze_max_y = None
        self._gaze_bounds_lock = QtCore.QMutex()

        # Mediapipe face mesh setup
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ---- Slots for UI to tweak image pre-filter parameters ----
    @QtCore.Slot(int)
    def set_brightness(self, value: int):
        with QtCore.QMutexLocker(self._brightness_lock):
            self._brightness = float(value)
            print(f"[Worker] Brightness set to {self._brightness}")

    @QtCore.Slot(int)
    def set_contrast_slider(self, value: int):
        with QtCore.QMutexLocker(self._contrast_lock):
            self._contrast = float(value) / 100.0
            print(f"[Worker] Contrast set to {self._contrast}")

    @QtCore.Slot(float)
    def set_contrast(self, value: float):
        with QtCore.QMutexLocker(self._contrast_lock):
            self._contrast = float(value)
            print(f"[Worker] Contrast set to {self._contrast}")

    @QtCore.Slot(bool)
    def set_show_landmarks_raw(self, enabled: bool):
        self._show_landmarks_raw = bool(enabled)

    @QtCore.Slot(bool)
    def set_show_landmarks_all(self, enabled: bool):
        self._show_landmarks_all = bool(enabled)

    @QtCore.Slot(bool)
    def set_show_landmarks_used(self, enabled: bool):
        self._show_landmarks_used = bool(enabled)
    
    @QtCore.Slot(bool)
    def set_left_eye_enabled(self, enabled: bool):
        with QtCore.QMutexLocker(self._eye_enable_lock):
            self._left_eye_enabled = bool(enabled)
            print(f"[Worker] Left eye {'enabled' if enabled else 'disabled'}")
    
    @QtCore.Slot(bool)
    def set_right_eye_enabled(self, enabled: bool):
        with QtCore.QMutexLocker(self._eye_enable_lock):
            self._right_eye_enabled = bool(enabled)
            print(f"[Worker] Right eye {'enabled' if enabled else 'disabled'}")
    
    @QtCore.Slot()
    def reset_gaze_bounds(self):
        """Reset gaze bounds tracking."""
        with QtCore.QMutexLocker(self._gaze_bounds_lock):
            self._gaze_min_x = None
            self._gaze_max_x = None
            self._gaze_min_y = None
            self._gaze_max_y = None
            print("[Worker] Gaze bounds reset")
    
    @QtCore.Slot(str, int)
    def set_filter_median_window(self, group: str, value: int):
        """Set median window for a filter group."""
        with QtCore.QMutexLocker(self._filter_lock):
            if group in self._filter_groups:
                self._filter_groups[group].median_window = value
                print(f"[Worker] {group} median window: {value}")
    
    @QtCore.Slot(str, float)
    def set_filter_ema_alpha(self, group: str, value: float):
        """Set EMA alpha for a filter group."""
        with QtCore.QMutexLocker(self._filter_lock):
            if group in self._filter_groups:
                self._filter_groups[group].ema_alpha = value
                print(f"[Worker] {group} EMA alpha: {value}")
    
    @QtCore.Slot(str, bool)
    def set_filter_enabled(self, group: str, enabled: bool):
        """Set filter enabled for a filter group."""
        with QtCore.QMutexLocker(self._filter_lock):
            if group in self._filter_groups:
                self._filter_groups[group].enabled = enabled
                print(f"[Worker] {group} filtering {'enabled' if enabled else 'disabled'}")

    @QtCore.Slot(object)
    def set_calibration_corners(self, corners: dict):
        """Set calibration corner data for plotting."""
        print(f"[Worker] set_calibration_corners called with: {corners}")
        with QtCore.QMutexLocker(self._cal_corners_lock):
            self._cal_corners = corners
            if corners:
                print(f"[Worker] Calibration corners set: {list(corners.keys())}")
            else:
                print("[Worker] Calibration corners cleared")
    
    # ---- Video playback control slots ----
    @QtCore.Slot()
    def toggle_pause(self):
        print("[Worker] toggle_pause called")
        with QtCore.QMutexLocker(self._playback_lock):
            self._paused = not self._paused
            print(f"[Worker] Paused: {self._paused}")

    @QtCore.Slot()
    def step_forward(self):
        print("[Worker] step_forward called")
        with QtCore.QMutexLocker(self._playback_lock):
            self._step_forward = True

    @QtCore.Slot()
    def step_backward(self):
        print("[Worker] step_backward called")
        with QtCore.QMutexLocker(self._playback_lock):
            self._step_backward = True

    @QtCore.Slot()
    def jump_to_start(self):
        print("[Worker] jump_to_start called")
        with QtCore.QMutexLocker(self._playback_lock):
            self._jump_to_start = True

    @QtCore.Slot()
    def jump_to_end(self):
        print("[Worker] jump_to_end called")
        with QtCore.QMutexLocker(self._playback_lock):
            self._jump_to_end = True

    # ---- Core loop ----
    @QtCore.Slot()
    def start(self):
        self._running = True
        if self._video_file:
            # Process video file
            self._process_video_file()
        elif self._usb_device or self._dev_device:
            # Process USB/V4L2 camera
            self._process_usb_camera()
        else:
            # Process TCP camera stream
            self._process_camera_stream()
        self.finished.emit()
        print("[Worker] Stopped")

    def _process_video_file(self):
        """Process a video file (for testing) with playback controls."""
        print(f"[Worker] Opening video file: {self._video_file}")
        
        # Determine frame delay
        if self._vfile_fps is not None:
            frame_delay = 1.0 / self._vfile_fps
        else:
            frame_delay = self._frame_delay

        while self._running:
            try:
                container = av.open(self._video_file)
            except Exception as e:
                msg = f"Failed to open video file: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
                break

            print(f"[Worker] Video file opened, frame delay: {frame_delay:.3f}s")

            try:
                # Get total frame count if possible
                stream = container.streams.video[0]
                total_frames = stream.frames if stream.frames > 0 else -1
                
                # Store all frames for random access (needed for backward stepping)
                frames_list = []
                print("[Worker] Loading frames into memory for playback control...")
                for frame in container.decode(video=0):
                    frames_list.append(frame.to_ndarray(format="bgr24"))
                
                total_frames = len(frames_list)
                print(f"[Worker] Loaded {total_frames} frames")
                
                current_frame_idx = 0
                
                while self._running and current_frame_idx < total_frames:
                    # Check playback controls
                    with QtCore.QMutexLocker(self._playback_lock):
                        if self._jump_to_start:
                            current_frame_idx = 0
                            self._jump_to_start = False
                        elif self._jump_to_end:
                            current_frame_idx = max(0, total_frames - 1)
                            self._jump_to_end = False
                        elif self._step_backward:
                            current_frame_idx = max(0, current_frame_idx - 1)
                            self._step_backward = False
                        elif self._step_forward:
                            current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
                            self._step_forward = False
                        elif self._paused:
                            time.sleep(0.033)  # Still check controls while paused
                            continue
                    
                    img = frames_list[current_frame_idx].copy()
                    
                    # Apply brightness/contrast adjustments
                    img = self._apply_image_adjustments(img)

                    # Run Mediapipe and draw overlays
                    gaze_data = None
                    try:
                        img, gaze_data = self._process_and_overlay(img)
                    except Exception as e:
                        self.error.emit(f"Processing error: {e}")

                    # Emit the annotated frame with frame info
                    self.frameReady.emit(img, current_frame_idx + 1, total_frames)
                    
                    # Emit gaze data for calibration
                    if gaze_data is not None:
                        self.gazeData.emit(img, gaze_data)

                    # Advance to next frame
                    current_frame_idx += 1
                    
                    # Apply frame delay
                    time.sleep(frame_delay)
                
                # End of video
                if self._loop and self._running:
                    print("[Worker] Looping video...")
                    continue
                else:
                    print("[Worker] Video finished")
                    break
            except Exception as e:
                msg = f"Decode error: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
            finally:
                try:
                    container.close()
                except Exception:
                    pass

            # Exit loop (unless looping)
            if not self._loop:
                break

    def _process_usb_camera(self):
        """Process USB/V4L2 camera device."""
        # Determine device path
        if self._dev_device:
            device = self._dev_device
        elif self._usb_device:
            # Try to find device by USB identifier
            device = self._find_device_by_usb_id(self._usb_device)
            if device is None:
                msg = f"Could not find device matching USB identifier: {self._usb_device}"
                print("[Worker]", msg)
                self.error.emit(msg)
                return
        else:
            return
        
        print(f"[Worker] Opening USB camera: {device}")
        
        while self._running:
            try:
                # Open with OpenCV (simpler for V4L2 devices)
                cap = cv2.VideoCapture(device)
                
                if not cap.isOpened():
                    msg = f"Failed to open camera device: {device}"
                    print("[Worker]", msg)
                    self.error.emit(msg)
                    time.sleep(1.0)
                    continue
                
                print(f"[Worker] Camera opened: {device}")
                
                frame_count = 0
                while self._running:
                    ret, img = cap.read()
                    
                    if not ret:
                        print("[Worker] Failed to read frame from camera")
                        break
                    
                    # Apply brightness/contrast adjustments
                    img = self._apply_image_adjustments(img)
                    
                    # Run Mediapipe and draw overlays
                    gaze_data = None
                    try:
                        img, gaze_data = self._process_and_overlay(img)
                    except Exception as e:
                        self.error.emit(f"Processing error: {e}")
                    
                    # Emit the annotated frame
                    frame_count += 1
                    self.frameReady.emit(img, frame_count, -1)
                    
                    # Emit gaze data for calibration
                    if gaze_data is not None:
                        self.gazeData.emit(img, gaze_data)
                
            except Exception as e:
                msg = f"Camera error: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
            
            finally:
                if 'cap' in locals():
                    cap.release()
            
            # If still running, try to reconnect
            if self._running:
                print("[Worker] Will attempt to reconnect in 1s")
                time.sleep(1.0)
    
    def _find_device_by_usb_id(self, usb_id: str) -> Optional[str]:
        """Find /dev/videoN device by USB identifier using v4l2-ctl."""
        try:
            import subprocess
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                print("[Worker] v4l2-ctl failed, trying direct match")
                # Fallback: try common device numbers
                for i in range(10):
                    dev = f"/dev/video{i}"
                    if Path(dev).exists():
                        # Try to match USB ID in device path
                        try:
                            realpath = Path(dev).resolve()
                            if usb_id in str(realpath):
                                return dev
                        except:
                            pass
                return None
            
            # Parse v4l2-ctl output
            lines = result.stdout.split('\n')
            current_device_name = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Device name lines don't start with /dev/
                if not line.startswith('/dev/'):
                    current_device_name = line.rstrip(':')
                else:
                    # This is a device path
                    if current_device_name and usb_id in current_device_name:
                        # Found a match
                        device_path = line.strip()
                        print(f"[Worker] Found USB device: {current_device_name} -> {device_path}")
                        return device_path
            
            return None
            
        except Exception as e:
            print(f"[Worker] Error finding USB device: {e}")
            return None

    def _process_camera_stream(self):
        """Process live camera stream."""
        host = camsettings.camhost
        port = camsettings.camport
        url = f"tcp://{host}:{port}"

        while self._running:
            try:
                print(f"[Worker] Opening stream {url} ...")
                container = av.open(
                    url,
                    options={
                        "fflags": "nobuffer",
                        "flags": "low_delay",
                        "probesize": "32",
                        "analyzeduration": "0",
                    },
                    timeout=5.0,
                )
            except Exception as e:
                msg = f"Failed to open stream: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
                time.sleep(1.0)
                continue

            print("[Worker] Stream opened, starting decode loop")
            frame_count = 0
            try:
                for frame in container.decode(video=0):
                    if not self._running:
                        break
                    img = frame.to_ndarray(format="bgr24")

                    # Apply brightness/contrast adjustments
                    img = self._apply_image_adjustments(img)

                    # Run Mediapipe and draw overlays
                    gaze_data = None
                    try:
                        img, gaze_data = self._process_and_overlay(img)
                    except Exception as e:
                        self.error.emit(f"Processing error: {e}")

                    # Emit the annotated frame (no frame count for streams)
                    frame_count += 1
                    self.frameReady.emit(img, frame_count, -1)
                    
                    # Emit gaze data for calibration
                    if gaze_data is not None:
                        self.gazeData.emit(img, gaze_data)

            except Exception as e:
                msg = f"Decode error or stream ended: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
            finally:
                try:
                    container.close()
                except Exception:
                    pass

            # If still running, reconnect
            if self._running:
                print("[Worker] Will attempt to reconnect in 1s")
                time.sleep(1.0)

    def _apply_image_adjustments(self, img: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments to the image."""
        with QtCore.QMutexLocker(self._contrast_lock):
            contrast = self._contrast
        with QtCore.QMutexLocker(self._brightness_lock):
            brightness = self._brightness
        
        # Apply the adjustments
        adjusted = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        return adjusted

    def stop(self):
        self._running = False

    # ---- Gaze / blink / overlay helpers ----
    def _process_and_overlay(self, img: np.ndarray) -> np.ndarray:
        """Run Mediapipe, compute simple per-eye gaze offsets, detect blinks
        via eye openness ratio, and draw debug overlays.
        """
        # Mirror flip horizontally for natural mirror-like view
        img = cv2.flip(img, 1)
        
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return img, None  # Return tuple - fix video freeze when no face detected, None

        face_landmarks = results.multi_face_landmarks[0]

        # Dynamic landmark scaling based on resolution
        base_lm_scaler = min(1.0, w / 750.0)
        lm_size = max(1, int(DEF_LM_SIZE * base_lm_scaler))
        lm_special_size = max(1, int(DEF_LM_SIZE * DEF_LM_SPECIAL_SCALE * base_lm_scaler))
        vector_end_size = max(1, int(DEF_VECTOR_END_SIZE * base_lm_scaler))

        def lm(idx: int):
            """Get filtered landmark position (uses appropriate filter group per landmark type)."""
            p = face_landmarks.landmark[idx]
            raw_point = np.array([p.x, p.y, p.z], dtype=np.float32)
            
            # Only filter landmarks we actually use
            if idx in self._used_landmark_indices:
                group_name = self._landmark_to_group.get(idx, 'face_orientation')
                with QtCore.QMutexLocker(self._filter_lock):
                    filtered_point = self._filter_groups[group_name].update(idx, raw_point)
                return filtered_point
            else:
                # Not used, return raw
                return raw_point
        
        def lm_avg(indices):
            """Average position of multiple landmarks."""
            pts = np.array([lm(i) for i in indices], dtype=np.float32)
            return pts.mean(axis=0)

        # Draw landmarks in order: raw -> filtered -> special (so they layer correctly)
        overlay = img.copy()
        
        # 1. Draw raw unfiltered landmarks if enabled (gray, drawn first so filtered draw over)
        if self._show_landmarks_raw:
            for idx, p in enumerate(face_landmarks.landmark):
                x = int(p.x * w)
                y = int(p.y * h)
                cv2.circle(overlay, (x, y), lm_size, (128, 128, 128), -1)  # Gray
        
        # 2. Draw filtered landmarks (all) if enabled (normal color, draws over raw)
        if self._show_landmarks_all:
            for idx, p in enumerate(face_landmarks.landmark):
                filt = lm(idx)  # Get filtered version
                x = int(filt[0] * w)
                y = int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_size, COLOR_LM_ALL, -1)

        # Helper to compute eye metrics given index sets
        def eye_info(iris_idx, pupil_idx, corners_idx, lids_idx):
            # Include pupil in iris center calculation
            iris_pts = np.array([lm(i) for i in iris_idx], dtype=np.float32)
            pupil_pt = lm(pupil_idx)
            all_iris_pts = np.vstack([iris_pts, pupil_pt.reshape(1, -1)])
            iris_center = all_iris_pts.mean(axis=0)

            # Eye center from all corner points
            corner_pts = np.array([lm(i) for i in corners_idx], dtype=np.float32)
            eye_center = corner_pts.mean(axis=0)

            # Horizontal eye width in normalized coords (use outermost corners)
            c0, c1 = corner_pts[0], corner_pts[-1]
            width = np.linalg.norm(c0[:2] - c1[:2]) + 1e-6

            lt, lb = lm(lids_idx[0]), lm(lids_idx[1])
            open_height = abs(lt[1] - lb[1])
            open_ratio = open_height / width

            # Relative iris position within eye, normalized
            # 0 = centered; +/- ~1 toward corners/lids
            horiz = (iris_center[0] - eye_center[0]) / (width / 2.0)
            # Invert vertical sign so looking up is negative vy
            vert = (iris_center[1] - eye_center[1]) / (width / 2.0)

            # Pixel coordinates
            eye_px = (int(eye_center[0] * w), int(eye_center[1] * h))

            return {
                "eye_center": eye_center,
                "iris_center": iris_center,
                "eye_px": eye_px,
                "open_ratio": open_ratio,
                "dir_norm": np.array([horiz, vert], dtype=np.float32),
                "corners_idx": corners_idx,
                "lids_idx": lids_idx,
                "iris_idx": iris_idx,
                "pupil_idx": pupil_idx,
            }

        left = eye_info(LEFT_IRIS, LEFT_PUPIL, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
        right = eye_info(RIGHT_IRIS, RIGHT_PUPIL, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)
        
        # ==== HEAD COORDINATE SYSTEM & GAZE TRANSFORMATION ====
        # Calculate head X-axis rotation from two sources and average
        # This compensates for camera angle and head tilt
        
        # 1. Eye corners axis: left outer -> right outer
        left_outer_pos = lm_avg(LEFT_EYE_CORNERS_OUTER)
        right_outer_pos = lm_avg(RIGHT_EYE_CORNERS_OUTER)
        eye_axis = right_outer_pos - left_outer_pos  # Left to right
        eye_axis_2d = eye_axis[:2]  # Just x, y
        eye_angle = np.arctan2(eye_axis_2d[1], eye_axis_2d[0])
        
        # 2. Forehead axis: left -> right  
        forehead_left = lm(FOREHEAD_TOP_L)
        forehead_right = lm(FOREHEAD_TOP_R)
        forehead_axis = forehead_right - forehead_left
        forehead_axis_2d = forehead_axis[:2]
        forehead_angle = np.arctan2(forehead_axis_2d[1], forehead_axis_2d[0])
        
        # Average the two angles for final head X-axis rotation
        head_angle = (eye_angle + forehead_angle) / 2.0
        
        # Rotation matrix to transform vectors into head-aligned coordinates
        cos_a = np.cos(-head_angle)  # Negative to rotate back to aligned
        sin_a = np.sin(-head_angle)
        rotation_matrix = np.array([[cos_a, -sin_a],
                                   [sin_a, cos_a]])
        
        # Calculate face-aim vector (we'll subtract this later)
        # Need to compute it now before using it
        nose_tip_point = lm_avg(NOSE_TIP_POINTS)
        cheek_avg = lm_avg(CHEEKBONE_R + CHEEKBONE_L)
        vector_cheeks_nose = nose_tip_point - cheek_avg
        vector_cheeks_nose = vector_cheeks_nose / (np.linalg.norm(vector_cheeks_nose) + 1e-6)
        
        forehead_l = lm(FOREHEAD_TOP_L)
        forehead_r = lm(FOREHEAD_TOP_R)
        chin_tip = lm_avg(CHIN_TIP_POINTS)
        v1 = chin_tip - forehead_l
        v2 = chin_tip - forehead_r
        vector_face_normal = np.cross(v1, v2)
        if vector_face_normal[2] > 0:
            vector_face_normal = -vector_face_normal
        vector_face_normal = vector_face_normal / (np.linalg.norm(vector_face_normal) + 1e-6)
        
        final_face_vector = 0.5 * (vector_cheeks_nose + vector_face_normal)
        final_face_vector = final_face_vector / (np.linalg.norm(final_face_vector) + 1e-6)
        face_dir_prelim = np.array([final_face_vector[0], final_face_vector[1]], dtype=np.float32)
        
        # Note: Individual eye vectors (left["dir_norm"], right["dir_norm"]) stay raw/unmodified
        # Face aim is added only to the final averaged gaze vector below
        
        # Check which eyes are enabled
        with QtCore.QMutexLocker(self._eye_enable_lock):
            left_eye_enabled = self._left_eye_enabled
            right_eye_enabled = self._right_eye_enabled

        # Highlight the specific landmarks we are using, if requested
        # Draw them with darker versions of their associated vector colors
        if self._show_landmarks_used:
            def darken_color(color_bgr, factor=DEF_LM_SPECIAL_BRIGHTNESS):
                """Make color darker by brightness factor."""
                return tuple(int(c * factor) for c in color_bgr)
            
            def shift_hue_bgr(color_bgr, shift_degrees=30):
                """Shift BGR color hue in HSV space."""
                # Convert single color to image format for cv2
                pixel = np.uint8([[color_bgr]])
                hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
                hsv[0, 0, 0] = (hsv[0, 0, 0] + shift_degrees) % 180  # Hue in OpenCV is 0-179
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                return tuple(int(x) for x in bgr[0, 0])
            
            def draw_used(info, vector_color, eyelid_color):
                # Iris + pupil: darker version of gaze color
                iris_color = darken_color(vector_color, 0.7)
                for idx in list(info["iris_idx"]) + [info["pupil_idx"]]:
                    filt = lm(idx)  # Use FILTERED position
                    x = int(filt[0] * w)
                    y = int(filt[1] * h)
                    cv2.circle(overlay, (x, y), lm_special_size, iris_color, -1)
                
                # Corners: darker + hue shifted for distinctiveness
                corner_color = shift_hue_bgr(darken_color(vector_color, 0.6), 40)
                for idx in info["corners_idx"]:
                    filt = lm(idx)  # Use FILTERED position
                    x = int(filt[0] * w)
                    y = int(filt[1] * h)
                    cv2.circle(overlay, (x, y), lm_special_size, corner_color, -1)
                
                # Lids: use dedicated eyelid color (swapped from opposite eye)
                lid_color = darken_color(eyelid_color, 0.65)
                for idx in info["lids_idx"]:
                    filt = lm(idx)  # Use FILTERED position
                    x = int(filt[0] * w)
                    y = int(filt[1] * h)
                    cv2.circle(overlay, (x, y), lm_special_size, lid_color, -1)

            draw_used(left, COLOR_GAZE_LEFT, COLOR_EYELID_LEFT)
            draw_used(right, COLOR_GAZE_RIGHT, COLOR_EYELID_RIGHT)

        # Blend overlay back onto image for landmarks
        img = cv2.addWeighted(overlay, ALPHA_LANDMARKS, img, 1.0 - ALPHA_LANDMARKS, 0)

        # Simple blink / eye-closed detection
        left_open = left["open_ratio"] > self._eye_open_thresh
        right_open = right["open_ratio"] > self._eye_open_thresh

        # Draw per-eye gaze vectors (in normalized eye space)
        def draw_eye_vector(info, is_open: bool, color):
            # Use iris center as origin, not eye center
            iris_center = info["iris_center"]
            cx, cy = int(iris_center[0] * w), int(iris_center[1] * h)
            vx, vy = info["dir_norm"]
            if is_open:
                end_pt = (
                    int(cx + vx * GAZE_VECTOR_SCALE_PX),
                    int(cy + vy * GAZE_VECTOR_SCALE_PX),
                )
                cv2.circle(img, (cx, cy), 2, color, -1)
                cv2.line(img, (cx, cy), end_pt, color, 2, cv2.LINE_AA)
                # distal dot
                cv2.circle(img, end_pt, vector_end_size, color, -1)
                # Text overlay with gaze values and eye open ratio
                label = f"[{vx:+.3f}, {vy:+.3f}] ({info['open_ratio']:.2f})"
                cv2.putText(
                    img,
                    label,
                    (end_pt[0] + 5, end_pt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            else:
                # Mark closed eye center with small x
                cv2.putText(
                    img,
                    "x",
                    (cx - 3, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        draw_eye_vector(left, left_open and left_eye_enabled, COLOR_GAZE_LEFT)
        draw_eye_vector(right, right_open and right_eye_enabled, COLOR_GAZE_RIGHT)

        # Combined gaze (average of open AND enabled eyes)
        open_dirs = []
        open_labels = []  # Track which eyes: 'left', 'right'
        centers_px = []
        for info, is_open, is_enabled, label in (
            (left, left_open, left_eye_enabled, 'left'),
            (right, right_open, right_eye_enabled, 'right')
        ):
            if is_open and is_enabled:
                open_dirs.append(info["dir_norm"])
                open_labels.append(label)
                # Use iris center for combined gaze origin
                iris_px = (int(info["iris_center"][0] * w), int(info["iris_center"][1] * h))
                centers_px.append(iris_px)

        avg_dir = None
        if len(open_dirs) == 2:
            # Both eyes - compute average and update bias estimates
            avg_dir = np.mean(open_dirs, axis=0)
            with QtCore.QMutexLocker(self._bias_lock):
                self._left_bias = (1 - self._bias_alpha) * self._left_bias + \
                                  self._bias_alpha * (left["dir_norm"] - avg_dir)
                self._right_bias = (1 - self._bias_alpha) * self._right_bias + \
                                   self._bias_alpha * (right["dir_norm"] - avg_dir)
        elif len(open_dirs) == 1:
            # Single eye - apply bias correction to match what binocular average would be
            raw_gaze = open_dirs[0]
            with QtCore.QMutexLocker(self._bias_lock):
                if open_labels[0] == 'left':
                    avg_dir = raw_gaze - self._left_bias
                else:
                    avg_dir = raw_gaze - self._right_bias
            
            # Apply single_eye filter for extra smoothing
            with QtCore.QMutexLocker(self._filter_lock):
                # Use index 0 for x, 1 for y in the single_eye filter
                avg_dir_3d = np.array([avg_dir[0], avg_dir[1], 0.0], dtype=np.float32)
                filtered = self._filter_groups['single_eye'].update(0, avg_dir_3d)
                avg_dir = np.array([filtered[0], filtered[1]], dtype=np.float32)
        # else: no eyes open - avg_dir stays None, no update
        
        if avg_dir is not None:
            # Save eye-only gaze before combining with face
            eye_only_dir = avg_dir.copy()
            
            # Add face aim to averaged gaze to get world-space direction
            # Rationale: If eyes look at world angle +20, head turns +5 toward target,
            # eye-relative gaze shifts to +15. World gaze = eye-relative + head = 15 + 5 = 20
            combined_gaze = avg_dir + face_dir_prelim
            avg_dir = combined_gaze  # Keep using avg_dir for rest of code
            
            avg_center = np.mean(np.array(centers_px, dtype=np.float32), axis=0)
            acx, acy = int(avg_center[0]), int(avg_center[1])
            end_pt = (
                int(acx + avg_dir[0] * GAZE_VECTOR_SCALE_PX * 1.2),
                int(acy + avg_dir[1] * GAZE_VECTOR_SCALE_PX * 1.2),
            )
            cv2.line(img, (acx, acy), end_pt, COLOR_GAZE_AVG, 2, cv2.LINE_AA)
            cv2.circle(img, end_pt, vector_end_size, COLOR_GAZE_AVG, -1)
            cv2.putText(
                img,
                f"avg [{avg_dir[0]:+.3f}, {avg_dir[1]:+.3f}]",
                (end_pt[0] + 5, end_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_GAZE_AVG,
                1,
                cv2.LINE_AA,
            )
            
            # Output plot data at level 1
            # EYES = eye gaze only, FACE = face aim only, GAZE = combined (eye + face)
            if _print_data_level >= 1:
                parts = [f"PLOT:EYES.x:{eye_only_dir[0]:.4f}",
                        f"EYES.y:{eye_only_dir[1]:.4f}",
                        f"FACE.x:{face_dir_prelim[0]:.4f}",
                        f"FACE.y:{face_dir_prelim[1]:.4f}",
                        f"GAZE.x:{combined_gaze[0]:.4f}",
                        f"GAZE.y:{combined_gaze[1]:.4f}",
                        f"N_EYES:{len(open_dirs)}"]
                
                # Add corner calibration values as constant reference lines
                with QtCore.QMutexLocker(self._cal_corners_lock):
                    if self._cal_corners:
                        for name in ('tl', 'tr', 'bl', 'br'):
                            corner = self._cal_corners.get(name)
                            if corner and corner.get('gaze'):
                                gx, gy = corner['gaze']
                                parts.append(f"C_{name}.x:{gx:.4f}")
                                parts.append(f"C_{name}.y:{gy:.4f}")
                
                termplot(1, ' '.join(parts))
            
            # Track persistent max gaze bounds
            with QtCore.QMutexLocker(self._gaze_bounds_lock):
                if self._gaze_min_x is None:
                    self._gaze_min_x = avg_dir[0]
                    self._gaze_max_x = avg_dir[0]
                    self._gaze_min_y = avg_dir[1]
                    self._gaze_max_y = avg_dir[1]
                else:
                    self._gaze_min_x = min(self._gaze_min_x, avg_dir[0])
                    self._gaze_max_x = max(self._gaze_max_x, avg_dir[0])
                    self._gaze_min_y = min(self._gaze_min_y, avg_dir[1])
                    self._gaze_max_y = max(self._gaze_max_y, avg_dir[1])
                
                # Draw bounding box if we have data
                if self._gaze_min_x is not None:
                    box_x, box_y = 10, 10
                    box_w, box_h = 220, 80
                    
                    # Draw opaque black background
                    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                                (0, 0, 0), -1)
                    
                    # Yellow border
                    cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), 
                                COLOR_GAZE_AVG, 2)
                    
                    # White text (fully opaque, readable)
                    cv2.putText(img, "avgxy bounds:", (box_x + 5, box_y + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, f"x: [{self._gaze_min_x:+.3f}, {self._gaze_max_x:+.3f}]", 
                              (box_x + 5, box_y + 45),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, f"y: [{self._gaze_min_y:+.3f}, {self._gaze_max_y:+.3f}]", 
                              (box_x + 5, box_y + 65),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Face direction already computed during gaze transformation
        # Just get it for display purposes
        nose_tip_point = lm_avg(NOSE_TIP_POINTS)  # For drawing origin point
        face_dir = face_dir_prelim  # Already computed above
        
        nx, ny = int(nose_tip_point[0] * w), int(nose_tip_point[1] * h)
        face_end = (
            int(nx + face_dir[0] * FACE_VECTOR_SCALE_PX),
            int(ny + face_dir[1] * FACE_VECTOR_SCALE_PX),
        )
        cv2.circle(img, (nx, ny), 3, COLOR_FACE_AIM, -1)
        cv2.line(img, (nx, ny), face_end, COLOR_FACE_AIM, 2, cv2.LINE_AA)
        cv2.circle(img, face_end, vector_end_size, COLOR_FACE_AIM, -1)
        
        # Draw face orientation landmarks with darker face aim color if requested
        if self._show_landmarks_used:
            darker_face_color = tuple(int(c * DEF_LM_SPECIAL_BRIGHTNESS) for c in COLOR_FACE_AIM)
            
            # Cheekbone points - use FILTERED
            for idx in CHEEKBONE_R + CHEEKBONE_L:
                filt = lm(idx)
                x, y = int(filt[0] * w), int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Nose tip points - use FILTERED
            for idx in NOSE_TIP_POINTS:
                filt = lm(idx)
                x, y = int(filt[0] * w), int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Forehead points - use FILTERED
            for idx in [FOREHEAD_TOP_R, FOREHEAD_TOP_L]:
                filt = lm(idx)
                x, y = int(filt[0] * w), int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Chin points - use FILTERED
            for idx in CHIN_TIP_POINTS:
                filt = lm(idx)
                x, y = int(filt[0] * w), int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Old eye corner landmarks (for reference/reminder) - use FILTERED
            for idx in RIGHT_EYE_CORNERS + LEFT_EYE_CORNERS:
                filt = lm(idx)
                x, y = int(filt[0] * w), int(filt[1] * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Blend this overlay update
            img = cv2.addWeighted(overlay, ALPHA_LANDMARKS, img, 1.0 - ALPHA_LANDMARKS, 0)

        # Package gaze data for calibration/logging
        gaze_data = {
            'timestamp': datetime.now().isoformat(),
            'left_gaze': left["dir_norm"].tolist() if (left_open and left_eye_enabled) else None,
            'right_gaze': right["dir_norm"].tolist() if (right_open and right_eye_enabled) else None,
            'avg_gaze': avg_dir.tolist() if open_dirs else None,
            'face_aim': face_dir.tolist(),
            'left_open': left_open,
            'right_open': right_open,
            'left_enabled': left_eye_enabled,
            'right_enabled': right_eye_enabled,
            'landmarks': {idx: [p.x, p.y, p.z] for idx, p in enumerate(face_landmarks.landmark)}
        }

        return img, gaze_data


class VideoWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._current_frame = 0
        self._total_frames = 0
        self.setMinimumSize(640, 480)

    @QtCore.Slot(np.ndarray, int, int)
    def update_frame(self, frame: np.ndarray, current_frame: int, total_frames: int):
        # frame is BGR uint8
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
        ).copy()  # copy to detach from numpy buffer
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self._current_frame = current_frame
        self._total_frames = total_frames
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtCore.Qt.black)
        if self._pixmap is not None:
            # scale to fit while keeping aspect ratio
            target = self._pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            x = (self.width() - target.width()) // 2
            y = (self.height() - target.height()) // 2
            painter.drawPixmap(x, y, target)
            
            # Draw frame counter if available
            if self._total_frames > 0:
                painter.setPen(QtGui.QColor(255, 255, 255))
                painter.setFont(QtGui.QFont("Monospace", 10))
                frame_text = f"Frame: {self._current_frame}/{self._total_frames}"
                painter.drawText(10, self.height() - 10, frame_text)
        painter.end()


class ColorLegendWidget(QtWidgets.QWidget):
    """Widget showing color legend for gaze visualization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(130)
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        
        # Legend items (no title)
        y_offset = 15
        line_height = 20
        
        legend_items = [
            ("Left Eye Gaze", COLOR_GAZE_LEFT),
            ("Right Eye Gaze", COLOR_GAZE_RIGHT),
            ("Average Gaze", COLOR_GAZE_AVG),
            ("Face Aim", COLOR_FACE_AIM),
            ("All Landmarks", COLOR_LM_ALL),
        ]
        
        painter.setFont(QtGui.QFont("Sans", 9))
        
        for label, color_bgr in legend_items:
            # Draw color box
            color = QtGui.QColor(color_bgr[2], color_bgr[1], color_bgr[0])  # BGR to RGB
            painter.fillRect(10, y_offset - 10, 15, 15, color)
            painter.setPen(QtCore.Qt.black)
            painter.drawRect(10, y_offset - 10, 15, 15)
            
            # Draw label
            painter.drawText(30, y_offset, label)
            y_offset += line_height
        
        # Add note about darker landmarks
        painter.setFont(QtGui.QFont("Sans", 8))
        painter.setPen(QtGui.QColor(80, 80, 80))
        painter.drawText(10, y_offset + 5, "Note: Feature landmarks are darker")


class HotkeyHelpWidget(QtWidgets.QWidget):
    """Widget showing keyboard shortcuts."""
    
    def __init__(self, is_video_file=False, parent=None):
        super().__init__(parent)
        self.is_video_file = is_video_file
        self.setMinimumHeight(120 if is_video_file else 60)
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        
        # Hotkeys
        y_offset = 15
        line_height = 16
        
        painter.setFont(QtGui.QFont("Monospace", 9))
        painter.setPen(QtCore.Qt.black)
        
        hotkeys = [
            ("C", "Start calibration"),
            ("M", "Toggle mouse control"),
            ("Q", "Quit"),
        ]
        
        if self.is_video_file:
            hotkeys.extend([
                ("Space", "Pause/Play"),
                ("←/→", "Step frame"),
                ("Home/End", "Jump start/end"),
            ])
        
        for key, desc in hotkeys:
            text = f"{key:8s} {desc}"
            painter.drawText(10, y_offset, text)
            y_offset += line_height


class FiltersDialog(QtWidgets.QDialog):
    """Popup dialog for configuring filters for each landmark group."""
    
    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker
        self.setWindowTitle("Filter Settings")
        self.setModal(False)  # Non-modal so user can see main window
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create filter controls for each group
        self.filter_groups = ['iris_pupil', 'eye_corners', 'eye_lids', 'face_orientation', 'single_eye']
        self.group_labels = {
            'iris_pupil': 'Iris/Pupil',
            'eye_corners': 'Eye Corners',
            'eye_lids': 'Eye Lids',
            'face_orientation': 'Face Orientation',
            'single_eye': 'Single Eye Mode'
        }
        
        # Store widget references for each group
        self.enable_cbs = {}
        self.median_sliders = {}
        self.median_spins = {}
        self.ema_sliders = {}
        self.ema_spins = {}
        
        for group in self.filter_groups:
            group_box = QtWidgets.QGroupBox(self.group_labels[group])
            group_layout = QtWidgets.QFormLayout()
            
            # Enable checkbox
            enable_cb = QtWidgets.QCheckBox("Enable filtering")
            enable_cb.setChecked(DEF_FILTER_ENABLED)
            enable_cb.toggled.connect(
                lambda checked, g=group: self.worker.set_filter_enabled(g, checked))
            group_layout.addRow(enable_cb)
            self.enable_cbs[group] = enable_cb
            
            # Median window
            median_layout = QtWidgets.QHBoxLayout()
            median_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            median_slider.setRange(1, 21)
            median_slider.setValue(DEF_MEDIAN_WINDOW)
            median_spin = QtWidgets.QSpinBox()
            median_spin.setRange(1, 21)
            median_spin.setValue(DEF_MEDIAN_WINDOW)
            # Capture references properly
            median_slider.valueChanged.connect(
                lambda val, spin=median_spin: spin.setValue(val))
            median_spin.valueChanged.connect(
                lambda val, slider=median_slider: slider.setValue(val))
            median_slider.valueChanged.connect(
                lambda val, g=group: self.worker.set_filter_median_window(g, val))
            median_layout.addWidget(median_slider)
            median_layout.addWidget(median_spin)
            group_layout.addRow("Median window:", median_layout)
            self.median_sliders[group] = median_slider
            self.median_spins[group] = median_spin
            
            # EMA alpha (smoothing)
            ema_layout = QtWidgets.QHBoxLayout()
            ema_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            ema_slider.setRange(1, 100)
            ema_slider.setValue(int(DEF_EMA_ALPHA * 100))
            ema_spin = QtWidgets.QDoubleSpinBox()
            ema_spin.setRange(0.01, 1.0)
            ema_spin.setSingleStep(0.01)
            ema_spin.setDecimals(2)
            ema_spin.setValue(DEF_EMA_ALPHA)
            # Capture references properly
            ema_slider.valueChanged.connect(
                lambda val, spin=ema_spin: spin.setValue(val / 100.0))
            ema_spin.valueChanged.connect(
                lambda val, slider=ema_slider: slider.setValue(int(val * 100)))
            ema_slider.valueChanged.connect(
                lambda val, g=group: self.worker.set_filter_ema_alpha(g, val / 100.0))
            ema_layout.addWidget(ema_slider)
            ema_layout.addWidget(ema_spin)
            group_layout.addRow("Smoothing (EMA α):", ema_layout)
            self.ema_sliders[group] = ema_slider
            self.ema_spins[group] = ema_spin
            
            group_box.setLayout(group_layout)
            layout.addWidget(group_box)
        
        # Close button
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.resize(400, 600)
        
        # Don't apply values here - let the caller decide when to apply
        # (either defaults on first open, or loaded settings)
    
    def apply_current_values(self):
        """Apply current UI values to worker (for initialization or after loading)."""
        for group in self.filter_groups:
            self.worker.set_filter_enabled(group, self.enable_cbs[group].isChecked())
            self.worker.set_filter_median_window(group, self.median_spins[group].value())
            self.worker.set_filter_ema_alpha(group, self.ema_spins[group].value())
    
    def get_settings(self):
        """Get current filter settings as dict."""
        settings = {}
        for group in self.filter_groups:
            settings[group] = {
                'enabled': self.enable_cbs[group].isChecked(),
                'median_window': self.median_spins[group].value(),
                'ema_alpha': self.ema_spins[group].value()
            }
        return settings
    
    def set_settings(self, settings):
        """Apply settings dict to UI widgets."""
        for group in self.filter_groups:
            if group in settings:
                s = settings[group]
                if 'enabled' in s:
                    self.enable_cbs[group].setChecked(s['enabled'])
                if 'median_window' in s:
                    self.median_spins[group].setValue(s['median_window'])
                if 'ema_alpha' in s:
                    self.ema_spins[group].setValue(s['ema_alpha'])


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, video_file=None, usb_device=None, dev_device=None,
                 frame_delay=DEF_VFILE_FRAME_DELAY,
                 vfile_fps=DEF_VFILE_FPS, loop=DEF_VFILE_LOOP,
                 mouse_control=False, cal_auto_collect_s=3.0,
                 audio_enabled=None, use_tts=None):
        super().__init__()
        self.setWindowTitle("Eye Gaze Tracker - Accessibility HID Controller")
        
        # Mouse control state
        self.mouse_control_enabled = mouse_control
        
        # Audio settings (None = use saved/default, True/False = forced)
        self._cli_audio_enabled = audio_enabled
        self._cli_use_tts = use_tts
        # These will be set properly in load_settings()
        self.audio_enabled = app_settings.DEF_AUDIO_ENABLED
        self.use_tts = app_settings.DEF_USE_TTS
        
        # Calibration auto-collect duration
        self.cal_auto_collect_s = cal_auto_collect_s
        
        # Active calibration preset
        self.active_preset = None
        self.preset_actions = []  # Toolbar preset buttons
        
        # Compute UI scales (hierarchical: global * specific)
        self.ui_scale_global = app_settings.DEF_UI_SCALE_GLOBAL
        self.ui_scale_nav = self.ui_scale_global * app_settings.DEF_UI_SCALE_MAIN_WINDOW * app_settings.DEF_UI_SCALE_NAV_BUTTONS
        self.ui_scale_panel = self.ui_scale_global * app_settings.DEF_UI_SCALE_MAIN_WINDOW * app_settings.DEF_UI_SCALE_CONTROL_PANEL

        # Preset management lists (loaded from settings)
        self.presets_order = []
        self.presets_rank = []

        # Central video widget
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Thread + worker setup (create before toolbar so signals can connect)
        self.thread = QtCore.QThread(self)
        self.worker = VideoWorker(
            video_file=video_file,
            usb_device=usb_device,
            dev_device=dev_device,
            frame_delay=frame_delay,
            vfile_fps=vfile_fps,
            loop=loop
        )
        self.worker.moveToThread(self.thread)
        
        # Store video file flag
        self.is_video_file = video_file is not None
        
        # Calibration window
        self.calibration_window = None

        # Toolbar (needs worker to exist for signal connections)
        self._create_toolbar(self.is_video_file)

        # Status bar for basic info / errors
        self.status = self.statusBar()
        if video_file:
            mode_str = "looping" if loop else "single-pass"
            fps_str = f"{vfile_fps} fps" if vfile_fps else f"{frame_delay}s/frame"
            self.status.showMessage(f"Video file ({mode_str}, {fps_str}): {video_file}")
        elif usb_device:
            self.status.showMessage(f"USB camera: {usb_device}")
        elif dev_device:
            self.status.showMessage(f"Camera device: {dev_device}")
        else:
            self.status.showMessage("Connecting to TCP camera...")

        # Connect signals
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self.video_widget.update_frame)
        self.worker.gazeData.connect(self.handle_gaze_data, QtCore.Qt.DirectConnection)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Settings dock
        self._create_settings_dock(self.is_video_file)
        
        # Setup global keyboard shortcuts
        self._setup_shortcuts()
        
        # Initialize filter dialog state BEFORE loading settings
        self.filters_dialog = None
        self.saved_filter_settings = None
        
        # Load saved settings (will populate saved_filter_settings if found)
        self.load_settings()

        self.thread.start()
    
    def open_filters_dialog(self):
        """Open the filters configuration dialog."""
        if self.filters_dialog is None:
            self.filters_dialog = FiltersDialog(self.worker, self)
            # Apply saved settings if we have them, otherwise apply defaults
            if self.saved_filter_settings is not None:
                self.filters_dialog.set_settings(self.saved_filter_settings)
            # Apply current UI values to worker (either loaded or defaults)
            self.filters_dialog.apply_current_values()
        self.filters_dialog.show()
        self.filters_dialog.raise_()
        self.filters_dialog.activateWindow()

    def _create_toolbar(self, is_video_file):
        self.toolbar = self.addToolBar("Main")
        
        # Apply nav button scaling
        nav_font_size = app_settings.scaled_font_size(10, app_settings.DEF_UI_SCALE_MAIN_WINDOW, 
                                                       app_settings.DEF_UI_SCALE_NAV_BUTTONS)
        toolbar_font = QtGui.QFont()
        toolbar_font.setPointSize(nav_font_size)
        self.toolbar.setFont(toolbar_font)
        
        # Video controls (only for file playback)
        if is_video_file:
            def make_handler(name, func):
                def handler():
                    print(f"[UI] {name}")
                    func()
                return handler
            
            play_pause_action = QtGui.QAction("⏯ Pause/Play", self)
            play_pause_action.triggered.connect(make_handler("Play/Pause clicked", self.worker.toggle_pause))
            self.toolbar.addAction(play_pause_action)
            
            step_back_action = QtGui.QAction("⏮ Step Back", self)
            step_back_action.triggered.connect(make_handler("Step Back clicked", self.worker.step_backward))
            self.toolbar.addAction(step_back_action)
            
            step_fwd_action = QtGui.QAction("⏭ Step Fwd", self)
            step_fwd_action.triggered.connect(make_handler("Step Forward clicked", self.worker.step_forward))
            self.toolbar.addAction(step_fwd_action)
            
            jump_start_action = QtGui.QAction("⏪ Start", self)
            jump_start_action.triggered.connect(make_handler("Jump Start clicked", self.worker.jump_to_start))
            self.toolbar.addAction(jump_start_action)
            
            jump_end_action = QtGui.QAction("⏩ End", self)
            jump_end_action.triggered.connect(make_handler("Jump End clicked", self.worker.jump_to_end))
            self.toolbar.addAction(jump_end_action)
            
            self.toolbar.addSeparator()
        
        # Mouse Control toggle (checkable button)
        self.mouse_control_action = QtGui.QAction("🖱 Mouse Control", self)
        self.mouse_control_action.setCheckable(True)
        self.mouse_control_action.setChecked(self.mouse_control_enabled)
        self.mouse_control_action.toggled.connect(self.toggle_mouse_control)
        self.toolbar.addAction(self.mouse_control_action)
        
        # Calibration/Visualization button
        calibrate_action = QtGui.QAction("📍 Calib/Viz", self)
        calibrate_action.triggered.connect(self.start_calibration)
        self.toolbar.addAction(calibrate_action)
        self.toolbar.addSeparator()
        
        # Exit
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        self.toolbar.addAction(exit_action)
        
        self.toolbar.addSeparator()
        
        # Presets Manager button
        presets_action = QtGui.QAction("📂 Presets", self)
        presets_action.triggered.connect(self.open_presets_manager)
        self.toolbar.addAction(presets_action)
        
        # Separator before favorite presets
        self.presets_separator = self.toolbar.addSeparator()
        
        # Add favorite presets as buttons
        self._refresh_preset_buttons()
        
        # Notice area (after preset buttons)
        self.toolbar.addSeparator()
        self.notice_label = QtWidgets.QLabel("")
        self.notice_label.setMinimumWidth(200)
        self.notice_label.setStyleSheet("padding: 4px;")
        self.toolbar.addWidget(self.notice_label)
    
    def _update_notice(self):
        """Update the notice label based on calibration state."""
        if self.active_preset is None:
            # Check if any presets exist
            has_presets = app_settings.CAL_PRESETS_DIR.exists() and \
                         any(app_settings.CAL_PRESETS_DIR.glob('*.yaml'))
            if has_presets:
                self.notice_label.setText("No Default Preset")
                self.notice_label.setStyleSheet("padding: 4px; color: #003050; font-weight: bold;")
            else:
                self.notice_label.setText("Calibration Needed")
                self.notice_label.setStyleSheet("padding: 4px; color: #003050; font-weight: bold;")
        else:
            self.notice_label.setText("")
            self.notice_label.setStyleSheet("padding: 4px;")

    def _setup_shortcuts(self):
        """Setup global keyboard shortcuts that work regardless of focus."""
        def make_handler(name, func):
            def handler():
                print(f"[UI] {name}")
                func()
            return handler
        
        # Always available
        self.shortcut_quit = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
        self.shortcut_quit.activated.connect(self.close)
        
        # Mouse control toggle
        self.shortcut_mouse = QtGui.QShortcut(QtGui.QKeySequence("M"), self)
        self.shortcut_mouse.activated.connect(lambda: self.mouse_control_action.toggle())
        
        # Calibration
        self.shortcut_calib = QtGui.QShortcut(QtGui.QKeySequence("C"), self)
        self.shortcut_calib.activated.connect(self.start_calibration)
        
        # Video controls (only for file playback)
        if self.is_video_file:
            self.shortcut_pause = QtGui.QShortcut(QtGui.QKeySequence("Space"), self)
            self.shortcut_pause.activated.connect(make_handler("Space pressed", self.worker.toggle_pause))
            
            self.shortcut_left = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
            self.shortcut_left.activated.connect(make_handler("Left pressed", self.worker.step_backward))
            
            self.shortcut_right = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
            self.shortcut_right.activated.connect(make_handler("Right pressed", self.worker.step_forward))
            
            self.shortcut_home = QtGui.QShortcut(QtGui.QKeySequence("Home"), self)
            self.shortcut_home.activated.connect(make_handler("Home pressed", self.worker.jump_to_start))
            
            self.shortcut_end = QtGui.QShortcut(QtGui.QKeySequence("End"), self)
            self.shortcut_end.activated.connect(make_handler("End pressed", self.worker.jump_to_end))

    def _create_settings_dock(self, is_video_file):
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        widget = QtWidgets.QWidget(dock)
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Apply control panel scaling
        cp_font_size = app_settings.scaled_font_size(9, app_settings.DEF_UI_SCALE_MAIN_WINDOW,
                                                      app_settings.DEF_UI_SCALE_CONTROL_PANEL)
        cp_font = QtGui.QFont()
        cp_font.setPointSize(cp_font_size)
        widget.setFont(cp_font)
        
        # Image adjustments group
        img_group = QtWidgets.QGroupBox("Image Adjustments")
        img_layout = QtWidgets.QFormLayout()

        # Brightness: -100..100
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_spin = QtWidgets.QSpinBox()
        self.brightness_spin.setRange(-100, 100)
        # Keep slider and spinbox in sync
        self.brightness_slider.valueChanged.connect(self.brightness_spin.setValue)
        self.brightness_spin.valueChanged.connect(self.brightness_slider.setValue)
        # Send updates to worker (DirectConnection since worker loop is blocking)
        self.brightness_slider.valueChanged.connect(
            self.worker.set_brightness, QtCore.Qt.DirectConnection)

        # Contrast: 0.5..2.0 mapped to slider 50..200
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 2.0)
        self.contrast_spin.setSingleStep(0.1)
        # Sync slider <-> spinbox
        def on_contrast_slider(val: int):
            self.contrast_spin.blockSignals(True)
            self.contrast_spin.setValue(val / 100.0)
            self.contrast_spin.blockSignals(False)
        def on_contrast_spin(val: float):
            slider_val = int(val * 100)
            self.contrast_slider.blockSignals(True)
            self.contrast_slider.setValue(slider_val)
            self.contrast_slider.blockSignals(False)
        self.contrast_slider.valueChanged.connect(on_contrast_slider)
        self.contrast_spin.valueChanged.connect(on_contrast_spin)
        # Send to worker (DirectConnection since worker loop is blocking)
        self.contrast_slider.valueChanged.connect(
            self.worker.set_contrast_slider, QtCore.Qt.DirectConnection)
        self.contrast_spin.valueChanged.connect(
            self.worker.set_contrast, QtCore.Qt.DirectConnection)

        img_layout.addRow("Brightness", self._hbox(self.brightness_slider, self.brightness_spin))
        img_layout.addRow("Contrast", self._hbox(self.contrast_slider, self.contrast_spin))
        img_group.setLayout(img_layout)
        
        # Overlay options group
        overlay_group = QtWidgets.QGroupBox("Overlay Options")
        overlay_layout = QtWidgets.QVBoxLayout()
        
        # Landmark toggles
        self.show_raw_lm_cb = QtWidgets.QCheckBox("Raw LMs (unfiltered)")
        self.show_raw_lm_cb.setChecked(False)
        self.show_raw_lm_cb.toggled.connect(
            self.worker.set_show_landmarks_raw, QtCore.Qt.DirectConnection)
        
        self.show_all_lm_cb = QtWidgets.QCheckBox("Show all landmarks")
        self.show_all_lm_cb.setChecked(True)  # Default ON
        self.show_used_lm_cb = QtWidgets.QCheckBox("Highlight gaze landmarks")
        self.show_used_lm_cb.setChecked(True)
        self.show_all_lm_cb.toggled.connect(
            self.worker.set_show_landmarks_all, QtCore.Qt.DirectConnection)
        self.show_used_lm_cb.toggled.connect(
            self.worker.set_show_landmarks_used, QtCore.Qt.DirectConnection)
        
        overlay_layout.addWidget(self.show_raw_lm_cb)
        overlay_layout.addWidget(self.show_all_lm_cb)
        overlay_layout.addWidget(self.show_used_lm_cb)
        overlay_group.setLayout(overlay_layout)
        
        # Hotkey help (compact)
        help_group = QtWidgets.QGroupBox("Hotkeys")
        help_layout = QtWidgets.QVBoxLayout()
        self.hotkey_widget = HotkeyHelpWidget(is_video_file=is_video_file)
        help_layout.addWidget(self.hotkey_widget)
        help_group.setLayout(help_layout)
        
        # Color legend (no title label)
        self.legend_widget = ColorLegendWidget()
        
        # Landmark filter controls group
        filter_group = QtWidgets.QGroupBox("Landmark Options")
        filter_layout = QtWidgets.QFormLayout()
        
        # Eye enable checkboxes
        eye_layout = QtWidgets.QHBoxLayout()
        self.left_eye_cb = QtWidgets.QCheckBox("Left eye")
        self.left_eye_cb.setChecked(True)
        self.left_eye_cb.toggled.connect(
            self.worker.set_left_eye_enabled, QtCore.Qt.DirectConnection)
        eye_layout.addWidget(self.left_eye_cb)
        
        self.right_eye_cb = QtWidgets.QCheckBox("Right eye")
        self.right_eye_cb.setChecked(True)
        self.right_eye_cb.toggled.connect(
            self.worker.set_right_eye_enabled, QtCore.Qt.DirectConnection)
        eye_layout.addWidget(self.right_eye_cb)
        
        filter_layout.addRow(eye_layout)
        
        # Filters button to open dialog
        self.filters_btn = QtWidgets.QPushButton("Filters...")
        self.filters_btn.clicked.connect(self.open_filters_dialog)
        filter_layout.addRow(self.filters_btn)
        
        filter_group.setLayout(filter_layout)
        
        # Save/Reset buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        
        self.reset_gaze_btn = QtWidgets.QPushButton("Reset Gaze")
        self.reset_gaze_btn.clicked.connect(self.reset_gaze_bounds)
        buttons_layout.addWidget(self.reset_gaze_btn)
        
        self.save_stats_btn = QtWidgets.QPushButton("Save Stats")
        self.save_stats_btn.clicked.connect(self.save_stats)
        buttons_layout.addWidget(self.save_stats_btn)
        
        self.reset_btn = QtWidgets.QPushButton("Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        buttons_layout.addWidget(self.reset_btn)
        
        self.save_btn = QtWidgets.QPushButton("Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        buttons_layout.addWidget(self.save_btn)
        
        # Add all groups to main layout
        layout.addWidget(img_group)
        layout.addWidget(overlay_group)
        layout.addWidget(filter_group)
        layout.addWidget(help_group)
        layout.addWidget(self.legend_widget)
        layout.addWidget(QtWidgets.QLabel())  # Spacer
        layout.addLayout(buttons_layout)
        layout.addStretch()

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        
        # Trigger initial values to worker (after all connections are made)
        self.brightness_slider.setValue(DEF_BRIGHTNESS)
        self.contrast_slider.setValue(int(DEF_CONTRAST * 100))

    @staticmethod
    def _hbox(*widgets):
        box = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w)
        return box

    @QtCore.Slot(str)
    def on_worker_error(self, msg: str):
        self.status.showMessage(msg)

    def reset_gaze_bounds(self):
        """Reset gaze bounds tracking."""
        self.worker.reset_gaze_bounds()
        self.status.showMessage("Gaze bounds reset", 2000)

    def save_stats(self):
        """Save gaze statistics to YAML file."""
        # Get current bounds from worker
        with QtCore.QMutexLocker(self.worker._gaze_bounds_lock):
            if self.worker._gaze_min_x is None:
                self.status.showMessage("No gaze data to save yet", 3000)
                return
            
            stats = {
                'timestamp': datetime.now().isoformat(),
                'gaze_bounds': {
                    'x_min': float(self.worker._gaze_min_x),
                    'x_max': float(self.worker._gaze_max_x),
                    'y_min': float(self.worker._gaze_min_y),
                    'y_max': float(self.worker._gaze_max_y),
                    'x_range': float(self.worker._gaze_max_x - self.worker._gaze_min_x),
                    'y_range': float(self.worker._gaze_max_y - self.worker._gaze_min_y),
                }
            }
        
        # Save to stats.yaml in config dir (parent of CAL_PRESETS_DIR)
        config_dir = app_settings.CAL_PRESETS_DIR.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        stats_file = config_dir / 'stats.yaml'
        try:
            with open(stats_file, 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
            print(f"[UI] Saved stats to {stats_file}")
            print(f"[UI] Gaze bounds: x=[{stats['gaze_bounds']['x_min']:.3f}, {stats['gaze_bounds']['x_max']:.3f}], y=[{stats['gaze_bounds']['y_min']:.3f}, {stats['gaze_bounds']['y_max']:.3f}]")
            self.status.showMessage(f"Stats saved to {stats_file.name}", 3000)
        except Exception as e:
            print(f"[UI] Error saving stats: {e}")
            self.status.showMessage(f"Error saving stats: {e}", 5000)

    def reset_to_defaults(self):
        """Reset all controls to default values."""
        # Image adjustments
        self.brightness_slider.setValue(DEF_BRIGHTNESS)
        self.contrast_slider.setValue(int(DEF_CONTRAST * 100))
        
        self.status.showMessage("Reset to defaults", 2000)

    def save_settings(self):
        """Save current settings to config file."""
        config = {
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value() / 100.0,
            'left_eye_enabled': self.left_eye_cb.isChecked(),
            'right_eye_enabled': self.right_eye_cb.isChecked(),
            'presets_order': self.presets_order,
            'presets_rank': self.presets_rank,
            'audio_enabled': self.audio_enabled,
            'use_tts': self.use_tts,
        }
        
        # Add filter settings if dialog exists
        if self.filters_dialog is not None:
            config['filters'] = self.filters_dialog.get_settings()
        
        # Save to settings.yaml in config dir (parent of CAL_PRESETS_DIR)
        config_dir = app_settings.CAL_PRESETS_DIR.parent
        config_dir.mkdir(parents=True, exist_ok=True)
        settings_file = config_dir / 'settings.yaml'
        try:
            with open(settings_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"[UI] Saved settings to {settings_file}")
            self.status.showMessage(f"Settings saved to {settings_file.name}", 3000)
        except Exception as e:
            print(f"[UI] Error saving settings: {e}")
            self.status.showMessage(f"Error saving: {e}", 5000)
    
    def load_settings(self):
        """Load settings from config file."""
        config_dir = app_settings.CAL_PRESETS_DIR.parent
        settings_file = config_dir / 'settings.yaml'
        if not settings_file.exists():
            self._apply_audio_cli_overrides()
            self._update_notice()
            return
        
        try:
            with open(settings_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config:
                self._apply_audio_cli_overrides()
                self._update_notice()
                return
            
            # Apply loaded settings
            if 'brightness' in config:
                self.brightness_slider.setValue(config['brightness'])
            if 'contrast' in config:
                self.contrast_slider.setValue(int(config['contrast'] * 100))
            if 'left_eye_enabled' in config:
                self.left_eye_cb.setChecked(config['left_eye_enabled'])
            if 'right_eye_enabled' in config:
                self.right_eye_cb.setChecked(config['right_eye_enabled'])
            
            # Load preset management lists
            if 'presets_order' in config:
                self.presets_order = config['presets_order']
            if 'presets_rank' in config:
                self.presets_rank = config['presets_rank']
            
            # Clean lists of missing files
            if app_settings.CAL_PRESETS_DIR.exists():
                existing = {f.name for f in app_settings.CAL_PRESETS_DIR.glob('*.yaml')}
                self.presets_order = [f for f in self.presets_order if f in existing]
                self.presets_rank = [f for f in self.presets_rank if f in existing]
            
            # Load default preset if available
            if self.presets_rank:
                default_file = app_settings.CAL_PRESETS_DIR / self.presets_rank[0]
                if default_file.exists():
                    try:
                        preset = calibration.CalibrationData.load(default_file)
                        self.select_preset(preset)
                        print(f"[UI] Loaded default preset: {preset.name}")
                    except Exception as e:
                        print(f"[UI] Error loading default preset: {e}")
            
            # Apply filter settings to worker immediately
            if 'filters' in config:
                self.saved_filter_settings = config['filters']
                # Send to worker right away (don't wait for dialog to open)
                for group, settings in self.saved_filter_settings.items():
                    if 'enabled' in settings:
                        self.worker.set_filter_enabled(group, settings['enabled'])
                    if 'median_window' in settings:
                        self.worker.set_filter_median_window(group, settings['median_window'])
                    if 'ema_alpha' in settings:
                        self.worker.set_filter_ema_alpha(group, settings['ema_alpha'])
                print(f"[UI] Applied filter settings from config")
            else:
                self.saved_filter_settings = None
            
            # Load audio settings (then apply CLI overrides if provided)
            if 'audio_enabled' in config:
                self.audio_enabled = config['audio_enabled']
            if 'use_tts' in config:
                self.use_tts = config['use_tts']
            
            self._apply_audio_cli_overrides()
            
            print(f"[UI] Loaded settings from {settings_file}")
            self.status.showMessage("Settings loaded", 2000)
            
            # Update notice
            self._update_notice()
            
        except Exception as e:
            print(f"[UI] Error loading settings: {e}")
            self._update_notice()

    def start_calibration(self):
        """Start calibration process."""
        if self.calibration_window is not None:
            return  # Already calibrating
        
        # Create calibration window
        screen = self.screen()
        screen_geom = screen.geometry()
        screen_size = screen_geom.size()  # Logical size for Qt drawing
        dpr = screen.devicePixelRatio()    # For converting to physical coordinates
        
        print(f"[UI] Screen DPR: {dpr}")
        print(f"[UI] Logical size (for drawing): {screen_size.width()}x{screen_size.height()}")
        print(f"[UI] Physical size (for mouse): {int(screen_size.width() * dpr)}x{int(screen_size.height() * dpr)}")
        
        self.calibration_window = calibration.CalibrationWindow(
            device_pixel_ratio=dpr,  # Pass DPR for coordinate conversion
            screen_size=screen_size,
            grid_size=app_settings.DEF_CAL_GRID_SIZE,
            active_preset=self.active_preset,
            cal_auto_collect_s=self.cal_auto_collect_s,
            audio_enabled=self.audio_enabled,
            use_tts=self.use_tts,
        )
        
        # Connect worker gaze data to calibration window (DirectConnection for responsiveness)
        self.worker.gazeData.connect(
            self.calibration_window.update_frame_and_data, QtCore.Qt.DirectConnection)
        
        # Connect calibration completion/cancellation
        self.calibration_window.calibrationComplete.connect(self.on_calibration_complete)
        self.calibration_window.calibrationCancelled.connect(self.on_calibration_cancelled)
        
        # Connect audio settings changes (to persist user's preference)
        self.calibration_window.audioSettingsChanged.connect(self._on_audio_settings_changed)
        
        # Show calibration window
        self.calibration_window.showFullScreen()
        
        # Process events to ensure window is fully shown and sized
        QtCore.QCoreApplication.processEvents()
        
        # Finalize calibration setup (regenerates grid if window size changed)
        self.calibration_window.finalize_after_show()
        
        self.status.showMessage("Calibration in progress...")
    
    def on_calibration_complete(self, cal_data: calibration.CalibrationData):
        """Handle calibration completion."""
        self.status.showMessage(f"Calibration complete: {cal_data.name}", 5000)
        self.calibration_window = None
        
        # Auto-select the new preset
        self.active_preset = cal_data
        
        # Send corner data to worker for plotting
        corners = cal_data.get_corners()
        self.worker.set_calibration_corners(corners)
        
        # Check if this was set as default (stored during save dialog)
        if hasattr(cal_data, '_set_as_default') and cal_data._set_as_default:
            filename = getattr(cal_data, '_filename', None)
            if filename:
                if filename in self.presets_rank:
                    self.presets_rank.remove(filename)
                self.presets_rank.insert(0, filename)
                self.save_settings()
        
        # Refresh favorite buttons
        self._refresh_preset_buttons()
        self._update_notice()
        
        self.status.showMessage(f"Active preset: {cal_data.name}", 3000)

    def on_calibration_cancelled(self):
        """Handle calibration cancellation."""
        self.status.showMessage("Calibration cancelled", 2000)
        self.calibration_window = None

    def _on_audio_settings_changed(self, audio_enabled: bool, use_tts: bool):
        """Handle audio settings change from calibration window."""
        self.audio_enabled = audio_enabled
        self.use_tts = use_tts
        print(f"[UI] Audio settings changed: enabled={audio_enabled}, use_tts={use_tts}")
    
    def _apply_audio_cli_overrides(self):
        """Apply CLI audio overrides and update global state."""
        # CLI overrides take precedence over loaded/default values
        if self._cli_audio_enabled is not None:
            self.audio_enabled = self._cli_audio_enabled
        if self._cli_use_tts is not None:
            self.use_tts = self._cli_use_tts
        
        # Update calibration module's global state
        calibration.set_audio_state(self.audio_enabled, self.use_tts)
        print(f"[UI] Audio: enabled={self.audio_enabled}, use_tts={self.use_tts}")

    def open_presets_manager(self):
        """Open the presets manager dialog."""
        dialog = calibration.PresetsManagerDialog(
            presets_order=self.presets_order,
            presets_rank=self.presets_rank,
            active_preset=self.active_preset,
            parent=self
        )
        dialog.presetSelected.connect(self.select_preset)
        dialog.presetsOrderChanged.connect(self._on_presets_order_changed)
        dialog.presetsRankChanged.connect(self._on_presets_rank_changed)
        dialog.exec()
        
        # Refresh favorite buttons after manager closes
        self._refresh_preset_buttons()
        self._update_notice()
    
    def _on_presets_order_changed(self, new_order: list):
        """Handle presets order change from manager."""
        self.presets_order = new_order
        self.save_settings()
    
    def _on_presets_rank_changed(self, new_rank: list):
        """Handle presets rank change from manager."""
        self.presets_rank = new_rank
        self.save_settings()
        # If rank changed and we have a new default, load it
        if new_rank and (self.active_preset is None or 
                        getattr(self.active_preset, '_filename', None) != new_rank[0]):
            default_file = app_settings.CAL_PRESETS_DIR / new_rank[0]
            if default_file.exists():
                try:
                    preset = calibration.CalibrationData.load(default_file)
                    self.select_preset(preset)
                except Exception as e:
                    print(f"[UI] Error loading new default: {e}")
        self._update_notice()
    
    def select_preset(self, preset: calibration.CalibrationData):
        """Select and activate a preset."""
        self.active_preset = preset
        self.status.showMessage(f"Active preset: {preset.name}", 3000)
        
        # Send corner data to worker for plotting
        if preset:
            corners = preset.get_corners()
            self.worker.set_calibration_corners(corners)
        else:
            self.worker.set_calibration_corners(None)
        
        self._update_notice()

    def _refresh_preset_buttons(self):
        """Refresh favorite preset buttons in toolbar."""
        print("[UI] Refreshing preset buttons...")
        start_time = time.time()
        
        # Remove old preset actions
        for action in self.preset_actions:
            self.toolbar.removeAction(action)
        self.preset_actions.clear()
        
        # Add favorite presets as buttons (only loads favorites, not all presets)
        favorites = calibration.CalibrationData.get_favorites()
        print(f"[UI] Found {len(favorites)} favorites in {time.time() - start_time:.2f}s")
        
        for preset in favorites:
            action = QtGui.QAction(f"⭐ {preset.name}", self)
            
            # Tooltip with descriptions
            tooltip = f"<b>{preset.name}</b>"
            if preset.camera_description:
                tooltip += f"<br>Camera: {preset.camera_description}"
            if preset.display_description:
                tooltip += f"<br>Display: {preset.display_description}"
            action.setToolTip(tooltip)
            
            # Capture preset in lambda properly
            action.triggered.connect(lambda checked, p=preset: self.select_preset(p))
            
            self.toolbar.addAction(action)
            self.preset_actions.append(action)
        
        print(f"[UI] Preset buttons refreshed in {time.time() - start_time:.2f}s")

    def toggle_mouse_control(self, enabled: bool):
        """Toggle mouse control on/off."""
        self.mouse_control_enabled = enabled
        if enabled:
            self.status.showMessage("Mouse control ENABLED - gaze will move cursor", 3000)
        else:
            self.status.showMessage("Mouse control disabled - visualization only", 3000)

    @QtCore.Slot(np.ndarray, dict)
    def handle_gaze_data(self, frame: np.ndarray, gaze_data: dict):
        """Handle gaze data for mouse control."""
        if not self.mouse_control_enabled:
            return
        
        if not gaze_data or gaze_data.get('avg_gaze') is None:
            return
        
        # Get screen size
        screen = self.screen()
        screen_w = screen.size().width()
        screen_h = screen.size().height()
        
        # Map gaze to screen coordinates
        if self.active_preset:
            # Use calibration data for mapping
            cx, cy = self._map_gaze_with_calibration(gaze_data, screen_w, screen_h)
        else:
            # Fallback: simple mapping (will be inaccurate)
            avg_gaze = gaze_data['avg_gaze']
            # Flip horizontal (gaze left = screen left, which means negative -> left side)
            cx = screen_w // 2 - int(avg_gaze[0] * screen_w * 0.5)
            cy = screen_h // 2 + int(avg_gaze[1] * screen_h * 0.5)
        
        # Clamp to screen bounds
        cx = max(0, min(screen_w - 1, cx))
        cy = max(0, min(screen_h - 1, cy))
        
        # Move cursor
        mouse_control.move_cursor(cx, cy)
    
    def _map_gaze_with_calibration(self, gaze_data: dict, screen_w: int, screen_h: int) -> Tuple[int, int]:
        """Map gaze data to screen coordinates using calibration preset."""
        if not self.active_preset:
            return screen_w // 2, screen_h // 2
        
        avg_gaze = gaze_data.get('avg_gaze')
        if avg_gaze is None:
            return screen_w // 2, screen_h // 2
        
        # Get averaged calibration data
        cal_points = self.active_preset.get_averaged_data()
        
        if not cal_points:
            # No calibration data, use fallback
            return screen_w // 2 - int(avg_gaze[0] * screen_w * 0.5), \
                   screen_h // 2 + int(avg_gaze[1] * screen_h * 0.5)
        
        # Simple nearest-neighbor interpolation for now
        # TODO: Implement proper polynomial interpolation
        gaze_x, gaze_y = avg_gaze[0], avg_gaze[1]
        
        # Find closest calibration points and interpolate
        # For now, use inverse distance weighting
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for (screen_x, screen_y), data in cal_points.items():
            if data.get('avg_gaze') is None:
                continue
            
            cal_gaze = data['avg_gaze']
            
            # Distance in gaze space
            dist = np.sqrt((gaze_x - cal_gaze[0])**2 + (gaze_y - cal_gaze[1])**2)
            
            if dist < 0.001:
                # Very close - just use this point
                return screen_x, screen_y
            
            # Inverse distance weight
            weight = 1.0 / (dist ** 2)
            total_weight += weight
            weighted_x += screen_x * weight
            weighted_y += screen_y * weight
        
        if total_weight > 0:
            return int(weighted_x / total_weight), int(weighted_y / total_weight)
        else:
            # Fallback
            return screen_w // 2, screen_h // 2

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Stop worker
        if hasattr(self, 'worker') and self.worker is not None:
            self.worker.stop()
        
        # Wait for thread to finish
        if hasattr(self, 'thread') and self.thread is not None:
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait(2000)
        
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Eye Gaze Tracker - Accessibility HID Controller Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Video File Controls:
  Space     Pause/Play
  ←/→       Step backward/forward one frame
  Home/End  Jump to start/end of video
  C         Start calibration
  M         Toggle mouse control
  Q         Quit application
  
Calibration Window Controls:
  D         Toggle audio on/off
  T         Toggle TTS/tones mode
  
Camera Sources:
  Default: TCP stream from camsettingsh264.py (camhost:camport)
  --usb: USB camera by identifier (from v4l2-ctl --list-devices)
  --dev: Direct device path (e.g., /dev/video0)
  -f/--file: Video file for testing
        """
    )
    parser.add_argument("-f", "--file", type=str, help="Path to video file for testing")
    parser.add_argument("--usb", type=str, help="USB camera identifier")
    parser.add_argument("--dev", type=str, help="Direct device path (e.g., /dev/video0)")
    parser.add_argument("--no-loop", action="store_true", help="Disable video looping")
    parser.add_argument("--vfile-fps", "--ffps", type=float, metavar="FPS", help="Playback FPS for video file")
    parser.add_argument("--vfile-frame-delay", "--fd", type=float, metavar="SECONDS", 
                       default=DEF_VFILE_FRAME_DELAY, help=f"Delay between video file frames (default: {DEF_VFILE_FRAME_DELAY})")
    parser.add_argument("--mouse", action="store_true", help="Enable mouse control on startup")
    parser.add_argument("--cal-auto-s", type=float, default=3.0, metavar="SECONDS",
                       help="Duration for 'C' key auto-collection in calibration (default: 3.0)")
    parser.add_argument("-P", "--print-data", type=int, default=0, metavar="LEVEL",
                       help="Verbosity level for plot data output (0=off, 1+=on)")
    
    # Audio control arguments
    audio_group = parser.add_mutually_exclusive_group()
    audio_group.add_argument("--no-audio", "-A", action="store_true",
                            help="Disable all audio at startup")
    audio_group.add_argument("--audio", action="store_true",
                            help="Force audio on (override saved settings)")
    
    tts_group = parser.add_mutually_exclusive_group()
    tts_group.add_argument("--no-tts", "-T", action="store_true",
                          help="Use tones instead of TTS")
    tts_group.add_argument("--tts", action="store_true",
                          help="Force TTS on (override saved settings)")
    
    args = parser.parse_args()
    
    # Set global print data level
    set_print_data_level(args.print_data)
    
    # Validate camera source options
    camera_sources = sum([args.file is not None, args.usb is not None, args.dev is not None])
    if camera_sources > 1:
        print("Error: Only one camera source can be specified (--file, --usb, or --dev)")
        sys.exit(1)
    
    # Process --dev shorthand
    dev_device = args.dev
    if dev_device and dev_device.isdigit():
        dev_device = f"/dev/video{dev_device}"
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(
        video_file=args.file,
        usb_device=args.usb,
        dev_device=dev_device,
        frame_delay=args.vfile_frame_delay,
        vfile_fps=args.vfile_fps,
        loop=(not args.no_loop),
        mouse_control=args.mouse,
        cal_auto_collect_s=args.cal_auto_s,
        # Audio settings: None means "use saved/default", True/False forces that value
        audio_enabled=False if args.no_audio else (True if args.audio else None),
        use_tts=False if args.no_tts else (True if args.tts else None),
    )
    win.resize(1400, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
