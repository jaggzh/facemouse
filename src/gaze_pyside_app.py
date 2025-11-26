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
import time
import math
import argparse
from pathlib import Path

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


# ============================================================================
# Configuration defaults
# ============================================================================
DEF_VFILE_FRAME_DELAY = 1.0  # Delay between frames in seconds (for video files)
DEF_VFILE_FPS = None  # FPS for video file playback (None = use frame delay)
DEF_VFILE_LOOP = True  # Loop video files by default

# Landmark filtering defaults
DEF_FILTER_ENABLED = True
DEF_MEDIAN_WINDOW = 3  # Number of samples for median filter
DEF_EMA_ALPHA = 0.3  # Exponential moving average (0=very smooth, 1=no smoothing)

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
COLOR_GAZE_LEFT = (0, 255, 0)
COLOR_GAZE_RIGHT = (0, 200, 255)
COLOR_GAZE_AVG = (255, 255, 0)
COLOR_FACE_AIM = (255, 0, 255)

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
CHIN_TIP_POINTS = [203, 428, 175]


class VideoWorker(QtCore.QObject):
    """Background worker: decodes H.264 or video file, applies pre-filter, runs Mediapipe,
    draws debug overlays, and emits annotated frames as NumPy arrays (BGR).
    """

    frameReady = QtCore.Signal(np.ndarray, int, int)  # frame, current_frame, total_frames
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, video_file=None, frame_delay=DEF_VFILE_FRAME_DELAY, 
                 vfile_fps=DEF_VFILE_FPS, loop=DEF_VFILE_LOOP, parent=None):
        super().__init__(parent)
        self._running = False
        self._video_file = video_file
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

        # Landmark filtering
        self._landmark_filter = filters.LandmarkFilterSet(
            median_window=DEF_MEDIAN_WINDOW,
            ema_alpha=DEF_EMA_ALPHA,
            enabled=DEF_FILTER_ENABLED
        )
        self._filter_lock = QtCore.QMutex()

        # Eye open ratio threshold (height / width) for blink / eye-closed detection
        self._eye_open_thresh = 0.20

        # Landmark visualization toggles
        self._show_landmarks_all = True  # Default ON
        self._show_landmarks_used = True

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
    def set_show_landmarks_all(self, enabled: bool):
        self._show_landmarks_all = bool(enabled)

    @QtCore.Slot(bool)
    def set_show_landmarks_used(self, enabled: bool):
        self._show_landmarks_used = bool(enabled)

    # ---- Slots for landmark filter parameters ----

    @QtCore.Slot(bool)
    def set_filter_enabled(self, enabled: bool):
        with QtCore.QMutexLocker(self._filter_lock):
            self._landmark_filter.set_parameters(enabled=enabled)
            print(f"[Worker] Filtering {'enabled' if enabled else 'disabled'}")

    @QtCore.Slot(int)
    def set_median_window(self, value: int):
        with QtCore.QMutexLocker(self._filter_lock):
            self._landmark_filter.set_parameters(median_window=value)
            print(f"[Worker] Median window set to {value}")

    @QtCore.Slot(float)
    def set_ema_alpha(self, value: float):
        with QtCore.QMutexLocker(self._filter_lock):
            self._landmark_filter.set_parameters(ema_alpha=value)
            print(f"[Worker] EMA alpha set to {value:.3f}")

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
        else:
            # Process camera stream
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
                    try:
                        img = self._process_and_overlay(img)
                    except Exception as e:
                        self.error.emit(f"Processing error: {e}")

                    # Emit the annotated frame with frame info
                    self.frameReady.emit(img, current_frame_idx + 1, total_frames)

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
                    try:
                        img = self._process_and_overlay(img)
                    except Exception as e:
                        self.error.emit(f"Processing error: {e}")

                    # Emit the annotated frame (no frame count for streams)
                    frame_count += 1
                    self.frameReady.emit(img, frame_count, -1)

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
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return img

        face_landmarks = results.multi_face_landmarks[0]

        # Dynamic landmark scaling based on resolution
        base_lm_scaler = min(1.0, w / 750.0)
        lm_size = max(1, int(DEF_LM_SIZE * base_lm_scaler))
        lm_special_size = max(1, int(DEF_LM_SIZE * DEF_LM_SPECIAL_SCALE * base_lm_scaler))
        vector_end_size = max(1, int(DEF_VECTOR_END_SIZE * base_lm_scaler))

        def lm(idx: int):
            p = face_landmarks.landmark[idx]
            raw_point = np.array([p.x, p.y, p.z], dtype=np.float32)
            # Apply filtering
            with QtCore.QMutexLocker(self._filter_lock):
                filtered_point = self._landmark_filter.update(idx, raw_point)
            return filtered_point
        
        def lm_avg(indices):
            """Average position of multiple landmarks."""
            pts = np.array([lm(i) for i in indices], dtype=np.float32)
            return pts.mean(axis=0)

        # Draw all landmarks (into a separate overlay) if enabled
        overlay = img.copy()
        if self._show_landmarks_all:
            for idx, p in enumerate(face_landmarks.landmark):
                x = int(p.x * w)
                y = int(p.y * h)
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
            
            def draw_used(info, vector_color):
                # Iris + pupil: darker version of gaze color
                iris_color = darken_color(vector_color, 0.7)
                for idx in list(info["iris_idx"]) + [info["pupil_idx"]]:
                    p = face_landmarks.landmark[idx]
                    x = int(p.x * w)
                    y = int(p.y * h)
                    cv2.circle(overlay, (x, y), lm_special_size, iris_color, -1)
                
                # Corners: darker + hue shifted for distinctiveness
                corner_color = shift_hue_bgr(darken_color(vector_color, 0.6), 40)
                for idx in info["corners_idx"]:
                    p = face_landmarks.landmark[idx]
                    x = int(p.x * w)
                    y = int(p.y * h)
                    cv2.circle(overlay, (x, y), lm_special_size, corner_color, -1)
                
                # Lids: similar to iris
                lid_color = darken_color(vector_color, 0.65)
                for idx in info["lids_idx"]:
                    p = face_landmarks.landmark[idx]
                    x = int(p.x * w)
                    y = int(p.y * h)
                    cv2.circle(overlay, (x, y), lm_special_size, lid_color, -1)

            draw_used(left, COLOR_GAZE_LEFT)
            draw_used(right, COLOR_GAZE_RIGHT)

        # Blend overlay back onto image for landmarks
        img = cv2.addWeighted(overlay, ALPHA_LANDMARKS, img, 1.0 - ALPHA_LANDMARKS, 0)

        # Simple blink / eye-closed detection
        left_open = left["open_ratio"] > self._eye_open_thresh
        right_open = right["open_ratio"] > self._eye_open_thresh

        # Draw per-eye gaze vectors (in normalized eye space)
        def draw_eye_vector(info, is_open: bool, color):
            cx, cy = info["eye_px"]
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

                label = f"{vx:+.2f},{vy:+.2f} ({info['open_ratio']:.2f})"
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

        draw_eye_vector(left, left_open, COLOR_GAZE_LEFT)
        draw_eye_vector(right, right_open, COLOR_GAZE_RIGHT)

        # Combined gaze (average of open eyes) – for now just a debug indicator
        open_dirs = []
        centers_px = []
        for info, is_open in ((left, left_open), (right, right_open)):
            if is_open:
                open_dirs.append(info["dir_norm"])
                centers_px.append(info["eye_px"])

        if open_dirs:
            avg_dir = np.mean(open_dirs, axis=0)
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
                f"avg {avg_dir[0]:+.2f},{avg_dir[1]:+.2f}",
                (end_pt[0] + 5, end_pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                COLOR_GAZE_AVG,
                1,
                cv2.LINE_AA,
            )

        # New face orientation calculation
        # Method 1: Cheekbone-to-nose vector
        cheekbone_r = lm_avg(CHEEKBONE_R)
        cheekbone_l = lm_avg(CHEEKBONE_L)
        cheekbone_center = 0.5 * (cheekbone_r + cheekbone_l)
        nose_tip_point = lm_avg(NOSE_TIP_POINTS)
        
        vector_cheeks_nose = nose_tip_point - cheekbone_center
        vector_cheeks_nose = vector_cheeks_nose / (np.linalg.norm(vector_cheeks_nose) + 1e-6)
        
        # Method 2: Forehead-chin plane normal
        forehead_r = lm(FOREHEAD_TOP_R)
        forehead_l = lm(FOREHEAD_TOP_L)
        chin_tip = lm_avg(CHIN_TIP_POINTS)
        
        v1 = forehead_l - forehead_r
        v2 = chin_tip - forehead_r
        vector_face_normal = np.cross(v1, v2)
        # Flip if needed so it points roughly forward (negative z)
        if vector_face_normal[2] > 0:
            vector_face_normal = -vector_face_normal
        vector_face_normal = vector_face_normal / (np.linalg.norm(vector_face_normal) + 1e-6)
        
        # Combine both methods
        final_face_vector = 0.5 * (vector_cheeks_nose + vector_face_normal)
        final_face_vector = final_face_vector / (np.linalg.norm(final_face_vector) + 1e-6)
        
        # Use x,y components as 2D direction for display
        face_dir = np.array([final_face_vector[0], final_face_vector[1]], dtype=np.float32)
        
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
            
            # Cheekbone points
            for idx in CHEEKBONE_R + CHEEKBONE_L:
                p = face_landmarks.landmark[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Nose tip points
            for idx in NOSE_TIP_POINTS:
                p = face_landmarks.landmark[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Forehead points
            for idx in [FOREHEAD_TOP_R, FOREHEAD_TOP_L]:
                p = face_landmarks.landmark[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Chin points
            for idx in CHIN_TIP_POINTS:
                p = face_landmarks.landmark[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Old eye corner landmarks (for reference/reminder)
            for idx in RIGHT_EYE_CORNERS + LEFT_EYE_CORNERS:
                p = face_landmarks.landmark[idx]
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(overlay, (x, y), lm_special_size, darker_face_color, -1)
            
            # Blend this overlay update
            img = cv2.addWeighted(overlay, ALPHA_LANDMARKS, img, 1.0 - ALPHA_LANDMARKS, 0)

        return img


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
        self.setMinimumHeight(150)  # Increased for note
        
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        
        # Title
        painter.setPen(QtCore.Qt.black)
        painter.setFont(QtGui.QFont("Sans", 10, QtGui.QFont.Bold))
        painter.drawText(10, 20, "Color Legend")
        
        # Legend items
        y_offset = 40
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, video_file=None, frame_delay=DEF_VFILE_FRAME_DELAY,
                 vfile_fps=DEF_VFILE_FPS, loop=DEF_VFILE_LOOP):
        super().__init__()
        self.setWindowTitle("Eye Gaze Tracker - Accessibility HID Controller")

        # Central video widget
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Thread + worker setup (create before toolbar so signals can connect)
        self.thread = QtCore.QThread(self)
        self.worker = VideoWorker(
            video_file=video_file,
            frame_delay=frame_delay,
            vfile_fps=vfile_fps,
            loop=loop
        )
        self.worker.moveToThread(self.thread)
        
        # Store video file flag
        self.is_video_file = video_file is not None

        # Toolbar (needs worker to exist for signal connections)
        self._create_toolbar(self.is_video_file)

        # Status bar for basic info / errors
        self.status = self.statusBar()
        if video_file:
            mode_str = "looping" if loop else "single-pass"
            fps_str = f"{vfile_fps} fps" if vfile_fps else f"{frame_delay}s/frame"
            self.status.showMessage(f"Video file ({mode_str}, {fps_str}): {video_file}")
        else:
            self.status.showMessage("Connecting to camera...")

        # Connect signals
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self.video_widget.update_frame)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Settings dock
        self._create_settings_dock(self.is_video_file)
        
        # Setup global keyboard shortcuts
        self._setup_shortcuts()

        self.thread.start()

    def _create_toolbar(self, is_video_file):
        toolbar = self.addToolBar("Main")
        
        # Video controls (only for file playback)
        if is_video_file:
            def make_handler(name, func):
                def handler():
                    print(f"[UI] {name}")
                    func()
                return handler
            
            play_pause_action = QtGui.QAction("⏯ Pause/Play", self)
            play_pause_action.triggered.connect(make_handler("Play/Pause clicked", self.worker.toggle_pause))
            toolbar.addAction(play_pause_action)
            
            step_back_action = QtGui.QAction("⏮ Step Back", self)
            step_back_action.triggered.connect(make_handler("Step Back clicked", self.worker.step_backward))
            toolbar.addAction(step_back_action)
            
            step_fwd_action = QtGui.QAction("⏭ Step Fwd", self)
            step_fwd_action.triggered.connect(make_handler("Step Forward clicked", self.worker.step_forward))
            toolbar.addAction(step_fwd_action)
            
            jump_start_action = QtGui.QAction("⏪ Start", self)
            jump_start_action.triggered.connect(make_handler("Jump Start clicked", self.worker.jump_to_start))
            toolbar.addAction(jump_start_action)
            
            jump_end_action = QtGui.QAction("⏩ End", self)
            jump_end_action.triggered.connect(make_handler("Jump End clicked", self.worker.jump_to_end))
            toolbar.addAction(jump_end_action)
            
            toolbar.addSeparator()
        
        # Exit
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        toolbar.addAction(exit_action)

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
        self.show_all_lm_cb = QtWidgets.QCheckBox("Show all landmarks")
        self.show_all_lm_cb.setChecked(True)  # Default ON
        self.show_used_lm_cb = QtWidgets.QCheckBox("Highlight gaze landmarks")
        self.show_used_lm_cb.setChecked(True)

        self.show_all_lm_cb.toggled.connect(
            self.worker.set_show_landmarks_all, QtCore.Qt.DirectConnection)
        self.show_used_lm_cb.toggled.connect(
            self.worker.set_show_landmarks_used, QtCore.Qt.DirectConnection)
        
        overlay_layout.addWidget(self.show_all_lm_cb)
        overlay_layout.addWidget(self.show_used_lm_cb)
        overlay_group.setLayout(overlay_layout)
        
        # Color legend
        legend_group = QtWidgets.QGroupBox("Color Legend")
        legend_layout = QtWidgets.QVBoxLayout()
        self.legend_widget = ColorLegendWidget()
        legend_layout.addWidget(self.legend_widget)
        legend_group.setLayout(legend_layout)
        
        # Video controls help (if video file)
        if is_video_file:
            help_group = QtWidgets.QGroupBox("Video Controls")
            help_layout = QtWidgets.QVBoxLayout()
            help_text = QtWidgets.QLabel(
                "Space: Pause/Play\n"
                "←/→: Step frame\n"
                "Home/End: Jump to start/end\n"
                "Q: Quit"
            )
            help_text.setFont(QtGui.QFont("Monospace", 9))
            help_layout.addWidget(help_text)
            help_group.setLayout(help_layout)
        
        # Landmark filter controls group
        filter_group = QtWidgets.QGroupBox("Landmark Filtering")
        filter_layout = QtWidgets.QFormLayout()
        
        # Filter enable toggle
        self.filter_enabled_cb = QtWidgets.QCheckBox("Enable filtering")
        self.filter_enabled_cb.toggled.connect(
            self.worker.set_filter_enabled, QtCore.Qt.DirectConnection)
        filter_layout.addRow(self.filter_enabled_cb)
        
        # Median window size: 1..10
        self.median_window_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.median_window_slider.setRange(1, 10)
        self.median_window_spin = QtWidgets.QSpinBox()
        self.median_window_spin.setRange(1, 10)
        
        # Sync slider and spinbox
        self.median_window_slider.valueChanged.connect(self.median_window_spin.setValue)
        self.median_window_spin.valueChanged.connect(self.median_window_slider.setValue)
        
        # Send to worker (DirectConnection since worker loop is blocking)
        self.median_window_slider.valueChanged.connect(
            self.worker.set_median_window, QtCore.Qt.DirectConnection)
        
        filter_layout.addRow("Median Window", self._hbox(self.median_window_slider, self.median_window_spin))
        
        # EMA alpha: 0.01..1.0 (mapped to slider 1..100)
        self.ema_alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.ema_alpha_slider.setRange(1, 100)
        self.ema_alpha_spin = QtWidgets.QDoubleSpinBox()
        self.ema_alpha_spin.setRange(0.01, 1.0)
        self.ema_alpha_spin.setSingleStep(0.01)
        
        # Sync slider <-> spinbox
        def on_ema_slider(val: int):
            self.ema_alpha_spin.blockSignals(True)
            self.ema_alpha_spin.setValue(val / 100.0)
            self.ema_alpha_spin.blockSignals(False)
        
        def on_ema_spin(val: float):
            slider_val = int(val * 100)
            self.ema_alpha_slider.blockSignals(True)
            self.ema_alpha_slider.setValue(slider_val)
            self.ema_alpha_slider.blockSignals(False)
        
        self.ema_alpha_slider.valueChanged.connect(on_ema_slider)
        self.ema_alpha_spin.valueChanged.connect(on_ema_spin)
        
        # Send to worker (DirectConnection since worker loop is blocking)
        self.ema_alpha_spin.valueChanged.connect(
            self.worker.set_ema_alpha, QtCore.Qt.DirectConnection)
        
        filter_layout.addRow("Smoothing (α)", self._hbox(self.ema_alpha_slider, self.ema_alpha_spin))
        
        # Add note about smoothing
        note_label = QtWidgets.QLabel("Lower α = smoother, higher = more responsive")
        note_label.setFont(QtGui.QFont("Sans", 8))
        note_label.setWordWrap(True)
        filter_layout.addRow(note_label)
        
        filter_group.setLayout(filter_layout)
        
        # Save/Reset buttons
        buttons_layout = QtWidgets.QHBoxLayout()
        
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
        layout.addWidget(legend_group)
        if is_video_file:
            layout.addWidget(help_group)
        layout.addWidget(QtWidgets.QLabel())  # Spacer
        layout.addLayout(buttons_layout)
        layout.addStretch()

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        
        # Trigger initial values to worker (after all connections are made)
        self.brightness_slider.setValue(DEF_BRIGHTNESS)
        self.contrast_slider.setValue(int(DEF_CONTRAST * 100))
        self.filter_enabled_cb.setChecked(DEF_FILTER_ENABLED)
        self.median_window_slider.setValue(DEF_MEDIAN_WINDOW)
        self.ema_alpha_slider.setValue(int(DEF_EMA_ALPHA * 100))

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

    def reset_to_defaults(self):
        """Reset all controls to default values."""
        # Image adjustments
        self.brightness_slider.setValue(DEF_BRIGHTNESS)
        self.contrast_slider.setValue(int(DEF_CONTRAST * 100))
        
        # Filters
        self.filter_enabled_cb.setChecked(DEF_FILTER_ENABLED)
        self.median_window_slider.setValue(DEF_MEDIAN_WINDOW)
        self.ema_alpha_slider.setValue(int(DEF_EMA_ALPHA * 100))
        
        self.status.showMessage("Reset to defaults", 2000)

    def save_settings(self):
        """Save current settings to config file (placeholder for now)."""
        # TODO: Implement YAML config save
        config = {
            'brightness': self.brightness_slider.value(),
            'contrast': self.contrast_slider.value() / 100.0,
            'filter_enabled': self.filter_enabled_cb.isChecked(),
            'median_window': self.median_window_slider.value(),
            'ema_alpha': self.ema_alpha_slider.value() / 100.0,
        }
        print(f"[UI] Would save config: {config}")
        self.status.showMessage("Settings saved (not yet implemented)", 2000)

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
  Q         Quit application
        """
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to video file for testing (instead of live camera stream)"
    )
    parser.add_argument(
        "--no-loop",
        action="store_true",
        help=f"Disable video looping (default: loop enabled)"
    )
    parser.add_argument(
        "--vfile-fps", "--ffps",
        type=float,
        metavar="FPS",
        help=f"Playback FPS for video file. Overrides --vfile-frame-delay. (default: use frame delay)"
    )
    parser.add_argument(
        "--vfile-frame-delay", "--fd",
        type=float,
        metavar="SECONDS",
        default=DEF_VFILE_FRAME_DELAY,
        help=f"Delay between video file frames in seconds. Float. (default: {DEF_VFILE_FRAME_DELAY})"
    )
    
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(
        video_file=args.file,
        frame_delay=args.vfile_frame_delay,
        vfile_fps=args.vfile_fps,
        loop=(not args.no_loop)
    )
    win.resize(1400, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()