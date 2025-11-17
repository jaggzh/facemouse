#!/usr/bin/env python3

import sys
import time
import math
import argparse

import av  # pip install av
import numpy as np
import cv2
import mediapipe as mp

from PySide6 import QtCore, QtGui, QtWidgets

import camsettingsh264 as camsettings


# ---- Visualization and processing constants ----

# Landmark colors (B, G, R)
COLOR_LM_ALL = (64, 64, 255)
COLOR_LM_USED = (0, 255, 255)

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
RIGHT_EYE_CORNERS = [33, 133] # Outer, inner
LEFT_EYE_CORNERS = [263, 362] # Outer, inner
LEFT_EYE_LIDS = [386, 374] # Top, bottom
RIGHT_EYE_LIDS = [159, 145] # Top, bottom
NOSE_TIP = 1


class VideoWorker(QtCore.QObject):
    """Background worker: decodes H.264 or video file, applies pre-filter, runs Mediapipe,
    draws debug overlays, and emits annotated frames as NumPy arrays (BGR).
    """

    frameReady = QtCore.Signal(np.ndarray)
    error = QtCore.Signal(str)
    finished = QtCore.Signal()

    def __init__(self, video_file=None, parent=None):
        super().__init__(parent)
        self._running = False
        self._video_file = video_file

        # Image pre-filter controls (alpha, beta for convertScaleAbs)
        self._contrast = 1.0  # alpha
        self._brightness = 0.0  # beta
        self._contrast_lock = QtCore.QMutex()
        self._brightness_lock = QtCore.QMutex()

        # Eye open ratio threshold (height / width) for blink / eye-closed detection
        self._eye_open_thresh = 0.20

        # Landmark visualization toggles
        self._show_landmarks_all = False
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
        # direct mapping: slider -100..100 -> beta -100..100
        with QtCore.QMutexLocker(self._brightness_lock):
            self._brightness = float(value)
            print(f"[Worker] Brightness set to {self._brightness}")

    @QtCore.Slot(int)
    def set_contrast_slider(self, value: int):
        # slider 50..200 -> alpha 0.5..2.0
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
        """Process a video file (for testing)."""
        print(f"[Worker] Opening video file: {self._video_file}")
        
        while self._running:
            try:
                container = av.open(self._video_file)
            except av.AVError as e:
                msg = f"Failed to open video file: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
                break

            print("[Worker] Video file opened, starting decode loop")

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

                    # Emit the annotated frame
                    self.frameReady.emit(img)

                    # For low FPS test videos, add a small delay to make it viewable
                    # Adjust this if your test video is truly 1 fps
                    #time.sleep(0.033)  # ~30 fps display rate
                    time.sleep(1)  # ~30 fps display rate

            except av.AVError as e:
                msg = f"Decode error: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)

            finally:
                try:
                    container.close()
                except Exception:
                    pass

            # For file playback, stop after one pass
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
            except av.AVError as e:
                msg = f"Failed to open stream: {e}"
                print("[Worker]", msg)
                self.error.emit(msg)
                time.sleep(1.0)
                continue

            print("[Worker] Stream opened, starting decode loop")

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

                    # Emit the annotated frame
                    self.frameReady.emit(img)

            except av.AVError as e:
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

        def lm(idx: int):
            p = face_landmarks.landmark[idx]
            return np.array([p.x, p.y, p.z], dtype=np.float32)

        # Draw all landmarks (into a separate overlay) if enabled
        overlay = img.copy()
        if self._show_landmarks_all:
            for idx, p in enumerate(face_landmarks.landmark):
                x = int(p.x * w)
                y = int(p.y * h)
                cv2.circle(overlay, (x, y), 1, COLOR_LM_ALL, -1)

        # Helper to compute eye metrics given index sets
        def eye_info(iris_idx, corners_idx, lids_idx):
            iris_pts = np.array([lm(i) for i in iris_idx], dtype=np.float32)
            iris_center = iris_pts.mean(axis=0)

            c0, c1 = lm(corners_idx[0]), lm(corners_idx[1])
            eye_center = 0.5 * (c0 + c1)

            # Horizontal eye width in normalized coords
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
            }

        left = eye_info(LEFT_IRIS, LEFT_EYE_CORNERS, LEFT_EYE_LIDS)
        right = eye_info(RIGHT_IRIS, RIGHT_EYE_CORNERS, RIGHT_EYE_LIDS)

        # Highlight the specific landmarks we are using, if requested
        if self._show_landmarks_used:
            def draw_used(info):
                for idx in (
                    list(info["iris_idx"])
                    + list(info["corners_idx"])
                    + list(info["lids_idx"])
                ):
                    p = face_landmarks.landmark[idx]
                    x = int(p.x * w)
                    y = int(p.y * h)
                    cv2.circle(overlay, (x, y), 2, COLOR_LM_USED, -1)

            draw_used(left)
            draw_used(right)

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
                cv2.circle(img, end_pt, 3, color, -1)

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
            cv2.circle(img, end_pt, 3, COLOR_GAZE_AVG, -1)
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

        # Approximate face aim using a simple plane normal from eyes + nose
        left_eye_center = 0.5 * (lm(LEFT_EYE_CORNERS[0]) + lm(LEFT_EYE_CORNERS[1]))
        right_eye_center = 0.5 * (
            lm(RIGHT_EYE_CORNERS[0]) + lm(RIGHT_EYE_CORNERS[1])
        )
        nose = lm(NOSE_TIP)

        v1 = right_eye_center - left_eye_center
        v2 = nose - left_eye_center
        n = np.cross(v1, v2)
        # Flip so "forward" roughly means toward camera (negative z)
        if n[2] > 0:
            n = -n

        # Use x,y components as a rough 2D direction for debug
        face_dir = np.array([n[0], n[1]], dtype=np.float32)
        norm = np.linalg.norm(face_dir) + 1e-6
        face_dir /= norm

        nx, ny = int(nose[0] * w), int(nose[1] * h)
        face_end = (
            int(nx + face_dir[0] * FACE_VECTOR_SCALE_PX),
            int(ny + face_dir[1] * FACE_VECTOR_SCALE_PX),
        )
        cv2.circle(img, (nx, ny), 3, COLOR_FACE_AIM, -1)
        cv2.line(img, (nx, ny), face_end, COLOR_FACE_AIM, 2, cv2.LINE_AA)
        cv2.circle(img, face_end, 3, COLOR_FACE_AIM, -1)

        return img


class VideoWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self.setMinimumSize(640, 480)

    @QtCore.Slot(np.ndarray)
    def update_frame(self, frame: np.ndarray):
        # frame is BGR uint8
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
        ).copy()  # copy to detach from numpy buffer
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
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

        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, video_file=None):
        super().__init__()
        self.setWindowTitle("Eye Gaze Prototype - Accessibility HID Controller")

        # Central video widget
        self.video_widget = VideoWidget(self)
        self.setCentralWidget(self.video_widget)

        # Simple toolbar with an Exit button for pre-MVP
        exit_action = QtGui.QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(exit_action)

        # Status bar for basic info / errors
        self.status = self.statusBar()
        if video_file:
            self.status.showMessage(f"Loading video file: {video_file}")
        else:
            self.status.showMessage("Connecting to camera...")

        # Thread + worker setup
        self.thread = QtCore.QThread(self)
        self.worker = VideoWorker(video_file=video_file)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self.video_widget.update_frame)
        self.worker.error.connect(self.on_worker_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Settings dock (brightness / contrast and landmark toggles)
        self._create_settings_dock()

        self.thread.start()

    def _create_settings_dock(self):
        dock = QtWidgets.QDockWidget("Image / Overlay Controls", self)
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )

        widget = QtWidgets.QWidget(dock)
        layout = QtWidgets.QFormLayout(widget)

        # Brightness: -100..100
        self.brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_spin = QtWidgets.QSpinBox()
        self.brightness_spin.setRange(-100, 100)
        self.brightness_spin.setValue(0)

        # Keep slider and spinbox in sync
        self.brightness_slider.valueChanged.connect(self.brightness_spin.setValue)
        self.brightness_spin.valueChanged.connect(self.brightness_slider.setValue)

        # Send updates to worker
        self.brightness_slider.valueChanged.connect(self.worker.set_brightness)

        # Contrast: 0.5..2.0 mapped to slider 50..200
        self.contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.contrast_slider.setRange(50, 200)
        self.contrast_slider.setValue(100)  # 1.0
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0.5, 2.0)
        self.contrast_spin.setSingleStep(0.1)
        self.contrast_spin.setValue(1.0)

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

        # Send to worker
        self.contrast_slider.valueChanged.connect(self.worker.set_contrast_slider)
        self.contrast_spin.valueChanged.connect(self.worker.set_contrast)

        # Landmark toggles
        self.show_all_lm_cb = QtWidgets.QCheckBox("Show all landmarks")
        self.show_all_lm_cb.setChecked(False)
        self.show_used_lm_cb = QtWidgets.QCheckBox("Highlight gaze landmarks")
        self.show_used_lm_cb.setChecked(True)

        self.show_all_lm_cb.toggled.connect(self.worker.set_show_landmarks_all)
        self.show_used_lm_cb.toggled.connect(self.worker.set_show_landmarks_used)

        layout.addRow("Brightness", self._hbox(self.brightness_slider, self.brightness_spin))
        layout.addRow("Contrast", self._hbox(self.contrast_slider, self.contrast_spin))
        layout.addRow(self.show_all_lm_cb)
        layout.addRow(self.show_used_lm_cb)

        widget.setLayout(layout)
        dock.setWidget(widget)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Stop worker and wait for thread to finish
        if self.worker is not None:
            self.worker.stop()
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        return super().closeEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)


def main():
    parser = argparse.ArgumentParser(
        description="Eye Gaze Tracker - Accessibility HID Controller Prototype"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to video file for testing (instead of live camera stream)"
    )
    
    args = parser.parse_args()
    
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(video_file=args.file)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
