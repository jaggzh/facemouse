#!/usr/bin/env python3
"""
Calibration system for gaze tracking.

Provides fullscreen calibration UI, data collection, and preset management.
"""

import random
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import settings


# TTS queue and lock for sequential playback
_tts_lock = threading.Lock()
_tts_process = None


def tts(msg: str):
    """Text-to-speech output. Waits for previous speech to complete."""
    if settings.DEF_CAL_NO_TTS:
        return
    
    def _speak():
        global _tts_process
        with _tts_lock:
            # Wait for any previous speech to finish
            if _tts_process is not None:
                try:
                    _tts_process.wait(timeout=10)
                except:
                    pass
            
            try:
                # Run synchronously within this thread
                _tts_process = subprocess.Popen(
                    ["vpi", "--", msg],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                _tts_process.wait()  # Wait for this speech to complete
            except Exception as e:
                print(f"[TTS] Failed: {e}")
    
    # Run in background thread so we don't block the UI
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()


def distant_with_noise(
    current_point: Tuple[int, int],
    remaining_points: List[Tuple[int, int]],
    choices: int = settings.DEF_CAL_DISTANT_CHOICES,
    larger_choices: int = settings.DEF_CAL_DISTANT_LARGER_CHOICES,
    large_set_prob: float = settings.DEF_CAL_DISTANT_LARGE_SET_PROB
) -> Tuple[int, int]:
    """
    Select next calibration point from distant options with controlled randomness.
    
    This keeps users from drifting toward center by preferring distant points,
    but adds variation to avoid being too predictable.
    """
    if not remaining_points:
        return None
    
    # Calculate distances from current point
    distances = []
    for pt in remaining_points:
        dist = np.sqrt((pt[0] - current_point[0])**2 + (pt[1] - current_point[1])**2)
        distances.append((dist, pt))
    
    # Sort by distance (farthest first)
    distances.sort(reverse=True, key=lambda x: x[0])
    
    # Decide on choice set size
    if random.random() <= large_set_prob:
        set_size = min(larger_choices, len(distances))
    else:
        set_size = min(choices, len(distances))
    
    # Randomly select from the farthest points
    selected = random.choice(distances[:set_size])
    return selected[1]


class CalibrationData:
    """Stores calibration data for a single preset."""
    
    def __init__(self, name: str = "", grid_size: int = settings.DEF_CAL_GRID_SIZE):
        self.name = name  # Short UI name
        self.camera_description = ""  # e.g., "Webcam medium high"
        self.display_description = ""  # e.g., "3-foot"
        self.is_favorite = False
        self.sort_order = 0
        self.grid_size = grid_size
        self.timestamp = datetime.now()
        self.points = {}  # (screen_x, screen_y) -> list of samples
        
    def add_sample(self, screen_pos: Tuple[int, int], sample_data: Dict[str, Any]):
        """Add a sample for a calibration point."""
        if screen_pos not in self.points:
            self.points[screen_pos] = []
        self.points[screen_pos].append(sample_data)
    
    def remove_last_sample(self, screen_pos: Tuple[int, int]) -> bool:
        """Remove the last sample from a point. Returns True if successful."""
        if screen_pos in self.points and self.points[screen_pos]:
            self.points[screen_pos].pop()
            if not self.points[screen_pos]:
                del self.points[screen_pos]
            return True
        return False
    
    def get_sample_count(self, screen_pos: Tuple[int, int]) -> int:
        """Get number of samples collected for a point."""
        return len(self.points.get(screen_pos, []))
    
    def get_averaged_data(self) -> Dict[Tuple[int, int], Dict]:
        """Get averaged gaze data for each calibration point."""
        averaged = {}
        for pos, samples in self.points.items():
            if not samples:
                continue
            
            # Average the gaze vectors
            avg_left = []
            avg_right = []
            avg_gaze = []
            avg_face = []
            
            for sample in samples:
                if sample.get('left_gaze'):
                    avg_left.append(sample['left_gaze'])
                if sample.get('right_gaze'):
                    avg_right.append(sample['right_gaze'])
                if sample.get('avg_gaze'):
                    avg_gaze.append(sample['avg_gaze'])
                if sample.get('face_aim'):
                    avg_face.append(sample['face_aim'])
            
            averaged[pos] = {
                'left_gaze': np.mean(avg_left, axis=0).tolist() if avg_left else None,
                'right_gaze': np.mean(avg_right, axis=0).tolist() if avg_right else None,
                'avg_gaze': np.mean(avg_gaze, axis=0).tolist() if avg_gaze else None,
                'face_aim': np.mean(avg_face, axis=0).tolist() if avg_face else None,
            }
        
        return averaged
    
    @staticmethod
    def _convert_to_python_types(obj):
        """Recursively convert numpy types to Python native types for YAML serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: CalibrationData._convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [CalibrationData._convert_to_python_types(item) for item in obj]
        else:
            return obj
    
    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        # Convert samples to ensure no numpy types
        converted_points = []
        for pos, samples in self.points.items():
            converted_samples = [
                self._convert_to_python_types(sample) for sample in samples
            ]
            converted_points.append({
                'screen_x': int(pos[0]),
                'screen_y': int(pos[1]),
                'samples': converted_samples
            })
        
        return {
            'name': self.name,
            'camera_description': self.camera_description,
            'display_description': self.display_description,
            'is_favorite': self.is_favorite,
            'sort_order': self.sort_order,
            'grid_size': self.grid_size,
            'timestamp': self.timestamp.isoformat(),
            'points': converted_points
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        """Create from dictionary loaded from YAML."""
        cal = cls(name=data.get('name', ''), grid_size=data.get('grid_size', 9))
        cal.camera_description = data.get('camera_description', '')
        cal.display_description = data.get('display_description', '')
        cal.is_favorite = data.get('is_favorite', False)
        cal.sort_order = data.get('sort_order', 0)
        cal.timestamp = datetime.fromisoformat(data['timestamp'])
        for pt_data in data.get('points', []):
            pos = (pt_data['screen_x'], pt_data['screen_y'])
            cal.points[pos] = pt_data['samples']
        return cal
    
    def save(self, filename: Optional[Path] = None) -> Path:
        """Save calibration data to YAML file."""
        if filename is None:
            # Create safe filename from name
            safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in self.name)
            safe_name = safe_name.strip() or 'unnamed'
            filename = settings.CAL_PRESETS_DIR / f"{safe_name}.yaml"
        
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        print(f"[Calibration] Saved to {filename}")
        return filename
    
    @classmethod
    def load(cls, filename: Path) -> 'CalibrationData':
        """Load calibration data from YAML file."""
        with open(filename, 'r') as f:
            try:
                # Try safe load first
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                # If that fails, try to load with full loader (for numpy types)
                f.seek(0)
                try:
                    data = yaml.unsafe_load(f)
                except Exception as e:
                    raise ValueError(f"Could not parse YAML: {e}")
        
        return cls.from_dict(data)
    
    @classmethod
    def list_presets(cls) -> List['CalibrationData']:
        """List all saved presets."""
        presets = []
        print(f"[Calibration] Looking for presets in {settings.CAL_PRESETS_DIR}")
        if settings.CAL_PRESETS_DIR.exists():
            yaml_files = list(settings.CAL_PRESETS_DIR.glob('*.yaml'))
            print(f"[Calibration] Found {len(yaml_files)} yaml files: {yaml_files}")
            for f in yaml_files:
                try:
                    preset = cls.load(f)
                    print(f"[Calibration] Loaded preset: {preset.name}")
                    presets.append(preset)
                except Exception as e:
                    print(f"[Calibration] Error loading {f}: {e}")
        else:
            print(f"[Calibration] Presets directory does not exist")
        
        # Sort by sort_order, then by name
        presets.sort(key=lambda p: (p.sort_order, p.name))
        return presets
    
    @classmethod
    def get_favorites(cls) -> List['CalibrationData']:
        """Get favorited presets in sort order."""
        return [p for p in cls.list_presets() if p.is_favorite]


class CalibrationWindow(QtWidgets.QWidget):
    """Fullscreen calibration window with video underlay."""
    
    # Signals
    calibrationComplete = QtCore.Signal(CalibrationData)
    calibrationCancelled = QtCore.Signal()
    
    def __init__(self, screen_size: QtCore.QSize, grid_size: int = settings.DEF_CAL_GRID_SIZE, 
                 active_preset: CalibrationData = None, parent=None):
        super().__init__(parent)
        self.screen_size = screen_size
        self.grid_size = grid_size
        self.active_preset = active_preset  # For cursor visualization
        
        # Calibration state
        self.cal_data = CalibrationData(grid_size=grid_size)
        self.grid_points = self._generate_grid_points()
        self.remaining_points = list(self.grid_points)
        self.current_point = None
        self.current_frame = None
        self.current_gaze_data = None
        
        # Computed cursor position (screen coordinates)
        self.cursor_screen_pos = None
        
        # UI setup
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.FramelessWindowHint)
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())
        # Keep cursor visible for assistant to click buttons
        self.setCursor(QtCore.Qt.ArrowCursor)
        
        # Create controls
        self._create_controls()
        
        # Start calibration
        self._next_point()
        tts(settings.CAL_TTS_MESSAGES['start'])
    
    def _generate_grid_points(self) -> List[Tuple[int, int]]:
        """Generate grid of calibration points."""
        rows, cols = settings.CAL_GRID_CONFIGS[self.grid_size]
        w, h = self.screen_size.width(), self.screen_size.height()
        
        points = []
        for row in range(rows):
            for col in range(cols):
                # Place points at edges and evenly spaced
                x = int((col / (cols - 1)) * w) if cols > 1 else w // 2
                y = int((row / (rows - 1)) * h) if rows > 1 else h // 2
                points.append((x, y))
        
        print(f"[Calibration] Generated {len(points)} grid points: {points[:3]}...")
        return points
    
    def _create_controls(self):
        """Create control buttons."""
        # Button panel at bottom center
        self.button_widget = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QHBoxLayout(self.button_widget)
        
        # Apply calibration button scaling
        btn_font_size = settings.scaled_font_size(12, settings.DEF_UI_SCALE_CAL_BUTTONS)
        btn_font = QtGui.QFont()
        btn_font.setPointSize(btn_font_size)
        btn_font.setBold(True)
        
        btn_min_w = settings.scaled_size(150, settings.DEF_UI_SCALE_CAL_BUTTONS)
        btn_min_h = settings.scaled_size(50, settings.DEF_UI_SCALE_CAL_BUTTONS)
        
        self.accept_btn = QtWidgets.QPushButton("Accept Point (A)")
        self.accept_btn.setFont(btn_font)
        self.accept_btn.setMinimumSize(btn_min_w, btn_min_h)
        self.accept_btn.clicked.connect(self.accept_point)
        
        self.undo_btn = QtWidgets.QPushButton("Undo Last (U)")
        self.undo_btn.setFont(btn_font)
        self.undo_btn.setMinimumSize(btn_min_w, btn_min_h)
        self.undo_btn.clicked.connect(self.undo_point)
        
        self.cancel_btn = QtWidgets.QPushButton("Cancel (Esc)")
        self.cancel_btn.setFont(btn_font)
        self.cancel_btn.setMinimumSize(btn_min_w, btn_min_h)
        self.cancel_btn.clicked.connect(self.cancel_calibration)
        
        button_layout.addWidget(self.accept_btn)
        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.cancel_btn)
        
        # Position at bottom center
        self.button_widget.adjustSize()
        self.button_widget.move(
            (self.screen_size.width() - self.button_widget.width()) // 2,
            self.screen_size.height() - self.button_widget.height() - 20
        )
    
    def _next_point(self):
        """Select and display next calibration point (sequential for MVP)."""
        if not self.remaining_points:
            # Calibration complete
            self._complete_calibration()
            return
        
        # Sequential order for simplicity (start corner, work through grid)
        self.current_point = self.remaining_points.pop(0)
        
        self.update()
        
        # TTS direction hint
        direction = self._get_direction_text(self.current_point)
        tts(settings.CAL_TTS_MESSAGES['point'].format(direction=direction))
        print(f"[Calibration] Next point: {self.current_point} ({direction})")
    
    def _get_direction_text(self, point: Tuple[int, int]) -> str:
        """Get human-readable direction for a point."""
        x, y = point
        w, h = self.screen_size.width(), self.screen_size.height()
        
        vert = "top" if y < h / 3 else ("bottom" if y > 2 * h / 3 else "middle")
        horiz = "left" if x < w / 3 else ("right" if x > 2 * w / 3 else "center")
        
        if vert == "middle" and horiz == "center":
            return "center"
        elif vert == "middle":
            return horiz
        elif horiz == "center":
            return vert
        else:
            return f"{vert} {horiz}"
    
    @QtCore.Slot(np.ndarray, dict)
    def update_frame_and_data(self, frame: np.ndarray, gaze_data: dict):
        """Update video frame and current gaze data."""
        self.current_frame = frame
        self.current_gaze_data = gaze_data
        
        # Compute cursor screen position from gaze data
        # For now, use a simple mapping - will be replaced with calibration interpolation
        if gaze_data and gaze_data.get('avg_gaze') is not None:
            avg_gaze = gaze_data['avg_gaze']
            # Map gaze vector (roughly -1 to 1) to screen coordinates
            # This is placeholder until calibration provides proper mapping
            w, h = self.screen_size.width(), self.screen_size.height()
            
            if self.active_preset:
                # Use calibration data for mapping
                cx, cy = self._map_gaze_with_preset(gaze_data, w, h)
            else:
                # Simple mapping (flip horizontal: gaze left = screen left)
                cx = w // 2 - int(avg_gaze[0] * w * 0.5)
                cy = h // 2 + int(avg_gaze[1] * h * 0.5)
            
            # Clamp to screen bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            self.cursor_screen_pos = (cx, cy)
        else:
            self.cursor_screen_pos = None
        
        self.update()
    
    def _map_gaze_with_preset(self, gaze_data: dict, screen_w: int, screen_h: int) -> Tuple[int, int]:
        """Map gaze data to screen coordinates using calibration preset."""
        avg_gaze = gaze_data.get('avg_gaze')
        if avg_gaze is None:
            return screen_w // 2, screen_h // 2
        
        # Get averaged calibration data
        cal_points = self.active_preset.get_averaged_data()
        
        if not cal_points:
            # No calibration data, use fallback
            return screen_w // 2 - int(avg_gaze[0] * screen_w * 0.5), \
                   screen_h // 2 + int(avg_gaze[1] * screen_h * 0.5)
        
        gaze_x, gaze_y = avg_gaze[0], avg_gaze[1]
        
        # Inverse distance weighting interpolation
        total_weight = 0.0
        weighted_x = 0.0
        weighted_y = 0.0
        
        for (screen_x, screen_y), data in cal_points.items():
            if data.get('avg_gaze') is None:
                continue
            
            cal_gaze = data['avg_gaze']
            dist = np.sqrt((gaze_x - cal_gaze[0])**2 + (gaze_y - cal_gaze[1])**2)
            
            if dist < 0.001:
                return screen_x, screen_y
            
            weight = 1.0 / (dist ** 2)
            total_weight += weight
            weighted_x += screen_x * weight
            weighted_y += screen_y * weight
        
        if total_weight > 0:
            return int(weighted_x / total_weight), int(weighted_y / total_weight)
        else:
            return screen_w // 2, screen_h // 2
    
    @QtCore.Slot()
    def accept_point(self):
        """Accept current calibration point and move to next."""
        if self.current_point is None:
            return
        
        # Collect sample
        if hasattr(self, 'current_gaze_data'):
            self.cal_data.add_sample(self.current_point, self.current_gaze_data)
            tts(settings.CAL_TTS_MESSAGES['accepted'])
            print(f"[Calibration] Accepted point {self.current_point}, samples: {self.cal_data.get_sample_count(self.current_point)}")
        
        # Move to next point immediately
        self._next_point()
    
    @QtCore.Slot()
    def undo_point(self):
        """Undo last sample."""
        if self.current_point and self.cal_data.remove_last_sample(self.current_point):
            tts(settings.CAL_TTS_MESSAGES['undo'])
            self.update()
    
    @QtCore.Slot()
    def cancel_calibration(self):
        """Cancel calibration process."""
        self.calibrationCancelled.emit()
        self.close()
    
    def _complete_calibration(self):
        """Complete calibration and emit data."""
        tts(settings.CAL_TTS_MESSAGES['complete'])
        
        # Show save dialog with name and descriptions
        dialog = PresetSaveDialog(self)
        if dialog.exec() == QtWidgets.QDialog.Accepted:
            self.cal_data.name = dialog.name_edit.text() or f"Calibration {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            self.cal_data.camera_description = dialog.camera_edit.text()
            self.cal_data.display_description = dialog.display_edit.text()
            self.cal_data.is_favorite = dialog.favorite_cb.isChecked()
            
            # Check if overwriting existing
            save_file = dialog.get_save_file()
            if save_file:
                # Overwriting - preserve sort_order from existing
                if dialog.selected_existing:
                    self.cal_data.sort_order = dialog.selected_existing.sort_order
                self.cal_data.save(save_file)
            else:
                # New preset
                self.cal_data.save()
        
        self.calibrationComplete.emit(self.cal_data)
        self.close()

    def paintEvent(self, event):
        """Draw calibration overlay."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        # Draw darkened video frame if available
        if self.current_frame is not None:
            h, w, ch = self.current_frame.shape
            
            # Darken frame
            darkened = (self.current_frame * (1.0 - settings.DEF_CAL_VID_DARKEN)).astype(np.uint8)
            
            bytes_per_line = ch * w
            qimg = QtGui.QImage(
                darkened.data, w, h, bytes_per_line, QtGui.QImage.Format_BGR888
            )
            pixmap = QtGui.QPixmap.fromImage(qimg)
            
            # Scale to fill screen
            scaled = pixmap.scaled(
                self.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
            
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            # Black background
            painter.fillRect(self.rect(), QtCore.Qt.black)
        
        # Draw all calibration points
        for point in self.grid_points:
            sample_count = self.cal_data.get_sample_count(point)
            
            if point == self.current_point:
                color = settings.CAL_TARGET_ACTIVE
            elif sample_count >= settings.DEF_CAL_SAMPLES:
                color = settings.CAL_TARGET_ACCEPTED
            else:
                color = settings.CAL_TARGET_INACTIVE
            
            self._draw_target(painter, point, color, sample_count)
        
        # Draw visualization cursor (gaze position indicator)
        if self.cursor_screen_pos is not None:
            self._draw_viz_cursor(painter, self.cursor_screen_pos)

    def _draw_viz_cursor(self, painter: QtGui.QPainter, pos: Tuple[int, int]):
        """Draw visualization cursor showing computed gaze position."""
        x, y = pos
        
        # Calculate cursor size based on screen width and scaling
        base_size = self.screen_size.width() / 64
        vs = int(base_size * settings.ui_scale(settings.DEF_UI_SCALE_CAL))
        
        # Color from settings (BGR to RGB for Qt)
        color = QtGui.QColor(
            settings.VIZ_CURSOR_COLOR[2],  # R
            settings.VIZ_CURSOR_COLOR[1],  # G
            settings.VIZ_CURSOR_COLOR[0]   # B
        )
        
        pen = QtGui.QPen(color, settings.VIZ_CURSOR_LINE_THICKNESS)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        
        # Draw crosshairs
        # Horizontal line
        painter.drawLine(x - vs, y, x + vs, y)
        # Vertical line
        painter.drawLine(x, y - vs, x, y + vs)
        
        # Draw outer circle
        painter.drawEllipse(QtCore.QPoint(x, y), vs, vs)
        
        # Draw inner circle
        inner_vs = (vs + 1) // 2
        painter.drawEllipse(QtCore.QPoint(x, y), inner_vs, inner_vs)

    def _draw_target(self, painter: QtGui.QPainter, point: Tuple[int, int], 
                    color: Tuple[int, int, int], sample_count: int):
        """Draw a calibration target point."""
        x, y = point
        
        # Outer circle
        painter.setPen(QtGui.QPen(QtGui.QColor(*reversed(color)), 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(
            QtCore.QPoint(x, y),
            settings.DEF_CAL_TARGET_SIZE // 2,
            settings.DEF_CAL_TARGET_SIZE // 2
        )
        
        # Center dot
        painter.setBrush(QtGui.QColor(*reversed(color)))
        painter.drawEllipse(
            QtCore.QPoint(x, y),
            settings.DEF_CAL_TARGET_CENTER_SIZE // 2,
            settings.DEF_CAL_TARGET_CENTER_SIZE // 2
        )
        
        # Sample count if any
        if sample_count > 0:
            # Position count based on location (avoid corners)
            w, h = self.width(), self.height()
            
            if y > h * 0.8:  # Bottom row
                if x < w * 0.2:  # Bottom left
                    text_pos = (x + 25, y - 25)
                elif x > w * 0.8:  # Bottom right
                    text_pos = (x - 40, y - 25)
                else:  # Bottom middle
                    text_pos = (x + 25, y - 25)
            else:  # Not bottom row
                text_pos = (x + 25, y + 25)
            
            painter.setPen(QtGui.QColor(*reversed(settings.CAL_TARGET_COUNT_COLOR)))
            painter.setFont(QtGui.QFont("Sans", 12, QtGui.QFont.Bold))
            painter.drawText(QtCore.QPoint(*text_pos), str(sample_count))

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        key = event.key()
        
        if key == QtCore.Qt.Key_Escape:
            self.cancel_calibration()
        elif key == QtCore.Qt.Key_A:
            self.accept_point()
        elif key == QtCore.Qt.Key_U:
            self.undo_point()
        else:
            super().keyPressEvent(event)


class PresetSaveDialog(QtWidgets.QDialog):
    """Dialog for saving a calibration preset with descriptions."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Calibration Preset")
        self.setMinimumWidth(500)
        self.setMinimumHeight(400)
        
        self.selected_existing = None  # If overwriting existing
        self.existing_presets = []
        self.existing_files = {}
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        info = QtWidgets.QLabel(
            "<b>Save your calibration as a preset.</b><br>"
            "<i>You can save as new or overwrite an existing preset.</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Existing presets table
        existing_group = QtWidgets.QGroupBox("Overwrite existing preset (optional - select one)")
        existing_layout = QtWidgets.QVBoxLayout(existing_group)
        
        self.existing_table = QtWidgets.QTableWidget()
        self.existing_table.setColumnCount(4)
        self.existing_table.setHorizontalHeaderLabels([
            "Name", "Camera", "Display", "Calibrated"
        ])
        self.existing_table.horizontalHeader().setStretchLastSection(True)
        self.existing_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.existing_table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.existing_table.itemSelectionChanged.connect(self._on_existing_selected)
        self.existing_table.setMaximumHeight(150)
        existing_layout.addWidget(self.existing_table)
        
        self.clear_selection_btn = QtWidgets.QPushButton("Clear Selection (Save as New)")
        self.clear_selection_btn.clicked.connect(self._clear_selection)
        existing_layout.addWidget(self.clear_selection_btn)
        
        layout.addWidget(existing_group)
        
        # New preset form
        form_group = QtWidgets.QGroupBox("Preset Details")
        form = QtWidgets.QFormLayout(form_group)
        
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("e.g., Laptop Table")
        self.name_edit.setText(f"Calibration {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        form.addRow("Short UI Name:", self.name_edit)
        
        self.camera_edit = QtWidgets.QLineEdit()
        self.camera_edit.setPlaceholderText("e.g., Webcam medium high (optional)")
        form.addRow("Camera Descr.:", self.camera_edit)
        
        self.display_edit = QtWidgets.QLineEdit()
        self.display_edit.setPlaceholderText("e.g., 3-foot distance (optional)")
        form.addRow("Display Descr.:", self.display_edit)
        
        self.favorite_cb = QtWidgets.QCheckBox("Add to favorites (show in main toolbar)")
        form.addRow(self.favorite_cb)
        
        layout.addWidget(form_group)
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Load existing presets
        self._load_existing_presets()
    
    def _load_existing_presets(self):
        """Load existing presets into table."""
        self.existing_presets = []
        self.existing_files = {}
        
        if settings.CAL_PRESETS_DIR.exists():
            for f in settings.CAL_PRESETS_DIR.glob('*.yaml'):
                try:
                    preset = CalibrationData.load(f)
                    self.existing_presets.append(preset)
                    self.existing_files[id(preset)] = f
                except Exception as e:
                    print(f"[Calibration] Error loading {f}: {e}")
        
        self.existing_presets.sort(key=lambda p: (p.sort_order, p.name))
        self.existing_table.setRowCount(len(self.existing_presets))
        
        for row, preset in enumerate(self.existing_presets):
            self.existing_table.setItem(row, 0, QtWidgets.QTableWidgetItem(preset.name))
            self.existing_table.setItem(row, 1, QtWidgets.QTableWidgetItem(preset.camera_description))
            self.existing_table.setItem(row, 2, QtWidgets.QTableWidgetItem(preset.display_description))
            self.existing_table.setItem(row, 3, QtWidgets.QTableWidgetItem(
                preset.timestamp.strftime("%Y-%m-%d %H:%M")
            ))
        
        self.existing_table.resizeColumnsToContents()
    
    def _on_existing_selected(self):
        """Handle selection of existing preset."""
        row = self.existing_table.currentRow()
        if row >= 0 and row < len(self.existing_presets):
            self.selected_existing = self.existing_presets[row]
            # Fill in the form with existing values
            self.name_edit.setText(self.selected_existing.name)
            self.camera_edit.setText(self.selected_existing.camera_description)
            self.display_edit.setText(self.selected_existing.display_description)
            self.favorite_cb.setChecked(self.selected_existing.is_favorite)
    
    def _clear_selection(self):
        """Clear existing preset selection."""
        self.existing_table.clearSelection()
        self.selected_existing = None
        # Reset to default new name
        self.name_edit.setText(f"Calibration {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    def get_save_file(self) -> Optional[Path]:
        """Get the file path to save to (existing or new)."""
        if self.selected_existing:
            return self.existing_files.get(id(self.selected_existing))
        return None


class PresetsManagerDialog(QtWidgets.QDialog):
    """Dialog for managing calibration presets."""
    
    presetSelected = QtCore.Signal(CalibrationData)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Presets Manager")
        self.setMinimumSize(750, 500)
        
        self.selected_preset = None
        self.preset_files = {}  # Map preset to its file path
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Instructions
        info = QtWidgets.QLabel(
            "<b>Double-click to select a preset.</b><br><br>"
            "Note: How the camera and display are positioned with respect to the user "
            "affects the eye/face tracking. So we allow you to type an <b><i>OPTIONAL</i></b> "
            "description for each.<br><br>"
            "Example:<br>"
            "&nbsp;&nbsp;Short UI Name: <b>Laptop Table</b><br>"
            "&nbsp;&nbsp;Cam Descr.: <i>(blank)</i><br>"
            "&nbsp;&nbsp;Display Descr.: Laptop on computer table<br><br>"
            "<i>If you favorite a preset, it will show in the Main Window as a button.</i>"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Expandable details box
        details_group = QtWidgets.QGroupBox("For those who love details (click to expand)")
        details_group.setCheckable(True)
        details_group.setChecked(False)
        details_layout = QtWidgets.QVBoxLayout(details_group)
        self.details_text = QtWidgets.QLabel(
            "The location/orientation matter because the eyes and head move in curves, "
            "which have to be \"translated\" to the 'linear' way the mouse cursor moves. "
            "Also, the angles of where the camera sees the range of eye movement matters as well. "
            "To help out, we let you put a description for each, as a reminder of where the camera "
            "and display/monitor are <i>with respect to the user</i>."
        )
        self.details_text.setWordWrap(True)
        self.details_text.setVisible(False)  # Start hidden
        details_layout.addWidget(self.details_text)
        
        # Connect toggle to show/hide content
        details_group.toggled.connect(self.details_text.setVisible)
        
        layout.addWidget(details_group)
        
        # Presets table - removed Sort column, added Calibrated column
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            "★", "Short UI Name", "Camera Descr.", "Display Descr.", "Calibrated"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)
        
        # Buttons
        btn_layout = QtWidgets.QHBoxLayout()
        
        self.select_btn = QtWidgets.QPushButton("Select")
        self.select_btn.clicked.connect(self._on_select)
        btn_layout.addWidget(self.select_btn)
        
        self.delete_btn = QtWidgets.QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.delete_btn)
        
        # Move up/down buttons
        self.move_up_btn = QtWidgets.QPushButton("Move ▲")
        self.move_up_btn.clicked.connect(self._on_move_up)
        btn_layout.addWidget(self.move_up_btn)
        
        self.move_down_btn = QtWidgets.QPushButton("Move ▼")
        self.move_down_btn.clicked.connect(self._on_move_down)
        btn_layout.addWidget(self.move_down_btn)
        
        btn_layout.addStretch()
        
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        
        # Load presets
        self._load_presets()
    
    def _load_presets(self):
        """Load and display all presets."""
        self.presets = []
        self.preset_files = {}
        
        # Load presets and track their file paths
        if settings.CAL_PRESETS_DIR.exists():
            for f in settings.CAL_PRESETS_DIR.glob('*.yaml'):
                try:
                    preset = CalibrationData.load(f)
                    self.presets.append(preset)
                    self.preset_files[id(preset)] = f
                except Exception as e:
                    print(f"[Calibration] Error loading {f}: {e}")
        
        # Sort by sort_order, then by name
        self.presets.sort(key=lambda p: (p.sort_order, p.name))
        
        self.table.setRowCount(len(self.presets))
        
        for row, preset in enumerate(self.presets):
            # Favorite checkbox
            fav_widget = QtWidgets.QWidget()
            fav_layout = QtWidgets.QHBoxLayout(fav_widget)
            fav_layout.setContentsMargins(2, 2, 2, 2)
            fav_layout.setAlignment(QtCore.Qt.AlignCenter)
            
            fav_cb = QtWidgets.QCheckBox()
            fav_cb.setChecked(preset.is_favorite)
            fav_cb.stateChanged.connect(lambda state, r=row: self._toggle_favorite(r, state))
            fav_layout.addWidget(fav_cb)
            self.table.setCellWidget(row, 0, fav_widget)
            
            # Name
            name_item = QtWidgets.QTableWidgetItem(preset.name)
            self.table.setItem(row, 1, name_item)
            
            # Camera description
            cam_item = QtWidgets.QTableWidgetItem(preset.camera_description)
            self.table.setItem(row, 2, cam_item)
            
            # Display description
            disp_item = QtWidgets.QTableWidgetItem(preset.display_description)
            self.table.setItem(row, 3, disp_item)
            
            # Calibration timestamp
            time_str = preset.timestamp.strftime("%Y-%m-%d %H:%M")
            time_item = QtWidgets.QTableWidgetItem(time_str)
            self.table.setItem(row, 4, time_item)
        
        self.table.resizeColumnsToContents()
    
    def _get_selected_row(self) -> int:
        """Get selected row, or show message if none selected."""
        row = self.table.currentRow()
        if row < 0 or row >= len(self.presets):
            QtWidgets.QMessageBox.information(
                self, "No Selection",
                "Please select a preset to use this action."
            )
            return -1
        return row
    
    def _on_move_up(self):
        """Move selected preset up."""
        row = self._get_selected_row()
        if row < 0 or row == 0:
            return
        self._swap_sort_order(row, row - 1)
        self.table.selectRow(row - 1)
    
    def _on_move_down(self):
        """Move selected preset down."""
        row = self._get_selected_row()
        if row < 0 or row >= len(self.presets) - 1:
            return
        self._swap_sort_order(row, row + 1)
        self.table.selectRow(row + 1)
    
    def _swap_sort_order(self, row1: int, row2: int):
        """Swap sort order of two presets and save."""
        p1, p2 = self.presets[row1], self.presets[row2]
        p1.sort_order, p2.sort_order = p2.sort_order, p1.sort_order
        
        # Save to their original files
        f1 = self.preset_files.get(id(p1))
        f2 = self.preset_files.get(id(p2))
        if f1:
            p1.save(f1)
        if f2:
            p2.save(f2)
        
        self._load_presets()
    
    def _toggle_favorite(self, row: int, state: int):
        """Toggle favorite status."""
        if row >= len(self.presets):
            return
        preset = self.presets[row]
        preset.is_favorite = (state == QtCore.Qt.Checked)
        
        # Save to original file
        f = self.preset_files.get(id(preset))
        if f:
            preset.save(f)
    
    def _on_double_click(self, index):
        """Handle double-click to select preset."""
        self._on_select()
    
    def _on_select(self):
        """Select current preset."""
        row = self._get_selected_row()
        if row < 0:
            return
        self.selected_preset = self.presets[row]
        self.presetSelected.emit(self.selected_preset)
        self.accept()
    
    def _on_delete(self):
        """Delete selected preset."""
        row = self._get_selected_row()
        if row < 0:
            return
        
        preset = self.presets[row]
        reply = QtWidgets.QMessageBox.question(
            self, "Delete Preset",
            f"Delete preset '{preset.name}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            # Delete using the tracked file path
            f = self.preset_files.get(id(preset))
            if f and f.exists():
                print(f"[Calibration] Deleting {f}")
                f.unlink()
            self._load_presets()
