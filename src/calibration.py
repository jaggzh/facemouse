#!/usr/bin/env python3
"""
Calibration system for gaze tracking.

Provides fullscreen calibration UI, data collection, and preset management.
"""

import random
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import yaml
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import settings


def tts(msg: str):
    """Text-to-speech output."""
    if settings.DEF_CAL_NO_TTS:
        return
    try:
        subprocess.Popen(["vpi", "--", msg], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[TTS] Failed: {e}")


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
        self.name = name
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
    
    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            'name': self.name,
            'grid_size': self.grid_size,
            'timestamp': self.timestamp.isoformat(),
            'points': [
                {
                    'screen_x': pos[0],
                    'screen_y': pos[1],
                    'samples': samples
                }
                for pos, samples in self.points.items()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        """Create from dictionary loaded from YAML."""
        cal = cls(name=data['name'], grid_size=data['grid_size'])
        cal.timestamp = datetime.fromisoformat(data['timestamp'])
        for pt_data in data['points']:
            pos = (pt_data['screen_x'], pt_data['screen_y'])
            cal.points[pos] = pt_data['samples']
        return cal
    
    def save(self, filename: Optional[Path] = None):
        """Save calibration data to YAML file."""
        if filename is None:
            filename = settings.CAL_PRESETS_DIR / f"{self.timestamp.strftime('%Y-%m-%d--%Hh%Mm%Ss')}.yaml"
        
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        
        print(f"[Calibration] Saved to {filename}")
        return filename
    
    @classmethod
    def load(cls, filename: Path) -> 'CalibrationData':
        """Load calibration data from YAML file."""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class CalibrationWindow(QtWidgets.QWidget):
    """Fullscreen calibration window with video underlay."""
    
    # Signals
    calibrationComplete = QtCore.Signal(CalibrationData)
    calibrationCancelled = QtCore.Signal()
    
    def __init__(self, screen_size: QtCore.QSize, grid_size: int = settings.DEF_CAL_GRID_SIZE, 
                 parent=None):
        super().__init__(parent)
        self.screen_size = screen_size
        self.grid_size = grid_size
        
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
            # Center of screen + gaze offset scaled to half screen
            cx = w // 2 + int(avg_gaze[0] * w * 0.5)
            cy = h // 2 + int(avg_gaze[1] * h * 0.5)
            # Clamp to screen bounds
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            self.cursor_screen_pos = (cx, cy)
        else:
            self.cursor_screen_pos = None
        
        self.update()
    
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
        
        # Prompt for preset name
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Calibration", "Enter preset name:",
            text=f"Calibration {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        if ok and name:
            self.cal_data.name = name
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
