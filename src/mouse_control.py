#!/usr/bin/env python3
"""
Mouse control module for gaze tracking.

Provides cursor positioning based on gaze data.
Uses Xlib for X11 systems (can be extended for other platforms).
"""

import subprocess
from typing import Optional, Tuple


class MouseController:
    """Controls mouse cursor position based on gaze input."""
    
    def __init__(self):
        self._method = None
        self._screen_size = None
        self._detect_method()
    
    def _detect_method(self):
        """Detect available mouse control method."""
        # Try xdotool first (most reliable on X11)
        try:
            result = subprocess.run(['which', 'xdotool'], capture_output=True)
            if result.returncode == 0:
                self._method = 'xdotool'
                print("[Mouse] Using xdotool for cursor control")
                return
        except:
            pass
        
        # Try xte (from xautomation)
        try:
            result = subprocess.run(['which', 'xte'], capture_output=True)
            if result.returncode == 0:
                self._method = 'xte'
                print("[Mouse] Using xte for cursor control")
                return
        except:
            pass
        
        # Try python-xlib
        try:
            from Xlib import X, display
            self._method = 'xlib'
            self._display = display.Display()
            self._screen = self._display.screen()
            print("[Mouse] Using python-xlib for cursor control")
            return
        except ImportError:
            pass
        
        print("[Mouse] WARNING: No mouse control method available!")
        print("        Install one of: xdotool, xautomation, or python-xlib")
        self._method = None
    
    def move_to(self, x: int, y: int):
        """Move cursor to absolute screen position."""
        if self._method is None:
            return
        
        try:
            if self._method == 'xdotool':
                subprocess.run(['xdotool', 'mousemove', str(x), str(y)],
                              capture_output=True, timeout=0.1)
            
            elif self._method == 'xte':
                subprocess.run(['xte', f'mousemove {x} {y}'],
                              capture_output=True, timeout=0.1)
            
            elif self._method == 'xlib':
                self._screen.root.warp_pointer(x, y)
                self._display.sync()
        
        except Exception as e:
            print(f"[Mouse] Error moving cursor: {e}")
    
    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        """Get screen dimensions."""
        if self._screen_size:
            return self._screen_size
        
        try:
            if self._method == 'xdotool':
                result = subprocess.run(['xdotool', 'getdisplaygeometry'],
                                       capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    parts = result.stdout.strip().split()
                    self._screen_size = (int(parts[0]), int(parts[1]))
                    return self._screen_size
            
            elif self._method == 'xlib':
                self._screen_size = (self._screen.width_in_pixels, 
                                    self._screen.height_in_pixels)
                return self._screen_size
        
        except Exception as e:
            print(f"[Mouse] Error getting screen size: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if mouse control is available."""
        return self._method is not None


# Global instance
_controller = None

def get_controller() -> MouseController:
    """Get or create the global mouse controller."""
    global _controller
    if _controller is None:
        _controller = MouseController()
    return _controller


def move_cursor(x: int, y: int):
    """Move cursor to absolute position (convenience function)."""
    get_controller().move_to(x, y)