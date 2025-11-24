"""
Fix for Windows camera backend issue.

LeRobot defaults to MSMF backend on Windows, but some cameras
work better with DSHOW. This module monkey-patches the backend
selection to use DSHOW instead.

Import this module BEFORE importing any LeRobot camera code.
"""

import cv2
import platform


def get_cv2_backend_dshow() -> int:
    """Override to use DSHOW instead of MSMF on Windows."""
    if platform.system() == "Windows":
        return int(cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF
    else:
        return int(cv2.CAP_ANY)


def apply_fix():
    """Apply the camera backend fix by monkey-patching LeRobot's get_cv2_backend function."""
    import lerobot.cameras.utils
    lerobot.cameras.utils.get_cv2_backend = get_cv2_backend_dshow
    print("[INFO] Applied camera backend fix: Using DSHOW instead of MSMF")


# Auto-apply when imported
apply_fix()
