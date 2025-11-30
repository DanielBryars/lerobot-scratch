"""
OpenCV Camera using DirectShow backend for Windows.
Some cameras don't work with MSMF (Windows Media Foundation) but work fine with DirectShow.
"""

import cv2
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig


class OpenCVCameraDSHOW(OpenCVCamera):
    """OpenCV camera that uses DirectShow backend instead of MSMF."""

    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)
        # Override the backend to use DirectShow
        self.backend = cv2.CAP_DSHOW
