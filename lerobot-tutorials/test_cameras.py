"""
Test camera connections and display feeds.
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Create and connect cameras (using DirectShow on Windows)
cameras = {}
for name, cam_cfg in config.get("cameras", {}).items():
    cfg = OpenCVCameraConfig(
        index_or_path=cam_cfg["index_or_path"],
        fps=cam_cfg.get("fps", 30),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
    )
    cam = OpenCVCamera(cfg)
    cam.backend = cv2.CAP_DSHOW  # Use DirectShow instead of MSMF on Windows
    cameras[name] = cam

print(f"Connecting {len(cameras)} cameras...")
for name, cam in cameras.items():
    print(f"  Connecting {name}...")
    cam.connect()
    print(f"  {name} connected!")

# Save test snapshots from each camera
print("\nCapturing test frames...")
for name, cam in cameras.items():
    frame = cam.async_read()
    # Convert RGB to BGR for saving with OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    filename = f"snapshot_{name}.jpg"
    cv2.imwrite(filename, frame_bgr)
    print(f"  Saved {filename} ({frame.shape[1]}x{frame.shape[0]})")

# Disconnect
for name, cam in cameras.items():
    cam.disconnect()
print("\nDone! Check the snapshot files.")
