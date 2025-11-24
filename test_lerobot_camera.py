#!/usr/bin/env python3
"""Test LeRobot's OpenCVCamera class directly."""

# Import fix BEFORE lerobot imports
import fix_camera_backend

import json
from pathlib import Path
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)

print("=" * 70)
print("LeRobot OpenCVCamera Test")
print("=" * 70)

config = load_config()

for name, cam_cfg in config["cameras"].items():
    print(f"\nTesting {name} (index {cam_cfg['index_or_path']})...")

    try:
        camera_config = OpenCVCameraConfig(
            index_or_path=cam_cfg["index_or_path"],
            width=cam_cfg["width"],
            height=cam_cfg["height"],
            fps=cam_cfg["fps"],
        )

        camera = OpenCVCamera(config=camera_config)
        print(f"  Camera object created")

        camera.connect()
        print(f"  [OK] Connected successfully")

        # Try to read a frame
        frame_dict = camera.read()
        print(f"  [OK] Frame read successfully")
        print(f"       Keys: {list(frame_dict.keys())}")
        if name in frame_dict:
            print(f"       Shape: {frame_dict[name].shape}")

        camera.disconnect()
        print(f"  [OK] Disconnected")

    except Exception as e:
        print(f"  [FAILED] {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
