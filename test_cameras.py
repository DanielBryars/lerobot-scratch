#!/usr/bin/env python3
"""
Test camera connections and frame capture.
"""

import json
import time
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Camera Test")
    print("=" * 70)
    print()

    config = load_config()

    for name, cam_config in config["cameras"].items():
        print(f"\nTesting {name}:")
        print(f"  Path: {cam_config['index_or_path']}")
        print(f"  Resolution: {cam_config['width']}x{cam_config['height']}")
        print(f"  FPS: {cam_config['fps']}")

        try:
            # Create camera config
            camera_cfg = OpenCVCameraConfig(
                index_or_path=cam_config["index_or_path"],
                width=cam_config["width"],
                height=cam_config["height"],
                fps=cam_config["fps"]
            )

            # Create and connect camera
            camera = OpenCVCamera(camera_cfg)
            camera.connect()
            print(f"  ✓ Camera connected")

            # Try to read 5 frames
            print(f"  Reading frames...")
            for i in range(5):
                start = time.time()
                frame = camera.async_read()
                elapsed = time.time() - start
                print(f"    Frame {i+1}: {frame.shape} ({elapsed*1000:.1f}ms)")
                time.sleep(0.1)

            camera.disconnect()
            print(f"  ✓ Camera test passed")

        except Exception as e:
            print(f"  ✗ Camera test failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Camera test complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
