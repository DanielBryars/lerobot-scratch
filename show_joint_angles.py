#!/usr/bin/env python3
"""
Show real-time joint angles to help calibrate position detection.

Usage:
    python show_joint_angles.py
"""

import fix_camera_backend

import time
import json
import sys
from pathlib import Path
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Joint Angle Monitor")
    print("=" * 70)
    print("\nThis tool shows real-time joint angles from the follower robot.")
    print("Move the arm to different positions and observe the values.")
    print("\nPress Ctrl+C to quit")
    print()

    # Load configuration
    config = load_config()

    # Camera configuration (minimal, just to connect)
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
        for name, cam in config["cameras"].items()
    }

    # Connect to follower
    follower_port = config["follower"]["port"]
    follower_cfg = SO100FollowerConfig(
        port=follower_port,
        id=config["follower"]["id"],
        cameras=camera_config
    )
    follower = SO100FollowerSTS3250(follower_cfg)

    print(f"Connecting to follower at {follower_port}...")
    try:
        follower.connect()
        print("[OK] Follower connected\n")
    except Exception as e:
        print(f"[FAILED] {e}")
        return 1

    # Joint names
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("=" * 70)
    print("Monitoring Joint Angles (updating every 0.5s)")
    print("=" * 70)
    print()

    try:
        while True:
            # Get observation
            obs = follower.get_observation()

            # Display all joint angles
            print("\r", end="")  # Return to start of line
            angle_str = " | ".join([
                f"{joint}: {obs.get(f'{joint}.pos', 0.0):6.1f}°"
                for joint in joint_names
            ])
            print(angle_str, end="", flush=True)

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[QUIT] Monitoring stopped")

    finally:
        try:
            if follower.is_connected:
                follower.disconnect()
        except Exception:
            pass

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
