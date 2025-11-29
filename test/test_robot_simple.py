#!/usr/bin/env python3
"""
Simple robot test using raw motor units (no calibration needed).
Use this to verify the robot is connected and responding.

Controls:
  1-6: Select joint (1=shoulder_pan, 2=shoulder_lift, 3=elbow_flex, 4=wrist_flex, 5=wrist_roll, 6=gripper)
  +/-: Increase/decrease selected joint by 50 motor units
  0: Return selected joint to center (2048)
  h: Go to home position (all joints to 2048)
  s: Show current joint positions
  q: Quit
"""

import sys
import termios
import tty
import json
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def get_key():
    """Get a single keypress from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    print("=" * 60)
    print("SO100 STS3250 Simple Test (Raw Motor Units)")
    print("=" * 60)

    # Load configuration from config.json
    config = load_config()

    # Configure robot
    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]

    # Camera configuration
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"]
        )
        for name, cam in config["cameras"].items()
    }

    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)

    print(f"\nConnecting to robot at {follower_port}...")
    robot = SO100FollowerSTS3250(robot_cfg)

    try:
        robot.connect()
        print("✓ Robot connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    # Joint names
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper"
    ]

    selected_joint = 0
    increment = 50  # motor units (STS3250 has 4096 steps, center is ~2048)
    center_position = 2048  # Middle of motor range

    print("\n" + "=" * 60)
    print("Controls:")
    print("  1-6: Select joint")
    print("  +/-: Move selected joint by ±50 motor units")
    print("  0: Center selected joint (2048)")
    print("  h: Home position (all joints to 2048)")
    print("  s: Show current positions")
    print("  q: Quit")
    print("=" * 60)
    print("\nNOTE: Motor units range from 0-4095, center is 2048")
    print(f"Selected joint: [{selected_joint+1}] {joint_names[selected_joint]}")
    print("Ready! Press a key...")

    try:
        while True:
            key = get_key()

            if key == 'q':
                print("\nQuitting...")
                break

            # Select joint
            elif key in '123456':
                selected_joint = int(key) - 1
                print(f"\nSelected joint: [{selected_joint+1}] {joint_names[selected_joint]}")

            # Move joint
            elif key == '+' or key == '=':
                try:
                    # Read raw motor position (normalize=False for raw units)
                    current_pos = robot.bus.read("Present_Position", joint_names[selected_joint], normalize=False)
                    # Calculate new position
                    new_pos = min(4095, current_pos + increment)  # Clamp to max
                    robot.bus.write("Goal_Position", joint_names[selected_joint], new_pos, normalize=False)
                    print(f"→ {joint_names[selected_joint]}: {current_pos} → {new_pos} units")
                except Exception as e:
                    print(f"Error moving joint: {e}")

            elif key == '-' or key == '_':
                try:
                    current_pos = robot.bus.read("Present_Position", joint_names[selected_joint], normalize=False)
                    new_pos = max(0, current_pos - increment)  # Clamp to min
                    robot.bus.write("Goal_Position", joint_names[selected_joint], new_pos, normalize=False)
                    print(f"→ {joint_names[selected_joint]}: {current_pos} → {new_pos} units")
                except Exception as e:
                    print(f"Error moving joint: {e}")

            # Center selected joint
            elif key == '0':
                try:
                    robot.bus.write("Goal_Position", joint_names[selected_joint], center_position, normalize=False)
                    print(f"→ {joint_names[selected_joint]}: centering to {center_position} units")
                except Exception as e:
                    print(f"Error centering joint: {e}")

            # Home position
            elif key == 'h':
                try:
                    for name in joint_names:
                        robot.bus.write("Goal_Position", name, center_position, normalize=False)
                    print(f"→ Moving all joints to home position ({center_position} units)")
                except Exception as e:
                    print(f"Error homing: {e}")

            # Show positions
            elif key == 's':
                try:
                    print("\nCurrent joint positions (motor units):")
                    for i, name in enumerate(joint_names):
                        pos = robot.bus.read("Present_Position", name, normalize=False)
                        # Show offset from center
                        offset = pos - center_position
                        print(f"  [{i+1}] {name:15s}: {pos:4d} units (center {offset:+5d})")
                except Exception as e:
                    print(f"Error reading positions: {e}")

            else:
                print(f"Unknown key: '{key}' (use 1-6, +/-, 0, h, s, or q)")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
