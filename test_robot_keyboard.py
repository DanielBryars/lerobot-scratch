#!/usr/bin/env python3
"""
Simple keyboard control test for SO100 robot with STS3250 motors.
Use this to verify the robot is connected and responding properly.

Controls:
  1-6: Select joint (1=shoulder_pan, 2=shoulder_lift, 3=elbow_flex, 4=wrist_flex, 5=wrist_roll, 6=gripper)
  +/-: Increase/decrease selected joint angle by 5 degrees
  0: Return selected joint to zero/center position
  h: Go to home position (all joints to zero)
  s: Show current joint positions
  q: Quit
"""

import sys
import termios
import tty
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250


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
    print("SO100 STS3250 Keyboard Control Test")
    print("=" * 60)

    # Configure robot
    follower_port = "/dev/ttyACM1"
    follower_id = "test_robot"

    # Minimal camera config (you can remove cameras if not testing them)
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path=4, width=640, height=480, fps=30),
        "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
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
    increment = 5.0  # degrees

    print("\n" + "=" * 60)
    print("Controls:")
    print("  1-6: Select joint")
    print("  +/-: Move selected joint by ±5°")
    print("  0: Center selected joint")
    print("  h: Home position (all joints to 0)")
    print("  s: Show current positions")
    print("  q: Quit")
    print("=" * 60)
    print(f"\nSelected joint: [{selected_joint+1}] {joint_names[selected_joint]}")
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
                    # Get current position
                    current_pos = robot.bus.read("Present_Position", joint_names[selected_joint])
                    # Calculate new position (increment in degrees)
                    new_pos = current_pos + increment
                    robot.bus.write("Goal_Position", joint_names[selected_joint], new_pos)
                    print(f"→ {joint_names[selected_joint]}: {current_pos:.1f}° → {new_pos:.1f}°")
                except Exception as e:
                    print(f"Error moving joint: {e}")

            elif key == '-' or key == '_':
                try:
                    current_pos = robot.bus.read("Present_Position", joint_names[selected_joint])
                    new_pos = current_pos - increment
                    robot.bus.write("Goal_Position", joint_names[selected_joint], new_pos)
                    print(f"→ {joint_names[selected_joint]}: {current_pos:.1f}° → {new_pos:.1f}°")
                except Exception as e:
                    print(f"Error moving joint: {e}")

            # Center selected joint
            elif key == '0':
                try:
                    robot.bus.write("Goal_Position", joint_names[selected_joint], 0.0)
                    print(f"→ {joint_names[selected_joint]}: centering to 0°")
                except Exception as e:
                    print(f"Error centering joint: {e}")

            # Home position
            elif key == 'h':
                try:
                    for name in joint_names:
                        robot.bus.write("Goal_Position", name, 0.0)
                    print("→ Moving all joints to home position (0°)")
                except Exception as e:
                    print(f"Error homing: {e}")

            # Show positions
            elif key == 's':
                try:
                    print("\nCurrent joint positions:")
                    for i, name in enumerate(joint_names):
                        pos = robot.bus.read("Present_Position", name)
                        print(f"  [{i+1}] {name:15s}: {pos:7.2f}°")
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
