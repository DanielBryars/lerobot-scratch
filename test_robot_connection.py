#!/usr/bin/env python3
"""
Simple script to test robot connection and movement.
"""

import json
import time
from pathlib import Path

import fix_camera_backend
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250


def main():
    print("=" * 70)
    print("Robot Connection Test")
    print("=" * 70)
    print()

    # Load config
    with open("config.json") as f:
        robot_config = json.load(f)

    print(f"Robot port: {robot_config['follower']['port']}")
    print()

    # Connect to robot
    print("Connecting to robot...")
    follower_config = SO100FollowerConfig(
        port=robot_config["follower"]["port"],
        id=robot_config["follower"]["id"],
    )
    robot = SO100FollowerSTS3250(follower_config)
    robot.connect()
    print("[OK] Robot connected")
    print()

    # Check what motors/keys the robot has
    print("Checking robot state...")
    try:
        state = robot.get_observation()
        print(f"Observation keys: {list(state.keys())}")
        print()
        print("Current positions:")
        for key, val in state.items():
            if 'pos' in key.lower():
                print(f"  {key}: {val:.1f}")
    except Exception as e:
        print(f"Error getting observation: {e}")
    print()

    # Check what the robot's motor names are
    print("Checking motor bus...")
    try:
        if hasattr(robot, 'bus'):
            print(f"Bus type: {type(robot.bus)}")
            if hasattr(robot.bus, 'motors'):
                print(f"Motors: {robot.bus.motors}")
            if hasattr(robot.bus, 'motor_names'):
                print(f"Motor names: {robot.bus.motor_names}")
    except Exception as e:
        print(f"Error checking bus: {e}")
    print()

    # Try to read current position
    print("Reading current position via bus...")
    try:
        if hasattr(robot, 'bus'):
            pos = robot.bus.read("Present_Position")
            print(f"Present positions: {pos}")
    except Exception as e:
        print(f"Error reading position: {e}")
    print()

    # Check what send_action expects
    print("Checking send_action signature...")
    try:
        import inspect
        sig = inspect.signature(robot.send_action)
        print(f"send_action signature: {sig}")
    except Exception as e:
        print(f"Error: {e}")
    print()

    # Try sending a simple action
    print("Attempting to send action...")

    # First, get current state to use as base
    try:
        state = robot.get_observation()

        # Build action dict from current state (no movement)
        # Keys MUST end with .pos for send_action to work!
        action_dict = {}
        for key, val in state.items():
            if '.pos' in key:
                action_dict[key] = float(val)  # Keep the .pos suffix!

        print(f"Action dict (current positions): {action_dict}")

        # Try sending
        robot.send_action(action_dict)
        print("[OK] send_action succeeded!")

    except Exception as e:
        print(f"Error sending action: {e}")
        import traceback
        traceback.print_exc()
    print()

    # Disconnect
    print("Disconnecting...")
    robot.disconnect()
    print("[OK] Done!")


if __name__ == "__main__":
    main()
