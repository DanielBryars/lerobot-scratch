#!/usr/bin/env python3
"""
Debug gripper teleoperation to see what values are being passed.
"""

import time
import sys
import json
from pathlib import Path
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250
from SO100LeaderSTS3250 import SO100LeaderSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Gripper Teleoperation Debug")
    print("=" * 70)
    print("\nThis will show the raw values being read from leader")
    print("and sent to follower for the gripper.\n")

    # Load configuration
    config = load_config()

    # Connect to leader
    leader_port = config["leader"]["port"]
    leader_cfg = SO100LeaderConfig(port=leader_port, id=config["leader"]["id"])
    leader = SO100LeaderSTS3250(leader_cfg)

    print(f"Connecting to leader at {leader_port}...")
    try:
        leader.connect()
        print("✓ Leader connected")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return 1

    # Connect to follower
    follower_port = config["follower"]["port"]
    follower_cfg = SO100FollowerConfig(port=follower_port, id=config["follower"]["id"], cameras={})
    follower = SO100FollowerSTS3250(follower_cfg)

    print(f"Connecting to follower at {follower_port}...")
    try:
        follower.connect()
        print("✓ Follower connected")
    except Exception as e:
        print(f"✗ Failed: {e}")
        leader.disconnect()
        return 1

    print("\n" + "=" * 70)
    print("Reading gripper values (with leader closed)")
    print("=" * 70)

    try:
        # Read raw positions from both servos
        leader_raw = leader.bus.read("Present_Position", "gripper", normalize=False)
        follower_raw = follower.bus.read("Present_Position", "gripper", normalize=False)

        print(f"\nRaw servo positions:")
        print(f"  Leader:   {leader_raw}")
        print(f"  Follower: {follower_raw}")

        # Read normalized positions (0-100%)
        leader_norm = leader.bus.read("Present_Position", "gripper", normalize=True)
        follower_norm = follower.bus.read("Present_Position", "gripper", normalize=True)

        print(f"\nNormalized positions (0-100%):")
        print(f"  Leader:   {leader_norm:.2f}%")
        print(f"  Follower: {follower_norm:.2f}%")

        # Get action from leader (what teleoperation uses)
        action = leader.get_action()

        print(f"\nAction from leader.get_action():")
        print(f"  gripper.pos: {action.get('gripper.pos', 'NOT FOUND')}")

        # Check calibration ranges stored in motor bus
        print(f"\nLeader motor calibration:")
        leader_motor = leader.bus.motors.get("gripper")
        if leader_motor:
            leader_calib = leader.bus.calibration.get("gripper")
            print(f"  Motor ID: {leader_motor.id}")
            print(f"  Norm mode: {leader_motor.norm_mode}")
            if leader_calib:
                print(f"  Calibration range_min: {leader_calib.range_min}")
                print(f"  Calibration range_max: {leader_calib.range_max}")
                print(f"  Calibration homing_offset: {leader_calib.homing_offset}")

        print(f"\nFollower motor calibration:")
        follower_motor = follower.bus.motors.get("gripper")
        if follower_motor:
            follower_calib = follower.bus.calibration.get("gripper")
            print(f"  Motor ID: {follower_motor.id}")
            print(f"  Norm mode: {follower_motor.norm_mode}")
            if follower_calib:
                print(f"  Calibration range_min: {follower_calib.range_min}")
                print(f"  Calibration range_max: {follower_calib.range_max}")
                print(f"  Calibration homing_offset: {follower_calib.homing_offset}")

        print("\n" + "=" * 70)
        print("Now sending leader action to follower...")
        print("=" * 70)

        follower.send_action(action)
        time.sleep(1)  # Wait for movement

        # Read follower position after action
        follower_raw_after = follower.bus.read("Present_Position", "gripper", normalize=False)
        follower_norm_after = follower.bus.read("Present_Position", "gripper", normalize=True)

        print(f"\nFollower position AFTER send_action:")
        print(f"  Raw: {follower_raw_after}")
        print(f"  Normalized: {follower_norm_after:.2f}%")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        leader.disconnect()
        follower.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
