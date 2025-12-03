#!/usr/bin/env python3
"""
Check current gripper calibration values stored in servo firmware.
Reads Min/Max Position Limits and current position from both robots.
"""

import sys
import json
from pathlib import Path
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from SO100LeaderSTS3250 import SO100LeaderSTS3250
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def check_robot_gripper(robot, name):
    """Read and display gripper calibration from a robot."""
    print(f"\n{'=' * 70}")
    print(f"{name.upper()} Gripper Calibration")
    print(f"{'=' * 70}")

    try:
        min_limit = robot.bus.read("Min_Position_Limit", "gripper", normalize=False)
        max_limit = robot.bus.read("Max_Position_Limit", "gripper", normalize=False)
        current_pos = robot.bus.read("Present_Position", "gripper", normalize=False)

        # Calculate percentage based on limits
        if max_limit > min_limit:
            percentage = ((current_pos - min_limit) / (max_limit - min_limit)) * 100
        else:
            percentage = 0

        print(f"  Min Position Limit (closed): {min_limit}")
        print(f"  Max Position Limit (open):   {max_limit}")
        print(f"  Current Position:            {current_pos} ({percentage:.1f}%)")
        print(f"  Range:                       {max_limit - min_limit} steps")

        return {
            'min': min_limit,
            'max': max_limit,
            'current': current_pos,
            'percentage': percentage
        }
    except Exception as e:
        print(f"  ✗ Failed to read gripper values: {e}")
        return None


def main():
    print("=" * 70)
    print("Gripper Calibration Check")
    print("=" * 70)
    print("\nReading calibration values from servo firmware...")

    # Load configuration
    config = load_config()

    # Connect to leader
    leader_port = config["leader"]["port"]
    leader_cfg = SO100LeaderConfig(port=leader_port, id=config["leader"]["id"])
    leader = SO100LeaderSTS3250(leader_cfg)

    print(f"\nConnecting to leader at {leader_port}...")
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

    # Check both grippers
    leader_values = check_robot_gripper(leader, "Leader")
    follower_values = check_robot_gripper(follower, "Follower")

    # Compare with expected values
    print(f"\n{'=' * 70}")
    print("Comparison with Expected Values")
    print(f"{'=' * 70}")

    expected_leader = {'min': 2089, 'max': 3221}
    expected_follower = {'min': 1537, 'max': 2741}

    if leader_values:
        print("\nLeader:")
        print(f"  Expected: Min={expected_leader['min']}, Max={expected_leader['max']}")
        print(f"  Actual:   Min={leader_values['min']}, Max={leader_values['max']}")
        if (leader_values['min'] == expected_leader['min'] and
            leader_values['max'] == expected_leader['max']):
            print("  ✓ Leader calibration matches expected values")
        else:
            print("  ✗ Leader calibration DOES NOT match")

    if follower_values:
        print("\nFollower:")
        print(f"  Expected: Min={expected_follower['min']}, Max={expected_follower['max']}")
        print(f"  Actual:   Min={follower_values['min']}, Max={follower_values['max']}")
        if (follower_values['min'] == expected_follower['min'] and
            follower_values['max'] == expected_follower['max']):
            print("  ✓ Follower calibration matches expected values")
        else:
            print("  ✗ Follower calibration DOES NOT match")

    print(f"\n{'=' * 70}")

    # Cleanup
    leader.disconnect()
    follower.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
