#!/usr/bin/env python3
"""
Fix gripper calibration based on diagnostic measurements.
Sets the correct min/max values for both leader and follower.
"""

import sys
import json
from pathlib import Path
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_leader_sts3250 import SO100LeaderSTS3250
from so100_sts3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Fix Gripper Calibration")
    print("=" * 70)
    print()
    print("This will set the correct min/max calibration values based on")
    print("the measurements from diagnose_grippers.py")
    print()
    print("Values to be set:")
    print("  Leader:   Min=2089 (closed), Max=3221 (open)")
    print("  Follower: Min=1537 (closed), Max=2741 (open)")
    print()

    response = input("Apply these calibration values? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Cancelled.")
        return 1

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

    print("\n" + "=" * 70)
    print("Updating LEADER Gripper Calibration")
    print("=" * 70)
    try:
        print("Writing Min Position: 2089")
        leader.bus.write("Min_Position_Limit", "gripper", 2089, normalize=False)
        print("Writing Max Position: 3221")
        leader.bus.write("Max_Position_Limit", "gripper", 3221, normalize=False)

        # Verify
        new_min = leader.bus.read("Min_Position_Limit", "gripper", normalize=False)
        new_max = leader.bus.read("Max_Position_Limit", "gripper", normalize=False)
        print(f"Verified: Min={new_min}, Max={new_max}")
        print("✓ Leader calibration updated")
    except Exception as e:
        print(f"✗ Failed to update leader: {e}")

    print("\n" + "=" * 70)
    print("Updating FOLLOWER Gripper Calibration")
    print("=" * 70)
    try:
        print("Writing Min Position: 1537")
        follower.bus.write("Min_Position_Limit", "gripper", 1537, normalize=False)
        print("Writing Max Position: 2741")
        follower.bus.write("Max_Position_Limit", "gripper", 2741, normalize=False)

        # Verify
        new_min = follower.bus.read("Min_Position_Limit", "gripper", normalize=False)
        new_max = follower.bus.read("Max_Position_Limit", "gripper", normalize=False)
        print(f"Verified: Min={new_min}, Max={new_max}")
        print("✓ Follower calibration updated")
    except Exception as e:
        print(f"✗ Failed to update follower: {e}")

    print("\n" + "=" * 70)
    print("✓ Calibration Complete!")
    print("=" * 70)
    print("\nNow run ./diagnose_grippers.py again with grippers closed,")
    print("and they should both read close to 0%")
    print("\nThen test teleoperation - the grippers should now match!")

    leader.disconnect()
    follower.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
