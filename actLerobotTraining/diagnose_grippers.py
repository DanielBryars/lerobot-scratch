#!/usr/bin/env python3
"""
Diagnose gripper calibration on both leader and follower.
Shows raw and normalized values for both grippers.
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


def main():
    print("=" * 70)
    print("Gripper Calibration Diagnostics")
    print("=" * 70)
    print()

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
    print("LEADER Gripper")
    print("=" * 70)
    try:
        leader_min = leader.bus.read("Min_Position_Limit", "gripper", normalize=False)
        leader_max = leader.bus.read("Max_Position_Limit", "gripper", normalize=False)
        leader_pos_raw = leader.bus.read("Present_Position", "gripper", normalize=False)
        leader_pos_norm = leader.bus.read("Present_Position", "gripper", normalize=True)

        print(f"Min Position Limit: {leader_min}")
        print(f"Max Position Limit: {leader_max}")
        print(f"Range: {leader_max - leader_min}")
        print(f"Current Position (raw): {leader_pos_raw}")
        print(f"Current Position (normalized 0-100): {leader_pos_norm:.2f}%")
    except Exception as e:
        print(f"Error reading leader: {e}")

    print("\n" + "=" * 70)
    print("FOLLOWER Gripper")
    print("=" * 70)
    try:
        follower_min = follower.bus.read("Min_Position_Limit", "gripper", normalize=False)
        follower_max = follower.bus.read("Max_Position_Limit", "gripper", normalize=False)
        follower_pos_raw = follower.bus.read("Present_Position", "gripper", normalize=False)
        follower_pos_norm = follower.bus.read("Present_Position", "gripper", normalize=True)

        print(f"Min Position Limit: {follower_min}")
        print(f"Max Position Limit: {follower_max}")
        print(f"Range: {follower_max - follower_min}")
        print(f"Current Position (raw): {follower_pos_raw}")
        print(f"Current Position (normalized 0-100): {follower_pos_norm:.2f}%")
    except Exception as e:
        print(f"Error reading follower: {e}")

    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    try:
        leader_range = leader_max - leader_min
        follower_range = follower_max - follower_min
        ratio = follower_range / leader_range if leader_range > 0 else 0

        print(f"Leader range: {leader_range} units")
        print(f"Follower range: {follower_range} units")
        print(f"Ratio (follower/leader): {ratio:.2f}x")
        print()
        print("Analysis:")
        if abs(ratio - 1.0) < 0.1:
            print("  ✓ Ranges are similar, grippers should match well")
        else:
            print(f"  ⚠ Follower range is {ratio:.1f}x the leader range")
            print(f"    This means follower will only use {(1/ratio)*100:.0f}% of its range")
            print(f"    when following the leader.")
    except:
        pass

    print()
    leader.disconnect()
    follower.disconnect()
    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
