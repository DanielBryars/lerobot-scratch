#!/usr/bin/env python3
"""
Test what values we're actually reading from the grippers.
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
    print("Test Gripper Reading")
    print("=" * 70)

    config = load_config()

    # Connect to leader
    leader_port = config["leader"]["port"]
    leader_cfg = SO100LeaderConfig(port=leader_port, id=config["leader"]["id"])
    leader = SO100LeaderSTS3250(leader_cfg)

    print(f"\nConnecting to leader at {leader_port}...")
    leader.connect()

    # Connect to follower
    follower_port = config["follower"]["port"]
    follower_cfg = SO100FollowerConfig(port=follower_port, id=config["follower"]["id"], cameras={})
    follower = SO100FollowerSTS3250(follower_cfg)

    print(f"Connecting to follower at {follower_port}...")
    follower.connect()

    print("\n" + "=" * 70)
    print("Reading via get_action() - what teleoperation sees")
    print("=" * 70)

    action = leader.get_action()
    print(f"\nLeader get_action() result:")
    for key, value in action.items():
        if 'gripper' in key:
            print(f"  {key}: {value}")

    print(f"\nFollower state:")
    try:
        obs = follower.get_observation()
        for key, value in obs.items():
            if 'gripper' in str(key):
                print(f"  {key}: {value}")
    except:
        pass

    print("\n" + "=" * 70)
    print("Direct bus reads")
    print("=" * 70)

    print(f"\nLeader gripper:")
    raw = leader.bus.read("Present_Position", "gripper", normalize=False)
    norm = leader.bus.read("Present_Position", "gripper", normalize=True)
    print(f"  Raw: {raw}")
    print(f"  Normalized: {norm}")

    print(f"\nFollower gripper:")
    raw = follower.bus.read("Present_Position", "gripper", normalize=False)
    norm = follower.bus.read("Present_Position", "gripper", normalize=True)
    print(f"  Raw: {raw}")
    print(f"  Normalized: {norm}")

    print(f"\nLeader motor config:")
    leader_motor = leader.bus.motors["gripper"]
    print(f"  Motor mode: {leader_motor.norm_mode}")

    print(f"\nFollower motor config:")
    follower_motor = follower.bus.motors["gripper"]
    print(f"  Motor mode: {follower_motor.norm_mode}")

    leader.disconnect()
    follower.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
