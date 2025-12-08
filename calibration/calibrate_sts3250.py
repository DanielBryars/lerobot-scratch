#!/usr/bin/env python
"""
Calibrate SO100 arms with STS3250 motors.

Uses lerobot's file-based calibration (JSON files).

Usage:
    python calibrate_sts3250.py --leader          # Calibrate leader only
    python calibrate_sts3250.py --follower        # Calibrate follower only
    python calibrate_sts3250.py --leader --follower  # Calibrate both
"""
import argparse
import json
from pathlib import Path

# Import our custom STS3250 classes (this registers them)
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config
from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config


def load_config():
    """Load config.json for COM ports."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def calibrate_leader(port: str, arm_id: str = "leader"):
    """Calibrate the leader arm."""
    print("=" * 60)
    print(f"CALIBRATING LEADER ARM")
    print(f"Port: {port}")
    print(f"ID: {arm_id}")
    print("=" * 60)

    config = SO100LeaderSTS3250Config(
        port=port,
        id=arm_id,
    )
    leader = SO100LeaderSTS3250(config)

    # Connect without auto-calibration
    leader.bus.connect()
    leader.bus.disable_torque()

    # Run calibration
    leader.calibrate()

    leader.disconnect()
    print(f"\nLeader calibration saved to: {leader.calibration_fpath}")


def calibrate_follower(port: str, arm_id: str = "follower"):
    """Calibrate the follower arm."""
    print("=" * 60)
    print(f"CALIBRATING FOLLOWER ARM")
    print(f"Port: {port}")
    print(f"ID: {arm_id}")
    print("=" * 60)

    config = SO100FollowerSTS3250Config(
        port=port,
        id=arm_id,
        cameras={},  # No cameras for calibration
    )
    follower = SO100FollowerSTS3250(config)

    # Connect without auto-calibration
    follower.bus.connect()
    follower.bus.disable_torque()

    # Run calibration
    follower.calibrate()

    follower.disconnect()
    print(f"\nFollower calibration saved to: {follower.calibration_fpath}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate SO100 arms with STS3250 motors")
    parser.add_argument("--leader", "-l", action="store_true", help="Calibrate leader arm")
    parser.add_argument("--follower", "-f", action="store_true", help="Calibrate follower arm")
    parser.add_argument("--leader-port", type=str, default=None, help="Leader port (default: from config.json)")
    parser.add_argument("--follower-port", type=str, default=None, help="Follower port (default: from config.json)")
    parser.add_argument("--leader-id", type=str, default="leader", help="Leader ID for calibration file")
    parser.add_argument("--follower-id", type=str, default="follower", help="Follower ID for calibration file")
    args = parser.parse_args()

    # Load config for default ports
    config = load_config()

    leader_port = args.leader_port
    follower_port = args.follower_port

    if config:
        if leader_port is None and "leader" in config:
            leader_port = config["leader"]["port"]
        if follower_port is None and "follower" in config:
            follower_port = config["follower"]["port"]

    # Default ports if still not set
    if leader_port is None:
        leader_port = "COM8"
    if follower_port is None:
        follower_port = "COM7"

    # If neither specified, show help
    if not args.leader and not args.follower:
        parser.print_help()
        print("\nExample:")
        print("  python calibrate_sts3250.py --leader")
        print("  python calibrate_sts3250.py --follower")
        print("  python calibrate_sts3250.py --leader --follower")
        return

    if args.leader:
        calibrate_leader(leader_port, args.leader_id)

    if args.follower:
        calibrate_follower(follower_port, args.follower_id)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("\nCalibration files saved to:")
    print("  ~/.cache/huggingface/lerobot/calibration/")


if __name__ == "__main__":
    main()
