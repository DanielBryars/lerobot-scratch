#!/usr/bin/env python3
"""
Adjust gripper zero position.
Use this to find the correct homing_offset for your gripper.

The script will:
1. Show you the current gripper position
2. Let you manually move it to find the correct zero (closed/open) position
3. Save the offset value you can use to update so100_sts3250.py
"""

import sys
import json
from pathlib import Path
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("Gripper Zero Position Adjustment")
    print("=" * 60)
    print()

    # Load configuration
    config = load_config()
    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]

    # Connect to robot (no cameras needed)
    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras={})
    robot = SO100FollowerSTS3250(robot_cfg)

    print(f"Connecting to follower arm at {follower_port}...")
    try:
        robot.connect()
        print("✓ Connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    print("\n" + "=" * 60)
    print("Gripper Information")
    print("=" * 60)

    try:
        # Read current raw position
        current_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
        print(f"\nCurrent gripper position (raw motor units): {current_pos}")
        print(f"Current homing_offset in code: 2048")
        print(f"Difference from center: {current_pos - 2048:+d}")

        # Determine gripper state
        if current_pos < 1500:
            state = "likely OPEN"
        elif current_pos > 2500:
            state = "likely CLOSED"
        else:
            state = "near CENTER"
        print(f"Gripper appears to be {state}")

        print("\n" + "=" * 60)
        print("Finding Zero Position")
        print("=" * 60)
        print("\nFor SO100, the gripper zero position is typically:")
        print("  - Fully OPEN (0%) = lower motor value (~1024 or less)")
        print("  - Fully CLOSED (100%) = higher motor value (~3072 or more)")
        print()

        # Manually position the gripper
        print("Let's find the correct position for your gripper:")
        print()
        print("Option 1 - Test current position as zero:")
        print(f"  Move gripper to desired zero position manually (power off or gentle force)")
        print(f"  Then measure the position")
        print()
        print("Option 2 - Move to specific position:")
        print("  Enter a target motor position (0-4095)")
        print()

        response = input("Choose option (1/2) or 'q' to quit: ").strip()

        if response == '1':
            input("\nMove the gripper to your desired zero position, then press Enter...")
            zero_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
            print(f"\nMeasured zero position: {zero_pos}")
            print(f"\nTo update the code, change gripper homing_offset in so100_sts3250.py:")
            print(f"  Line 58: homing_offset={zero_pos},  # was 2048")

        elif response == '2':
            target = input("\nEnter target position (0-4095): ").strip()
            try:
                target_pos = int(target)
                if 0 <= target_pos <= 4095:
                    robot.bus.write("Goal_Position", "gripper", target_pos, normalize=False)
                    print(f"✓ Moving gripper to {target_pos}")
                    input("\nPress Enter when movement is complete...")

                    actual_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
                    print(f"Gripper reached position: {actual_pos}")
                    print(f"\nIf this is your desired zero position, update so100_sts3250.py:")
                    print(f"  Line 58: homing_offset={actual_pos},  # was 2048")
                else:
                    print("Position must be between 0-4095")
            except ValueError:
                print("Invalid number")

        print("\n" + "=" * 60)
        print("Quick Reference")
        print("=" * 60)
        print("\nCommon gripper positions for SO100:")
        print("  Fully open:   ~1024-1500 (varies by gripper)")
        print("  Center:       ~2048")
        print("  Fully closed: ~2700-3072 (varies by gripper)")
        print("\nThe homing_offset should be set to where you want")
        print("the gripper to be when commanded to position 0 (or 0%)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nDisconnecting...")
        robot.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
