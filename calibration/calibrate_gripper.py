#!/usr/bin/env python3
"""
Calibrate just the gripper on the SO100 follower arm.
This will write proper min/max limits to the servo firmware.
"""

import sys
import json
from pathlib import Path
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("SO100 Gripper Calibration")
    print("=" * 70)
    print()
    print("This will calibrate the gripper min/max limits.")
    print()
    print("IMPORTANT:")
    print("  - Make sure the gripper has space to open and close fully")
    print("  - Don't hold the gripper during calibration")
    print()

    response = input("Ready to calibrate the gripper? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Calibration cancelled.")
        return 1

    # Load configuration
    config = load_config()
    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]

    # Connect to robot
    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras={})
    robot = SO100FollowerSTS3250(robot_cfg)

    print(f"\nConnecting to follower arm at {follower_port}...")
    try:
        robot.connect()
        print("✓ Connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    print("\n" + "=" * 70)
    print("Starting Gripper Calibration")
    print("=" * 70)
    print()

    try:
        # Read current calibration values
        print("Current calibration values in servo firmware:")
        try:
            current_min = robot.bus.read("Min_Position_Limit", "gripper", normalize=False)
            current_max = robot.bus.read("Max_Position_Limit", "gripper", normalize=False)
            current_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
            print(f"  Min Position Limit: {current_min}")
            print(f"  Max Position Limit: {current_max}")
            print(f"  Current Position: {current_pos}")
        except Exception as e:
            print(f"  Could not read current values: {e}")
        print()
        # Manual calibration process for gripper
        print("Step 1: Finding CLOSED position")
        print("  Manually close the gripper completely (or let it auto-close)")
        input("  Press Enter when gripper is CLOSED...")

        # Read closed position
        closed_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
        print(f"  ✓ Closed position: {closed_pos}")

        print("\nStep 2: Finding OPEN position")
        print("  Manually open the gripper completely")
        input("  Press Enter when gripper is fully OPEN...")

        # Read open position
        open_pos = robot.bus.read("Present_Position", "gripper", normalize=False)
        print(f"  ✓ Open position: {open_pos}")

        # Determine min/max
        if closed_pos < open_pos:
            min_pos = closed_pos
            max_pos = open_pos
            print(f"\n  Gripper closes at lower values ({closed_pos})")
        else:
            min_pos = open_pos
            max_pos = closed_pos
            print(f"\n  Gripper closes at higher values ({closed_pos})")

        print("\n" + "=" * 70)
        print("Writing Calibration to Servo")
        print("=" * 70)

        # Write min/max positions to servo EEPROM
        print(f"\nWriting Min Position: {min_pos}")
        robot.bus.write("Min_Position_Limit", "gripper", min_pos, normalize=False)

        print(f"Writing Max Position: {max_pos}")
        robot.bus.write("Max_Position_Limit", "gripper", max_pos, normalize=False)

        print("\n✓ Calibration written to servo firmware!")

        # Read back the values to confirm
        print("\nVerifying calibration values in servo firmware:")
        try:
            new_min = robot.bus.read("Min_Position_Limit", "gripper", normalize=False)
            new_max = robot.bus.read("Max_Position_Limit", "gripper", normalize=False)
            print(f"  Min Position Limit: {new_min}")
            print(f"  Max Position Limit: {new_max}")
            if new_min == min_pos and new_max == max_pos:
                print("  ✓ Values confirmed!")
            else:
                print("  ⚠ Warning: Read values don't match written values")
        except Exception as e:
            print(f"  Could not verify values: {e}")

        print("\nStep 3: Testing calibration")
        print("  Moving gripper through range...")

        # Test: move to 0% (should be min/closed)
        robot.bus.write("Goal_Position", "gripper", 0, normalize=True)
        import time
        time.sleep(1)
        print("  → Moved to 0% (closed)")

        # Test: move to 100% (should be max/open)
        robot.bus.write("Goal_Position", "gripper", 100, normalize=True)
        time.sleep(1)
        print("  → Moved to 100% (open)")

        # Back to closed
        robot.bus.write("Goal_Position", "gripper", 0, normalize=True)
        time.sleep(1)
        print("  → Moved back to 0% (closed)")

        print("\n" + "=" * 70)
        print("✓ Gripper calibration complete!")
        print("=" * 70)
        print("\nCalibration values:")
        print(f"  Min (closed): {min_pos}")
        print(f"  Max (open): {max_pos}")
        print("\nThe gripper should now work correctly with teleoperation!")

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
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
