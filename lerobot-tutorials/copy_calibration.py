"""
Copy calibration from leader motors to follower motors.
This writes homing_offset and position limits to the follower's motor EEPROM.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)


def create_bus(port: str) -> FeetechMotorsBus:
    return FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )


def main():
    print("=" * 60)
    print("Copy Calibration: Leader -> Follower")
    print("=" * 60)

    leader_bus = create_bus(config["leader"]["port"])
    follower_bus = create_bus(config["follower"]["port"])

    print(f"\nConnecting to leader ({config['leader']['port']})...")
    leader_bus.connect()

    print(f"Connecting to follower ({config['follower']['port']})...")
    follower_bus.connect()

    # Read calibration from leader
    print("\nReading calibration from LEADER motors...")
    leader_calibration = leader_bus.read_calibration()

    print("\nLeader calibration:")
    for motor, calib in leader_calibration.items():
        print(f"  {motor:15s}: homing_offset={calib.homing_offset:5d}, "
              f"range=[{calib.range_min:4d}, {calib.range_max:4d}]")

    # Read current follower calibration
    print("\nCurrent FOLLOWER calibration:")
    follower_calibration = follower_bus.read_calibration()
    for motor, calib in follower_calibration.items():
        print(f"  {motor:15s}: homing_offset={calib.homing_offset:5d}, "
              f"range=[{calib.range_min:4d}, {calib.range_max:4d}]")

    # Confirm before writing
    print("\n" + "=" * 60)
    response = input("Write leader calibration to follower motors? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        leader_bus.disconnect()
        follower_bus.disconnect()
        return

    # Write calibration to follower
    print("\nWriting calibration to FOLLOWER motors...")
    follower_bus.write_calibration(leader_calibration)

    # Verify by reading back
    print("\nVerifying - reading back from follower...")
    follower_calibration_new = follower_bus.read_calibration()

    print("\nNew FOLLOWER calibration:")
    for motor, calib in follower_calibration_new.items():
        print(f"  {motor:15s}: homing_offset={calib.homing_offset:5d}, "
              f"range=[{calib.range_min:4d}, {calib.range_max:4d}]")

    # Check if they match
    match = True
    for motor in leader_calibration:
        l = leader_calibration[motor]
        f = follower_calibration_new[motor]
        if l.homing_offset != f.homing_offset or l.range_min != f.range_min or l.range_max != f.range_max:
            print(f"\nWARNING: {motor} doesn't match!")
            match = False

    if match:
        print("\n✓ Calibration copied successfully!")
    else:
        print("\n✗ Some values didn't match - check above")

    leader_bus.disconnect()
    follower_bus.disconnect()
    print("\nDone.")


if __name__ == "__main__":
    main()
