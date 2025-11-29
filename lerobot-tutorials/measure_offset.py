"""
Measure the offset between leader and follower when physically aligned.
Put both arms in the EXACT same position, then run this script.
It will compute the offset needed to make them match.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

CALIBRATION_DIR = Path(__file__).parent.parent / "calibration"
CALIBRATION_DIR.mkdir(exist_ok=True)


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
    print("="*60)
    print("MEASURE OFFSET BETWEEN LEADER AND FOLLOWER")
    print("="*60)
    print("""
Put BOTH arms in the EXACT same physical position.
This script will measure the raw value difference and create
calibration that accounts for assembly differences.
""")

    input("Press ENTER when both arms are aligned...")

    follower_bus = create_bus(config["follower"]["port"])
    leader_bus = create_bus(config["leader"]["port"])

    print("\nConnecting...")
    follower_bus.connect()
    leader_bus.connect()

    # Read raw positions
    print("\nReading raw positions...")

    print(f"\n{'Motor':<15} | {'Follower':>8} | {'Leader':>8} | {'Diff':>8}")
    print(f"{'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    offsets = {}
    for motor in follower_bus.motors:
        f_raw = follower_bus.read("Present_Position", motor, normalize=False)
        l_raw = leader_bus.read("Present_Position", motor, normalize=False)
        diff = f_raw - l_raw
        offsets[motor] = diff
        print(f"{motor:<15} | {f_raw:>8} | {l_raw:>8} | {diff:>8}")

    print("\n" + "="*60)
    print("COMPUTED OFFSETS (follower - leader)")
    print("="*60)
    print("""
These offsets represent the assembly difference between arms.
We can apply these in software so normalized values match.
""")

    for motor, offset in offsets.items():
        print(f"  {motor}: {offset}")

    # Create calibration files
    # For follower: use raw values as-is (offset=0)
    # For leader: apply offset so it matches follower

    print("\n" + "="*60)
    response = input("Save calibration files? (yes/no): ")

    if response.lower() == "yes":
        # Both use same range, but leader gets homing_offset to match follower
        follower_cal = {}
        leader_cal = {}

        for motor in follower_bus.motors:
            m = follower_bus.motors[motor]
            # Follower: no offset needed
            follower_cal[motor] = {
                "id": m.id,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            # Leader: apply offset so when aligned, both normalize the same
            # If follower_raw = leader_raw + offset, then leader needs homing_offset = -offset
            # So that: leader_adjusted = leader_raw - homing_offset = leader_raw + offset = follower_raw
            leader_cal[motor] = {
                "id": m.id,
                "drive_mode": 0,
                "homing_offset": -offsets[motor],  # Note: negative!
                "range_min": 0,
                "range_max": 4095,
            }

        # Save files
        follower_file = CALIBRATION_DIR / "follower_so100.json"
        leader_file = CALIBRATION_DIR / "leader_so100.json"

        with open(follower_file, "w") as f:
            json.dump(follower_cal, f, indent=2)
        print(f"Saved: {follower_file}")

        with open(leader_file, "w") as f:
            json.dump(leader_cal, f, indent=2)
        print(f"Saved: {leader_file}")

        print("""
Calibration files saved!

NOTE: These are SOFTWARE calibration files only.
The STS3250 classes need to be updated to load these files
instead of using FIXED_CALIBRATION.
""")

    follower_bus.disconnect()
    leader_bus.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
