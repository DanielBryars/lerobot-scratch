#!/usr/bin/env python
"""Quick debug of follower readings."""
import time
from pathlib import Path
import draccus

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION


def create_bus(port: str):
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
    # Load calibration files
    leader_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / "leader_so100.json"
    follower_path = HF_LEROBOT_CALIBRATION / "robots" / "so100_follower_sts3250" / "follower_so100.json"

    print("=" * 70)
    print("FOLLOWER DEBUG")
    print("=" * 70)

    print(f"\nLeader calibration: {leader_path}")
    print(f"Follower calibration: {follower_path}")

    # Load and show calibrations
    print("\n--- LEADER CALIBRATION ---")
    if leader_path.exists():
        with open(leader_path) as f, draccus.config_type("json"):
            leader_cal = draccus.load(dict[str, MotorCalibration], f)
        for motor, cal in leader_cal.items():
            print(f"  {motor:<15}: range=[{cal.range_min:5d}, {cal.range_max:5d}], drive_mode={cal.drive_mode}")
    else:
        print("  NOT FOUND!")

    print("\n--- FOLLOWER CALIBRATION ---")
    if follower_path.exists():
        with open(follower_path) as f, draccus.config_type("json"):
            follower_cal = draccus.load(dict[str, MotorCalibration], f)
        for motor, cal in follower_cal.items():
            print(f"  {motor:<15}: range=[{cal.range_min:5d}, {cal.range_max:5d}], drive_mode={cal.drive_mode}")
    else:
        print("  NOT FOUND!")
        return

    # Connect to follower
    print("\nConnecting to follower (COM7)...")
    bus = create_bus("COM7")
    bus.connect()
    bus.disable_torque()
    bus.calibration = follower_cal

    print("\n--- LIVE READINGS (move the arm) ---")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            # Read raw and normalized
            raw = {}
            norm = {}
            for motor in bus.motors:
                raw[motor] = bus.read("Present_Position", motor, normalize=False)
                norm[motor] = bus.read("Present_Position", motor, normalize=True)

            print("\033[H\033[J")  # Clear screen
            print("=" * 70)
            print("FOLLOWER LIVE READINGS")
            print("=" * 70)
            print(f"\n{'Motor':<15} {'Raw':>8} {'Normalized':>12} {'Range':>20}")
            print("-" * 60)

            for motor in bus.motors:
                cal = follower_cal[motor]
                range_str = f"[{cal.range_min}, {cal.range_max}]"

                # Check if raw is within range
                in_range = cal.range_min <= raw[motor] <= cal.range_max
                flag = "" if in_range else " <-- OUT OF RANGE!"

                print(f"{motor:<15} {raw[motor]:>8} {norm[motor]:>12.2f} {range_str:>20}{flag}")

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass

    bus.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
