#!/usr/bin/env python
"""
Calibrate SO100 arms by positioning at zero pose only.

The min/max limits are calculated from the simulation's known joint limits,
not by physically moving to the limits. Only the offset (where the motor
happened to be at assembly) needs to be measured.

Usage:
    python calibrate_from_zero.py --leader
    python calibrate_from_zero.py --follower
    python calibrate_from_zero.py --leader --follower
"""
import argparse
import json
from pathlib import Path

import draccus
import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION


# =============================================================================
# SIMULATION JOINT LIMITS (from SO101 MuJoCo XML)
# These are the same for all SO101 arms - defined by the physical design
# =============================================================================
SIM_LIMITS_RAD = {
    #                    (min_rad,    max_rad)
    "shoulder_pan":  (-1.91986,  1.91986),
    "shoulder_lift": (-1.74533,  1.74533),
    "elbow_flex":    (-1.69,     1.69),
    "wrist_flex":    (-1.65806,  1.65806),
    "wrist_roll":    (-2.74385,  2.84121),
    "gripper":       (-0.17453,  1.74533),
}

# Motor specifications
MOTOR_RESOLUTION = 4096  # counts per revolution
COUNTS_PER_RAD = MOTOR_RESOLUTION / (2 * np.pi)  # ~651.9 counts/rad


def load_config():
    """Load config.json for COM ports."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def create_bus(port: str):
    """Create motor bus for arm."""
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


def print_zero_pose():
    """Print the expected zero pose for reference."""
    print("\n" + "=" * 60)
    print("ZERO POSE REFERENCE (from simulation)")
    print("=" * 60)
    print("""
Position your arm to match this pose:

  - shoulder_pan:  0° - pointing forward
  - shoulder_lift: 0° - horizontal
  - elbow_flex:    0° - mid-bend
  - wrist_flex:    0° - level with forearm
  - wrist_roll:    0° - gripper jaws VERTICAL (like gripping a horizontal bar)
  - gripper:       CLOSED

Visual reference:
  - Arm extended forward, roughly horizontal
  - Gripper jaws vertical (pinching from above/below, like gripping a bar)
  - Gripper closed

Run 'python show_zero_pose.py' in lerobot-gym for visual reference.
""")
    print("=" * 60)


def calibrate_arm(port: str, arm_type: str, arm_id: str):
    """Calibrate arm by reading zero pose position."""

    print(f"\n{'=' * 60}")
    print(f"CALIBRATING {arm_type.upper()}: {arm_id}")
    print(f"Port: {port}")
    print("=" * 60)

    print_zero_pose()

    # Connect to arm
    print(f"\nConnecting to {arm_type} on {port}...")
    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()
    print("Connected! Torque disabled - you can move the arm freely.")

    # Show current EEPROM settings
    print("\n" + "-" * 60)
    print("CURRENT EEPROM SETTINGS (from motor firmware):")
    print("-" * 60)
    eeprom_cal = bus.read_calibration()
    for motor, cal in eeprom_cal.items():
        print(f"  {motor:<15}: homing_offset={cal.homing_offset:5d}, range=[{cal.range_min:5d}, {cal.range_max:5d}]")

    # Show current file-based calibration (if exists)
    if arm_type == "leader":
        existing_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / f"{arm_id}.json"
    else:
        existing_path = HF_LEROBOT_CALIBRATION / "robots" / "so100_follower_sts3250" / f"{arm_id}.json"

    print("\n" + "-" * 60)
    print(f"CURRENT FILE CALIBRATION ({existing_path}):")
    print("-" * 60)
    if existing_path.exists():
        with open(existing_path) as f, draccus.config_type("json"):
            existing_cal = draccus.load(dict[str, MotorCalibration], f)
        for motor, cal in existing_cal.items():
            inv = " [INV]" if cal.drive_mode == 1 else ""
            print(f"  {motor:<15}: range=[{cal.range_min:5d}, {cal.range_max:5d}], drive_mode={cal.drive_mode}{inv}")
    else:
        print("  (no existing calibration file)")

    # Wait for user to position arm
    print("\n>>> Position the arm at the ZERO POSE shown above <<<")
    print("\nPress Enter when the arm is positioned...")
    input()

    # Read raw positions at zero pose
    print("\nReading motor positions...")

    # Read raw counts (without normalization)
    raw_positions = {}
    for motor in bus.motors:
        # Read Present_Position without any calibration applied
        raw = bus.read("Present_Position", motor, normalize=False)
        raw_positions[motor] = raw

    print("\nRaw motor counts at zero pose:")
    print("-" * 40)
    for motor, raw in raw_positions.items():
        print(f"  {motor:<15}: {raw:5d} counts")

    # Calculate calibration based on sim limits
    print("\n" + "-" * 60)
    print("NEW CALIBRATION (calculated from zero pose + sim limits):")
    print("-" * 60)

    calibration = {}
    for motor in bus.motors:
        zero_counts = raw_positions[motor]
        min_rad, max_rad = SIM_LIMITS_RAD[motor]

        if motor == "gripper":
            # Special case: gripper zero pose = CLOSED = min_rad (not 0 rad)
            # So zero_counts corresponds to min_rad position
            range_min = zero_counts
            range_max = int(zero_counts + (max_rad - min_rad) * COUNTS_PER_RAD)
            inverted = False
        else:
            # Normal joints: zero pose = 0 radians
            # Calculate range assuming normal motor direction
            range_min = int(zero_counts + min_rad * COUNTS_PER_RAD)
            range_max = int(zero_counts + max_rad * COUNTS_PER_RAD)

            # Check if motor is inverted (range goes outside valid bounds)
            # Motor counts should be in [0, 4095] range
            if range_min < -500 or range_max > 4595:
                # Motor is inverted - flip the direction
                # For inverted motor: increasing angle = decreasing counts
                range_min = int(zero_counts - min_rad * COUNTS_PER_RAD)
                range_max = int(zero_counts - max_rad * COUNTS_PER_RAD)
                # Swap so min < max
                range_min, range_max = range_max, range_min
                inverted = True
            else:
                inverted = False

        # Calculate total range in degrees for display
        total_deg = np.degrees(max_rad - min_rad)
        inv_marker = " [INVERTED]" if inverted else ""

        print(f"  {motor:<15}: range [{range_min:5d}, {range_max:5d}] ({total_deg:.1f}° range){inv_marker}")

        calibration[motor] = MotorCalibration(
            id=bus.motors[motor].id,
            drive_mode=1 if inverted else 0,  # 1 = inverted
            homing_offset=0,  # Not using EEPROM offset
            range_min=range_min,
            range_max=range_max,
        )

    # Determine calibration file path (use same path as existing_path)
    calib_dir = existing_path.parent
    calib_dir.mkdir(parents=True, exist_ok=True)
    calib_path = existing_path

    # Save calibration
    with open(calib_path, "w") as f, draccus.config_type("json"):
        draccus.dump(calibration, f, indent=4)

    print(f"\nCalibration saved to: {calib_path}")

    # Verify by applying calibration and reading normalized values
    print("\nVerifying calibration...")
    bus.calibration = calibration

    normalized = bus.sync_read("Present_Position")
    print("\nNormalized values at zero pose:")
    print("-" * 60)
    for motor, val in normalized.items():
        min_rad, max_rad = SIM_LIMITS_RAD[motor]

        if motor == "gripper":
            # Gripper: zero pose = closed = 0 normalized (RANGE_0_100)
            expected = 0.0
        else:
            # Other joints: calculate expected normalized for 0 radians
            # For RANGE_M100_100: normalized = (rad - min) / (max - min) * 200 - 100
            # At 0 rad: expected = (0 - min) / (max - min) * 200 - 100
            expected = (-min_rad) / (max_rad - min_rad) * 200 - 100

        diff = abs(val - expected)
        status = "✓" if diff < 2.0 else "✗"
        print(f"  {motor:<15}: {val:7.2f} (expected: {expected:6.2f}) {status}")

    bus.disconnect()
    print(f"\n{arm_type.capitalize()} calibration complete!")

    return calib_path


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SO100 arms from zero pose only"
    )
    parser.add_argument("--leader", "-l", action="store_true", help="Calibrate leader")
    parser.add_argument("--follower", "-f", action="store_true", help="Calibrate follower")
    parser.add_argument("--leader-port", type=str, default=None)
    parser.add_argument("--follower-port", type=str, default=None)
    parser.add_argument("--leader-id", type=str, default="leader_so100")
    parser.add_argument("--follower-id", type=str, default="follower_so100")
    args = parser.parse_args()

    # Load config
    config = load_config()
    leader_port = args.leader_port or (config and config.get("leader", {}).get("port")) or "COM8"
    follower_port = args.follower_port or (config and config.get("follower", {}).get("port")) or "COM7"
    leader_id = args.leader_id if args.leader_id != "leader_so100" else (config and config.get("leader", {}).get("id")) or "leader_so100"
    follower_id = args.follower_id if args.follower_id != "follower_so100" else (config and config.get("follower", {}).get("id")) or "follower_so100"

    if not args.leader and not args.follower:
        parser.print_help()
        print("\nExample:")
        print("  python calibrate_from_zero.py --leader")
        print("  python calibrate_from_zero.py --follower")
        print("  python calibrate_from_zero.py --leader --follower")
        return

    print("=" * 60)
    print("ZERO-POSE CALIBRATION")
    print("=" * 60)
    print("""
This calibration method:
1. You position the arm at the known 'zero' pose
2. Script reads the raw motor counts
3. Min/max are CALCULATED from simulation limits (not physical movement)

This ensures the angle mapping matches the simulation exactly.
""")

    paths = []

    if args.leader:
        path = calibrate_arm(leader_port, "leader", leader_id)
        paths.append(path)

    if args.follower:
        path = calibrate_arm(follower_port, "follower", follower_id)
        paths.append(path)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print("\nCalibration files created:")
    for p in paths:
        print(f"  {p}")
    print("\nThe angle mapping now matches the simulation exactly.")


if __name__ == "__main__":
    main()
