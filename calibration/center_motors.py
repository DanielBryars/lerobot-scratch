#!/usr/bin/env python
"""
Center motor readings by setting EEPROM homing_offset.

This makes both arms read ~2048 at zero pose, avoiding wraparound issues.
Run this BEFORE calibrate_from_zero.py.

Usage:
    python center_motors.py --leader
    python center_motors.py --follower
    python center_motors.py --leader --follower
"""
import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


TARGET_CENTER = 2048  # Midpoint of 4096-count encoder


def load_config():
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


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


def center_arm(port: str, arm_name: str):
    """Set homing_offset so zero pose reads ~2048."""

    print(f"\n{'=' * 60}")
    print(f"CENTERING {arm_name.upper()} ARM")
    print(f"Port: {port}")
    print("=" * 60)

    print("""
ZERO POSE REFERENCE:
  - All joints at 0° (arm extended forward, horizontal)
  - Gripper jaws VERTICAL (like gripping a horizontal bar)
  - Gripper CLOSED
""")

    print(f"Connecting to {arm_name} on {port}...")
    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()
    print("Connected!")

    # Show current state
    print("\n--- CURRENT STATE (before centering) ---")
    print(f"{'Motor':<15} {'Position':>10} {'Homing_Offset':>15} {'Distance from 2048':>20}")
    print("-" * 65)

    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        offset = bus.read("Homing_Offset", motor, normalize=False)
        dist = pos - TARGET_CENTER
        flag = " <-- FAR!" if abs(dist) > 1000 else ""
        print(f"{motor:<15} {pos:>10} {offset:>15} {dist:>+20}{flag}")

    print("\n>>> Position the arm at ZERO POSE <<<")
    print("Press Enter when ready...")
    input()

    # Read positions at zero pose
    print("\nReading positions at zero pose...")
    zero_positions = {}
    for motor in bus.motors:
        # Read the RAW position (without homing offset applied)
        # Present_Position already has homing_offset applied by firmware
        # To get raw: raw = present + current_homing_offset
        present = bus.read("Present_Position", motor, normalize=False)
        current_offset = bus.read("Homing_Offset", motor, normalize=False)

        # Handle signed offset (it's stored as signed in EEPROM)
        if current_offset > 2048:
            current_offset = current_offset - 4096

        raw = present + current_offset
        if raw < 0:
            raw += 4096
        elif raw >= 4096:
            raw -= 4096

        zero_positions[motor] = (present, raw)

    print(f"\n{'Motor':<15} {'Present':>10} {'Raw':>10}")
    print("-" * 40)
    for motor, (present, raw) in zero_positions.items():
        print(f"{motor:<15} {present:>10} {raw:>10}")

    # Calculate new homing offsets to center around 2048
    print("\n--- CALCULATING NEW HOMING OFFSETS ---")
    print(f"Target: All motors read {TARGET_CENTER} at zero pose")
    print()

    new_offsets = {}
    for motor, (present, raw) in zero_positions.items():
        # We want: present_position_after = TARGET_CENTER
        # present_position = raw_position - homing_offset
        # So: homing_offset = raw_position - TARGET_CENTER
        new_offset = raw - TARGET_CENTER

        # Handle wraparound for offset
        if new_offset > 2048:
            new_offset = new_offset - 4096
        elif new_offset < -2048:
            new_offset = new_offset + 4096

        new_offsets[motor] = new_offset

        expected_new_pos = raw - new_offset
        if expected_new_pos < 0:
            expected_new_pos += 4096
        elif expected_new_pos >= 4096:
            expected_new_pos -= 4096

        print(f"{motor:<15}: raw={raw:4d}, new_offset={new_offset:+5d}, will read={expected_new_pos:4d}")

    print(f"\nThis will write homing_offset to EEPROM for {arm_name}.")
    print("Press Enter to continue, or Ctrl+C to cancel...")
    input()

    # Write new offsets
    print("\nWriting homing offsets to EEPROM...")
    for motor, offset in new_offsets.items():
        # Convert signed offset to value for writing
        write_val = offset if offset >= 0 else offset + 4096
        bus.write("Homing_Offset", motor, write_val, normalize=False)
        print(f"  {motor}: wrote {offset:+d} (as {write_val})")

    # Verify
    print("\n--- VERIFICATION ---")
    print(f"{'Motor':<15} {'New Position':>12} {'Target':>10} {'Diff':>10}")
    print("-" * 50)

    all_ok = True
    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        diff = pos - TARGET_CENTER
        ok = "OK" if abs(diff) < 50 else "CHECK!"
        if abs(diff) >= 50:
            all_ok = False
        print(f"{motor:<15} {pos:>12} {TARGET_CENTER:>10} {diff:>+10} {ok}")

    if all_ok:
        print("\n[OK] All motors centered successfully!")
    else:
        print("\n[!] Some motors may need adjustment. Check positions.")

    bus.disconnect()
    print(f"\n{arm_name.capitalize()} centering complete!")
    print("Now run: python calibrate_from_zero.py --" + arm_name)


def main():
    parser = argparse.ArgumentParser(description="Center motor readings via EEPROM homing_offset")
    parser.add_argument("--leader", "-l", action="store_true", help="Center leader arm")
    parser.add_argument("--follower", "-f", action="store_true", help="Center follower arm")
    parser.add_argument("--leader-port", type=str, default=None)
    parser.add_argument("--follower-port", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    leader_port = args.leader_port or (config and config.get("leader", {}).get("port")) or "COM8"
    follower_port = args.follower_port or (config and config.get("follower", {}).get("port")) or "COM7"

    if not args.leader and not args.follower:
        parser.print_help()
        print("\nExample:")
        print("  python center_motors.py --leader")
        print("  python center_motors.py --follower")
        print("  python center_motors.py --leader --follower")
        return

    print("=" * 60)
    print("MOTOR CENTERING")
    print("=" * 60)
    print("""
This script sets EEPROM homing_offset so motors read ~2048 at zero pose.
This prevents wraparound issues with file-based calibration.

After centering, run calibrate_from_zero.py to create the calibration file.
""")

    if args.leader:
        center_arm(leader_port, "leader")

    if args.follower:
        center_arm(follower_port, "follower")

    print("\n" + "=" * 60)
    print("CENTERING COMPLETE")
    print("=" * 60)
    print("\nNext step: python calibrate_from_zero.py --leader --follower")


if __name__ == "__main__":
    main()
