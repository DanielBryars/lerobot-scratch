#!/usr/bin/env python
"""
Write homing offsets to center motors at 2048.

This script unlocks EEPROM, writes offsets, and verifies.
"""
import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def load_config():
    config_path = Path(__file__).parent.parent / "config.json"
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


def encode_homing_offset(value):
    """Encode signed value to sign-magnitude (bit 11 = sign)."""
    if value < 0:
        return 0x800 | (abs(value) & 0x7FF)
    else:
        return value & 0x7FF


def decode_homing_offset(raw):
    """Decode sign-magnitude to signed value."""
    if raw & 0x800:
        return -(raw & 0x7FF)
    return raw & 0x7FF


def center_arm(port: str, name: str):
    """Center all motors on an arm."""
    print(f"\n{'=' * 60}")
    print(f"CENTERING {name.upper()} ({port})")
    print("=" * 60)

    bus = create_bus(port)
    bus.connect()

    # Step 1: Disable torque (required for EEPROM writes)
    print("\n1. Disabling torque...")
    bus.disable_torque()
    print("   Done")

    # Step 2: Unlock EEPROM (Lock register = 0)
    print("\n2. Unlocking EEPROM (Lock=0)...")
    for motor in bus.motors:
        try:
            bus.write("Lock", motor, 0, normalize=False)
            print(f"   {motor}: unlocked")
        except Exception as e:
            print(f"   {motor}: {e}")

    # Step 3: Read current positions
    print("\n3. Reading current positions...")
    positions = {}
    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        positions[motor] = pos
        print(f"   {motor}: {pos}")

    # Step 4: Calculate offsets
    print("\n4. Calculating homing offsets (target=2048)...")
    offsets = {}
    for motor, pos in positions.items():
        offset = pos - 2048
        offsets[motor] = offset
        encoded = encode_homing_offset(offset)
        print(f"   {motor}: offset={offset:+d}, encoded=0x{encoded:04X} ({encoded})")

    # Step 5: Write offsets using lerobot's method
    print("\n5. Writing homing offsets...")
    for motor, offset in offsets.items():
        try:
            # lerobot handles the sign-magnitude encoding internally
            bus.write("Homing_Offset", motor, offset, normalize=False)
            print(f"   {motor}: wrote {offset:+d}")
        except Exception as e:
            print(f"   {motor}: FAILED - {e}")

    # Step 6: Lock EEPROM (Lock register = 1)
    print("\n6. Locking EEPROM (Lock=1)...")
    for motor in bus.motors:
        try:
            bus.write("Lock", motor, 1, normalize=False)
            print(f"   {motor}: locked")
        except Exception as e:
            print(f"   {motor}: {e}")

    # Step 7: Verify by re-reading
    print("\n7. Verifying...")
    print(f"   {'Motor':<15} {'Expected':>10} {'Read':>10} {'Decoded':>10} {'Status':>10}")
    print("   " + "-" * 55)

    all_ok = True
    for motor, expected in offsets.items():
        raw = bus.read("Homing_Offset", motor, normalize=False)
        decoded = decode_homing_offset(raw)
        match = abs(decoded - expected) < 5
        status = "OK" if match else "MISMATCH!"
        if not match:
            all_ok = False
        print(f"   {motor:<15} {expected:>+10} {raw:>10} {decoded:>+10} {status:>10}")

    # Step 8: Check new positions (should be ~2048)
    print("\n8. Checking new positions (should be ~2048)...")
    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        diff = pos - 2048
        status = "OK" if abs(diff) < 50 else "NOT CENTERED!"
        print(f"   {motor}: {pos} (diff: {diff:+d}) {status}")

    bus.disconnect()

    if all_ok:
        print(f"\n[OK] {name} centered successfully!")
    else:
        print(f"\n[!] {name} centering may have failed - check above")

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Write homing offsets to center motors")
    parser.add_argument("--leader", "-l", action="store_true")
    parser.add_argument("--follower", "-f", action="store_true")
    parser.add_argument("--leader-port", type=str, default=None)
    parser.add_argument("--follower-port", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    leader_port = args.leader_port or (config and config.get("leader", {}).get("port")) or "COM8"
    follower_port = args.follower_port or (config and config.get("follower", {}).get("port")) or "COM7"

    if not args.leader and not args.follower:
        print("Usage: python write_homing_offset.py --leader --follower")
        print("\nThis will set EEPROM homing_offset so motors read 2048 at current position.")
        print("\n!!! POSITION ARMS AT ZERO POSE FIRST !!!")
        return

    print("=" * 60)
    print("WRITE HOMING OFFSETS")
    print("=" * 60)
    print("\nThis writes to motor EEPROM to center readings at 2048.")
    print("Make sure arms are at ZERO POSE!")
    print("\nPress Enter to continue, Ctrl+C to cancel...")
    input()

    if args.leader:
        center_arm(leader_port, "Leader")

    if args.follower:
        center_arm(follower_port, "Follower")

    print("\n" + "=" * 60)
    print("DONE - Now run: python calibration/calibrate_from_zero.py --leader --follower")
    print("=" * 60)


if __name__ == "__main__":
    main()
