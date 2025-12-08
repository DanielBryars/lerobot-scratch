#!/usr/bin/env python
"""
Read and display all EEPROM values from STS3250 motors.

Usage:
    python read_eeprom.py              # Read both arms
    python read_eeprom.py --leader     # Read leader only
    python read_eeprom.py --follower   # Read follower only
"""
import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


# Key EEPROM registers to read
EEPROM_REGISTERS = [
    "ID",
    "Baud_Rate",
    "Min_Position_Limit",
    "Max_Position_Limit",
    "Homing_Offset",
    "Operating_Mode",
    "P_Coefficient",
    "D_Coefficient",
    "I_Coefficient",
    "Max_Torque_Limit",
    "Max_Temperature_Limit",
    "CW_Dead_Zone",
    "CCW_Dead_Zone",
]

# Current state registers (SRAM)
SRAM_REGISTERS = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Torque_Enable",
]


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


def decode_homing_offset(raw_value):
    """Decode sign-magnitude encoded homing offset (bit 11 is sign)."""
    if raw_value & 0x800:  # Bit 11 set = negative
        return -(raw_value & 0x7FF)
    else:
        return raw_value & 0x7FF


def read_arm(port: str, name: str):
    """Read all EEPROM values from an arm."""
    print(f"\n{'=' * 70}")
    print(f"{name.upper()} ARM ({port})")
    print("=" * 70)

    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()

    # Read EEPROM registers
    print("\n--- EEPROM REGISTERS ---")
    print(f"{'Register':<25} ", end="")
    for motor in bus.motors:
        print(f"{motor[:8]:>10}", end="")
    print()
    print("-" * 85)

    for reg in EEPROM_REGISTERS:
        print(f"{reg:<25} ", end="")
        for motor in bus.motors:
            try:
                val = bus.read(reg, motor, normalize=False)

                # Special handling for Homing_Offset (sign-magnitude)
                if reg == "Homing_Offset":
                    decoded = decode_homing_offset(val)
                    print(f"{decoded:>+10}", end="")
                else:
                    print(f"{val:>10}", end="")
            except Exception as e:
                print(f"{'ERR':>10}", end="")
        print()

    # Read current state
    print("\n--- CURRENT STATE (SRAM) ---")
    print(f"{'Register':<25} ", end="")
    for motor in bus.motors:
        print(f"{motor[:8]:>10}", end="")
    print()
    print("-" * 85)

    for reg in SRAM_REGISTERS:
        print(f"{reg:<25} ", end="")
        for motor in bus.motors:
            try:
                val = bus.read(reg, motor, normalize=False)
                print(f"{val:>10}", end="")
            except Exception as e:
                print(f"{'ERR':>10}", end="")
        print()

    # Summary for calibration
    print("\n--- CALIBRATION SUMMARY ---")
    print(f"{'Motor':<15} {'Position':>10} {'Homing_Offset':>15} {'Effective Pos':>15} {'Dist from 2048':>15}")
    print("-" * 75)

    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        raw_offset = bus.read("Homing_Offset", motor, normalize=False)
        offset = decode_homing_offset(raw_offset)

        # Effective position = what the motor reports (already has offset applied)
        # Raw position would be: pos + offset
        effective = pos
        dist = effective - 2048

        flag = "" if abs(dist) < 200 else " <-- NOT CENTERED"
        print(f"{motor:<15} {pos:>10} {offset:>+15} {effective:>15} {dist:>+15}{flag}")

    bus.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Read EEPROM values from STS3250 motors")
    parser.add_argument("--leader", "-l", action="store_true", help="Read leader only")
    parser.add_argument("--follower", "-f", action="store_true", help="Read follower only")
    parser.add_argument("--leader-port", type=str, default=None)
    parser.add_argument("--follower-port", type=str, default=None)
    args = parser.parse_args()

    config = load_config()
    leader_port = args.leader_port or (config and config.get("leader", {}).get("port")) or "COM8"
    follower_port = args.follower_port or (config and config.get("follower", {}).get("port")) or "COM7"

    # Default to both if neither specified
    if not args.leader and not args.follower:
        args.leader = True
        args.follower = True

    print("=" * 70)
    print("EEPROM READER - STS3250 Motors")
    print("=" * 70)

    if args.leader:
        read_arm(leader_port, "Leader")

    if args.follower:
        read_arm(follower_port, "Follower")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
