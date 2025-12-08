#!/usr/bin/env python
"""
Reset motor EEPROM to factory defaults.

This resets:
- Homing_Offset to 0
- Min_Position_Limit to 0
- Max_Position_Limit to 4095 (full range)

After running this, use lerobot's calibration procedure which stores
calibration in JSON files instead of EEPROM.

Usage:
    python reset_eeprom.py           # Reset both arms
    python reset_eeprom.py --leader  # Reset leader only
    python reset_eeprom.py --follower # Reset follower only
"""
import argparse
import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def load_config():
    """Load config.json for COM ports."""
    config_path = Path("config.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def create_bus(port: str, motor_model: str = "sts3250"):
    """Create a FeetechMotorsBus for an arm."""
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, motor_model, MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, motor_model, MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, motor_model, MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, motor_model, MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, motor_model, MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, motor_model, MotorNormMode.RANGE_0_100),
        },
    )
    return bus


def read_and_print_calibration(bus: FeetechMotorsBus, name: str):
    """Read and display current EEPROM calibration values."""
    print(f"\n{name} EEPROM values:")
    print("-" * 60)
    print(f"{'Motor':<15} {'Homing_Offset':>14} {'Min_Pos':>10} {'Max_Pos':>10}")
    print("-" * 60)

    for motor in bus.motors:
        try:
            offset = bus.read("Homing_Offset", motor, normalize=False)
        except Exception:
            offset = "N/A"
        min_pos = bus.read("Min_Position_Limit", motor, normalize=False)
        max_pos = bus.read("Max_Position_Limit", motor, normalize=False)
        print(f"{motor:<15} {offset:>14} {min_pos:>10} {max_pos:>10}")


def reset_arm(port: str, name: str, motor_model: str = "sts3250"):
    """Reset EEPROM calibration for one arm."""
    print(f"\n{'='*60}")
    print(f"Resetting {name} on {port}")
    print("=" * 60)

    bus = create_bus(port, motor_model)
    bus.connect()

    # Show current values
    print("\nBEFORE reset:")
    read_and_print_calibration(bus, name)

    # Disable torque first (required to write EEPROM)
    print(f"\nDisabling torque...")
    bus.disable_torque()

    # Reset calibration - this sets:
    # - Homing_Offset to 0
    # - Min/Max_Position_Limit to full range (0 to resolution-1)
    print(f"Resetting EEPROM to factory defaults...")
    bus.reset_calibration()

    # Show new values
    print("\nAFTER reset:")
    read_and_print_calibration(bus, name)

    bus.disconnect()
    print(f"\n{name} reset complete!")


def main():
    parser = argparse.ArgumentParser(description="Reset motor EEPROM to defaults")
    parser.add_argument("--leader", "-l", action="store_true", help="Reset leader arm only")
    parser.add_argument("--follower", "-f", action="store_true", help="Reset follower arm only")
    parser.add_argument("--leader-port", type=str, default=None, help="Leader port (default: from config)")
    parser.add_argument("--follower-port", type=str, default=None, help="Follower port (default: from config)")
    parser.add_argument("--motor-model", type=str, default="sts3250", help="Motor model (default: sts3250)")
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

    # Determine which arms to reset
    reset_leader = args.leader or (not args.leader and not args.follower)
    reset_follower = args.follower or (not args.leader and not args.follower)

    print("=" * 60)
    print("EEPROM RESET TOOL")
    print("=" * 60)
    print("\nThis will reset motor EEPROM to factory defaults:")
    print("  - Homing_Offset -> 0")
    print("  - Min_Position_Limit -> 0")
    print("  - Max_Position_Limit -> 4095")
    print("\nAfter this, use lerobot's calibration procedure.")

    if reset_leader:
        reset_arm(leader_port, "Leader", args.motor_model)

    if reset_follower:
        reset_arm(follower_port, "Follower", args.motor_model)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run lerobot calibration for leader:")
    print(f"   lerobot-calibrate --teleop.type=so100_leader --teleop.port={leader_port} --teleop.id=leader")
    print("\n2. Run lerobot calibration for follower:")
    print(f"   lerobot-calibrate --robot.type=so100_follower --robot.port={follower_port} --robot.id=follower")
    print("\nCalibration will be saved to ~/.cache/huggingface/lerobot/calibration/")


if __name__ == "__main__":
    main()
