"""
Align both arms by setting EEPROM homing_offset so they report the same values.

Calibrates one arm at a time:
1. Put FOLLOWER in middle position, press Enter
2. Put LEADER in the SAME middle position, press Enter
3. Both will now report ~2048 when in that position
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


def calibrate_arm(bus: FeetechMotorsBus, name: str):
    """Calibrate a single arm."""
    print(f"\n{'='*60}")
    print(f"CALIBRATING {name.upper()}")
    print(f"{'='*60}")

    # Reset existing calibration
    print("Resetting existing EEPROM calibration...")
    bus.reset_calibration()

    # Show current raw position
    print(f"\nCurrent raw positions:")
    for motor in bus.motors:
        raw = bus.read("Present_Position", motor, normalize=False)
        print(f"  {motor}: {raw}")

    print(f"""
Move {name.upper()} to the MIDDLE of its range:
- All joints roughly centered
- Arm in "L" shape from the side
- Gripper half open

This position will become "2048" (center) for all joints.
""")
    input(f"Press ENTER when {name.upper()} is in position...")

    # Set half-turn homings
    print(f"\nWriting homing_offset to {name} EEPROM...")
    offsets = bus.set_half_turn_homings()

    print(f"\nHoming offsets written:")
    for motor, offset in offsets.items():
        print(f"  {motor}: {offset}")

    # Verify
    print(f"\nVerifying - raw positions should now be ~2047:")
    for motor in bus.motors:
        raw = bus.read("Present_Position", motor, normalize=False)
        print(f"  {motor}: {raw}")


def main():
    print("="*60)
    print("ALIGN ARMS - One at a time")
    print("="*60)
    print("""
This script calibrates each arm separately.
Put each arm in the SAME physical "middle" position.
After calibration, both will report ~2048 at that position.
""")

    # Connect to both
    follower_bus = create_bus(config["follower"]["port"])
    leader_bus = create_bus(config["leader"]["port"])

    print("Connecting to both arms...")
    follower_bus.connect()
    leader_bus.connect()
    print("Connected!\n")

    # Calibrate follower first
    calibrate_arm(follower_bus, "follower")

    # Then leader
    calibrate_arm(leader_bus, "leader")

    # Final verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    print("\nPut both arms in the same position and check the values:")

    input("\nPress ENTER to read both arms...")

    print(f"\n{'Motor':<15} | {'Follower':>8} | {'Leader':>8} | {'Diff':>8}")
    print(f"{'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for motor in follower_bus.motors:
        f_raw = follower_bus.read("Present_Position", motor, normalize=False)
        l_raw = leader_bus.read("Present_Position", motor, normalize=False)
        diff = f_raw - l_raw
        print(f"{motor:<15} | {f_raw:>8} | {l_raw:>8} | {diff:>8}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print("""
If the Diff column is close to 0, calibration is good!
Now try:
  python echojointangles.py  (to monitor)
  python teleoperate.py      (to test)
""")

    follower_bus.disconnect()
    leader_bus.disconnect()


if __name__ == "__main__":
    main()
