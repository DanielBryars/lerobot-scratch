"""
Calibrate a single SO100 arm with STS3250 motors.
Run this for BOTH leader and follower, with each arm in the same "middle" position.

Usage:
    python calibrate_arm.py leader
    python calibrate_arm.py follower
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

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


def calibrate(bus: FeetechMotorsBus, arm_name: str):
    """Run calibration process for an arm."""

    print(f"\n{'='*60}")
    print(f"CALIBRATING: {arm_name.upper()}")
    print(f"{'='*60}")

    # Disable torque so arm can be moved freely
    print("\nDisabling torque...")
    bus.disable_torque()

    # Set operating mode
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # Show current positions
    print("\nCurrent raw positions:")
    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        print(f"  {motor}: {pos}")

    # Step 1: Set homing offsets
    print("\n" + "="*60)
    print("STEP 1: Set middle position")
    print("="*60)
    print("""
Move the arm to the MIDDLE of its range of motion:
- All joints roughly centered
- Arm should look like an "L" shape from the side
- Gripper half open

This position will become "zero" for all joints.
""")
    input("Press ENTER when arm is in middle position...")

    # Reset calibration first
    print("\nResetting calibration...")
    bus.reset_calibration()

    # Read current positions and compute homing offsets
    print("Reading positions and computing homing offsets...")
    homing_offsets = bus.set_half_turn_homings()

    print("\nHoming offsets set:")
    for motor, offset in homing_offsets.items():
        print(f"  {motor}: {offset}")

    # Step 2: Record range of motion
    print("\n" + "="*60)
    print("STEP 2: Record range of motion")
    print("="*60)
    print("""
Now move EACH joint (except wrist_roll) through its FULL range:
- Move shoulder_pan left and right to its limits
- Move shoulder_lift up and down to its limits
- Move elbow_flex through its full range
- Move wrist_flex through its full range
- Open and close gripper fully

wrist_roll can do full rotations, so we'll use 0-4095 for it.
""")

    # Record ranges (excluding wrist_roll which can do full turns)
    full_turn_motor = "wrist_roll"
    unknown_range_motors = [m for m in bus.motors if m != full_turn_motor]

    input("Press ENTER to start recording, then move joints. Press ENTER again to stop...")
    range_mins, range_maxes = bus.record_ranges_of_motion(unknown_range_motors)

    # wrist_roll can do full rotation
    range_mins[full_turn_motor] = 0
    range_maxes[full_turn_motor] = 4095

    print("\nRecorded ranges:")
    for motor in bus.motors:
        print(f"  {motor}: [{range_mins[motor]}, {range_maxes[motor]}]")

    # Build calibration
    calibration = {}
    for motor, m in bus.motors.items():
        calibration[motor] = MotorCalibration(
            id=m.id,
            drive_mode=0,
            homing_offset=homing_offsets[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )

    # Write to motors
    print("\nWriting calibration to motor EEPROM...")
    bus.write_calibration(calibration)

    # Save to file
    calib_file = CALIBRATION_DIR / f"{arm_name}_so100.json"
    print(f"Saving calibration to {calib_file}...")

    # Convert to serializable format
    calib_dict = {}
    for motor, cal in calibration.items():
        calib_dict[motor] = {
            "id": cal.id,
            "drive_mode": cal.drive_mode,
            "homing_offset": cal.homing_offset,
            "range_min": cal.range_min,
            "range_max": cal.range_max,
        }

    with open(calib_file, "w") as f:
        json.dump(calib_dict, f, indent=2)

    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)

    # Verify by reading back
    print("\nVerifying - reading calibration from motors:")
    verify = bus.read_calibration()
    for motor, cal in verify.items():
        print(f"  {motor}: offset={cal.homing_offset}, range=[{cal.range_min}, {cal.range_max}]")

    return calibration


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["leader", "follower"]:
        print("Usage: python calibrate_arm.py <leader|follower>")
        sys.exit(1)

    arm_name = sys.argv[1]
    port = config[arm_name]["port"]

    print(f"Calibrating {arm_name} arm on port {port}")

    bus = create_bus(port)

    try:
        print("Connecting...")
        bus.connect()

        calibrate(bus, arm_name)

    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
    finally:
        bus.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
