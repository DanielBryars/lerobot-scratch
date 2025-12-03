"""Restore follower EEPROM calibration from saved values."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

# Calibration values from calibration_20251130_195943.json (follower "after")
FOLLOWER_CALIBRATION = {
    "shoulder_pan": 1744,
    "shoulder_lift": 770,
    "elbow_flex": 2047,
    "wrist_flex": -601,
    "wrist_roll": -4,
    "gripper": -680,
}

FOLLOWER_PORT = "COM7"

def main():
    print("=" * 60)
    print("RESTORE FOLLOWER CALIBRATION")
    print("=" * 60)
    print("\nValues to restore:")
    for motor, offset in FOLLOWER_CALIBRATION.items():
        print(f"  {motor}: {offset}")

    input("\nPress ENTER to connect and restore...")

    bus = FeetechMotorsBus(
        port=FOLLOWER_PORT,
        motors={
            "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
        },
    )

    bus.connect()
    print("\nConnected! Writing calibration to EEPROM...")

    for motor, offset in FOLLOWER_CALIBRATION.items():
        bus.write("Homing_Offset", motor, offset, normalize=False)
        print(f"  {motor}: wrote {offset}")

    print("\nVerifying...")
    cal = bus.read_calibration()
    for motor, c in cal.items():
        expected = FOLLOWER_CALIBRATION[motor]
        actual = c.homing_offset
        status = "OK" if actual == expected else "MISMATCH!"
        print(f"  {motor}: {actual} (expected {expected}) {status}")

    bus.disconnect()
    print("\nDone! Follower calibration restored.")

if __name__ == "__main__":
    main()
