#!/usr/bin/env python
"""Check if motors are properly centered."""
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


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


print("=" * 70)
print("CENTERING CHECK")
print("=" * 70)

print("\nPut BOTH arms at ZERO POSE, then press Enter...")
input()

for name, port in [("Leader", "COM8"), ("Follower", "COM7")]:
    print(f"\n--- {name} ({port}) ---")
    bus = create_bus(port)
    bus.connect()
    bus.disable_torque()

    print(f"{'Motor':<15} {'Position':>10} {'Homing_Offset':>15} {'Dist from 2048':>15}")
    print("-" * 60)

    for motor in bus.motors:
        pos = bus.read("Present_Position", motor, normalize=False)
        offset = bus.read("Homing_Offset", motor, normalize=False)
        # Handle signed offset
        if offset > 2048:
            offset_signed = offset - 4096
        else:
            offset_signed = offset

        dist = pos - 2048
        flag = " <-- NOT CENTERED!" if abs(dist) > 200 else ""
        print(f"{motor:<15} {pos:>10} {offset_signed:>+15} {dist:>+15}{flag}")

    bus.disconnect()

print("\nIf any motor shows 'NOT CENTERED', run center_motors.py again for that arm.")
