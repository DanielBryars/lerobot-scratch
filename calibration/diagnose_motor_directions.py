#!/usr/bin/env python
"""
Diagnose motor directions on leader and follower arms.

This script helps identify if motors are inverted between the two arms
by having you move each joint and comparing the direction of count changes.

Usage:
    python diagnose_motor_directions.py
"""
import time
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


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


def read_all_positions(bus):
    """Read raw positions for all motors."""
    positions = {}
    for motor in bus.motors:
        positions[motor] = bus.read("Present_Position", motor, normalize=False)
    return positions


def main():
    print("=" * 70)
    print("MOTOR DIRECTION DIAGNOSTIC")
    print("=" * 70)
    print("""
This tool compares motor directions between leader and follower arms.

For each joint, you'll:
1. See the current position on both arms
2. Move BOTH arms in the SAME physical direction
3. The script will show if counts went UP or DOWN

If a motor is inverted, counts will go opposite directions on the two arms.
""")

    # Connect to both arms
    print("Connecting to leader (COM8)...")
    leader = create_bus("COM8")
    leader.connect()
    leader.disable_torque()

    print("Connecting to follower (COM7)...")
    follower = create_bus("COM7")
    follower.connect()
    follower.disable_torque()

    print("\nBoth arms connected!\n")

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Movement instructions for each joint
    instructions = {
        "shoulder_pan": "Rotate the arm LEFT (counter-clockwise when viewed from above)",
        "shoulder_lift": "Lift the shoulder UP",
        "elbow_flex": "Bend the elbow (bring forearm UP)",
        "wrist_flex": "Bend the wrist UP",
        "wrist_roll": "Roll the wrist CLOCKWISE (when looking down the arm)",
        "gripper": "OPEN the gripper",
    }

    results = {}

    print("=" * 70)
    print("CURRENT POSITIONS (before any movement)")
    print("=" * 70)
    leader_pos = read_all_positions(leader)
    follower_pos = read_all_positions(follower)

    print(f"{'Joint':<15} {'Leader':>10} {'Follower':>10} {'Diff':>10}")
    print("-" * 50)
    for joint in joint_names:
        diff = follower_pos[joint] - leader_pos[joint]
        print(f"{joint:<15} {leader_pos[joint]:>10} {follower_pos[joint]:>10} {diff:>+10}")

    print("\n" + "=" * 70)
    print("DIRECTION TEST")
    print("=" * 70)

    for joint in joint_names:
        print(f"\n--- Testing: {joint} ---")
        print(f"Instruction: {instructions[joint]}")

        # Read before
        leader_before = leader.read("Present_Position", joint, normalize=False)
        follower_before = follower.read("Present_Position", joint, normalize=False)

        print(f"\nBefore - Leader: {leader_before}, Follower: {follower_before}")
        print(f"\nMove BOTH arms: {instructions[joint]}")
        print("Press Enter when done moving...")
        input()

        # Read after
        leader_after = leader.read("Present_Position", joint, normalize=False)
        follower_after = follower.read("Present_Position", joint, normalize=False)

        leader_delta = leader_after - leader_before
        follower_delta = follower_after - follower_before

        # Handle wraparound (if delta > 2048 or < -2048, it wrapped)
        if leader_delta > 2048:
            leader_delta -= 4096
        elif leader_delta < -2048:
            leader_delta += 4096
        if follower_delta > 2048:
            follower_delta -= 4096
        elif follower_delta < -2048:
            follower_delta += 4096

        leader_dir = "UP" if leader_delta > 0 else "DOWN" if leader_delta < 0 else "NO CHANGE"
        follower_dir = "UP" if follower_delta > 0 else "DOWN" if follower_delta < 0 else "NO CHANGE"

        print(f"\nAfter  - Leader: {leader_after}, Follower: {follower_after}")
        print(f"Delta  - Leader: {leader_delta:+d} ({leader_dir}), Follower: {follower_delta:+d} ({follower_dir})")

        # Check if same direction
        same_sign = (leader_delta > 0 and follower_delta > 0) or (leader_delta < 0 and follower_delta < 0) or (leader_delta == 0 and follower_delta == 0)

        if abs(leader_delta) < 50 or abs(follower_delta) < 50:
            status = "INSUFFICIENT MOVEMENT - move more!"
            results[joint] = "?"
        elif same_sign:
            status = "SAME DIRECTION - OK"
            results[joint] = "OK"
        else:
            status = "OPPOSITE DIRECTION - INVERTED!"
            results[joint] = "INVERTED"

        print(f"\nResult: {status}")
        print("-" * 50)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Joint':<15} {'Status':>15}")
    print("-" * 35)
    for joint, status in results.items():
        marker = "" if status == "OK" else " <-- FIX NEEDED" if status == "INVERTED" else " <-- RETEST"
        print(f"{joint:<15} {status:>15}{marker}")

    inverted_joints = [j for j, s in results.items() if s == "INVERTED"]
    if inverted_joints:
        print(f"\nInverted joints on follower: {inverted_joints}")
        print("These joints need drive_mode=1 in calibration.")
    else:
        print("\nNo inverted joints detected!")
        print("The issue might be wraparound (motor range crosses 0/4095 boundary).")

    leader.disconnect()
    follower.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
