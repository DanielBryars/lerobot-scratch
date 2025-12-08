#!/usr/bin/env python
"""
Compare motor configurations between leader and follower arms.
"""
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


def read_motor_info(bus, motor_name):
    """Read various motor parameters."""
    info = {}

    # Read raw position
    info["Present_Position"] = bus.read("Present_Position", motor_name, normalize=False)

    # Read EEPROM calibration values
    info["Homing_Offset"] = bus.read("Homing_Offset", motor_name, normalize=False)
    info["Min_Position_Limit"] = bus.read("Min_Position_Limit", motor_name, normalize=False)
    info["Max_Position_Limit"] = bus.read("Max_Position_Limit", motor_name, normalize=False)

    # Try to read other useful params
    try:
        info["Mode"] = bus.read("Operating_Mode", motor_name, normalize=False)
    except:
        info["Mode"] = "N/A"

    return info


def main():
    print("=" * 80)
    print("MOTOR COMPARISON: Leader vs Follower")
    print("=" * 80)

    # Connect to both arms
    print("\nConnecting to leader (COM8)...")
    leader = create_bus("COM8")
    leader.connect()
    leader.disable_torque()

    print("Connecting to follower (COM7)...")
    follower = create_bus("COM7")
    follower.connect()
    follower.disable_torque()

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\n" + "=" * 80)
    print("MOTOR PARAMETERS COMPARISON")
    print("=" * 80)

    for joint in joint_names:
        print(f"\n--- {joint} ---")

        leader_info = read_motor_info(leader, joint)
        follower_info = read_motor_info(follower, joint)

        print(f"{'Parameter':<20} {'Leader':>12} {'Follower':>12} {'Diff':>12}")
        print("-" * 60)

        for param in leader_info:
            l_val = leader_info[param]
            f_val = follower_info[param]

            if isinstance(l_val, (int, float)) and isinstance(f_val, (int, float)):
                diff = f_val - l_val
                diff_str = f"{diff:+d}" if isinstance(diff, int) else f"{diff:+.1f}"

                # Flag large differences
                flag = ""
                if param == "Present_Position" and abs(diff) > 1000:
                    flag = " <-- LARGE DIFF!"

                print(f"{param:<20} {l_val:>12} {f_val:>12} {diff_str:>12}{flag}")
            else:
                print(f"{param:<20} {str(l_val):>12} {str(f_val):>12}")

    # Also ping to verify model numbers
    print("\n" + "=" * 80)
    print("MOTOR MODEL VERIFICATION (via ping)")
    print("=" * 80)
    print(f"\n{'Joint':<15} {'Leader Model':>15} {'Follower Model':>15} {'Match':>10}")
    print("-" * 60)

    for joint in joint_names:
        motor_id = leader.motors[joint].id

        leader_model = leader.ping(motor_id)
        follower_model = follower.ping(motor_id)

        match = "YES" if leader_model == follower_model else "NO!"
        print(f"{joint:<15} {leader_model:>15} {follower_model:>15} {match:>10}")

    leader.disconnect()
    follower.disconnect()
    print("\nDone!")


if __name__ == "__main__":
    main()
