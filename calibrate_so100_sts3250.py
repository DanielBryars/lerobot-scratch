#!/usr/bin/env python3
"""
Calibration script for SO100 with STS3250 motors.
This will guide you through the LeRobot calibration process to set min/max limits in the servos.
"""

import sys
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250


def main():
    print("=" * 70)
    print("SO100 STS3250 LeRobot Calibration")
    print("=" * 70)
    print()
    print("This will calibrate the min/max joint limits and store them in the servo firmware.")
    print()
    print("IMPORTANT: Make sure you have enough space around the robot!")
    print("The robot will move to its full range of motion during calibration.")
    print()

    response = input("Ready to proceed? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Calibration cancelled.")
        return 1

    # Configure robot
    follower_port = "/dev/ttyACM1"
    follower_id = "so100_sts3250"

    # Minimal camera config (calibration doesn't need cameras)
    camera_config = {}

    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)

    print(f"\nConnecting to robot at {follower_port}...")
    robot = SO100FollowerSTS3250(robot_cfg)

    try:
        robot.connect()
        print("✓ Robot connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return 1

    print("\n" + "=" * 70)
    print("Starting Calibration Process")
    print("=" * 70)

    try:
        # Call the calibrate method from the SO100Follower parent class
        print("\nStarting calibration...")
        print("Follow the on-screen instructions carefully.")
        print()

        robot.calibrate()

        print("\n" + "=" * 70)
        print("✓ Calibration completed successfully!")
        print("=" * 70)
        print("\nThe min/max limits have been written to the servo firmware.")
        print("Your robot is now ready to use with LeRobot and Pi0!")

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
