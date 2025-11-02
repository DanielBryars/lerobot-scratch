#!/usr/bin/env python3
"""
SO100 Leader-Follower Teleoperation Script for STS3250 motors.

This script allows you to control a follower robot arm using a leader arm.
The follower will mirror the movements of the leader in real-time.

Usage:
    python teleoperate_so100.py

Controls:
    - Move the leader arm physically, and the follower will mirror the movements
    - Press Ctrl+C to exit
"""

import time
import sys
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from so100_sts3250 import SO100FollowerSTS3250
from so100_leader_sts3250 import SO100LeaderSTS3250


def main():
    print("=" * 70)
    print("SO100 STS3250 Leader-Follower Teleoperation")
    print("=" * 70)
    print()
    print("This will set up teleoperation where the follower mirrors the leader.")
    print("Move the leader arm physically, and the follower will follow.")
    print()

    # Leader arm configuration (serial ending 835)
    leader_port = "/dev/ttyACM0"  # Update this after connecting the leader
    leader_id = "leader_so100"

    # Follower arm configuration (serial 5AB0181764)
    follower_port = "/dev/ttyACM1"
    follower_id = "follower_so100"

    # Camera configuration for follower (optional for teleoperation, but useful for recording)
    camera_config = {
        "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=360, fps=30),  # Nuroum V11 (overhead)
        "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=30),  # USB2.0_CAM1 (wrist)
    }

    print(f"Leader arm port: {leader_port}")
    print(f"Follower arm port: {follower_port}")
    print()

    # Check if running interactively
    import sys
    if sys.stdin.isatty():
        response = input("Ready to connect? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Teleoperation cancelled.")
            return 1
    else:
        print("Running in non-interactive mode, proceeding automatically...")

    # Configure and connect leader
    print(f"\nConnecting to leader arm at {leader_port}...")
    leader_cfg = SO100LeaderConfig(port=leader_port, id=leader_id)
    leader = SO100LeaderSTS3250(leader_cfg)

    try:
        leader.connect()
        print("✓ Leader arm connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect to leader: {e}")
        return 1

    # Configure and connect follower
    print(f"\nConnecting to follower arm at {follower_port}...")
    follower_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    follower = SO100FollowerSTS3250(follower_cfg)

    try:
        follower.connect()
        print("✓ Follower arm connected successfully!")
    except Exception as e:
        print(f"✗ Failed to connect to follower: {e}")
        leader.disconnect()
        return 1

    print("\n" + "=" * 70)
    print("Teleoperation Active!")
    print("=" * 70)
    print("\nMove the leader arm, and the follower will mirror your movements.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        step = 0
        while True:
            # Read action from leader (current joint positions)
            action = leader.get_action()

            # Send action to follower
            follower.send_action(action)

            # Print status every 100 steps
            if step % 100 == 0:
                print(f"Step {step}: Teleoperation running...")

            step += 1

            # Small delay to avoid overwhelming the servos
            time.sleep(0.01)  # 100Hz update rate

    except KeyboardInterrupt:
        print("\n\nTeleoperation stopped by user.")
    except Exception as e:
        print(f"\n\n✗ Error during teleoperation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        print("\nDisconnecting robots...")
        follower.disconnect()
        leader.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
