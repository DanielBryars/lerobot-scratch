#!/usr/bin/env python
"""
Test script for SO100 simulation teleop with VR.

Uses the lerobot_robot_sim plugin directly without going through lerobot CLI.
This is useful for testing and debugging the sim + VR + leader arm integration.

Usage:
    python test_sim_teleop.py                  # Use leader arm with VR
    python test_sim_teleop.py --no-vr          # Use leader arm without VR
    python test_sim_teleop.py --test           # VR only, no arm
    python test_sim_teleop.py --port COM8      # Specify port
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the sim plugin (this registers so100_sim with lerobot)
import lerobot_robot_sim

from lerobot_robot_sim import (
    SO100Sim,
    SO100SimConfig,
    MOTOR_NAMES,
)


def load_config():
    """Load config.json for COM port settings."""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def load_calibration(arm_id: str = "leader_so100"):
    """Load calibration from JSON file (same as teleop_sim.py)."""
    import draccus
    from lerobot.motors import MotorCalibration
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

    calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / f"{arm_id}.json"

    if not calib_path.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path}\n"
            f"Run calibration first or check HF_LEROBOT_CALIBRATION path."
        )

    print(f"Loading calibration from: {calib_path}")
    with open(calib_path) as f, draccus.config_type("json"):
        return draccus.load(dict[str, MotorCalibration], f)


def create_leader_bus(port: str):
    """Create motor bus for leader arm with STS3250 motors."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
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
    return bus


def run_teleop(port: str, enable_vr: bool = True, fps: int = 30):
    """Run teleop with leader arm controlling simulation."""

    # Create simulated robot
    print("Creating SO100 simulation...")
    sim_config = SO100SimConfig(
        id="sim_test",
        sim_cameras=["wrist_cam"],
        camera_width=640,
        camera_height=480,
        enable_vr=enable_vr,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)

    print("Connecting simulation...")
    sim_robot.connect()
    print(f"Simulation connected! VR: {'enabled' if enable_vr else 'disabled'}")

    # Connect leader arm
    print(f"Connecting to leader arm on {port}...")
    bus = create_leader_bus(port)
    bus.connect()

    # Load calibration from JSON file (same as teleop_sim.py)
    bus.calibration = load_calibration("leader_so100")
    bus.disable_torque()
    print("Leader arm connected!")

    print("\n" + "="*60)
    print("Teleop Test Started!")
    print("Move the leader arm to control the simulation")
    if enable_vr:
        print("")
        print("VR Controller Controls:")
        print("  Left Thumbstick:  Forward/back (Y), Left/right (X)")
        print("  Right Thumbstick: Up/down (Y), Strafe left/right (X)")
        print("  A Button (right): Recenter robot in front of you")
    print("")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    frame_time = 1.0 / fps
    step_count = 0

    try:
        while True:
            loop_start = time.time()

            # Read leader arm positions
            positions = bus.sync_read("Present_Position")

            # Build action dict for sim robot
            action = {f"{motor}.pos": positions[motor] for motor in MOTOR_NAMES}

            # Send to simulation
            sim_robot.send_action(action)

            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                elapsed = time.time() - loop_start
                actual_fps = 1.0 / elapsed if elapsed > 0 else 0

                # Get observation to check sim state
                obs = sim_robot.get_observation()
                sim_pos = [obs[f"{m}.pos"] for m in MOTOR_NAMES]
                leader_pos = [positions[m] for m in MOTOR_NAMES]

                print(f"Step {step_count:5d} | FPS: {actual_fps:5.1f}")
                print(f"  Leader:  [{leader_pos[0]:6.1f}, {leader_pos[1]:6.1f}, {leader_pos[2]:6.1f}, {leader_pos[3]:6.1f}, {leader_pos[4]:6.1f}, {leader_pos[5]:5.1f}]")
                print(f"  Sim:     [{sim_pos[0]:6.1f}, {sim_pos[1]:6.1f}, {sim_pos[2]:6.1f}, {sim_pos[3]:6.1f}, {sim_pos[4]:6.1f}, {sim_pos[5]:5.1f}]")

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        sim_robot.disconnect()
        bus.disconnect()
        print("Done!")


def run_vr_test(fps: int = 30):
    """Run VR test mode - simulation only, no arm required."""

    print("Creating SO100 simulation with VR...")
    sim_config = SO100SimConfig(
        id="sim_test",
        sim_cameras=["wrist_cam"],
        camera_width=640,
        camera_height=480,
        enable_vr=True,
        n_sim_steps=10,
    )
    sim_robot = SO100Sim(sim_config)

    print("Connecting simulation...")
    sim_robot.connect()
    print("Simulation connected with VR!")

    print("\n" + "="*60)
    print("VR Test Mode (no arm required)")
    print("")
    print("VR Controller Controls:")
    print("  Left Thumbstick:  Forward/back (Y), Left/right (X)")
    print("  Right Thumbstick: Up/down (Y), Strafe left/right (X)")
    print("  A Button (right): Recenter robot in front of you")
    print("")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    frame_time = 1.0 / fps
    step_count = 0

    # Set a default pose (slightly bent arm)
    default_action = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 40.0,
        "wrist_flex.pos": -20.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 50.0,
    }

    try:
        while True:
            loop_start = time.time()

            # Send default action (just to trigger VR render)
            sim_robot.send_action(default_action)

            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                elapsed = time.time() - loop_start
                print(f"Step {step_count:5d} | FPS: {1/elapsed:.1f}")

            # Maintain frame rate
            elapsed = time.time() - loop_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        sim_robot.disconnect()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Test SO100 simulation teleop with VR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_sim_teleop.py                  # Full teleop with VR
  python test_sim_teleop.py --no-vr          # Teleop without VR
  python test_sim_teleop.py --test           # VR test, no arm needed
  python test_sim_teleop.py --port COM8      # Specify leader arm port
        """
    )
    parser.add_argument("--port", "-p", type=str, default=None,
                        help="Serial port for leader arm (default: from config.json)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target frame rate (default: 30)")
    parser.add_argument("--no-vr", action="store_true",
                        help="Disable VR output")
    parser.add_argument("--test", "-t", action="store_true",
                        help="Test mode: VR only, no arm required")

    args = parser.parse_args()

    # VR test mode
    if args.test:
        run_vr_test(args.fps)
        return

    # Get port from config if not specified
    port = args.port
    if port is None:
        config = load_config()
        if config and "leader" in config:
            port = config["leader"]["port"]
            print(f"Using leader port from config: {port}")
        else:
            port = "COM8"
            print(f"Using default leader port: {port}")

    run_teleop(port, enable_vr=not args.no_vr, fps=args.fps)


if __name__ == "__main__":
    main()
