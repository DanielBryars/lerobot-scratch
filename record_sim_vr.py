#!/usr/bin/env python
"""
Record demonstrations using the SO100 simulation with VR display.

Uses the leader arm for teleoperation and records to lerobot dataset format.
The simulation acts as a virtual follower robot.

Usage:
    python record_sim_vr.py --task "Pick up the cube" --num_episodes 10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add lerobot-gym to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot-gym"))

from so100_sim_follower import (
    SO100SimFollower,
    SO100SimFollowerConfig,
    MOTOR_NAMES,
    normalized_to_radians,
    radians_to_normalized,
)

# Import lerobot dataset utilities
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_branch
from lerobot.utils.hub import HfApi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Load config.json for leader arm port."""
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def create_leader_bus(port: str):
    """Create motor bus for leader arm."""
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


def record_episode(sim_robot, leader_bus, episode_idx: int, fps: int = 30):
    """Record a single episode."""
    print(f"\n--- Episode {episode_idx + 1} ---")
    print("Move leader arm to starting position, then press ENTER to start recording...")
    input()

    frames = []
    frame_time = 1.0 / fps

    print("Recording... Press ENTER to stop.")

    import threading
    stop_flag = threading.Event()

    def wait_for_enter():
        input()
        stop_flag.set()

    input_thread = threading.Thread(target=wait_for_enter, daemon=True)
    input_thread.start()

    step = 0
    while not stop_flag.is_set():
        loop_start = time.time()

        # Read leader arm position
        positions = leader_bus.sync_read("Present_Position")
        action = {f"{motor}.pos": positions[motor] for motor in MOTOR_NAMES}

        # Send to simulation
        sim_robot.send_action(action)

        # Get observation from simulation
        obs = sim_robot.get_observation()

        # Store frame data
        frame_data = {
            "observation": obs,
            "action": action,
            "timestamp": time.time(),
        }
        frames.append(frame_data)

        step += 1
        if step % 30 == 0:
            print(f"  Step {step}...")

        # Maintain frame rate
        elapsed = time.time() - loop_start
        if elapsed < frame_time:
            time.sleep(frame_time - elapsed)

    print(f"Episode recorded: {len(frames)} frames")

    # Ask if episode should be saved
    save = input("Save this episode? [Y/n]: ").strip().lower()
    if save == 'n':
        print("Episode discarded.")
        return None

    return frames


def save_to_lerobot_format(
    episodes: list,
    repo_id: str,
    task: str,
    fps: int,
    robot_type: str = "so100_sim_follower",
    push_to_hub: bool = False,
):
    """Save recorded episodes to lerobot dataset format."""
    print(f"\nSaving {len(episodes)} episodes to {repo_id}...")

    # Create dataset directory
    local_dir = Path("datasets") / repo_id.replace("/", "_")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Determine features from first episode
    first_obs = episodes[0][0]["observation"]
    camera_keys = [k for k in first_obs.keys() if not k.endswith(".pos")]
    motor_keys = [k for k in first_obs.keys() if k.endswith(".pos")]

    # Build info dict
    info = {
        "robot_type": robot_type,
        "fps": fps,
        "task": task,
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(motor_keys),),
                "names": motor_keys,
            },
            "action": {
                "dtype": "float32",
                "shape": (len(motor_keys),),
                "names": motor_keys,
            },
        },
    }

    # Add camera features
    for cam_key in camera_keys:
        sample_img = first_obs[cam_key]
        info["features"][f"observation.images.{cam_key}"] = {
            "dtype": "uint8",
            "shape": sample_img.shape,
        }

    # Save info
    with open(local_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # Save episodes as parquet and video files
    all_data = []
    for ep_idx, frames in enumerate(episodes):
        print(f"  Processing episode {ep_idx + 1}...")

        for frame_idx, frame in enumerate(frames):
            obs = frame["observation"]
            action = frame["action"]

            # Extract state and action vectors
            state = np.array([obs[k] for k in motor_keys], dtype=np.float32)
            action_vec = np.array([action[k] for k in motor_keys], dtype=np.float32)

            row = {
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "timestamp": frame["timestamp"],
                "observation.state": state.tolist(),
                "action": action_vec.tolist(),
            }

            # Add images
            for cam_key in camera_keys:
                row[f"observation.images.{cam_key}"] = obs[cam_key].tolist()

            all_data.append(row)

    # Save as simple JSON for now (can convert to parquet/video later)
    with open(local_dir / "data.json", "w") as f:
        json.dump(all_data, f)

    print(f"Dataset saved to {local_dir}")

    if push_to_hub:
        print("Push to hub not implemented yet - use lerobot tools to upload")

    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Record simulation demos with VR")
    parser.add_argument("--task", "-t", type=str, required=True,
                        help="Task description")
    parser.add_argument("--num_episodes", "-n", type=int, default=10,
                        help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS")
    parser.add_argument("--repo_id", type=str, default=None,
                        help="HuggingFace repo ID (default: auto-generated)")
    parser.add_argument("--leader_port", type=str, default=None,
                        help="Leader arm port (default: from config.json)")
    parser.add_argument("--enable_vr", action="store_true",
                        help="Enable VR display")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push to HuggingFace Hub")

    args = parser.parse_args()

    # Get leader port
    leader_port = args.leader_port
    if leader_port is None:
        config = load_config()
        if config and "leader" in config:
            leader_port = config["leader"]["port"]
        else:
            leader_port = "COM8"
    print(f"Leader port: {leader_port}")

    # Generate repo ID if not specified
    repo_id = args.repo_id
    if repo_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        repo_id = f"sim_recording_{timestamp}"
    print(f"Repo ID: {repo_id}")

    # Create simulated robot
    print("\nInitializing simulation...")
    sim_config = SO100SimFollowerConfig(
        id="sim_follower",
        cameras={"wrist_cam": None},  # Use MuJoCo camera
        enable_vr=args.enable_vr,
    )
    sim_robot = SO100SimFollower(sim_config)
    sim_robot.connect()

    # Connect leader arm
    print(f"Connecting to leader arm on {leader_port}...")
    leader_bus = create_leader_bus(leader_port)
    leader_bus.connect()

    # Load calibration
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION
    import draccus
    from lerobot.motors import MotorCalibration

    calib_path = HF_LEROBOT_CALIBRATION / "teleoperators" / "so100_leader_sts3250" / "leader_so100.json"
    if calib_path.exists():
        with open(calib_path) as f, draccus.config_type("json"):
            leader_bus.calibration = draccus.load(dict[str, MotorCalibration], f)
    leader_bus.disable_torque()
    print("Leader arm connected!")

    print(f"\n{'='*50}")
    print(f"Recording: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"FPS: {args.fps}")
    if args.enable_vr:
        print("VR: Enabled")
    print(f"{'='*50}\n")

    # Record episodes
    episodes = []
    try:
        for ep_idx in range(args.num_episodes):
            frames = record_episode(sim_robot, leader_bus, ep_idx, args.fps)
            if frames is not None:
                episodes.append(frames)

            if ep_idx < args.num_episodes - 1:
                cont = input(f"\nContinue to episode {ep_idx + 2}? [Y/n]: ").strip().lower()
                if cont == 'n':
                    break

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    finally:
        # Save if we have episodes
        if episodes:
            save_to_lerobot_format(
                episodes,
                repo_id,
                args.task,
                args.fps,
                push_to_hub=args.push_to_hub,
            )
        else:
            print("No episodes recorded.")

        # Cleanup
        sim_robot.disconnect()
        leader_bus.disconnect()
        print("\nDone!")


if __name__ == "__main__":
    main()
