#!/usr/bin/env python3
"""
Replay an episode from a LeRobot dataset on the actual robot.

This helps verify:
- Dataset actions are valid
- Actions would complete the task
- Problem is with model learning, not data quality

Usage:
    python replay_dataset_episode.py --dataset ./datasets/20251124_233735 --episode 0
"""

import argparse
import time
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq

# IMPORTANT: Import camera backend fix BEFORE any lerobot imports
import fix_camera_backend

from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Replay dataset episode on robot")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (1.0 = realtime)")
    parser.add_argument("--show-images", action="store_true", help="Save images during replay")

    args = parser.parse_args()

    dataset_root = Path(args.dataset)

    print("=" * 70)
    print("Replay Dataset Episode on Robot")
    print("=" * 70)
    print(f"\nDataset: {dataset_root}")
    print(f"Episode: {args.episode}")
    print(f"Speed: {args.speed}x")
    print()

    # Load dataset metadata
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)

    fps = info.get("fps", 30)
    print(f"Dataset FPS: {fps}")

    # Load episode data
    print("Loading episode data...")
    data_file = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_file).to_pandas()

    # Filter to specific episode
    episode_df = df[df['episode_index'] == args.episode]

    if len(episode_df) == 0:
        print(f"ERROR: Episode {args.episode} not found!")
        print(f"Available episodes: 0 to {df['episode_index'].max()}")
        return 1

    print(f"Episode {args.episode}: {len(episode_df)} frames ({len(episode_df)/fps:.1f}s)")

    # Extract actions and states
    actions = [np.array(a) for a in episode_df['action']]
    states = [np.array(s) for s in episode_df['observation.state']]

    print(f"\nFirst action: {actions[0]}")
    print(f"Last action: {actions[-1]}")
    print()

    # Load robot configuration
    config = load_config()

    # Camera configuration
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
        for name, cam in config["cameras"].items()
    }

    # Connect to follower
    follower_port = config["follower"]["port"]
    follower_cfg = SO100FollowerConfig(
        port=follower_port,
        id=config["follower"]["id"],
        cameras=camera_config
    )
    follower = SO100FollowerSTS3250(follower_cfg)

    print(f"Connecting to follower at {follower_port}...")
    try:
        follower.connect()
        print("[OK] Follower connected")
    except Exception as e:
        print(f"[FAILED] {e}")
        return 1

    print()
    print("=" * 70)
    print("Ready to Replay Episode")
    print("=" * 70)
    print("\nThis will send the recorded actions to the robot.")
    print("Make sure the robot is in a safe starting position!")
    print()
    input("Press ENTER to start replay...")

    # Replay episode
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\n[REPLAYING] Episode in progress...")
    print()

    frame_time = (1.0 / fps) / args.speed  # Adjust for speed multiplier

    try:
        for i, (action, state) in enumerate(zip(actions, states)):
            frame_start = time.time()

            # Build action dict for robot
            action_dict = {}
            for j, joint in enumerate(joint_names):
                action_dict[f"{joint}.pos"] = float(action[j])

            # Send to robot
            follower.send_action(action_dict)

            # Get current state
            current_obs = follower.get_observation()
            current_state = [current_obs.get(f"{joint}.pos", 0.0) for joint in joint_names]

            # Print progress every 30 frames (1 second at 30fps)
            if i % 30 == 0:
                print(f"  Frame {i}/{len(actions)} ({i/fps:.1f}s)")
                print(f"    Target: {[f'{a:.1f}' for a in action]}")
                print(f"    Actual: {[f'{s:.1f}' for s in current_state]}")

            # Maintain frame rate
            elapsed = time.time() - frame_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Replay stopped")

    finally:
        print("\nCleaning up...")
        try:
            if follower.is_connected:
                follower.disconnect()
        except Exception:
            pass

    print("\n[DONE] Episode replay complete!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
