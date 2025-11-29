#!/usr/bin/env python3
"""
Record multiple training demonstrations with random object selection.

Usage:
    python record_training_set.py --runs 10 --repo-id danbhf/pick_place_dataset
"""

# IMPORTANT: Import camera backend fix BEFORE any lerobot imports
import fix_camera_backend  # This fixes Windows MSMF -> DSHOW issue

import argparse
import time
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from SO100FollowerSTS3250 import SO100FollowerSTS3250
from SO100LeaderSTS3250 import SO100LeaderSTS3250


# Define available objects
OBJECTS = [
    "left_orange_box",
    "right_orange_box",
    "blue_triangle",
    "red_cross"
]


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def choose_next_object(current_object):
    """Choose a random object that's different from the current one."""
    available = [obj for obj in OBJECTS if obj != current_object]
    return random.choice(available)


def record_episode(dataset, leader, follower, task_description, fps):
    """Record a single episode."""
    print(f"\n[REC] RECORDING - {task_description}")
    print("Perform the task now. Press Ctrl+C when finished.\n")

    # Create episode buffer
    dataset.create_episode_buffer()

    # Sync follower to leader
    action = leader.get_action()
    follower.send_action(action)

    step = 0
    start_time = time.time()

    try:
        while True:
            step_start = time.time()

            try:
                # Read action from leader
                action = leader.get_action()

                # Get observation from follower
                observation = follower.get_observation()

                # Send action to follower
                follower.send_action(action)

                # Build properly formatted frames
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                action_frame = build_dataset_frame(dataset.features, action, prefix="action")

                # Add frame to dataset with task
                dataset.add_frame({
                    **observation_frame,
                    **action_frame,
                    "task": task_description,
                })

                step += 1

                # Print status every second
                if step % fps == 0:
                    elapsed = time.time() - start_time
                    print(f"  Recording... {elapsed:.1f}s ({step} frames)")

            except (ConnectionError, TimeoutError) as e:
                print(f"\n[WARNING] Error at step {step}: {e}")
                print("Skipping this frame and continuing...")
                time.sleep(0.1)
                continue

            # Maintain fps
            elapsed = time.time() - step_start
            sleep_time = (1.0 / fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[STOPPED] Recording stopped ({step} frames)")
        print("\nType 's' to SAVE or 'd' to DISCARD: ", end='')
        choice = input().strip().lower()

        if choice == 's':
            print("Saving episode...")
            dataset.save_episode()
            print(f"[OK] Episode saved! ({step} frames, {step / fps:.1f}s)")
            return True
        else:
            print("[DISCARDED] Episode discarded")
            return False


def main():
    parser = argparse.ArgumentParser(description="Record multiple training demonstrations")
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of demonstrations to record (default: 10)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Dataset repository ID (e.g., 'danbhf/pick_place_training'). If not provided, will auto-generate."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency in Hz (default: 30)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./datasets",
        help="Root directory for dataset storage (default: ./datasets)"
    )
    parser.add_argument(
        "--start-object",
        type=str,
        choices=OBJECTS,
        default="left_orange_box",
        help="Starting object location (default: left_orange_box)"
    )

    args = parser.parse_args()

    # Add timestamp to root directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.root = f"{args.root}/{timestamp}"

    # Auto-generate repo_id if not provided
    if not args.repo_id:
        args.repo_id = f"danbhf/pick_place_training_{timestamp}"

    print("=" * 70)
    print("Record Training Set - Multiple Demonstrations")
    print("=" * 70)
    print(f"\nDataset: {args.repo_id}")
    print(f"Storage: {args.root}")
    print(f"Runs: {args.runs}")
    print(f"FPS: {args.fps}")
    print(f"Starting object: {args.start_object}")
    print(f"\nAvailable objects:")
    for obj in OBJECTS:
        print(f"  - {obj}")
    print()

    # Load hardware configuration
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

    # Connect to leader
    leader_port = config["leader"]["port"]
    leader_cfg = SO100LeaderConfig(port=leader_port, id=config["leader"]["id"])
    leader = SO100LeaderSTS3250(leader_cfg)

    print(f"Connecting to leader at {leader_port}...")
    try:
        leader.connect()
        print("[OK] Leader connected")
    except Exception as e:
        print(f"[FAILED] {e}")
        return 1

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
        leader.disconnect()
        return 1

    # Warmup cameras
    print("\nWarming up cameras...")
    for i in range(5):
        try:
            _ = follower.get_observation()
            time.sleep(0.1)
        except Exception as e:
            print(f"  Warning: {e}")
            time.sleep(0.2)
    print("[OK] Cameras ready")

    # Create dataset
    print(f"\nCreating dataset: {args.repo_id}")
    try:
        # Get features from robot
        action_features = hw_to_dataset_features(follower.action_features, "action")
        obs_features = hw_to_dataset_features(follower.observation_features, "observation")
        features = {**action_features, **obs_features}

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type="so100_follower",
            image_writer_threads=4,
            features=features,
        )
        print("[OK] Dataset created")
    except Exception as e:
        print(f"[FAILED] Failed to create dataset: {e}")
        follower.disconnect()
        leader.disconnect()
        return 1

    print("\n" + "=" * 70)
    print("Ready to Record Training Set")
    print("=" * 70)
    print("\nInstructions:")
    print("1. For each run, you'll be told which object to pick and place to")
    print("2. Position the robot at the starting object")
    print("3. Press ENTER to start recording")
    print("4. Perform the pick and place task")
    print("5. Press Ctrl+C when done")
    print("6. Choose 's' to save or 'd' to discard and retry")
    print()

    input("Press ENTER to begin the training session...")

    # Track progress
    current_object = args.start_object
    completed_runs = 0
    total_frames = 0

    try:
        for run_num in range(1, args.runs + 1):
            print("\n" + "=" * 70)
            print(f"Run {run_num}/{args.runs}")
            print("=" * 70)

            # Choose next destination
            destination_object = choose_next_object(current_object)
            task_description = f"Pick from {current_object} and place to {destination_object}"

            print(f"\nCurrent location: {current_object}")
            print(f"Destination: {destination_object}")
            print(f"Task: {task_description}")
            print()

            # Record until successful
            while True:
                print(f"Position robot at {current_object} and press ENTER to record...")
                input()

                success = record_episode(dataset, leader, follower, task_description, args.fps)

                if success:
                    completed_runs += 1
                    total_frames += dataset.num_frames
                    current_object = destination_object  # Update current location
                    print(f"\n[OK] Progress: {completed_runs}/{args.runs} runs completed")
                    break
                else:
                    print("\nRetrying this run...")
                    # Don't update current_object, try same task again

    except KeyboardInterrupt:
        print("\n\n[STOPPED] Training session interrupted by user")
        print(f"Completed {completed_runs}/{args.runs} runs")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Safely disconnect devices
        try:
            if follower.is_connected:
                follower.disconnect()
        except Exception:
            pass

        try:
            if leader.is_connected:
                leader.disconnect()
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 70)
    print("Training Session Summary")
    print("=" * 70)
    print(f"Completed runs: {completed_runs}/{args.runs}")
    print(f"Total frames: {total_frames}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Dataset location: {Path(args.root) / args.repo_id}")
    print()

    if completed_runs == 0:
        print("No episodes recorded. Exiting.")
        return 0

    # Upload to HuggingFace
    print("=" * 70)
    print("Upload to HuggingFace Hub?")
    print("=" * 70)
    print(f"\nThis will upload your dataset to: https://huggingface.co/datasets/{args.repo_id}")
    print("You can then view it in the HuggingFace dataset viewer.")
    print("\nUpload? (y/n): ", end='')

    if input().strip().lower() == 'y':
        print("\nUploading to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            print(f"\n[OK] Upload complete!")
            print(f"\nView your dataset at:")
            print(f"https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"\n[FAILED] Upload failed: {e}")
            print("\nYou can upload manually later with:")
            print(f"  python -m lerobot.scripts.push_dataset_to_hub --repo-id {args.repo_id} --local-dir {Path(args.root) / args.repo_id}")
            return 1
    else:
        print("\nSkipping upload. You can upload later with:")
        print(f"  python -m lerobot.scripts.push_dataset_to_hub --repo-id {args.repo_id} --local-dir {Path(args.root) / args.repo_id}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
