#!/usr/bin/env python3
"""
Record a single demonstration episode for testing.

Usage:
    python record_single_demo.py --repo-id danbhf/test_demo --task "pick and place"
"""

import argparse
import time
import sys
import json
from pathlib import Path
from datetime import datetime
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
from so100_sts3250 import SO100FollowerSTS3250
from so100_leader_sts3250 import SO100LeaderSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Record a single demonstration")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Dataset repository ID (e.g., 'danbhf/test_demo'). If not provided, will auto-generate with timestamp."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="demonstration",
        help="Task description"
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

    args = parser.parse_args()

    # Add timestamp to root directory to make each run unique
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.root = f"{args.root}/{timestamp}"

    # Auto-generate repo_id with timestamp if not provided
    if not args.repo_id:
        task_slug = args.task.lower().replace(" ", "_")[:30]  # First 30 chars
        args.repo_id = f"danbhf/{task_slug}_{timestamp}"

    print("=" * 70)
    print("Record Single Demonstration")
    print("=" * 70)
    print(f"\nDataset: {args.repo_id}")
    print(f"Storage: {args.root}")
    print(f"Task: {args.task}")
    print(f"FPS: {args.fps}")
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
        print("✓ Leader connected")
    except Exception as e:
        print(f"✗ Failed: {e}")
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
        print("✓ Follower connected")
    except Exception as e:
        print(f"✗ Failed: {e}")
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
    print("✓ Cameras ready")

    # Create dataset
    print(f"\nCreating dataset: {args.repo_id}")
    try:
        # Get features from robot
        action_features = hw_to_dataset_features(follower.action_features, "action")
        obs_features = hw_to_dataset_features(follower.observation_features, "observation")
        features = {**action_features, **obs_features}

        # LeRobot will create the directories itself
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type="so100_follower",
            image_writer_threads=4,
            features=features,
        )
        print("✓ Dataset created")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        follower.disconnect()
        leader.disconnect()
        return 1

    print("\n" + "=" * 70)
    print("Ready to Record")
    print("=" * 70)
    print("\n1. Position the robot at the start of your demonstration")
    print("2. Press ENTER to start recording")
    print("3. Perform the task by moving the leader arm")
    print("4. Press Ctrl+C when done")
    print("5. Choose 's' to save or 'd' to discard")
    print()

    try:
        input("Press ENTER to start recording...")

        print("\n🔴 RECORDING - Perform the task now...")
        print("Press Ctrl+C when finished\n")

        # Create episode buffer
        dataset.create_episode_buffer()

        # Sync follower to leader
        action = leader.get_action()
        follower.send_action(action)

        step = 0
        start_time = time.time()

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
                    "task": args.task,
                })

                step += 1

                # Print status every second
                if step % args.fps == 0:
                    elapsed = time.time() - start_time
                    print(f"  Recording... {elapsed:.1f}s ({step} frames)")

            except (ConnectionError, TimeoutError) as e:
                print(f"\n⚠️  Warning: Error at step {step}: {e}")
                print("Skipping this frame and continuing...")
                time.sleep(0.1)
                continue

            # Maintain fps
            elapsed = time.time() - step_start
            sleep_time = (1.0 / args.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n\n⏸️  Recording stopped ({step} frames)")
        print("\nType 's' to SAVE or 'd' to DISCARD: ", end='')
        choice = input().strip().lower()

        if choice == 's':
            print("\nSaving episode...")
            dataset.save_episode()
            print(f"✓ Episode saved!")
            print(f"\nDataset location: {Path(args.root) / args.repo_id}")
            print(f"Frames recorded: {step}")
            print(f"Duration: {step / args.fps:.1f} seconds")
        else:
            print("✗ Episode discarded")
            follower.disconnect()
            leader.disconnect()
            return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        follower.disconnect()
        leader.disconnect()

    # Now upload to HuggingFace
    print("\n" + "=" * 70)
    print("Upload to HuggingFace Hub?")
    print("=" * 70)
    print(f"\nThis will upload your dataset to: https://huggingface.co/datasets/{args.repo_id}")
    print("You can then view it in the HuggingFace dataset viewer.")
    print("\nUpload? (y/n): ", end='')

    if input().strip().lower() == 'y':
        print("\nUploading to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            print(f"\n✓ Upload complete!")
            print(f"\nView your dataset at:")
            print(f"https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"\n✗ Upload failed: {e}")
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
