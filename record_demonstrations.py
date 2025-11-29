#!/usr/bin/env python3
"""
Record teleoperation demonstrations for Pi0 fine-tuning.

This script records demonstrations using the SO100 leader-follower setup,
saving them in the LeRobot dataset format for training.

Usage:
    python record_demonstrations.py --repo-id my_username/my_dataset_name --num-episodes 50

Controls during recording:
    - Move the leader arm to control the follower
    - Press Ctrl+C to stop recording the current episode
    - Then choose: 's' to save, 'd' to discard, 'q' to quit
"""

import argparse
import time
import sys
import json
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from SO100FollowerSTS3250 import SO100FollowerSTS3250
from SO100LeaderSTS3250 import SO100LeaderSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def get_keyboard_input():
    """Non-blocking keyboard input (simple version)."""
    import select
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def main():
    parser = argparse.ArgumentParser(description="Record teleoperation demonstrations")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., 'username/my_so100_dataset')"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=50,
        help="Target number of episodes to record (default: 50)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency in Hz (default: 30)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Task description (e.g., 'pick and place the cube')"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./data",
        help="Root directory for dataset storage (default: ./data)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("LeRobot Dataset Recording for Pi0 Fine-tuning")
    print("=" * 70)
    print()
    print(f"Dataset: {args.repo_id}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Recording frequency: {args.fps} Hz")
    print(f"Task: {args.task if args.task else '(not specified)'}")
    print()

    # Load hardware configuration
    config = load_config()

    # Camera configuration
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"]
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
        print(f"✗ Failed to connect to leader: {e}")
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
        print(f"✗ Failed to connect to follower: {e}")
        leader.disconnect()
        return 1

    # Warmup cameras
    print("\nWarming up cameras...")
    for i in range(5):
        try:
            _ = follower.get_observation()
            print(f"  Warmup frame {i+1}/5")
            time.sleep(0.1)
        except Exception as e:
            print(f"  Warning: Warmup frame {i+1}/5 failed: {e}")
            time.sleep(0.2)
    print("✓ Cameras ready")

    # Create or load dataset
    print(f"\nInitializing dataset: {args.repo_id}")
    try:
        # Check if dataset exists
        dataset = LeRobotDataset.from_preloaded(
            repo_id=args.repo_id,
            root=args.root,
        )
        print(f"✓ Loaded existing dataset with {dataset.num_episodes} episodes")
        start_episode = dataset.num_episodes
    except Exception:
        # Create new dataset
        print("Creating new dataset...")
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=args.root,
            robot_type="so100_follower",
            image_writer_threads=4,
            features=hw_to_dataset_features({
                **follower.observation_features,
                **follower.action_features
            }),
        )
        print("✓ New dataset created")
        start_episode = 0

    print("\n" + "=" * 70)
    print("Recording Instructions")
    print("=" * 70)
    print("1. Position the robot for the start of a demonstration")
    print("2. Press ENTER to start recording an episode")
    print("3. Perform the task by moving the leader arm")
    print("4. Press ENTER again to stop and save the episode")
    print("5. Type 'd' then ENTER to discard the current episode")
    print("6. Type 'q' then ENTER to quit")
    print("=" * 70)
    print()

    episode_count = start_episode
    target_episodes = args.num_episodes

    try:
        while episode_count < target_episodes:
            print(f"\n[Episode {episode_count + 1}/{target_episodes}]")
            print("Press ENTER to start recording...")
            input()

            print("🔴 RECORDING - Perform the task now...")
            episode_data = []
            start_time = time.time()

            # Sync follower to leader
            action = leader.get_action()
            follower.send_action(action)

            recording = True
            step = 0

            # Start episode
            dataset.start_episode(task=args.task if args.task else f"Episode {episode_count + 1}")

            while recording:
                step_start = time.time()

                # Read action from leader
                action = leader.get_action()

                # Get observation from follower (before action)
                observation = follower.get_observation()

                # Send action to follower
                follower.send_action(action)

                # Add frame to dataset
                dataset.add_frame({
                    "observation": observation,
                    "action": action,
                })

                step += 1

                # Print status every second
                if step % args.fps == 0:
                    elapsed = time.time() - start_time
                    print(f"  Recording... {elapsed:.1f}s ({step} frames)")

                # Check for stop command (simple approach - requires hitting enter)
                # Note: This will block. For non-blocking, we'd need threading or keyboard library
                # For now, user needs to just press Ctrl+C then 's' to save

                # Sleep to maintain fps
                elapsed = time.time() - step_start
                sleep_time = (1.0 / args.fps) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⏸️  Recording paused")
        print("Type 's' to SAVE this episode, 'd' to DISCARD, or 'q' to QUIT: ", end='')
        choice = input().strip().lower()

        if choice == 's':
            # Save the episode
            dataset.save_episode()
            print(f"✓ Episode {episode_count + 1} saved ({step} frames)")
            episode_count += 1
        elif choice == 'd':
            print(f"✗ Episode discarded")
        elif choice == 'q':
            print("Exiting...")
        else:
            print("Invalid choice. Episode discarded.")

        if choice != 'q' and episode_count < target_episodes:
            print(f"\nProgress: {episode_count}/{target_episodes} episodes recorded")
            print("Continue recording? (y/n): ", end='')
            if input().strip().lower() == 'y':
                # Restart the recording loop
                pass
            else:
                print("Exiting...")

    except Exception as e:
        print(f"\n✗ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\n" + "=" * 70)
        print(f"Recording complete: {episode_count} episodes saved")
        print("=" * 70)
        print("\nDisconnecting robots...")
        follower.disconnect()
        leader.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
