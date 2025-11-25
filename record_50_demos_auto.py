#!/usr/bin/env python3
"""
Record 50 demonstrations of the same task with automatic position-based triggers.

The robot arm position acts as a trigger:
- Move arm to "ready" position (wrist straight up) to START recording
- Move arm back to "ready" position to STOP recording
- No keyboard input needed during recording!

Usage:
    python record_50_demos_auto.py --task "move the block from the left to the right"
"""

# IMPORTANT: Import camera backend fix BEFORE any lerobot imports
import fix_camera_backend

import argparse
import time
import sys
import json
import cv2
import numpy as np
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


def is_ready_position(observation, threshold_deg=-50.0):
    """
    Check if robot is in 'ready' position (arm straight up).

    Ready position: elbow_flex < -50 degrees (bent backward)
    This detects when the arm is raised up/back, which is easy to reach before/after each demo.

    Typical values:
    - Ready (straight up): elbow_flex ~ -65°
    - Working (task): elbow_flex ~ +16°
    """
    elbow_flex = observation.get("elbow_flex.pos", 0.0)
    return elbow_flex < threshold_deg


def print_state_change(state, demo_num, total_demos):
    """Print status message when state changes."""
    if state == "WAITING":
        print(f"\n[WAITING] Demo {demo_num}/{total_demos}")
        print("  Position block, then move arm STRAIGHT UP/BACK to start")
    elif state == "READY":
        print("[READY] Arm detected in ready position - hold steady...")
    elif state == "RECORDING":
        print("[RECORDING] Started! Perform the task now...")
    elif state == "STOPPED":
        print("[STOPPED] Recording complete!")


def record_episode_auto(dataset, leader, follower, cameras_dict, demo_num, total_demos, fps, ready_threshold=-50.0):
    """
    Record a single episode with automatic position-based triggers.

    State machine:
    WAITING -> (arm up) -> READY -> (wait 0.5s) -> RECORDING -> (arm up) -> STOPPED
    """
    print(f"\n{'='*70}")
    print(f"Demo {demo_num}/{total_demos}")
    print(f"{'='*70}")
    print("\n1. Adjust block position")
    print("2. Move robot arm STRAIGHT UP/BACK (elbow < -50°) to start")
    print("3. Perform the task")
    print("4. Move robot arm STRAIGHT UP/BACK again to stop")
    print("\nPress Ctrl+C anytime to quit\n")

    state = "WAITING"
    prev_state = None
    ready_start_time = None
    ready_hold_duration = 0.5  # Hold ready pose for 0.5s to confirm

    step = 0
    recording_start_time = None

    # Create episode buffer (but don't start recording yet)
    dataset.create_episode_buffer()

    try:
        while True:
            step_start = time.time()

            # Get action from leader
            action = leader.get_action()

            # Get observation from follower
            observation = follower.get_observation()

            # Send action to follower
            follower.send_action(action)

            # Check if in ready position
            in_ready_pos = is_ready_position(observation, ready_threshold)

            # State machine
            if state == "WAITING":
                if state != prev_state:
                    print_state_change(state, demo_num, total_demos)
                    prev_state = state

                if in_ready_pos:
                    state = "READY"
                    ready_start_time = time.time()

            elif state == "READY":
                if state != prev_state:
                    print_state_change(state, demo_num, total_demos)
                    prev_state = state

                if not in_ready_pos:
                    # Left ready position too early
                    state = "WAITING"
                    ready_start_time = None
                    print("[INFO] Arm moved - move back to ready position to start")
                elif time.time() - ready_start_time > ready_hold_duration:
                    # Held ready position long enough, start recording!
                    state = "RECORDING"
                    recording_start_time = time.time()
                    step = 0

            elif state == "RECORDING":
                if state != prev_state:
                    print_state_change(state, demo_num, total_demos)
                    prev_state = state

                # Build properly formatted frames
                observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")
                action_frame = build_dataset_frame(dataset.features, action, prefix="action")

                # Add frame to dataset
                dataset.add_frame({
                    **observation_frame,
                    **action_frame,
                    "task": dataset.current_task,
                })

                step += 1

                # Print status every second
                if step % fps == 0:
                    elapsed = time.time() - recording_start_time
                    elbow_angle = observation.get("elbow_flex.pos", 0.0)
                    print(f"  Recording... {elapsed:.1f}s ({step} frames) | elbow: {elbow_angle:.1f}°")

                # Check if back in ready position (to stop)
                if in_ready_pos and step > fps * 2:  # Must record at least 2 seconds
                    state = "STOPPED"
                    print_state_change(state, demo_num, total_demos)
                    print(f"  Total: {step} frames ({step/fps:.1f}s)")
                    break

            # Maintain fps
            elapsed = time.time() - step_start
            sleep_time = (1.0 / fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Recording interrupted")
        return None

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None

    # Ask to save or discard
    if step > 0:
        print("\nSave this demo? (Y/n): ", end='')
        choice = input().strip().lower()

        if choice == '' or choice == 'y':
            print("Saving episode...")
            dataset.save_episode()
            print(f"[OK] Demo {demo_num} saved!")
            return True
        else:
            print("[DISCARDED] Episode discarded")
            return False
    else:
        print("[SKIPPED] No frames recorded")
        return False


def main():
    parser = argparse.ArgumentParser(description="Record 50 demos with auto position triggers")
    parser.add_argument(
        "--task",
        type=str,
        default="move the block from the left to the right",
        help="Task description"
    )
    parser.add_argument(
        "--num-demos",
        type=int,
        default=50,
        help="Number of demonstrations to record (default: 50)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frequency in Hz (default: 30)"
    )
    parser.add_argument(
        "--ready-threshold",
        type=float,
        default=-50.0,
        help="Elbow flex angle (degrees) for ready position (default: -50, arm up/back)"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=False,
        help="Dataset repository ID. If not provided, will auto-generate."
    )

    args = parser.parse_args()

    # Auto-generate repo_id with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.repo_id:
        task_slug = args.task.lower().replace(" ", "_")[:30]
        args.repo_id = f"danbhf/{task_slug}_{timestamp}"

    dataset_root = f"./datasets/{timestamp}"

    print("=" * 70)
    print("Record 50 Demos with Auto Position Triggers")
    print("=" * 70)
    print(f"\nDataset: {args.repo_id}")
    print(f"Storage: {dataset_root}")
    print(f"Task: {args.task}")
    print(f"Number of demos: {args.num_demos}")
    print(f"FPS: {args.fps}")
    print(f"Ready position threshold: elbow_flex < {args.ready_threshold}°")
    print()

    # Load hardware configuration
    config = load_config()

    # Camera configuration for LeRobot
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"],
        )
        for name, cam in config["cameras"].items()
    }

    # Also open cameras directly for preview (OpenCV VideoCapture)
    cameras_dict = {}
    print("Opening cameras for preview...")
    for cam_name, cam_cfg in config["cameras"].items():
        cap = cv2.VideoCapture(cam_cfg["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
        if cap.isOpened():
            cameras_dict[cam_name] = cap
            print(f"  [OK] {cam_name}")
        else:
            print(f"  [WARNING] Failed to open {cam_name}")

    if not cameras_dict:
        print("\n[ERROR] No cameras available")
        return 1
    print()

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
        for cap in cameras_dict.values():
            cap.release()
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
        for cap in cameras_dict.values():
            cap.release()
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
        action_features = hw_to_dataset_features(follower.action_features, "action")
        obs_features = hw_to_dataset_features(follower.observation_features, "observation")
        features = {**action_features, **obs_features}

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            root=dataset_root,
            robot_type="so100_follower",
            image_writer_threads=4,
            features=features,
        )
        # Store task description for later use
        dataset.current_task = args.task
        print("[OK] Dataset created")
    except Exception as e:
        print(f"[FAILED] Failed to create dataset: {e}")
        follower.disconnect()
        leader.disconnect()
        for cap in cameras_dict.values():
            cap.release()
        return 1

    # Camera validation preview
    print("\n" + "=" * 70)
    print("Camera Validation")
    print("=" * 70)
    print("\nCapturing test images from cameras...")

    # Capture frames from all cameras
    frames = []
    for cam_name, cap in cameras_dict.items():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            # Save individual camera image
            test_img_path = f"camera_test_{cam_name}.jpg"
            cv2.imwrite(test_img_path, frame)
            print(f"  [OK] {cam_name}: {frame.shape}")
        else:
            print(f"  [ERROR] Failed to read from {cam_name}")

    if not frames:
        print("\n[ERROR] No camera frames captured!")
        follower.disconnect()
        leader.disconnect()
        for cap in cameras_dict.values():
            cap.release()
        return 1

    # Combine frames horizontally
    if len(frames) > 1:
        combined = np.concatenate(frames, axis=1)
    else:
        combined = frames[0]

    # Save combined preview
    preview_path = "camera_preview_combined.jpg"
    cv2.imwrite(preview_path, combined)
    print(f"\n[OK] Saved combined preview to: {preview_path}")
    print(f"     Resolution: {combined.shape[1]}x{combined.shape[0]}")

    # Open the preview with default Windows viewer
    import subprocess
    try:
        subprocess.Popen(['start', preview_path], shell=True)
        print(f"\n[INFO] Opening preview in default image viewer...")
    except Exception as e:
        print(f"[WARNING] Could not auto-open preview: {e}")
        print(f"         Please manually open: {preview_path}")

    print("\nPlease check the camera preview image:")
    print("- Both cameras are working")
    print("- Camera angles are correct")
    print("- Lighting is adequate")
    print("- Objects are visible")
    print()
    print("Continue? (y/n): ", end='')

    choice = input().strip().lower()
    if choice != 'y':
        print("\n[QUIT] Camera validation cancelled")
        follower.disconnect()
        leader.disconnect()
        for cap in cameras_dict.values():
            cap.release()
        return 0

    print("[OK] Cameras validated!")

    print("\n" + "=" * 70)
    print("Ready to Record!")
    print("=" * 70)
    print("\nHow it works:")
    print("1. Position the block for each demo")
    print("2. Move robot arm STRAIGHT UP/BACK (elbow < -50°) and hold for 0.5s")
    print("3. Recording starts automatically")
    print("4. Perform the task")
    print("5. Move robot arm STRAIGHT UP/BACK again to stop")
    print("6. Confirm save (y) or discard (n)")
    print("7. Repeat for all demos!")
    print()

    input("Press ENTER to begin recording session...")

    # Recording loop
    completed_demos = 0
    demo_num = 1

    try:
        while completed_demos < args.num_demos:
            result = record_episode_auto(
                dataset,
                leader,
                follower,
                cameras_dict,
                demo_num,
                args.num_demos,
                args.fps,
                args.ready_threshold
            )

            if result is None:
                # User quit
                print("\n[QUIT] Recording session ended by user")
                break
            elif result:
                # Successfully saved
                completed_demos += 1
                demo_num += 1
                print(f"\n[PROGRESS] {completed_demos}/{args.num_demos} demos completed")
            else:
                # Discarded, retry same demo number
                print(f"\n[RETRY] Retrying demo {demo_num}...")

            # Brief pause between demos
            if completed_demos < args.num_demos:
                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Recording session interrupted")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up...")

        for cap in cameras_dict.values():
            cap.release()

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
    print("Recording Session Summary")
    print("=" * 70)
    print(f"Completed demos: {completed_demos}/{args.num_demos}")
    print(f"Total episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"Dataset location: {Path(dataset_root) / args.repo_id}")
    print()

    if completed_demos == 0:
        print("No episodes recorded. Exiting.")
        return 0

    # Optional upload
    print("=" * 70)
    print("Upload to HuggingFace Hub?")
    print("=" * 70)
    print(f"\nThis will upload to: https://huggingface.co/datasets/{args.repo_id}")
    print("\nUpload? (y/n): ", end='')

    if input().strip().lower() == 'y':
        print("\nUploading to HuggingFace Hub...")
        try:
            dataset.push_to_hub()
            print(f"\n[OK] Upload complete!")
            print(f"View at: https://huggingface.co/datasets/{args.repo_id}")
        except Exception as e:
            print(f"\n[FAILED] Upload failed: {e}")
            print(f"\nUpload manually with:")
            print(f"  python -m lerobot.scripts.push_dataset_to_hub --repo-id {args.repo_id} --local-dir {Path(dataset_root) / args.repo_id}")
    else:
        print("\nSkipping upload. You can upload later with:")
        print(f"  python -m lerobot.scripts.push_dataset_to_hub --repo-id {args.repo_id} --local-dir {Path(dataset_root) / args.repo_id}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
