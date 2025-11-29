#!/usr/bin/env python3
"""
Replay model predictions on a validation episode to debug learning.

This shows:
- What the model predicted for each frame
- How predictions compare to ground truth
- If predictions would actually work on the robot

Usage:
    python replay_model_predictions.py --dataset ./datasets/20251124_233735 --episode 5
"""

import argparse
import time
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq

# Fix Windows console encoding
import os
os.environ['PYTHONUTF8'] = '1'

import torch
from PIL import Image

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


def build_video_frame_map(dataset_root, image_key):
    """Build mapping from global index to video file."""
    video_dir = dataset_root / "videos" / image_key
    video_files = sorted(video_dir.glob("**/*.mp4"))

    video_map = []
    current_idx = 0

    for vf in video_files:
        cap = cv2.VideoCapture(str(vf))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_map.append({
                'start_idx': current_idx,
                'end_idx': current_idx + frame_count - 1,
                'path': vf,
                'frame_count': frame_count,
            })
            current_idx += frame_count
            cap.release()

    return video_map


def get_frame_from_video(video_map, global_idx):
    """Extract a specific frame from the video files."""
    # Find which video contains this frame
    for video_info in video_map:
        if video_info['start_idx'] <= global_idx <= video_info['end_idx']:
            frame_in_video = global_idx - video_info['start_idx']

            cap = cv2.VideoCapture(str(video_info['path']))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()
            cap.release()

            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None

    return None


def main():
    parser = argparse.ArgumentParser(description="Replay model predictions on validation episode")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model", type=str, default="./outputs/openvla_finetuned/best_checkpoint", help="Path to model")
    parser.add_argument("--episode", type=int, default=5, help="Episode index to test")
    parser.add_argument("--no-robot", action="store_true", help="Don't replay on robot, just show predictions")
    parser.add_argument("--speed", type=float, default=0.5, help="Playback speed (default 0.5x)")

    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    model_path = Path(args.model)

    print("=" * 70)
    print("Replay Model Predictions on Validation Episode")
    print("=" * 70)
    print(f"\nDataset: {dataset_root}")
    print(f"Model: {model_path}")
    print(f"Episode: {args.episode}")
    print()

    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("[OK] Model loaded")
    print()

    # Load normalization stats
    norm_stats_path = model_path / "action_norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)
    action_mins = np.array(norm_stats['action_mins'])
    action_maxs = np.array(norm_stats['action_maxs'])
    print(f"Action normalization:")
    print(f"  Min: {action_mins}")
    print(f"  Max: {action_maxs}")
    print()

    # Load dataset
    print("Loading episode data...")
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)
    fps = info.get("fps", 30)

    data_file = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_file).to_pandas()

    # Load task description
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    tasks_table = pq.read_table(tasks_path)
    tasks_df = tasks_table.to_pandas()
    task_desc = str(tasks_df.index[0])  # Get first task description
    print(f"Task: {task_desc}")
    print()

    # Filter to episode
    episode_df = df[df['episode_index'] == args.episode]
    if len(episode_df) == 0:
        print(f"ERROR: Episode {args.episode} not found!")
        return 1

    print(f"Episode {args.episode}: {len(episode_df)} frames ({len(episode_df)/fps:.1f}s)")
    print()

    # Build video frame maps
    print("Building video frame map...")
    robot_config = load_config()
    camera_keys = [f"observation.images.{name}" for name in robot_config["cameras"].keys()]

    video_maps = {}
    for cam_key in camera_keys:
        video_maps[cam_key] = build_video_frame_map(dataset_root, cam_key)

    print(f"[OK] Mapped {len(camera_keys)} cameras")
    print()

    # Run model inference on each frame
    print("Running model inference on episode...")
    predicted_actions = []
    ground_truth_actions = []

    for idx, row in episode_df.iterrows():
        global_idx = int(row['index'])

        # Get images from all cameras
        frames = []
        for cam_key in camera_keys:
            frame = get_frame_from_video(video_maps[cam_key], global_idx)
            if frame is not None:
                frames.append(frame)

        if not frames:
            print(f"Warning: No frames for index {global_idx}")
            continue

        # Combine horizontally
        if len(frames) > 1:
            combined_frame = np.concatenate(frames, axis=1)
        else:
            combined_frame = frames[0]

        # Convert to PIL
        pil_image = Image.fromarray(combined_frame)

        # Format prompt
        prompt = f"In: What action should the robot take to {task_desc}?\nOut:"

        # Process inputs
        inputs = processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Predict action
        with torch.no_grad():
            action = model.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=False,
            )

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action[0]

        # Denormalize
        normalized_action = np.clip(action[:6], -1.0, 1.0)
        denorm_action = (normalized_action + 1.0) / 2.0
        denorm_action = denorm_action * (action_maxs - action_mins) + action_mins

        predicted_actions.append(denorm_action)
        ground_truth_actions.append(np.array(row['action']))

        if len(predicted_actions) % 30 == 0:
            print(f"  Processed {len(predicted_actions)}/{len(episode_df)} frames...")

    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)

    print(f"[OK] Generated {len(predicted_actions)} predictions")
    print()

    # Compute statistics
    print("=" * 70)
    print("Prediction Statistics")
    print("=" * 70)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\nPredicted action statistics:")
    for i, joint in enumerate(joint_names):
        pred = predicted_actions[:, i]
        gt = ground_truth_actions[:, i]
        error = np.abs(pred - gt)

        print(f"\n{joint}:")
        print(f"  Predicted: min={pred.min():.1f}, max={pred.max():.1f}, mean={pred.mean():.1f}, std={pred.std():.1f}")
        print(f"  Ground truth: min={gt.min():.1f}, max={gt.max():.1f}, mean={gt.mean():.1f}, std={gt.std():.1f}")
        print(f"  Error: mean={error.mean():.1f}, max={error.max():.1f}")

        # Check if predictions are constant
        if pred.std() < 1.0:
            print(f"  ⚠️  WARNING: Predictions are nearly constant!")

    # Overall error
    mae = np.abs(predicted_actions - ground_truth_actions).mean()
    print(f"\n\nOverall Mean Absolute Error: {mae:.2f}°")

    # Check if model is predicting constant
    pred_variance = predicted_actions.var(axis=0).mean()
    print(f"Average prediction variance: {pred_variance:.2f}")

    if pred_variance < 10:
        print("⚠️  WARNING: Model is predicting nearly constant actions!")
        print("   This means it's ignoring visual input.")

    print()

    # Replay on robot
    if not args.no_robot:
        print("=" * 70)
        print("Replay Model Predictions on Robot")
        print("=" * 70)
        print("\nThis will replay the MODEL'S predictions on the robot.")
        print("(Not the ground truth - we want to see what the model learned)")
        print()
        input("Press ENTER to start replay...")

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

        print("\n[REPLAYING] Model predictions...")
        print()

        frame_time = (1.0 / fps) / args.speed

        try:
            for i, (pred_action, gt_action) in enumerate(zip(predicted_actions, ground_truth_actions)):
                frame_start = time.time()

                # Build action dict for robot
                action_dict = {}
                for j, joint in enumerate(joint_names):
                    action_dict[f"{joint}.pos"] = float(pred_action[j])

                # Send to robot
                follower.send_action(action_dict)

                # Print progress every 30 frames
                if i % 30 == 0:
                    print(f"  Frame {i}/{len(predicted_actions)} ({i/fps:.1f}s)")
                    print(f"    Predicted: {[f'{a:.1f}' for a in pred_action]}")
                    print(f"    Ground truth: {[f'{a:.1f}' for a in gt_action]}")

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

    print("\n[DONE] Analysis complete!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
