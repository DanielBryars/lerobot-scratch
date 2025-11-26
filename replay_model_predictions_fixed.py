#!/usr/bin/env python3
"""
Replay model predictions on a validation episode to debug learning (FIXED VERSION).

This shows:
- What the model predicted for each frame
- How predictions compare to ground truth
- Whether predictions are varying or constant

Usage:
    python replay_model_predictions_fixed.py --episode 45
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq

import os
os.environ['PYTHONUTF8'] = '1'

import torch
from PIL import Image


class ActionTokenizer:
    """Decode action tokens back to continuous actions."""
    def __init__(self, tokenizer, bins: int = 256, min_action: int = -1, max_action: int = 1):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]


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
            })
            current_idx += frame_count
            cap.release()

    return video_map


def get_frame_from_video(video_map, global_idx):
    """Extract a specific frame from the video files."""
    for video_info in video_map:
        if video_info['start_idx'] <= global_idx <= video_info['end_idx']:
            frame_in_video = global_idx - video_info['start_idx']
            cap = cv2.VideoCapture(str(video_info['path']))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./datasets/20251124_233735_5hz")
    parser.add_argument("--model", type=str, default="./outputs/openvla_fixed/best_checkpoint")
    parser.add_argument("--episode", type=int, default=45, help="Episode to test (use validation episode)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    model_path = Path(args.model)

    print("=" * 70)
    print("Replay Model Predictions (FIXED VERSION)")
    print("=" * 70)
    print(f"\nDataset: {dataset_root}")
    print(f"Model: {model_path}")
    print(f"Episode: {args.episode}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # Load model with LoRA
    print("Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)

    # Load base model then LoRA weights
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.to(device)
    model.eval()
    print("[OK] Model loaded")
    print()

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load normalization stats (q01/q99)
    norm_stats_path = model_path / "action_norm_stats.json"
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    action_q01 = np.array(norm_stats['action_q01'])
    action_q99 = np.array(norm_stats['action_q99'])
    print(f"Action normalization (q01/q99):")
    print(f"  q01: {action_q01}")
    print(f"  q99: {action_q99}")
    print()

    # Load dataset
    print("Loading episode data...")
    data_file = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_file).to_pandas()

    # Load task description
    tasks_path = dataset_root / "meta" / "tasks.parquet"
    tasks_table = pq.read_table(tasks_path)
    task_desc = str(tasks_table.to_pandas().index[0])
    print(f"Task: {task_desc}")

    # Filter to episode
    episode_df = df[df['episode_index'] == args.episode]
    if len(episode_df) == 0:
        print(f"ERROR: Episode {args.episode} not found!")
        print(f"Available episodes: {df['episode_index'].unique()}")
        return 1

    print(f"Episode {args.episode}: {len(episode_df)} frames")
    print()

    # Build video frame maps
    print("Building video frame map...")
    with open("config.json") as f:
        robot_config = json.load(f)
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

    prompt = f"In: What action should the robot take to {task_desc}?\nOut:"

    for i, (idx, row) in enumerate(episode_df.iterrows()):
        global_idx = int(row['index'])

        # Get images from all cameras
        frames = []
        for cam_key in camera_keys:
            frame = get_frame_from_video(video_maps[cam_key], global_idx)
            if frame is not None:
                frames.append(frame)

        if not frames:
            continue

        # Combine horizontally
        combined_frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
        pil_image = Image.fromarray(combined_frame)

        # Process inputs
        inputs = processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        # Generate action tokens
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )

        # Extract 6 action tokens BEFORE the EOS token
        # Model outputs: [action1, ..., action6, EOS] so we need -7:-1
        generated_ids = output_ids[0, -7:-1].cpu().numpy()

        # Decode to continuous normalized actions [-1, 1]
        normalized_action = action_tokenizer.decode_token_ids_to_actions(generated_ids)

        # Denormalize using q01/q99
        denorm_action = (normalized_action + 1.0) / 2.0 * (action_q99 - action_q01) + action_q01

        predicted_actions.append(denorm_action)
        ground_truth_actions.append(np.array(row['action']))

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(episode_df)} frames...")

    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)

    print(f"[OK] Generated {len(predicted_actions)} predictions")
    print()

    # Compute statistics
    print("=" * 70)
    print("Prediction Statistics")
    print("=" * 70)

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\nPer-joint analysis:")
    for i, joint in enumerate(joint_names):
        pred = predicted_actions[:, i]
        gt = ground_truth_actions[:, i]
        error = np.abs(pred - gt)

        print(f"\n{joint}:")
        print(f"  Predicted: min={pred.min():.1f}, max={pred.max():.1f}, mean={pred.mean():.1f}, std={pred.std():.1f}")
        print(f"  Ground truth: min={gt.min():.1f}, max={gt.max():.1f}, mean={gt.mean():.1f}, std={gt.std():.1f}")
        print(f"  Error: mean={error.mean():.1f}, max={error.max():.1f}")

        if pred.std() < 1.0:
            print(f"  WARNING: Predictions are nearly constant!")

    # Overall metrics
    mae = np.abs(predicted_actions - ground_truth_actions).mean()
    print(f"\n\nOverall Mean Absolute Error: {mae:.2f} deg")

    pred_variance = predicted_actions.var(axis=0).mean()
    print(f"Average prediction variance: {pred_variance:.2f}")

    if pred_variance < 10:
        print("\nWARNING: Model is predicting nearly constant actions!")
        print("This means it's ignoring visual input.")
    else:
        print("\nPredictions are varying - model appears to be learning!")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
