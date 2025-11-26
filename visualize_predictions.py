#!/usr/bin/env python3
"""
Visualize model predictions vs ground truth as plots.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

import os
os.environ['PYTHONUTF8'] = '1'

import torch
from PIL import Image


class ActionTokenizer:
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
    parser.add_argument("--episode", type=int, default=45)
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    model_path = Path(args.model)

    print("Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(model_path))
    model = model.to(device)
    model.eval()

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load normalization stats
    with open(model_path / "action_norm_stats.json") as f:
        norm_stats = json.load(f)
    action_q01 = np.array(norm_stats['action_q01'])
    action_q99 = np.array(norm_stats['action_q99'])

    # Load dataset
    data_file = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_file).to_pandas()

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    tasks_table = pq.read_table(tasks_path)
    task_desc = str(tasks_table.to_pandas().index[0])

    episode_df = df[df['episode_index'] == args.episode]
    print(f"Episode {args.episode}: {len(episode_df)} frames")

    # Build video frame maps
    with open("config.json") as f:
        robot_config = json.load(f)
    camera_keys = [f"observation.images.{name}" for name in robot_config["cameras"].keys()]

    video_maps = {}
    for cam_key in camera_keys:
        video_maps[cam_key] = build_video_frame_map(dataset_root, cam_key)

    # Run inference
    print("Running inference...")
    prompt = f"In: What action should the robot take to {task_desc}?\nOut:"

    predicted_actions = []
    ground_truth_actions = []
    normalized_preds = []

    for i, (idx, row) in enumerate(episode_df.iterrows()):
        global_idx = int(row['index'])

        frames = []
        for cam_key in camera_keys:
            frame = get_frame_from_video(video_maps[cam_key], global_idx)
            if frame is not None:
                frames.append(frame)

        if not frames:
            continue

        combined_frame = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
        pil_image = Image.fromarray(combined_frame)

        inputs = processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

        generated_ids = output_ids[0, -6:].cpu().numpy()
        normalized_action = action_tokenizer.decode_token_ids_to_actions(generated_ids)
        denorm_action = (normalized_action + 1.0) / 2.0 * (action_q99 - action_q01) + action_q01

        predicted_actions.append(denorm_action)
        ground_truth_actions.append(np.array(row['action']))
        normalized_preds.append(normalized_action)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(episode_df)}")

    predicted_actions = np.array(predicted_actions)
    ground_truth_actions = np.array(ground_truth_actions)
    normalized_preds = np.array(normalized_preds)

    # Plot
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (ax, joint) in enumerate(zip(axes, joint_names)):
        frames = np.arange(len(predicted_actions))
        ax.plot(frames, ground_truth_actions[:, i], 'b-', label='Ground Truth', linewidth=2)
        ax.plot(frames, predicted_actions[:, i], 'r--', label='Predicted', linewidth=2)
        ax.set_title(f'{joint}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Degrees')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add error annotation
        mae = np.abs(predicted_actions[:, i] - ground_truth_actions[:, i]).mean()
        ax.annotate(f'MAE: {mae:.1f}°', xy=(0.02, 0.98), xycoords='axes fraction',
                   fontsize=10, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Episode {args.episode}: Predicted vs Ground Truth Actions\n'
                 f'Overall MAE: {np.abs(predicted_actions - ground_truth_actions).mean():.1f}°',
                 fontsize=12)
    plt.tight_layout()

    output_path = f"prediction_comparison_ep{args.episode}.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")

    # Also plot normalized predictions to check tokenization
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10))
    axes2 = axes2.flatten()

    for i, (ax, joint) in enumerate(zip(axes2, joint_names)):
        frames = np.arange(len(normalized_preds))
        ax.plot(frames, normalized_preds[:, i], 'g-', linewidth=2)
        ax.axhline(y=-1, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title(f'{joint} (normalized)')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Normalized [-1, 1]')
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Episode {args.episode}: Normalized Predictions (before denormalization)', fontsize=12)
    plt.tight_layout()

    output_path2 = f"normalized_predictions_ep{args.episode}.png"
    plt.savefig(output_path2, dpi=150)
    print(f"Saved normalized plot to: {output_path2}")

    # Open the plots
    import subprocess
    subprocess.Popen(["start", "", output_path], shell=True)
    subprocess.Popen(["start", "", output_path2], shell=True)

    print("\nDone!")


if __name__ == "__main__":
    main()
