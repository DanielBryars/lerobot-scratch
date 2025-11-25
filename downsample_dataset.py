#!/usr/bin/env python3
"""
Downsample a LeRobot dataset from 30Hz to 5Hz for OpenVLA training.

OpenVLA performs best with 5-10Hz control frequency. High-frequency data
(30Hz) has too much temporal redundancy and confuses the model.

Usage:
    python downsample_dataset.py --input ./datasets/20251124_233735 --output ./datasets/20251124_233735_5hz
"""

import argparse
import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import cv2
from tqdm import tqdm


def downsample_videos(input_dir: Path, output_dir: Path, skip_factor: int):
    """Downsample video files by keeping every Nth frame."""
    video_files = list(input_dir.glob("**/*.mp4"))

    for video_path in tqdm(video_files, desc=f"  Downsampling videos in {input_dir.name}"):
        # Get relative path
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open input video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            continue

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        new_fps = fps / skip_factor
        out = cv2.VideoWriter(str(output_path), fourcc, new_fps, (width, height))

        frame_idx = 0
        frames_written = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Keep every Nth frame
            if frame_idx % skip_factor == 0:
                out.write(frame)
                frames_written += 1

            frame_idx += 1

        cap.release()
        out.release()


def detect_idle_frames(actions, threshold_deg=2.0):
    """
    Detect idle frames where robot is not moving significantly.

    Returns a boolean array where True = idle, False = moving
    """
    # Compute action velocity (difference between consecutive actions)
    action_diffs = np.diff(actions, axis=0)
    action_speeds = np.abs(action_diffs).max(axis=1)  # Max change across all joints

    # Frames where max joint change < threshold are considered idle
    idle_mask = np.concatenate([[False], action_speeds < threshold_deg])  # First frame always kept

    return idle_mask


def trim_idle_edges(episode_df, idle_threshold_deg=2.0):
    """
    Trim idle frames from start and end of an episode.

    OpenVLA guidance: "Eliminate pauses and minimal movements from demonstrations.
    The model may 'get stuck' during inference when exposed to idle actions."
    """
    actions = np.array([np.array(a) for a in episode_df['action']])
    idle_mask = detect_idle_frames(actions, idle_threshold_deg)

    # Find first and last non-idle frames
    non_idle_indices = np.where(~idle_mask)[0]

    if len(non_idle_indices) == 0:
        # Entire episode is idle - keep it anyway (better than empty)
        return episode_df, 0

    start_idx = non_idle_indices[0]
    end_idx = non_idle_indices[-1] + 1

    # Trim the episode
    trimmed_df = episode_df.iloc[start_idx:end_idx].copy()

    # Reset frame indices
    trimmed_df['frame_index'] = range(len(trimmed_df))

    frames_removed = len(episode_df) - len(trimmed_df)

    return trimmed_df, frames_removed


def downsample_dataset(input_path: str, output_path: str, target_fps: int = 5, trim_idle: bool = True, idle_threshold: float = 2.0):
    """Downsample entire LeRobot dataset to target FPS."""
    input_root = Path(input_path)
    output_root = Path(output_path)

    print("=" * 70)
    print("Downsample LeRobot Dataset for OpenVLA")
    print("=" * 70)
    print(f"\nInput:  {input_root}")
    print(f"Output: {output_root}")
    print()

    # Load metadata
    with open(input_root / "meta" / "info.json") as f:
        info = json.load(f)

    original_fps = info.get("fps", 30)
    skip_factor = int(original_fps / target_fps)

    print(f"Original FPS: {original_fps}")
    print(f"Target FPS:   {target_fps}")
    print(f"Skip factor:  {skip_factor} (keep 1 out of every {skip_factor} frames)")
    print(f"Trim idle:    {trim_idle} (threshold: {idle_threshold}° max joint change)")
    print()

    if skip_factor <= 1:
        print("ERROR: Target FPS must be lower than original FPS")
        return 1

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy and update metadata
    print("Copying metadata...")
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Update info.json with new FPS
    info['fps'] = target_fps
    info['original_fps'] = original_fps
    info['downsampled_from'] = str(input_root)
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    # Copy tasks.parquet
    shutil.copy(input_root / "meta" / "tasks.parquet", meta_dir / "tasks.parquet")
    print("[OK] Metadata copied")
    print()

    # Downsample parquet data
    print("Downsampling parquet data...")
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    input_parquet = input_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(input_parquet).to_pandas()

    print(f"  Original frames: {len(df)}")

    # Downsample and trim: keep every Nth frame per episode, optionally trim idle edges
    downsampled_rows = []
    total_trimmed = 0

    for episode_idx in df['episode_index'].unique():
        episode_df = df[df['episode_index'] == episode_idx]

        # Trim idle frames from start/end if requested
        if trim_idle:
            episode_df, frames_removed = trim_idle_edges(episode_df, idle_threshold)
            total_trimmed += frames_removed

        # Keep every Nth frame
        selected_indices = list(range(0, len(episode_df), skip_factor))
        downsampled_episode = episode_df.iloc[selected_indices].copy()

        # Reset frame indices within episode
        downsampled_episode['frame_index'] = range(len(downsampled_episode))

        downsampled_rows.append(downsampled_episode)

    downsampled_df = pd.concat(downsampled_rows, ignore_index=True)

    # Reset global index
    downsampled_df['index'] = range(len(downsampled_df))

    print(f"  Downsampled frames: {len(downsampled_df)}")
    if trim_idle:
        print(f"  Trimmed idle frames: {total_trimmed}")
    print(f"  Total reduction: {len(df) - len(downsampled_df)} frames ({100 * (len(df) - len(downsampled_df)) / len(df):.1f}%)")

    # Save downsampled parquet
    output_parquet_dir = data_dir / "chunk-000"
    output_parquet_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(downsampled_df)
    pq.write_table(table, output_parquet_dir / "file-000.parquet")
    print("[OK] Parquet data downsampled")
    print()

    # Downsample videos
    print("Downsampling videos...")
    videos_dir = output_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Find all camera directories
    input_videos = input_root / "videos"
    for camera_dir in input_videos.iterdir():
        if camera_dir.is_dir():
            print(f"\n  Processing {camera_dir.name}...")
            output_camera_dir = videos_dir / camera_dir.name
            downsample_videos(camera_dir, output_camera_dir, skip_factor)

    print("\n[OK] Videos downsampled")
    print()

    # Update info.json with new frame count
    info['total_frames'] = len(downsampled_df)
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("=" * 70)
    print("Downsampling Complete!")
    print("=" * 70)
    print(f"\nNew dataset location: {output_root}")
    print(f"Total frames: {len(downsampled_df)}")
    print(f"FPS: {target_fps}")
    print()
    print("Next steps:")
    print("1. Validate the downsampled dataset:")
    print(f"   python validate_dataset.py {output_root}")
    print()
    print("2. Test replay a downsampled episode:")
    print(f"   python replay_dataset_episode.py --dataset {output_root} --episode 0")
    print()
    print("3. Train OpenVLA with the downsampled dataset:")
    print(f"   Update openvla_finetune.py: dataset_root = '{output_root}'")
    print("=" * 70)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Downsample LeRobot dataset for OpenVLA")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output", type=str, required=True, help="Output dataset path")
    parser.add_argument("--fps", type=int, default=5, help="Target FPS (default: 5)")
    parser.add_argument("--trim-idle", action="store_true", default=True, help="Trim idle frames from episode edges (default: True)")
    parser.add_argument("--no-trim-idle", action="store_false", dest="trim_idle", help="Don't trim idle frames")
    parser.add_argument("--idle-threshold", type=float, default=2.0, help="Max joint change (degrees) to consider idle (default: 2.0)")

    args = parser.parse_args()

    return downsample_dataset(args.input, args.output, args.fps, args.trim_idle, args.idle_threshold)


if __name__ == "__main__":
    sys.exit(main())
