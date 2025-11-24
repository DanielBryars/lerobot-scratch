#!/usr/bin/env python3
"""
Pre-extract video frames to disk for faster training.

This converts the LeRobot video-based dataset to extracted PNG frames,
which load much faster during training (no video decoding overhead).

Usage:
    python preextract_frames.py
"""

import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm


def extract_frames_from_video(video_path: Path, output_dir: Path, start_idx: int) -> int:
    """Extract all frames from a video file to output directory."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted = 0

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        global_idx = start_idx + i
        output_path = output_dir / f"{global_idx:06d}.jpg"
        cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        extracted += 1

    cap.release()
    return extracted


def main():
    dataset_root = Path("./datasets/merged_training_set")
    output_root = Path("./datasets/merged_training_set_frames")

    print("=" * 70)
    print("Pre-extracting Video Frames")
    print("=" * 70)
    print()
    print(f"Source: {dataset_root}")
    print(f"Output: {output_root}")
    print()

    # Load dataset info
    with open(dataset_root / "meta" / "info.json") as f:
        info = json.load(f)

    # Find image keys
    features = info.get("features", {})
    image_keys = [k for k in features.keys() if k.startswith("observation.images.")]

    print(f"Found {len(image_keys)} cameras: {image_keys}")
    print(f"Total frames expected: {info['total_frames']}")
    print()

    # Create output directories
    output_root.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    import shutil
    meta_src = dataset_root / "meta"
    meta_dst = output_root / "meta"
    if meta_dst.exists():
        shutil.rmtree(meta_dst)
    shutil.copytree(meta_src, meta_dst)

    # Copy data (parquet files)
    data_src = dataset_root / "data"
    data_dst = output_root / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    shutil.copytree(data_src, data_dst)

    print("Copied metadata and parquet files")
    print()

    # Extract frames for each camera
    for image_key in image_keys:
        print(f"Extracting {image_key}...")

        video_dir = dataset_root / "videos" / image_key
        frames_dir = output_root / "frames" / image_key
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Find all video files and build frame map
        video_files = sorted(video_dir.glob("**/*.mp4"))

        current_idx = 0
        total_extracted = 0

        for video_file in tqdm(video_files, desc=f"  Videos"):
            # Get frame count
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Extract frames
            extracted = extract_frames_from_video(video_file, frames_dir, current_idx)
            total_extracted += extracted
            current_idx += frame_count

        print(f"  Extracted {total_extracted} frames to {frames_dir}")
        print()

    # Update info.json to indicate frames are extracted
    info['frames_extracted'] = True
    info['frames_format'] = 'jpg'
    with open(output_root / "meta" / "info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("=" * 70)
    print("Done! Use the new dataset path for training:")
    print(f"  dataset_root = \"{output_root}\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
