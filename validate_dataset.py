#!/usr/bin/env python3
"""
Validate LeRobot dataset integrity.

Checks:
- Required files exist (info.json, tasks.parquet, data, videos)
- Parquet data is readable and has expected columns
- Video files exist and are readable
- Frame counts match between parquet and videos
- Sample frames can be extracted from videos
- Task descriptions are present

Usage:
    python validate_dataset.py [dataset_path]
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pyarrow.parquet as pq


def validate_dataset(dataset_root: str, robot_config_path: str = "./config.json") -> bool:
    """
    Validate a LeRobot dataset.

    Returns True if dataset is valid, False otherwise.
    """
    dataset_root = Path(dataset_root)
    all_ok = True

    print("=" * 70)
    print(f"Validating Dataset: {dataset_root}")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. Check required files exist
    # =========================================================================
    print("[1/6] Checking required files...")

    required_files = [
        "meta/info.json",
        "meta/tasks.parquet",
    ]

    for rf in required_files:
        path = dataset_root / rf
        if path.exists():
            print(f"  [OK] {rf}")
        else:
            print(f"  [ERROR] Missing: {rf}")
            all_ok = False

    required_dirs = ["data", "videos"]
    for rd in required_dirs:
        path = dataset_root / rd
        if path.exists() and path.is_dir():
            print(f"  [OK] {rd}/")
        else:
            print(f"  [ERROR] Missing directory: {rd}/")
            all_ok = False

    if not all_ok:
        print("\nCritical files missing. Cannot continue validation.")
        return False
    print()

    # =========================================================================
    # 2. Load and validate info.json
    # =========================================================================
    print("[2/6] Validating info.json...")

    with open(dataset_root / "meta/info.json") as f:
        info = json.load(f)

    print(f"  Codebase version: {info.get('codebase_version', 'unknown')}")
    print(f"  Robot type: {info.get('robot_type', 'unknown')}")
    print(f"  Total episodes: {info.get('total_episodes', 'unknown')}")
    print(f"  Total frames: {info.get('total_frames', 'unknown')}")
    print(f"  FPS: {info.get('fps', 'unknown')}")

    # Check features
    features = info.get("features", {})
    image_keys = [k for k in features.keys() if k.startswith("observation.images.")]
    print(f"  Image features: {image_keys}")

    if not image_keys:
        print("  [WARNING] No image features found!")
    print()

    # =========================================================================
    # 3. Validate parquet data files
    # =========================================================================
    print("[3/6] Validating parquet data files...")

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")

    if not parquet_files:
        print("  [ERROR] No parquet files found!")
        return False

    # Load all data
    all_dfs = []
    for pf in parquet_files:
        try:
            df = pq.read_table(pf).to_pandas()
            all_dfs.append(df)
            print(f"  [OK] {pf.relative_to(dataset_root)}: {len(df)} rows")
        except Exception as e:
            print(f"  [ERROR] {pf.relative_to(dataset_root)}: {e}")
            all_ok = False

    if not all_dfs:
        print("  [ERROR] Could not load any parquet files!")
        return False

    import pandas as pd
    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total rows: {len(full_df)}")

    # Check columns
    expected_cols = ['action', 'observation.state', 'frame_index', 'episode_index', 'index', 'task_index']
    missing_cols = [c for c in expected_cols if c not in full_df.columns]
    if missing_cols:
        print(f"  [ERROR] Missing columns: {missing_cols}")
        all_ok = False
    else:
        print(f"  [OK] All expected columns present")

    # Check index continuity
    indices = sorted(full_df['index'].unique())
    expected_indices = list(range(len(indices)))
    if indices != expected_indices:
        print(f"  [WARNING] Index not continuous: {indices[:5]}...{indices[-5:]}")
    else:
        print(f"  [OK] Index is continuous (0 to {len(indices)-1})")

    # Episode summary
    episodes = sorted(full_df['episode_index'].unique())
    print(f"  Episodes: {len(episodes)} (indices: {min(episodes)} to {max(episodes)})")
    print()

    # =========================================================================
    # 4. Validate video files
    # =========================================================================
    print("[4/6] Validating video files...")

    for image_key in image_keys:
        print(f"\n  Camera: {image_key}")
        video_dir = dataset_root / "videos" / image_key

        if not video_dir.exists():
            print(f"    [ERROR] Video directory missing: {video_dir}")
            all_ok = False
            continue

        video_files = sorted(video_dir.glob("**/*.mp4"))
        print(f"    Found {len(video_files)} video files")

        total_video_frames = 0
        video_frame_map = []  # List of (start_idx, end_idx, video_path)

        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                video_frame_map.append((total_video_frames, total_video_frames + frame_count - 1, vf))
                total_video_frames += frame_count

                print(f"    [OK] {vf.relative_to(video_dir)}: {frame_count} frames, {width}x{height}@{fps:.1f}fps")
                cap.release()
            else:
                print(f"    [ERROR] Cannot open: {vf.relative_to(video_dir)}")
                all_ok = False

        # Compare with parquet
        expected_frames = info.get('total_frames', len(full_df))
        if total_video_frames == expected_frames:
            print(f"    [OK] Video frames ({total_video_frames}) match expected ({expected_frames})")
        else:
            print(f"    [WARNING] Video frames ({total_video_frames}) != expected ({expected_frames})")

    print()

    # =========================================================================
    # 5. Test frame extraction
    # =========================================================================
    print("[5/6] Testing frame extraction...")

    # Build video frame map for first camera
    if image_keys:
        test_image_key = image_keys[0]
        video_dir = dataset_root / "videos" / test_image_key
        video_files = sorted(video_dir.glob("**/*.mp4"))

        video_frame_map = []
        current_idx = 0
        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_frame_map.append({
                    'start_idx': current_idx,
                    'end_idx': current_idx + frame_count - 1,
                    'path': vf,
                    'frame_count': frame_count,
                })
                current_idx += frame_count
                cap.release()

        # Test extracting frames from different parts of the dataset
        test_indices = [0, len(full_df) // 2, len(full_df) - 1]

        for test_idx in test_indices:
            global_idx = full_df.iloc[test_idx]['index']

            # Find which video file contains this frame
            video_info = None
            for vi in video_frame_map:
                if vi['start_idx'] <= global_idx <= vi['end_idx']:
                    video_info = vi
                    break

            if video_info is None:
                print(f"  [ERROR] No video found for index {global_idx}")
                all_ok = False
                continue

            # Calculate frame position within this video
            frame_in_video = global_idx - video_info['start_idx']

            cap = cv2.VideoCapture(str(video_info['path']))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                print(f"  [OK] Index {global_idx}: extracted from {video_info['path'].name} frame {frame_in_video} ({frame.shape})")
            else:
                print(f"  [ERROR] Index {global_idx}: failed to extract frame")
                all_ok = False

    print()

    # =========================================================================
    # 6. Validate tasks
    # =========================================================================
    print("[6/6] Validating tasks...")

    tasks_path = dataset_root / "meta" / "tasks.parquet"
    try:
        tasks_table = pq.read_table(tasks_path)
        tasks_df = tasks_table.to_pandas()
        print(f"  Found {len(tasks_df)} tasks")

        # Show task descriptions
        # Task description is the row index, task_index is the column
        for task_desc, row in tasks_df.iterrows():
            task_idx = row['task_index']
            task_desc_str = str(task_desc)
            print(f"    Task {task_idx}: {task_desc_str[:50]}{'...' if len(task_desc_str) > 50 else ''}")

        # Check all task indices in data are valid
        data_task_indices = set(full_df['task_index'].unique())
        task_indices = set(tasks_df['task_index'].unique())

        missing_tasks = data_task_indices - task_indices
        if missing_tasks:
            print(f"  [WARNING] Data references undefined tasks: {missing_tasks}")
        else:
            print(f"  [OK] All task indices are valid")

    except Exception as e:
        print(f"  [ERROR] Failed to load tasks: {e}")
        all_ok = False

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    if all_ok:
        print("VALIDATION PASSED - Dataset is ready for training!")
    else:
        print("VALIDATION FAILED - Please fix the issues above")
    print("=" * 70)

    return all_ok


def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./datasets/merged_training_set"

    success = validate_dataset(dataset_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
