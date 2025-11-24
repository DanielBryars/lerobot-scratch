#!/usr/bin/env python3
"""
Merge multiple local LeRobot datasets into a single dataset.

Usage:
    python merge_datasets.py
"""

import shutil
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq

# Configuration
DATASETS_DIR = Path("datasets")
DATASET_1 = "20251115_232405"  # 10 episodes, 7691 frames
DATASET_2 = "20251123_141805"  # 10 episodes, 5250 frames
OUTPUT_NAME = "merged_training_set"


def load_parquet_as_df(path):
    """Load a parquet file as a pandas DataFrame."""
    return pq.read_table(path).to_pandas()


def save_df_as_parquet(df, path):
    """Save a pandas DataFrame as a parquet file."""
    import pyarrow as pa
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)


def merge_datasets():
    ds1_path = DATASETS_DIR / DATASET_1
    ds2_path = DATASETS_DIR / DATASET_2
    output_path = DATASETS_DIR / OUTPUT_NAME

    # Clean output if exists
    if output_path.exists():
        print(f"Removing existing {output_path}")
        shutil.rmtree(output_path)

    print(f"Merging {DATASET_1} and {DATASET_2} into {OUTPUT_NAME}")

    # Load metadata
    with open(ds1_path / "meta" / "info.json") as f:
        info1 = json.load(f)
    with open(ds2_path / "meta" / "info.json") as f:
        info2 = json.load(f)

    print(f"Dataset 1: {info1['total_episodes']} episodes, {info1['total_frames']} frames")
    print(f"Dataset 2: {info2['total_episodes']} episodes, {info2['total_frames']} frames")

    # Create output directory structure
    output_path.mkdir(parents=True)
    (output_path / "meta").mkdir()
    (output_path / "data" / "chunk-000").mkdir(parents=True)
    (output_path / "videos" / "observation.images.base_0_rgb" / "chunk-000").mkdir(parents=True)
    (output_path / "videos" / "observation.images.left_wrist_0_rgb" / "chunk-000").mkdir(parents=True)

    # Load and merge data parquet files
    print("Loading data files...")
    df1 = load_parquet_as_df(ds1_path / "data" / "chunk-000" / "file-000.parquet")
    df2 = load_parquet_as_df(ds2_path / "data" / "chunk-000" / "file-000.parquet")

    print(f"Dataset 1 data shape: {df1.shape}")
    print(f"Dataset 2 data shape: {df2.shape}")

    # Load tasks
    tasks1_df = load_parquet_as_df(ds1_path / "meta" / "tasks.parquet")
    tasks2_df = load_parquet_as_df(ds2_path / "meta" / "tasks.parquet")

    # Get unique tasks from both datasets (use string as key)
    all_tasks = {}
    for _, row in tasks1_df.iterrows():
        task_str = row.name if isinstance(row.name, str) else str(row.name)
        all_tasks[task_str] = row['task_index']

    # Map dataset 2 tasks - may need new indices if tasks differ
    task2_mapping = {}  # old_task_index -> new_task_index
    next_task_idx = max(all_tasks.values()) + 1 if all_tasks else 0

    for _, row in tasks2_df.iterrows():
        task_str = row.name if isinstance(row.name, str) else str(row.name)
        old_idx = row['task_index']
        if task_str in all_tasks:
            task2_mapping[old_idx] = all_tasks[task_str]
        else:
            task2_mapping[old_idx] = next_task_idx
            all_tasks[task_str] = next_task_idx
            next_task_idx += 1

    print(f"Total unique tasks: {len(all_tasks)}")

    # Offset dataset 2 indices
    num_episodes_1 = info1['total_episodes']
    num_frames_1 = info1['total_frames']

    # Adjust dataset 2
    df2_adjusted = df2.copy()
    df2_adjusted['episode_index'] = df2_adjusted['episode_index'] + num_episodes_1
    df2_adjusted['index'] = df2_adjusted['index'] + num_frames_1
    df2_adjusted['task_index'] = df2_adjusted['task_index'].map(task2_mapping)

    # Concatenate
    merged_df = pd.concat([df1, df2_adjusted], ignore_index=True)
    print(f"Merged data shape: {merged_df.shape}")

    # Save merged data
    print("Saving merged data...")
    save_df_as_parquet(merged_df, output_path / "data" / "chunk-000" / "file-000.parquet")

    # Create merged tasks parquet
    tasks_data = {
        'task_index': list(all_tasks.values()),
    }
    tasks_df = pd.DataFrame(tasks_data)
    tasks_df.index = pd.Index(list(all_tasks.keys()))
    save_df_as_parquet(tasks_df, output_path / "meta" / "tasks.parquet")

    # Copy and rename video files
    print("Copying video files...")

    # For simplicity, we'll concatenate videos using ffmpeg or just reference them
    # LeRobot v3 uses chunked format - each chunk can have multiple episodes
    # We need to handle this properly

    # Copy dataset 1 videos as chunk-000
    for video_key in ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]:
        src = ds1_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
        dst = output_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
        shutil.copy2(src, dst)
        print(f"  Copied {video_key} from dataset 1")

    # Copy dataset 2 videos as chunk-001
    (output_path / "videos" / "observation.images.base_0_rgb" / "chunk-001").mkdir(parents=True)
    (output_path / "videos" / "observation.images.left_wrist_0_rgb" / "chunk-001").mkdir(parents=True)

    for video_key in ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]:
        src = ds2_path / "videos" / video_key / "chunk-000" / "file-000.mp4"
        dst = output_path / "videos" / video_key / "chunk-001" / "file-000.mp4"
        shutil.copy2(src, dst)
        print(f"  Copied {video_key} from dataset 2")

    # Create merged info.json
    merged_info = info1.copy()
    merged_info['total_episodes'] = info1['total_episodes'] + info2['total_episodes']
    merged_info['total_frames'] = info1['total_frames'] + info2['total_frames']
    merged_info['total_tasks'] = len(all_tasks)
    merged_info['splits'] = {
        'train': f"0:{merged_info['total_episodes']}"
    }

    with open(output_path / "meta" / "info.json", 'w') as f:
        json.dump(merged_info, f, indent=4)

    # Merge episodes metadata
    print("Merging episodes metadata...")
    (output_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True)

    episodes1_df = load_parquet_as_df(ds1_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    episodes2_df = load_parquet_as_df(ds2_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")

    print(f"  Dataset 1 episodes: {len(episodes1_df)}")
    print(f"  Dataset 2 episodes: {len(episodes2_df)}")

    # Adjust episode 2 indices
    episodes2_adjusted = episodes2_df.copy()
    episodes2_adjusted['episode_index'] = episodes2_adjusted['episode_index'] + num_episodes_1
    episodes2_adjusted['dataset_from_index'] = episodes2_adjusted['dataset_from_index'] + num_frames_1
    episodes2_adjusted['dataset_to_index'] = episodes2_adjusted['dataset_to_index'] + num_frames_1

    # Update video chunk indices for dataset 2 (they go in chunk-001)
    for video_key in ["observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb"]:
        chunk_col = f'videos/{video_key}/chunk_index'
        if chunk_col in episodes2_adjusted.columns:
            episodes2_adjusted[chunk_col] = 1  # All dataset 2 videos are in chunk-001

    # Update episode stats indices
    if 'stats/episode_index/min' in episodes2_adjusted.columns:
        episodes2_adjusted['stats/episode_index/min'] = episodes2_adjusted['stats/episode_index/min'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/max'] = episodes2_adjusted['stats/episode_index/max'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/mean'] = episodes2_adjusted['stats/episode_index/mean'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/q01'] = episodes2_adjusted['stats/episode_index/q01'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/q10'] = episodes2_adjusted['stats/episode_index/q10'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/q50'] = episodes2_adjusted['stats/episode_index/q50'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/q90'] = episodes2_adjusted['stats/episode_index/q90'] + num_episodes_1
        episodes2_adjusted['stats/episode_index/q99'] = episodes2_adjusted['stats/episode_index/q99'] + num_episodes_1

    if 'stats/index/min' in episodes2_adjusted.columns:
        episodes2_adjusted['stats/index/min'] = episodes2_adjusted['stats/index/min'] + num_frames_1
        episodes2_adjusted['stats/index/max'] = episodes2_adjusted['stats/index/max'] + num_frames_1
        episodes2_adjusted['stats/index/mean'] = episodes2_adjusted['stats/index/mean'] + num_frames_1
        episodes2_adjusted['stats/index/q01'] = episodes2_adjusted['stats/index/q01'] + num_frames_1
        episodes2_adjusted['stats/index/q10'] = episodes2_adjusted['stats/index/q10'] + num_frames_1
        episodes2_adjusted['stats/index/q50'] = episodes2_adjusted['stats/index/q50'] + num_frames_1
        episodes2_adjusted['stats/index/q90'] = episodes2_adjusted['stats/index/q90'] + num_frames_1
        episodes2_adjusted['stats/index/q99'] = episodes2_adjusted['stats/index/q99'] + num_frames_1

    # Update meta/episodes chunk index for dataset 2
    if 'meta/episodes/chunk_index' in episodes2_adjusted.columns:
        # Keep all in chunk-000 for the merged episodes file
        pass

    # Concatenate episodes
    merged_episodes_df = pd.concat([episodes1_df, episodes2_adjusted], ignore_index=True)
    save_df_as_parquet(merged_episodes_df, output_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    print(f"  Merged episodes: {len(merged_episodes_df)}")

    # Calculate and save stats
    print("Calculating statistics...")
    import numpy as np
    stats = {}

    for col in ['action', 'observation.state']:
        if col in merged_df.columns:
            # Handle array columns
            values = merged_df[col].tolist()
            arr = np.array(values)
            stats[col] = {
                'min': arr.min(axis=0).tolist(),
                'max': arr.max(axis=0).tolist(),
                'mean': arr.mean(axis=0).tolist(),
                'std': arr.std(axis=0).tolist(),
            }

    with open(output_path / "meta" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=4)

    print()
    print("=" * 60)
    print("Merge complete!")
    print(f"Output: {output_path}")
    print(f"Total episodes: {merged_info['total_episodes']}")
    print(f"Total frames: {merged_info['total_frames']}")
    print(f"Total tasks: {merged_info['total_tasks']}")
    print("=" * 60)


if __name__ == "__main__":
    merge_datasets()
