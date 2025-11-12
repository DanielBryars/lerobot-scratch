#!/usr/bin/env python3
"""
Upload a local dataset to HuggingFace Hub.

Usage:
    python upload_dataset.py --repo-id danbhf/my_dataset --local-dir ./datasets/20251112_233551/danbhf/my_dataset
"""

import argparse
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., 'danbhf/my_dataset')"
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        required=True,
        help="Local directory containing the dataset"
    )

    args = parser.parse_args()

    local_path = Path(args.local_dir)

    if not local_path.exists():
        print(f"✗ Error: Local directory does not exist: {local_path}")
        return 1

    print("=" * 70)
    print("Upload Dataset to HuggingFace Hub")
    print("=" * 70)
    print(f"\nRepo ID: {args.repo_id}")
    print(f"Local dir: {local_path}")
    print()

    try:
        # Load the dataset
        print("Loading dataset...")

        # The local_dir is the root where the dataset is stored
        # Structure is: local_dir/data, local_dir/videos, local_dir/meta
        dataset = LeRobotDataset(
            repo_id=args.repo_id,
            root=str(local_path),
        )

        print(f"✓ Dataset loaded")
        print(f"  Episodes: {dataset.num_episodes}")
        print(f"  Frames: {dataset.num_frames}")
        print(f"  FPS: {dataset.fps}")
        print()

        # Upload to hub
        print("Uploading to HuggingFace Hub...")
        print("This may take several minutes depending on dataset size...")

        dataset.push_to_hub()

        print()
        print("=" * 70)
        print("✓ Upload complete!")
        print("=" * 70)
        print(f"\nView your dataset at:")
        print(f"https://huggingface.co/datasets/{args.repo_id}")

        return 0

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        import traceback
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Ensure your token has WRITE permissions")
        print("3. Check your internet connection")

        return 1


if __name__ == "__main__":
    sys.exit(main())
