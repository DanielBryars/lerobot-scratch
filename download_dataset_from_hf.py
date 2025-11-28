#!/usr/bin/env python3
"""
Download a LeRobot dataset from Hugging Face Hub.

Usage:
    python download_dataset_from_hf.py --repo-id YOUR_USERNAME/so100-block-move

Requirements:
    pip install huggingface_hub
    # For private repos: huggingface-cli login
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download LeRobot dataset from Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: datasets/<repo-name>)"
    )
    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        # Use the repo name as the folder name under datasets/
        repo_name = args.repo_id.split("/")[-1]
        output_path = Path("datasets") / repo_name

    print(f"Downloading dataset: {args.repo_id}")
    print(f"To local path: {output_path}")
    print()

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download the dataset
    print("Downloading from Hugging Face Hub...")
    local_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(output_path),
        local_dir_use_symlinks=False  # Copy files instead of symlinking
    )

    print(f"\nDone! Dataset downloaded to: {local_path}")

    # Verify the download
    meta_file = output_path / "meta" / "info.json"
    if meta_file.exists():
        import json
        with open(meta_file) as f:
            info = json.load(f)
        print(f"\nDataset info:")
        print(f"  Total episodes: {info.get('total_episodes', 'N/A')}")
        print(f"  Total frames: {info.get('total_frames', 'N/A')}")
        print(f"  FPS: {info.get('fps', 'N/A')}")
    else:
        print(f"\nNote: Could not find meta/info.json to verify download")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
