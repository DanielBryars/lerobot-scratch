#!/usr/bin/env python3
"""
Upload a local LeRobot dataset to Hugging Face Hub.

Usage:
    python upload_dataset_to_hf.py --dataset datasets/20251124_233735_5hz --repo-id YOUR_USERNAME/so100-block-move

Requirements:
    pip install huggingface_hub
    huggingface-cli login  # Login first with your HF token
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload LeRobot dataset to Hugging Face")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/20251124_233735_5hz",
        help="Path to local dataset folder"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return 1

    print(f"Uploading dataset: {dataset_path}")
    print(f"To Hugging Face repo: {args.repo_id}")
    print(f"Private: {args.private}")
    print()

    # Initialize HF API
    api = HfApi()

    # Create repo if it doesn't exist
    print("Creating/checking repository...")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True
        )
        print(f"  Repository ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"  Warning: {e}")

    # Upload the entire folder
    print("\nUploading files...")
    api.upload_folder(
        folder_path=str(dataset_path),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload LeRobot dataset"
    )

    print(f"\nDone! Dataset uploaded to:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")
    print(f"\nTo download on another machine:")
    print(f"  python download_dataset_from_hf.py --repo-id {args.repo_id}")

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
