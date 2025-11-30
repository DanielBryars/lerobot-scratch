"""
Train a policy on a recorded dataset.

Usage:
    python train_policy.py

Or with CLI overrides:
    python train_policy.py --steps=50000 --batch_size=32
"""

import sys
from pathlib import Path

# Register STS3250 classes (needed if dataset was recorded with custom robot type)
sys.path.insert(0, str(Path(__file__).parent.parent))
import sts3250_plugin  # noqa: F401

from lerobot.scripts.lerobot_train import train

if __name__ == "__main__":
    # Training is configured via command line args or config files
    # Example command:
    #
    # python train_policy.py \
    #     --dataset.repo_id=YOUR_USERNAME/your_dataset \
    #     --policy.type=act \
    #     --steps=100000 \
    #     --batch_size=8
    #
    # Or for diffusion policy:
    #
    # python train_policy.py \
    #     --dataset.repo_id=YOUR_USERNAME/your_dataset \
    #     --policy.type=diffusion \
    #     --steps=100000 \
    #     --batch_size=64

    print("Starting training...")
    print("\nExample usage:")
    print("  python train_policy.py --dataset.repo_id=user/dataset --policy.type=act --steps=100000")
    print()

    train()
