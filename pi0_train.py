#!/usr/bin/env python3
"""
Fine-tune Pi0 model on custom SO100 dataset.

Usage:
    python train_pi0.py --dataset-repo-id username/my_dataset --output-dir ./outputs/pi0_finetuned

This script fine-tunes the Pi0 model using your recorded demonstrations.
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Pi0 model")
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repository ID (e.g., 'username/my_so100_dataset')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/pi0_finetuned",
        help="Output directory for model checkpoints (default: ./outputs/pi0_finetuned)"
    )
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="lerobot/pi0_base",
        help="Pretrained model path (default: lerobot/pi0_base)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training (default: 8, reduce if out of memory)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Number of training steps (default: 3000)"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=500,
        help="Evaluation frequency in steps (default: 500)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1000,
        help="Checkpoint save frequency in steps (default: 1000)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory where dataset is stored (default: ./data)"
    )

    args = parser.parse_args()

    # Build the training command
    cmd_parts = [
        "python", "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        f"--dataset.root={args.data_root}",
        "--policy.type=pi0",
        f"--policy.pretrained_path={args.pretrained_path}",
        "--policy.compile_model=true",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--output_dir={args.output_dir}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--eval_freq={args.eval_freq}",
        f"--save_freq={args.save_freq}",
        f"--learning_rate={args.learning_rate}",
        "--device=cuda",
    ]

    print("=" * 70)
    print("Pi0 Fine-tuning")
    print("=" * 70)
    print()
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training steps: {args.steps}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    print("Configuration:")
    print("  - Model compilation: enabled (faster training)")
    print("  - Gradient checkpointing: enabled (reduced memory)")
    print("  - Mixed precision: bfloat16 (faster training)")
    print("  - Device: CUDA (GPU)")
    print()
    print("=" * 70)
    print()

    # Execute training
    import subprocess
    try:
        result = subprocess.run(cmd_parts, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())
