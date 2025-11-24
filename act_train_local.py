#!/usr/bin/env python3
"""
Train ACT model on local dataset.

Usage:
    python train_local.py
"""

import os
import sys

# Fix Windows console encoding issues
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'

# Also set for subprocess
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

def main():
    # Training configuration
    dataset_root = "./datasets/merged_training_set"
    output_dir = "./outputs/act_finetuned"
    batch_size = 8  # ACT uses more memory per batch
    steps = 50000  # ACT typically needs more steps
    learning_rate = 1e-5

    print("=" * 70)
    print("ACT Training on Local Dataset")
    print("=" * 70)
    print()
    print(f"Dataset: {dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Batch size: {batch_size}")
    print(f"Steps: {steps}")
    print(f"Learning rate: {learning_rate}")
    print()

    # Build command - ACT policy
    cmd = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local_dataset",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        "--policy.repo_id=danbhf/act_so100_finetuned",
        "--policy.push_to_hub=false",
        "--policy.device=cuda",
        # ACT specific settings
        "--policy.chunk_size=100",
        "--policy.n_action_steps=100",
        f"--output_dir={output_dir}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--eval_freq=5000",
        "--save_freq=5000",
        "--log_freq=100",
        f"--optimizer.lr={learning_rate}",
        # Enable wandb
        "--wandb.enable=true",
        "--wandb.project=lerobot-so100",
    ]

    print("Command:")
    print(" ".join(cmd))
    print()
    print("=" * 70)
    print()

    import subprocess
    try:
        # Use Popen to handle output properly
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Stream output
        for line in process.stdout:
            print(line, end='')

        process.wait()
        return process.returncode

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        process.terminate()
        return 1
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
