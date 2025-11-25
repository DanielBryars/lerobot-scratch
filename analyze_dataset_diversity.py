#!/usr/bin/env python3
"""
Analyze action diversity in a dataset to diagnose learning issues.

Checks:
- Action value ranges per joint
- Action variance (are demos all the same?)
- Per-episode action statistics
- Visual diversity (if possible)

Usage:
    python analyze_dataset_diversity.py [dataset_path]
"""

import sys
import json
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def analyze_dataset(dataset_root: str):
    """Analyze action diversity in the dataset."""
    dataset_root = Path(dataset_root)

    print("=" * 70)
    print(f"Dataset Diversity Analysis: {dataset_root}")
    print("=" * 70)
    print()

    # Load data
    data_file = dataset_root / "data" / "chunk-000" / "file-000.parquet"
    df = pq.read_table(data_file).to_pandas()

    print(f"Total frames: {len(df)}")
    print(f"Total episodes: {df['episode_index'].nunique()}")
    print()

    # Extract actions
    actions = np.array([np.array(a) for a in df['action']])
    print(f"Action shape: {actions.shape}")
    print()

    # Overall statistics
    print("=" * 70)
    print("Overall Action Statistics")
    print("=" * 70)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    for i, joint in enumerate(joint_names):
        joint_actions = actions[:, i]
        print(f"\n{joint}:")
        print(f"  Min:    {joint_actions.min():8.2f}°")
        print(f"  Max:    {joint_actions.max():8.2f}°")
        print(f"  Mean:   {joint_actions.mean():8.2f}°")
        print(f"  Std:    {joint_actions.std():8.2f}°")
        print(f"  Range:  {joint_actions.max() - joint_actions.min():8.2f}°")

    # Check variance - low variance = all demos are similar
    print("\n" + "=" * 70)
    print("Action Variance Analysis")
    print("=" * 70)
    print("\nVariance (higher = more diverse):")
    for i, joint in enumerate(joint_names):
        variance = actions[:, i].var()
        print(f"  {joint:20s}: {variance:10.2f}")

    # Per-episode analysis
    print("\n" + "=" * 70)
    print("Per-Episode Statistics")
    print("=" * 70)

    episodes = df['episode_index'].unique()
    print(f"\nAnalyzing {len(episodes)} episodes...")

    # For each episode, compute mean action
    episode_means = []
    episode_lengths = []

    for ep_idx in episodes:
        ep_data = df[df['episode_index'] == ep_idx]
        ep_actions = np.array([np.array(a) for a in ep_data['action']])
        ep_mean = ep_actions.mean(axis=0)
        episode_means.append(ep_mean)
        episode_lengths.append(len(ep_data))

    episode_means = np.array(episode_means)
    episode_lengths = np.array(episode_lengths)

    print(f"\nEpisode length statistics:")
    print(f"  Min:  {episode_lengths.min()} frames ({episode_lengths.min()/30:.1f}s)")
    print(f"  Max:  {episode_lengths.max()} frames ({episode_lengths.max()/30:.1f}s)")
    print(f"  Mean: {episode_lengths.mean():.0f} frames ({episode_lengths.mean()/30:.1f}s)")

    print(f"\nVariance across episode means (higher = more diverse starting positions):")
    for i, joint in enumerate(joint_names):
        ep_variance = episode_means[:, i].var()
        print(f"  {joint:20s}: {ep_variance:10.2f}")

    # Key diagnostic: If episode mean variance is LOW, all demos started from same position
    avg_ep_variance = episode_means.var(axis=0).mean()
    print(f"\nAverage episode variance: {avg_ep_variance:.2f}")

    if avg_ep_variance < 50:
        print("  ⚠️  WARNING: Very low variance across episodes!")
        print("     This suggests all demos have similar trajectories.")
        print("     The block may have been in the same position every time.")
    elif avg_ep_variance < 200:
        print("  ⚠️  CAUTION: Moderate variance across episodes.")
        print("     There is some diversity but may not be enough.")
    else:
        print("  ✓ Good: Episodes have diverse trajectories.")

    # Show first few episode means
    print("\n" + "=" * 70)
    print("Sample Episode Mean Actions (first 5 episodes)")
    print("=" * 70)
    for ep_idx in range(min(5, len(episodes))):
        print(f"\nEpisode {ep_idx}:")
        print(f"  Frames: {episode_lengths[ep_idx]}")
        print(f"  Mean action: {episode_means[ep_idx]}")

    # Create visualization
    if not HAS_MATPLOTLIB:
        print("\n⚠️  matplotlib not installed, skipping plots")
        print("   Install with: pip install matplotlib")
    else:
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle("Action Distributions per Joint", fontsize=16)

            for i, (ax, joint) in enumerate(zip(axes.flat, joint_names)):
                ax.hist(actions[:, i], bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(joint)
                ax.set_xlabel("Angle (degrees)")
                ax.set_ylabel("Frequency")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = dataset_root / "action_distribution.png"
            plt.savefig(output_path, dpi=150)
            print(f"\n✓ Saved action distribution plot: {output_path}")

            # Episode variance plot
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(joint_names))
            ax.bar(x, episode_means.var(axis=0))
            ax.set_xticks(x)
            ax.set_xticklabels(joint_names, rotation=45, ha='right')
            ax.set_ylabel("Variance")
            ax.set_title("Variance Across Episode Means (Diversity Indicator)")
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            output_path = dataset_root / "episode_variance.png"
            plt.savefig(output_path, dpi=150)
            print(f"✓ Saved episode variance plot: {output_path}")

        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


def main():
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "./datasets/20251124_233735"

    analyze_dataset(dataset_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
