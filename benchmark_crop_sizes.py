#!/usr/bin/env python3
"""
Quick benchmark to compare training speed at different crop sizes.
Runs a few training steps and measures time per step.
"""

import torch
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType

def benchmark_crop_size(crop_size, num_steps=20):
    """Benchmark training speed for a given crop size."""
    dataset_path = Path("datasets/20251124_233735_5hz")
    device = torch.device("cuda")

    # Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="local",
        root=str(dataset_path),
    )

    # Extract features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Configure policy with specific crop size
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        use_group_norm=False,  # Required for pretrained weights
        crop_shape=(crop_size, crop_size),
        crop_is_random=False,  # Consistent for benchmarking
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        num_train_timesteps=100,
    )

    # Create policy
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())

    # Create dummy batch (simulating what dataloader would provide)
    batch_size = 64

    # Fake batch with correct shapes
    batch = {
        "observation.state": torch.randn(batch_size, 2, 6, device=device),  # [B, n_obs_steps, state_dim]
        "observation.images.base_0_rgb": torch.rand(batch_size, 2, 3, crop_size, crop_size, device=device),
        "observation.images.left_wrist_0_rgb": torch.rand(batch_size, 2, 3, crop_size, crop_size, device=device),
        "action": torch.randn(batch_size, 16, 6, device=device),  # [B, horizon, action_dim]
        "action_is_pad": torch.zeros(batch_size, 16, dtype=torch.bool, device=device),  # No padding
    }

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    # Warmup
    for _ in range(3):
        loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_steps):
        loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.time() - start

    ms_per_step = (elapsed / num_steps) * 1000
    steps_per_sec = num_steps / elapsed

    # Memory usage
    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "crop_size": crop_size,
        "params_M": total_params / 1e6,
        "ms_per_step": ms_per_step,
        "steps_per_sec": steps_per_sec,
        "vram_gb": mem_gb,
    }

def main():
    print("=" * 60)
    print("Diffusion Policy Crop Size Benchmark")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Batch size: 64")
    print()

    crop_sizes = [84, 128, 224]
    results = []

    for size in crop_sizes:
        print(f"Testing {size}x{size}...")
        torch.cuda.reset_peak_memory_stats()
        try:
            result = benchmark_crop_size(size)
            results.append(result)
            print(f"  {result['ms_per_step']:.1f} ms/step, {result['steps_per_sec']:.1f} steps/sec, {result['vram_gb']:.1f} GB VRAM")
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Crop':>8} {'Params':>10} {'ms/step':>10} {'steps/sec':>12} {'VRAM':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['crop_size']:>8} {r['params_M']:>9.1f}M {r['ms_per_step']:>10.1f} {r['steps_per_sec']:>12.1f} {r['vram_gb']:>9.1f}GB")

    # Estimate training time for 50k steps
    print()
    print("Estimated time for 50,000 steps:")
    for r in results:
        hours = (50000 / r['steps_per_sec']) / 3600
        print(f"  {r['crop_size']}x{r['crop_size']}: {hours:.1f} hours")

if __name__ == "__main__":
    main()
