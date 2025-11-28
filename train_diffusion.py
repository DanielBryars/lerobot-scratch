#!/usr/bin/env python
"""
Train Diffusion Policy on local LeRobot dataset.

Adapted from LeRobot examples for our SO-100 block moving task.
"""

from pathlib import Path
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import wandb

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType


class CachedDataset(torch.utils.data.Dataset):
    """Wrapper that pre-caches all dataset samples in memory for fast access."""

    def __init__(self, dataset, device=None, resize_images_to=None):
        self.device = device
        self.resize_to = resize_images_to
        print(f"\nPre-caching {len(dataset)} samples to memory...")
        if resize_images_to:
            print(f"  Resizing images to {resize_images_to}x{resize_images_to}")

        # Pre-load all samples
        self.samples = []
        for i in tqdm(range(len(dataset)), desc="Caching dataset"):
            sample = dataset[i]
            # Move tensors to specified device (or keep on CPU)
            cached_sample = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    # Resize images if requested
                    if resize_images_to and "images" in k and v.dim() >= 3:
                        # v shape could be [n_obs_steps, C, H, W] or [C, H, W]
                        if v.dim() == 4:  # [n_obs_steps, C, H, W]
                            v = F.interpolate(v, size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False)
                        elif v.dim() == 3:  # [C, H, W]
                            v = F.interpolate(v.unsqueeze(0), size=(resize_images_to, resize_images_to),
                                            mode='bilinear', align_corners=False).squeeze(0)

                    if device is not None:
                        cached_sample[k] = v.to(device)
                    else:
                        cached_sample[k] = v.clone()
                else:
                    cached_sample[k] = v
            self.samples.append(cached_sample)

        print(f"Cached {len(self.samples)} samples")

        # Estimate memory usage
        total_bytes = 0
        for sample in self.samples[:1]:  # Check one sample
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    total_bytes += v.element_size() * v.numel()
        total_gb = (total_bytes * len(self.samples)) / (1024**3)
        print(f"Estimated cache size: {total_gb:.2f} GB")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    parser.add_argument("--dataset", type=str, default="datasets/20251124_233735_5hz",
                       help="Path to dataset folder")
    parser.add_argument("--output", type=str, default="outputs/diffusion_so100",
                       help="Output directory for checkpoints")
    parser.add_argument("--steps", type=int, default=50000,
                       help="Number of training steps")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Image size (images resized to this)")
    args = parser.parse_args()

    # Configuration from args
    dataset_path = Path(args.dataset)
    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training hyperparameters (from args)
    training_steps = args.steps
    batch_size = args.batch_size
    image_size = args.image_size
    learning_rate = 1e-4
    log_freq = 100
    save_freq = 10000
    num_workers = 0  # No workers needed with cached dataset

    # WandB configuration
    wandb_project = "diffusion-so100"
    wandb_run_name = f"diffusion_{image_size}x{image_size}"

    # Load dataset metadata
    print(f"\nLoading dataset from: {dataset_path}")
    dataset_metadata = LeRobotDatasetMetadata(
        repo_id="local",
        root=str(dataset_path),
    )

    print(f"  Episodes: {dataset_metadata.total_episodes}")
    print(f"  Frames: {dataset_metadata.total_frames}")
    print(f"  FPS: {dataset_metadata.fps}")
    print(f"  Features: {list(dataset_metadata.features.keys())}")

    # Extract policy features from dataset
    features = dataset_to_policy_features(dataset_metadata.features)
    print(f"\nPolicy features:")
    for key, ft in features.items():
        print(f"  {key}: type={ft.type}, shape={ft.shape}")

    # Separate input and output features
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"\nInput features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Configure diffusion policy (image_size from args, default 224)
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Vision backbone
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        crop_shape=(image_size, image_size),  # Match resized image size (no additional cropping)
        crop_is_random=False,  # No random cropping - we resize the full image instead
        use_group_norm=False,  # Can't use group_norm with pretrained weights
        spatial_softmax_num_keypoints=32,
        use_separate_rgb_encoder_per_camera=False,
        # Diffusion parameters
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        # Training
        num_train_timesteps=100,
        noise_scheduler_type="DDPM",
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
    )

    print(f"\nDiffusion config:")
    print(f"  n_obs_steps: {cfg.n_obs_steps}")
    print(f"  horizon: {cfg.horizon}")
    print(f"  n_action_steps: {cfg.n_action_steps}")
    print(f"  vision_backbone: {cfg.vision_backbone}")

    # Create policy (no dataset_stats - normalization handled by preprocessor)
    print("\nCreating Diffusion Policy...")
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    # Create preprocessor for normalization
    print("Creating preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        dataset_stats=dataset_metadata.stats,
    )

    # Count parameters
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Define temporal frame sampling based on FPS (5 Hz)
    fps = dataset_metadata.fps
    frame_duration = 1.0 / fps

    # Observation: current and previous frame
    obs_timestamps = [i * frame_duration for i in range(1 - cfg.n_obs_steps, 1)]

    # Action: from (1-n_obs_steps) to (1-n_obs_steps + horizon)
    action_timestamps = [i * frame_duration for i in range(1 - cfg.n_obs_steps, 1 - cfg.n_obs_steps + cfg.horizon)]

    delta_timestamps = {
        "observation.images.base_0_rgb": obs_timestamps,
        "observation.images.left_wrist_0_rgb": obs_timestamps,
        "observation.state": obs_timestamps,
        "action": action_timestamps,
    }

    print(f"\nDelta timestamps (fps={fps}):")
    for key, ts in delta_timestamps.items():
        print(f"  {key}: {ts}")

    # Load dataset with temporal sampling
    print("\nLoading dataset...")
    raw_dataset = LeRobotDataset(
        repo_id="local",
        root=str(dataset_path),
        delta_timestamps=delta_timestamps,
    )
    print(f"  Dataset length: {len(raw_dataset)}")

    # Pre-cache entire dataset to CPU memory for fast training
    # This eliminates video decoding bottleneck during training
    # Also resize images to image_size x image_size to capture entire scene
    dataset = CachedDataset(raw_dataset, device=None, resize_images_to=image_size)

    # Show a sample
    sample = dataset[0]
    print(f"\nSample keys: {list(sample.keys())}")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__}")

    # Initialize WandB
    print("\nInitializing WandB...")
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "training_steps": training_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_workers": num_workers,
            "dataset_episodes": dataset_metadata.total_episodes,
            "dataset_frames": dataset_metadata.total_frames,
            "dataset_fps": dataset_metadata.fps,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "vision_backbone": cfg.vision_backbone,
            "n_obs_steps": cfg.n_obs_steps,
            "horizon": cfg.horizon,
            "n_action_steps": cfg.n_action_steps,
            "num_train_timesteps": cfg.num_train_timesteps,
            "noise_scheduler_type": cfg.noise_scheduler_type,
            "crop_shape": cfg.crop_shape,
        },
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-6,
    )

    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=training_steps, eta_min=1e-6)

    # Create dataloader - no workers needed since data is pre-cached
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # No workers needed - data is already cached
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,  # Already in CPU memory, no need to pin
        drop_last=True,
    )

    # Training loop
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}")
    print(f"  Steps: {training_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Save frequency: {save_freq}")
    print(f"  WandB project: {wandb_project}")

    step = 0
    best_loss = float('inf')
    running_loss = 0.0
    start_time = time.time()

    # Create infinite dataloader
    def cycle(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iter = cycle(dataloader)

    pbar = tqdm(total=training_steps, desc="Training")

    while step < training_steps:
        batch = next(data_iter)

        # Apply preprocessor (normalization)
        batch = preprocessor(batch)

        # Move batch to device
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # Forward pass
        loss, output_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)

        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        step += 1
        pbar.update(1)

        # Log to WandB every step
        wandb.log({
            "train/loss": loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
        }, step=step)

        # Logging
        if step % log_freq == 0:
            avg_loss = running_loss / log_freq
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            eta_seconds = (training_steps - step) / steps_per_sec
            eta_minutes = eta_seconds / 60

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'eta': f'{eta_minutes:.1f}m'
            })

            # Log averaged metrics to WandB
            wandb.log({
                "train/avg_loss": avg_loss,
                "train/steps_per_sec": steps_per_sec,
                "train/eta_minutes": eta_minutes,
            }, step=step)

            running_loss = 0.0

            # Track best loss
            if avg_loss < best_loss:
                best_loss = avg_loss
                wandb.log({"train/best_loss": best_loss}, step=step)

        # Save checkpoint
        if step % save_freq == 0 or step == training_steps:
            checkpoint_dir = output_directory / f"checkpoint_{step:06d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  Saving checkpoint to {checkpoint_dir}")
            policy.save_pretrained(checkpoint_dir)
            preprocessor.save_pretrained(checkpoint_dir)
            postprocessor.save_pretrained(checkpoint_dir)

            # Save training state
            torch.save({
                'step': step,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
            }, checkpoint_dir / "training_state.pt")

            # Log checkpoint to WandB
            wandb.log({"checkpoint/step": step}, step=step)

    pbar.close()

    # Save final model
    final_dir = output_directory / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {final_dir}")
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Final model: {final_dir}")

    # Log final metrics to WandB
    wandb.log({
        "final/total_time_minutes": elapsed / 60,
        "final/best_loss": best_loss,
    })

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
