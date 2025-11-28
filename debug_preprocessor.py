#!/usr/bin/env python3
"""
Debug script to test the diffusion policy inference without robot/camera.
"""

import torch
import numpy as np
from pathlib import Path

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

def main():
    model_path = "outputs/diffusion_so100/final"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("Diffusion Policy Inference Debug Script")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    policy = DiffusionPolicy.from_pretrained(model_path)
    policy = policy.to(device)
    policy.eval()
    print(f"[OK] Model loaded")
    print(f"  n_obs_steps: {policy.config.n_obs_steps}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")
    print(f"  crop_shape: {policy.config.crop_shape}")

    # Check image features from config
    print(f"\nImage features from config:")
    image_features = policy.config.image_features
    if image_features:
        for k, v in image_features.items():
            print(f"  {k}: shape={v.shape}")
    else:
        print("  No image features!")
        return 1

    # Load postprocessor
    print("\nLoading postprocessor...")
    _, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=model_path,
    )
    print(f"[OK] Postprocessor loaded")

    # Create observation WITH batch dimension
    crop_h, crop_w = policy.config.crop_shape

    print(f"\nCreating observation with batch dim=1, cropped to {crop_h}x{crop_w}...")

    # State: [batch, state_dim] = [1, 6]
    fake_state = torch.randn(1, 6, dtype=torch.float32, device=device)

    # Images: [batch, C, H, W] = [1, 3, 84, 84]
    fake_base_img = torch.rand(1, 3, crop_h, crop_w, dtype=torch.float32, device=device)
    fake_wrist_img = torch.rand(1, 3, crop_h, crop_w, dtype=torch.float32, device=device)

    batch = {
        OBS_STATE: fake_state,
        "observation.images.base_0_rgb": fake_base_img,
        "observation.images.left_wrist_0_rgb": fake_wrist_img,
    }

    print("\nBatch format (batch dim=1):")
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")

    # Trace what happens on line 129 of select_action:
    # batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
    print("\nTracing image stacking (line 129):")
    img_list = [batch[key] for key in policy.config.image_features]
    print(f"  Image tensors to stack: {[img.shape for img in img_list]}")

    stacked = torch.stack(img_list, dim=-4)
    print(f"  After torch.stack(..., dim=-4): {stacked.shape}")
    # Expected: [batch, n_cameras, C, H, W] = [1, 2, 3, 84, 84]

    # Test policy.select_action
    print("\n" + "=" * 70)
    print("Testing policy.select_action()...")
    print("=" * 70)

    try:
        policy.reset()

        # Call select_action multiple times
        print("\nCalling select_action to fill observation queue...")
        for step in range(3):
            with torch.no_grad():
                action = policy.select_action(batch)
            print(f"  Step {step+1}: action shape={action.shape}, sample={action[:3].cpu().numpy()}")

        print(f"\n[OK] Action output:")
        print(f"  Type: {type(action)}")
        print(f"  Shape: {action.shape}")
        print(f"  Full action: {action.cpu().numpy()}")

    except Exception as e:
        print(f"\n[ERROR] select_action failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test postprocessor
    print("\n" + "=" * 70)
    print("Testing postprocessor...")
    print("=" * 70)

    try:
        action_post = postprocessor(action)

        print(f"\n[OK] Postprocessed action:")
        print(f"  Type: {type(action_post)}")
        if hasattr(action_post, 'shape'):
            print(f"  Shape: {action_post.shape}")
        if hasattr(action_post, 'action'):
            print(f"  .action shape: {action_post.action.shape}")
            print(f"  .action values: {action_post.action.cpu().numpy()}")

    except Exception as e:
        print(f"\n[ERROR] Postprocessor failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print("\nKey insights for inference script:")
    print(f"  1. Include batch dimension: shape [1, ...] not [...]")
    print(f"  2. State: [1, 6], Images: [1, 3, {crop_h}, {crop_w}]")
    print(f"  3. Use separate image keys: 'observation.images.base_0_rgb', etc.")
    print(f"  4. Images cropped to {crop_h}x{crop_w}, normalized to [0, 1]")
    print(f"  5. Call policy.reset() at start of each episode")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
