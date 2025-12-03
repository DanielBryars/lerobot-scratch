"""
Policy interface for loading and running LeRobot policies in simulation.

Provides a unified interface to load ACT, SmolVLA, and other lerobot policies
and run inference in the simulation environment.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Union
from omegaconf import OmegaConf


class PolicyInterface:
    """
    Interface for loading and running LeRobot policies.

    Supports loading policies from:
    - HuggingFace Hub (e.g., "danbhf/act_so100_pick_place")
    - Local checkpoints (e.g., "./outputs/train/checkpoint_100000")
    """

    def __init__(
        self,
        policy_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        camera_names: Optional[list[str]] = None,
    ):
        """
        Initialize the policy interface.

        Args:
            policy_path: HuggingFace repo ID or local path to policy
            device: Device to run inference on
            camera_names: List of camera names expected by policy
        """
        self.policy_path = policy_path
        self.device = device
        self.camera_names = camera_names or ["camera1", "camera2"]

        self.policy = None
        self.config = None
        self._action_queue = []

    def load(self) -> None:
        """Load the policy from HuggingFace Hub or local path."""
        try:
            from lerobot.policies.factory import make_policy
            from lerobot.configs.policies import PreTrainedConfig
        except ImportError:
            raise ImportError(
                "lerobot not installed. Run: pip install lerobot"
            )

        # Check if it's a local path or HuggingFace repo
        local_path = Path(self.policy_path)
        if local_path.exists():
            pretrained_path = str(local_path)
        else:
            # Assume it's a HuggingFace repo ID
            pretrained_path = self.policy_path

        # Fix Windows path separators for HuggingFace Hub
        pretrained_path = pretrained_path.replace("\\", "/")

        # Load using lerobot factory
        print(f"Loading policy from: {pretrained_path}")

        # Create a minimal config for loading
        config = PreTrainedConfig(pretrained_path=pretrained_path)

        self.policy = make_policy(config, ds_meta=None)
        self.policy.to(self.device)
        self.policy.eval()

        print(f"Policy loaded: {type(self.policy).__name__}")
        print(f"Running on: {self.device}")

    def reset(self) -> None:
        """Reset the policy state for a new episode."""
        self._action_queue = []
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def get_action(self, observation: dict) -> np.ndarray:
        """
        Get action from the policy given an observation.

        Args:
            observation: Dict with keys like:
                - "observation.images.camera1": (H, W, 3) uint8 array
                - "observation.images.camera2": (H, W, 3) uint8 array
                - "observation.state": (6,) float32 array

        Returns:
            action: (6,) array of joint positions
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        # If we have queued actions (for action chunking policies), return next one
        if self._action_queue:
            return self._action_queue.pop(0)

        # Prepare observation for policy
        batch = self._prepare_observation(observation)

        # Run inference
        with torch.no_grad():
            action = self.policy.select_action(batch)

        # Handle action chunking - some policies return multiple timesteps
        if action.dim() > 1 and action.shape[0] > 1:
            # Queue extra actions for subsequent calls
            actions = action.cpu().numpy()
            self._action_queue = list(actions[1:])
            return actions[0]
        else:
            return action.squeeze().cpu().numpy()

    def _prepare_observation(self, observation: dict) -> dict:
        """
        Convert observation dict to policy input format.

        Args:
            observation: Raw observation from environment

        Returns:
            batch: Dict of tensors ready for policy
        """
        batch = {}

        # Process images
        for cam_name in self.camera_names:
            key = f"observation.images.{cam_name}"
            if key in observation:
                img = observation[key]
                # Convert to tensor: (H, W, C) -> (1, C, H, W)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.float() / 255.0  # Normalize to [0, 1]
                batch[key] = img_tensor.to(self.device)

        # Process state
        if "observation.state" in observation:
            state = observation["observation.state"]
            state_tensor = torch.from_numpy(state).unsqueeze(0).float()
            batch["observation.state"] = state_tensor.to(self.device)

        return batch

    def get_policy_info(self) -> dict:
        """Get information about the loaded policy."""
        if self.policy is None:
            return {"loaded": False}

        info = {
            "loaded": True,
            "type": type(self.policy).__name__,
            "device": str(self.device),
            "path": self.policy_path,
        }

        # Add config info if available
        if hasattr(self.policy, 'config'):
            cfg = self.policy.config
            if hasattr(cfg, 'chunk_size'):
                info["chunk_size"] = cfg.chunk_size
            if hasattr(cfg, 'n_action_steps'):
                info["n_action_steps"] = cfg.n_action_steps

        return info


class RandomPolicy:
    """Random policy for baseline comparisons."""

    def __init__(self, action_dim: int = 6, action_range: tuple = (-1.0, 1.0)):
        self.action_dim = action_dim
        self.action_range = action_range

    def load(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_action(self, observation: dict) -> np.ndarray:
        """Return random action."""
        low, high = self.action_range
        return np.random.uniform(low, high, size=self.action_dim).astype(np.float32)

    def get_policy_info(self) -> dict:
        return {
            "loaded": True,
            "type": "RandomPolicy",
            "action_dim": self.action_dim,
        }


class ReplayPolicy:
    """
    Policy that replays actions from a recorded dataset.
    Useful for testing the simulation matches real-world behavior.
    """

    def __init__(self, dataset_path: str, episode_index: int = 0):
        self.dataset_path = dataset_path
        self.episode_index = episode_index
        self.actions = None
        self.step_idx = 0

    def load(self) -> None:
        """Load actions from dataset."""
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError("lerobot not installed")

        dataset = LeRobotDataset(self.dataset_path)

        # Get actions for specified episode
        episode_actions = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample.get("episode_index", 0) == self.episode_index:
                episode_actions.append(sample["action"].numpy())

        self.actions = episode_actions
        print(f"Loaded {len(self.actions)} actions from episode {self.episode_index}")

    def reset(self) -> None:
        self.step_idx = 0

    def get_action(self, observation: dict) -> np.ndarray:
        if self.actions is None:
            raise RuntimeError("Actions not loaded. Call load() first.")

        if self.step_idx >= len(self.actions):
            # Repeat last action if we run out
            return self.actions[-1]

        action = self.actions[self.step_idx]
        self.step_idx += 1
        return action

    def get_policy_info(self) -> dict:
        return {
            "loaded": self.actions is not None,
            "type": "ReplayPolicy",
            "dataset": self.dataset_path,
            "episode": self.episode_index,
            "total_actions": len(self.actions) if self.actions else 0,
        }
