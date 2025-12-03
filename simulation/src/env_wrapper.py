"""
Environment wrapper for SO100 simulation.

Wraps gym-lowcostrobot environments to provide a consistent interface
for policy evaluation and adds features like observation preprocessing.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Optional
import cv2


class SO100SimEnv:
    """
    Wrapper around gym-lowcostrobot environments for SO100 simulation.

    Provides:
    - Consistent observation format matching lerobot policies
    - Camera image preprocessing
    - Action space normalization
    - Episode tracking
    """

    AVAILABLE_TASKS = [
        "PickPlaceCube-v0",
        "LiftCube-v0",
        "PushCube-v0",
        "ReachCube-v0",
        "StackTwoCubes-v0",
    ]

    def __init__(
        self,
        task: str = "PickPlaceCube-v0",
        render_mode: str = "rgb_array",
        camera_width: int = 640,
        camera_height: int = 480,
        max_episode_steps: int = 500,
        randomize_positions: bool = True,
    ):
        """
        Initialize the simulation environment.

        Args:
            task: Task name (e.g., "PickPlaceCube-v0")
            render_mode: "human" for visualization, "rgb_array" for headless
            camera_width: Width of camera images
            camera_height: Height of camera images
            max_episode_steps: Maximum steps per episode
            randomize_positions: Whether to randomize object positions on reset
        """
        if task not in self.AVAILABLE_TASKS:
            raise ValueError(f"Task {task} not available. Choose from: {self.AVAILABLE_TASKS}")

        self.task = task
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_episode_steps = max_episode_steps
        self.randomize_positions = randomize_positions

        # Import gym-lowcostrobot
        try:
            import gym_lowcostrobot
        except ImportError:
            raise ImportError(
                "gym-lowcostrobot not installed. Run:\n"
                "pip install git+https://github.com/perezjln/gym-lowcostrobot.git"
            )

        # Create the environment
        self.env = gym.make(
            task,
            render_mode=render_mode,
            observation_mode="both",  # Get both images and state
        )

        # Get action/state dimensions from environment
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.action_dim  # State matches action dim for joint positions

        # Episode tracking
        self.step_count = 0
        self.episode_count = 0
        self._last_obs = None

    @property
    def observation_space(self) -> gym.spaces.Dict:
        """Return observation space matching lerobot format."""
        return gym.spaces.Dict({
            "observation.images.camera1": gym.spaces.Box(
                low=0, high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8
            ),
            "observation.images.camera2": gym.spaces.Box(
                low=0, high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8
            ),
            "observation.state": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.state_dim,),  # Joint positions
                dtype=np.float32
            ),
        })

    @property
    def action_space(self) -> gym.spaces.Box:
        """Return action space (6 joint positions)."""
        return self.env.action_space

    def reset(self, seed: Optional[int] = None) -> tuple[dict, dict]:
        """
        Reset the environment.

        Returns:
            observation: Dict with images and state
            info: Additional info dict
        """
        obs, info = self.env.reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1

        processed_obs = self._process_observation(obs)
        self._last_obs = processed_obs

        return processed_obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Take a step in the environment.

        Args:
            action: Joint position commands (6 values)

        Returns:
            observation: Dict with images and state
            reward: Reward value
            terminated: Whether episode ended due to success/failure
            truncated: Whether episode ended due to time limit
            info: Additional info dict
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        # Check for max steps
        if self.step_count >= self.max_episode_steps:
            truncated = True

        processed_obs = self._process_observation(obs)
        self._last_obs = processed_obs

        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, obs: dict) -> dict:
        """
        Process raw observation into lerobot-compatible format.

        Args:
            obs: Raw observation from gym environment

        Returns:
            Processed observation dict
        """
        # Handle different observation formats
        if isinstance(obs, dict):
            # Extract images
            if "pixels" in obs:
                # Single camera
                img = obs["pixels"]
                img1 = self._resize_image(img)
                img2 = img1.copy()  # Duplicate if only one camera
            elif "image" in obs:
                img1 = self._resize_image(obs["image"])
                img2 = self._resize_image(obs.get("wrist_image", obs["image"]))
            else:
                # Generate placeholder images
                img1 = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
                img2 = img1.copy()

            # Extract state (joint positions)
            if "agent_pos" in obs:
                state = np.array(obs["agent_pos"], dtype=np.float32)
            elif "qpos" in obs:
                state = np.array(obs["qpos"][:self.state_dim], dtype=np.float32)
            else:
                state = np.zeros(self.state_dim, dtype=np.float32)
        else:
            # Array observation - assume it's state only
            state = np.array(obs[:self.state_dim], dtype=np.float32)
            img1 = self.env.render()
            img1 = self._resize_image(img1)
            img2 = img1.copy()

        return {
            "observation.images.camera1": img1,
            "observation.images.camera2": img2,
            "observation.state": state,
        }

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        if img is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

        if img.shape[:2] != (self.camera_height, self.camera_width):
            img = cv2.resize(img, (self.camera_width, self.camera_height))

        return img.astype(np.uint8)

    def get_cube_position(self) -> Optional[np.ndarray]:
        """Get the current cube position (if available)."""
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'get_body_com'):
            try:
                return self.env.unwrapped.get_body_com("cube")
            except:
                pass
        return None

    def get_target_position(self) -> Optional[np.ndarray]:
        """Get the target position (if available)."""
        if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'goal'):
            return np.array(self.env.unwrapped.goal)
        return None

    def render(self) -> np.ndarray:
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
