"""
Utility functions for SO100 simulation.
"""

import numpy as np
from pathlib import Path
from typing import Optional


def normalize_action(action: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """
    Normalize action to specified range.

    Args:
        action: Raw action array
        low: Lower bound of output range
        high: Upper bound of output range

    Returns:
        Normalized action
    """
    # Assume input is in some default range, normalize to [low, high]
    action = np.clip(action, -np.pi, np.pi)  # Clip to reasonable joint range
    normalized = (action + np.pi) / (2 * np.pi)  # [0, 1]
    return normalized * (high - low) + low


def denormalize_action(action: np.ndarray, low: float = -1.0, high: float = 1.0) -> np.ndarray:
    """
    Denormalize action from [low, high] to joint range.

    Args:
        action: Normalized action in [low, high]
        low: Lower bound of input range
        high: Upper bound of input range

    Returns:
        Denormalized action in joint space
    """
    normalized = (action - low) / (high - low)  # [0, 1]
    return normalized * (2 * np.pi) - np.pi  # [-pi, pi]


def compute_optimal_path_length(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    via_height: Optional[float] = None,
) -> float:
    """
    Compute optimal (straight-line) path length for a pick-and-place task.

    Args:
        start_pos: Starting end-effector position
        target_pos: Target placement position
        via_height: If specified, compute path with lift to this height

    Returns:
        Optimal path length in meters
    """
    if via_height is None:
        # Direct path
        return np.linalg.norm(target_pos - start_pos)

    # Path with lift: down to object, up, across, down to target
    # Simplified: start -> above start -> above target -> target
    above_start = start_pos.copy()
    above_start[2] = via_height

    above_target = target_pos.copy()
    above_target[2] = via_height

    path_length = (
        np.linalg.norm(above_start - start_pos) +  # Go up
        np.linalg.norm(above_target - above_start) +  # Go across
        np.linalg.norm(target_pos - above_target)  # Go down
    )

    return path_length


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.1f}s"


def get_latest_checkpoint(output_dir: str) -> Optional[Path]:
    """
    Find the latest checkpoint in an output directory.

    Args:
        output_dir: Directory containing checkpoint folders

    Returns:
        Path to latest checkpoint, or None if not found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Look for checkpoint directories
    checkpoints = list(output_path.glob("checkpoint_*"))
    if not checkpoints:
        return None

    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def ensure_directory(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_config(path: str) -> dict:
    """Load a YAML configuration file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class RateLimiter:
    """Simple rate limiter for controlling loop frequency."""

    def __init__(self, fps: float):
        """
        Initialize rate limiter.

        Args:
            fps: Target frames per second
        """
        import time
        self.period = 1.0 / fps
        self.last_time = time.time()
        self._time = time

    def wait(self) -> None:
        """Wait until next frame time."""
        current = self._time.time()
        elapsed = current - self.last_time
        if elapsed < self.period:
            self._time.sleep(self.period - elapsed)
        self.last_time = self._time.time()
