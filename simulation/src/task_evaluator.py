"""
Task evaluator for detecting success conditions in SO100 simulation.

Provides success detection for various manipulation tasks like
pick-and-place, lifting, pushing, and reaching.
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Status of task execution."""
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class TaskResult:
    """Result of a task evaluation."""
    status: TaskStatus
    success: bool
    steps_taken: int
    completion_time: float  # seconds
    final_distance: Optional[float] = None
    grasp_achieved: bool = False
    placement_error: Optional[float] = None
    path_length: float = 0.0
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class TaskEvaluator:
    """
    Evaluates task success for manipulation tasks.

    Supports multiple task types with configurable success thresholds.
    """

    def __init__(
        self,
        task: str,
        success_threshold: float = 0.05,  # 5cm default
        height_threshold: float = 0.1,    # 10cm for lift tasks
        grasp_threshold: float = 0.02,    # 2cm for grasp detection
        timestep: float = 0.02,           # 20ms per step (50Hz)
    ):
        """
        Initialize the task evaluator.

        Args:
            task: Task name (e.g., "PickPlaceCube-v0")
            success_threshold: Distance threshold for success (meters)
            height_threshold: Height threshold for lift tasks
            grasp_threshold: Distance for grasp detection
            timestep: Simulation timestep in seconds
        """
        self.task = task
        self.success_threshold = success_threshold
        self.height_threshold = height_threshold
        self.grasp_threshold = grasp_threshold
        self.timestep = timestep

        # Episode tracking
        self.step_count = 0
        self.path_length = 0.0
        self.last_ee_pos = None
        self.grasp_achieved = False
        self.initial_cube_pos = None
        self.target_pos = None

        # Get task-specific evaluator
        self._evaluate_fn = self._get_evaluator(task)

    def _get_evaluator(self, task: str) -> Callable:
        """Get the appropriate evaluation function for a task."""
        evaluators = {
            "PickPlaceCube-v0": self._evaluate_pick_place,
            "LiftCube-v0": self._evaluate_lift,
            "PushCube-v0": self._evaluate_push,
            "ReachCube-v0": self._evaluate_reach,
            "StackTwoCubes-v0": self._evaluate_stack,
        }
        return evaluators.get(task, self._evaluate_generic)

    def reset(
        self,
        initial_cube_pos: Optional[np.ndarray] = None,
        target_pos: Optional[np.ndarray] = None
    ) -> None:
        """Reset the evaluator for a new episode."""
        self.step_count = 0
        self.path_length = 0.0
        self.last_ee_pos = None
        self.grasp_achieved = False
        self.initial_cube_pos = initial_cube_pos
        self.target_pos = target_pos

    def step(
        self,
        cube_pos: Optional[np.ndarray] = None,
        ee_pos: Optional[np.ndarray] = None,
        gripper_closed: bool = False,
        reward: float = 0.0,
        info: Optional[dict] = None,
    ) -> TaskStatus:
        """
        Evaluate a single step of the task.

        Args:
            cube_pos: Current cube position [x, y, z]
            ee_pos: Current end-effector position [x, y, z]
            gripper_closed: Whether gripper is closed
            reward: Reward from environment
            info: Additional info from environment

        Returns:
            Current task status
        """
        self.step_count += 1

        # Track path length
        if ee_pos is not None:
            if self.last_ee_pos is not None:
                self.path_length += np.linalg.norm(ee_pos - self.last_ee_pos)
            self.last_ee_pos = ee_pos.copy()

        # Check for grasp
        if cube_pos is not None and ee_pos is not None:
            dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
            if dist_to_cube < self.grasp_threshold and gripper_closed:
                self.grasp_achieved = True

        # Run task-specific evaluation
        return self._evaluate_fn(cube_pos, ee_pos, gripper_closed, reward, info)

    def get_result(
        self,
        status: TaskStatus,
        cube_pos: Optional[np.ndarray] = None,
    ) -> TaskResult:
        """
        Get the final result of the episode.

        Args:
            status: Final task status
            cube_pos: Final cube position

        Returns:
            TaskResult with all metrics
        """
        completion_time = self.step_count * self.timestep

        # Calculate final distance to target
        final_distance = None
        if cube_pos is not None and self.target_pos is not None:
            final_distance = np.linalg.norm(cube_pos - self.target_pos)

        # Calculate placement error for pick-place tasks
        placement_error = None
        if status == TaskStatus.SUCCESS and final_distance is not None:
            placement_error = final_distance

        return TaskResult(
            status=status,
            success=(status == TaskStatus.SUCCESS),
            steps_taken=self.step_count,
            completion_time=completion_time,
            final_distance=final_distance,
            grasp_achieved=self.grasp_achieved,
            placement_error=placement_error,
            path_length=self.path_length,
            info={
                "task": self.task,
                "threshold": self.success_threshold,
            }
        )

    def _evaluate_pick_place(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Evaluate pick-and-place task."""
        if cube_pos is None or self.target_pos is None:
            return TaskStatus.IN_PROGRESS

        dist_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])  # XY distance

        # Success: cube is at target location
        if dist_to_target < self.success_threshold:
            return TaskStatus.SUCCESS

        return TaskStatus.IN_PROGRESS

    def _evaluate_lift(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Evaluate lift task."""
        if cube_pos is None or self.initial_cube_pos is None:
            return TaskStatus.IN_PROGRESS

        # Success: cube lifted above threshold
        height_gain = cube_pos[2] - self.initial_cube_pos[2]
        if height_gain > self.height_threshold:
            return TaskStatus.SUCCESS

        return TaskStatus.IN_PROGRESS

    def _evaluate_push(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Evaluate push task."""
        if cube_pos is None or self.target_pos is None:
            return TaskStatus.IN_PROGRESS

        # Success: cube pushed to target (XY only)
        dist_to_target = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        if dist_to_target < self.success_threshold:
            return TaskStatus.SUCCESS

        return TaskStatus.IN_PROGRESS

    def _evaluate_reach(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Evaluate reach task."""
        if ee_pos is None or self.target_pos is None:
            return TaskStatus.IN_PROGRESS

        # Success: end-effector reached target
        dist_to_target = np.linalg.norm(ee_pos - self.target_pos)
        if dist_to_target < self.success_threshold:
            return TaskStatus.SUCCESS

        return TaskStatus.IN_PROGRESS

    def _evaluate_stack(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Evaluate stack task (simplified - uses reward signal)."""
        # Stacking is complex - rely on environment reward
        if info and info.get("is_success", False):
            return TaskStatus.SUCCESS
        if reward > 0.9:  # High reward indicates success
            return TaskStatus.SUCCESS
        return TaskStatus.IN_PROGRESS

    def _evaluate_generic(
        self,
        cube_pos: Optional[np.ndarray],
        ee_pos: Optional[np.ndarray],
        gripper_closed: bool,
        reward: float,
        info: Optional[dict],
    ) -> TaskStatus:
        """Generic evaluation using reward signal."""
        if info and info.get("is_success", False):
            return TaskStatus.SUCCESS
        return TaskStatus.IN_PROGRESS


class MultiTaskEvaluator:
    """
    Evaluator for multi-task scenarios.

    Tracks success across multiple task types in a single session.
    """

    def __init__(self, tasks: list[str], **kwargs):
        """
        Initialize multi-task evaluator.

        Args:
            tasks: List of task names
            **kwargs: Arguments passed to individual TaskEvaluators
        """
        self.evaluators = {
            task: TaskEvaluator(task, **kwargs)
            for task in tasks
        }
        self.current_task = None
        self.results_by_task = {task: [] for task in tasks}

    def set_task(self, task: str) -> None:
        """Set the current task for evaluation."""
        if task not in self.evaluators:
            raise ValueError(f"Unknown task: {task}")
        self.current_task = task

    def get_evaluator(self) -> TaskEvaluator:
        """Get the evaluator for the current task."""
        if self.current_task is None:
            raise RuntimeError("No task set. Call set_task() first.")
        return self.evaluators[self.current_task]

    def record_result(self, result: TaskResult) -> None:
        """Record a result for the current task."""
        if self.current_task:
            self.results_by_task[self.current_task].append(result)

    def get_summary(self) -> dict:
        """Get summary statistics for all tasks."""
        summary = {}
        for task, results in self.results_by_task.items():
            if not results:
                continue

            successes = sum(1 for r in results if r.success)
            total = len(results)

            summary[task] = {
                "total_episodes": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0,
                "avg_completion_time": np.mean([r.completion_time for r in results if r.success]) if successes > 0 else None,
                "avg_steps": np.mean([r.steps_taken for r in results]),
            }

        return summary
