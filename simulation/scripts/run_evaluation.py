#!/usr/bin/env python3
"""
Single policy evaluation script for SO100 simulation.

Usage:
    python run_evaluation.py --policy danbhf/act_so100_pick_place --task PickPlaceCube-v0 --episodes 10
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from tqdm import tqdm

from src.env_wrapper import SO100SimEnv
from src.policy_interface import PolicyInterface, RandomPolicy
from src.task_evaluator import TaskEvaluator, TaskStatus
from src.metrics import MetricsCollector


def run_evaluation(
    policy_path: str,
    task: str,
    episodes: int,
    max_steps: int,
    render: bool,
    output_dir: str,
    seed: int,
) -> None:
    """
    Run evaluation of a policy on a task.

    Args:
        policy_path: Path to policy (HuggingFace repo or local)
        task: Task name
        episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        render: Whether to render visualization
        output_dir: Directory for results
        seed: Random seed
    """
    # Set up environment
    print(f"\nSetting up environment: {task}")
    render_mode = "human" if render else "rgb_array"

    try:
        env = SO100SimEnv(
            task=task,
            render_mode=render_mode,
            max_episode_steps=max_steps,
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nMake sure gym-lowcostrobot is installed:")
        print("  pip install git+https://github.com/perezjln/gym-lowcostrobot.git")
        return

    # Load policy
    print(f"\nLoading policy: {policy_path}")
    action_dim = env.action_space.shape[0]
    if policy_path.lower() == "random":
        policy = RandomPolicy(action_dim=action_dim)
    else:
        policy = PolicyInterface(policy_path)

    try:
        policy.load()
    except Exception as e:
        print(f"Error loading policy: {e}")
        return

    print(f"Policy info: {policy.get_policy_info()}")

    # Set up evaluator and metrics
    evaluator = TaskEvaluator(task)
    metrics = MetricsCollector(
        output_dir=output_dir,
        policy_name=Path(policy_path).name,
        task_name=task,
    )

    # Run episodes
    print(f"\nRunning {episodes} episodes...")
    np.random.seed(seed)

    for episode in tqdm(range(episodes), desc="Episodes"):
        # Reset everything
        obs, info = env.reset(seed=seed + episode)
        policy.reset()

        # Get initial positions
        cube_pos = env.get_cube_position()
        target_pos = env.get_target_position()
        evaluator.reset(initial_cube_pos=cube_pos, target_pos=target_pos)

        status = TaskStatus.IN_PROGRESS
        total_reward = 0

        for step in range(max_steps):
            # Get action from policy
            try:
                action = policy.get_action(obs)
            except Exception as e:
                print(f"\nError getting action: {e}")
                status = TaskStatus.FAILURE
                break

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Update evaluator
            cube_pos = env.get_cube_position()
            ee_pos = None  # Would need to extract from env
            # Check last action dimension for gripper (if present)
            gripper_closed = action[-1] < 0 if len(action) >= 1 else False

            status = evaluator.step(
                cube_pos=cube_pos,
                ee_pos=ee_pos,
                gripper_closed=gripper_closed,
                reward=reward,
                info=info,
            )

            # Check termination
            if status == TaskStatus.SUCCESS:
                break
            if terminated or truncated:
                if status == TaskStatus.IN_PROGRESS:
                    status = TaskStatus.TIMEOUT
                break

        # Record episode result
        result = evaluator.get_result(status, cube_pos)
        metrics.record_episode(episode, result)

        if render:
            print(f"\nEpisode {episode}: {status.value} "
                  f"(steps={result.steps_taken}, reward={total_reward:.2f})")

    # Save and print results
    env.close()

    paths = metrics.save_results()
    print(f"\nResults saved to:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    metrics.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a LeRobot policy in SO100 simulation"
    )
    parser.add_argument(
        "--policy", "-p",
        type=str,
        required=True,
        help="Policy path (HuggingFace repo ID or local path). Use 'random' for baseline."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="PickPlaceCube-v0",
        choices=SO100SimEnv.AVAILABLE_TASKS,
        help="Task to evaluate"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Show visualization"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    run_evaluation(
        policy_path=args.policy,
        task=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
