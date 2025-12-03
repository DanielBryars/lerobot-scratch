#!/usr/bin/env python3
"""
Interactive visualization for SO100 simulation.

Run a policy with real-time rendering to see how it performs.

Usage:
    python visualize.py --policy danbhf/act_so100_pick_place --task PickPlaceCube-v0
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.env_wrapper import SO100SimEnv
from src.policy_interface import PolicyInterface, RandomPolicy
from src.task_evaluator import TaskEvaluator, TaskStatus


def run_visualization(
    policy_path: str,
    task: str,
    episodes: int,
    max_steps: int,
    fps: int,
    seed: int,
) -> None:
    """
    Run interactive visualization of a policy.

    Args:
        policy_path: Path to policy
        task: Task name
        episodes: Number of episodes
        max_steps: Max steps per episode
        fps: Target frames per second
        seed: Random seed
    """
    # Create environment with human rendering
    print(f"\nSetting up visualization: {task}")

    try:
        env = SO100SimEnv(
            task=task,
            render_mode="human",
            max_episode_steps=max_steps,
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("\nMake sure gym-lowcostrobot is installed:")
        print("  pip install git+https://github.com/perezjln/gym-lowcostrobot.git")
        return

    # Load policy
    print(f"Loading policy: {policy_path}")
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

    # Set up evaluator
    evaluator = TaskEvaluator(task)

    # Run episodes
    frame_time = 1.0 / fps
    np.random.seed(seed)

    print(f"\nRunning {episodes} episodes (press Ctrl+C to stop)...")
    print("-" * 50)

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            policy.reset()

            cube_pos = env.get_cube_position()
            target_pos = env.get_target_position()
            evaluator.reset(initial_cube_pos=cube_pos, target_pos=target_pos)

            print(f"\nEpisode {episode + 1}/{episodes}")
            status = TaskStatus.IN_PROGRESS
            total_reward = 0

            for step in range(max_steps):
                step_start = time.time()

                # Get and execute action
                try:
                    action = policy.get_action(obs)
                except Exception as e:
                    print(f"Error getting action: {e}")
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                # Update evaluator
                cube_pos = env.get_cube_position()
                gripper_closed = action[-1] < 0 if len(action) >= 1 else False

                status = evaluator.step(
                    cube_pos=cube_pos,
                    ee_pos=None,
                    gripper_closed=gripper_closed,
                    reward=reward,
                    info=info,
                )

                # Render (automatic with human mode)
                env.render()

                # Print status periodically
                if step % 50 == 0:
                    status_str = f"Step {step:4d} | Reward: {total_reward:6.2f}"
                    if cube_pos is not None and target_pos is not None:
                        dist = np.linalg.norm(cube_pos[:2] - target_pos[:2])
                        status_str += f" | Dist: {dist:.3f}m"
                    print(f"\r{status_str}", end="", flush=True)

                # Check termination
                if status == TaskStatus.SUCCESS:
                    print(f"\n  SUCCESS at step {step}!")
                    break
                if terminated or truncated:
                    break

                # Frame rate control
                elapsed = time.time() - step_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            # Episode summary
            result = evaluator.get_result(status, cube_pos)
            print(f"\n  Result: {status.value}")
            print(f"  Steps: {result.steps_taken}, Time: {result.completion_time:.2f}s")
            print(f"  Total Reward: {total_reward:.2f}")
            if result.final_distance is not None:
                print(f"  Final Distance: {result.final_distance:.3f}m")

            # Pause between episodes
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        env.close()
        print("\nVisualization complete")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a policy running in SO100 simulation"
    )
    parser.add_argument(
        "--policy", "-p",
        type=str,
        required=True,
        help="Policy path (HuggingFace repo ID or local). Use 'random' for baseline."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="PickPlaceCube-v0",
        choices=SO100SimEnv.AVAILABLE_TASKS,
        help="Task to run"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes"
    )
    parser.add_argument(
        "--max-steps", "-s",
        type=int,
        default=500,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target frames per second"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    run_visualization(
        policy_path=args.policy,
        task=args.task,
        episodes=args.episodes,
        max_steps=args.max_steps,
        fps=args.fps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
