#!/usr/bin/env python3
"""
Batch benchmark runner for comparing multiple policies.

Usage:
    python benchmark_runner.py --config configs/benchmark_config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.env_wrapper import SO100SimEnv
from src.policy_interface import PolicyInterface, RandomPolicy
from src.task_evaluator import TaskEvaluator, TaskStatus
from src.metrics import MetricsCollector, ComparisonReport, plot_success_rates


def load_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_benchmark(config: dict, output_dir: str) -> None:
    """
    Run full benchmark suite.

    Args:
        config: Benchmark configuration dict
        output_dir: Directory for results
    """
    tasks = config.get("tasks", [])
    policies = config.get("policies", [])
    global_settings = config.get("settings", {})

    if not tasks:
        print("No tasks specified in config")
        return

    if not policies:
        print("No policies specified in config")
        return

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create comparison report
    comparison = ComparisonReport(output_dir=str(run_dir))

    # Run all policy/task combinations
    total_combinations = len(policies) * len(tasks)
    print(f"\nRunning benchmark: {len(policies)} policies x {len(tasks)} tasks")
    print(f"Total combinations: {total_combinations}")
    print(f"Results will be saved to: {run_dir}\n")

    for policy_cfg in policies:
        policy_name = policy_cfg["name"]
        policy_type = policy_cfg.get("type", "auto")

        print(f"\n{'='*60}")
        print(f"Policy: {policy_name}")
        print(f"{'='*60}")

        # Policy will be loaded per-task for random (to get correct action dim)
        is_random = policy_name.lower() == "random"
        policy = None

        if not is_random:
            policy = PolicyInterface(policy_name)
            try:
                policy.load()
            except Exception as e:
                print(f"Error loading policy {policy_name}: {e}")
                continue

        for task_cfg in tasks:
            task_name = task_cfg["name"]
            episodes = task_cfg.get("episodes", global_settings.get("episodes", 100))
            max_steps = task_cfg.get("max_steps", global_settings.get("max_steps", 500))
            seed = task_cfg.get("seed", global_settings.get("seed", 42))

            print(f"\n  Task: {task_name}")
            print(f"  Episodes: {episodes}, Max steps: {max_steps}")

            # Create environment
            try:
                env = SO100SimEnv(
                    task=task_name,
                    render_mode="rgb_array",
                    max_episode_steps=max_steps,
                )
            except Exception as e:
                print(f"  Error creating environment: {e}")
                continue

            # Create random policy with correct action dim (if needed)
            if is_random:
                action_dim = env.action_space.shape[0]
                policy = RandomPolicy(action_dim=action_dim)
                policy.load()

            # Set up evaluator and metrics
            evaluator = TaskEvaluator(task_name)
            metrics = MetricsCollector(
                output_dir=str(run_dir),
                policy_name=Path(policy_name).name,
                task_name=task_name,
            )

            # Run episodes
            np.random.seed(seed)

            for episode in tqdm(range(episodes), desc=f"  {task_name}", leave=False):
                obs, info = env.reset(seed=seed + episode)
                policy.reset()

                cube_pos = env.get_cube_position()
                target_pos = env.get_target_position()
                evaluator.reset(initial_cube_pos=cube_pos, target_pos=target_pos)

                status = TaskStatus.IN_PROGRESS

                for step in range(max_steps):
                    try:
                        action = policy.get_action(obs)
                    except Exception as e:
                        status = TaskStatus.FAILURE
                        break

                    obs, reward, terminated, truncated, info = env.step(action)

                    cube_pos = env.get_cube_position()
                    gripper_closed = action[-1] < 0 if len(action) >= 1 else False

                    status = evaluator.step(
                        cube_pos=cube_pos,
                        ee_pos=None,
                        gripper_closed=gripper_closed,
                        reward=reward,
                        info=info,
                    )

                    if status == TaskStatus.SUCCESS:
                        break
                    if terminated or truncated:
                        if status == TaskStatus.IN_PROGRESS:
                            status = TaskStatus.TIMEOUT
                        break

                result = evaluator.get_result(status, cube_pos)
                metrics.record_episode(episode, result)

            # Save results for this policy/task
            env.close()
            metrics.save_results()

            # Add to comparison
            aggregated = metrics.get_aggregated_metrics()
            comparison.add_result(aggregated)

            # Print quick summary
            print(f"    Success rate: {aggregated.success_rate:.1%} "
                  f"({aggregated.success_count}/{aggregated.total_episodes})")

    # Generate comparison report
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    comparison.print_comparison()
    comparison_path = comparison.save_comparison("comparison")
    print(f"\nComparison saved to: {comparison_path}")

    # Generate plots
    try:
        plot_path = run_dir / "success_rates.png"
        plot_success_rates(comparison.results, str(plot_path))
    except Exception as e:
        print(f"Could not generate plot: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark comparing multiple policies"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to benchmark config YAML"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    run_benchmark(config, args.output)


if __name__ == "__main__":
    main()
