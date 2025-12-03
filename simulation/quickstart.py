#!/usr/bin/env python3
"""
Quick start script to verify SO100 simulation is working.

This script tests the basic components:
1. Environment creation
2. Random policy baseline
3. Metrics collection

Usage:
    cd simulation
    python quickstart.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    print("=" * 60)
    print("SO100 Simulation - Quick Start Test")
    print("=" * 60)

    # Test 1: Check dependencies
    print("\n[1/4] Checking dependencies...")

    missing = []
    try:
        import numpy as np
        print(f"  numpy: {np.__version__}")
    except ImportError:
        missing.append("numpy")

    try:
        import gymnasium as gym
        print(f"  gymnasium: {gym.__version__}")
    except ImportError:
        missing.append("gymnasium")

    try:
        import cv2
        print(f"  opencv: {cv2.__version__}")
    except ImportError:
        missing.append("opencv-python")

    try:
        import mujoco
        print(f"  mujoco: {mujoco.__version__}")
    except ImportError:
        missing.append("mujoco")

    try:
        import gym_lowcostrobot
        print("  gym-lowcostrobot: installed")
    except ImportError:
        missing.append("gym-lowcostrobot (pip install git+https://github.com/perezjln/gym-lowcostrobot.git)")

    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print("  Please install missing packages and try again.")
        return False

    # Test 2: Create environment
    print("\n[2/4] Creating environment...")
    try:
        from src.env_wrapper import SO100SimEnv

        env = SO100SimEnv(
            task="ReachCube-v0",
            render_mode="rgb_array",
            max_episode_steps=100,
        )
        print(f"  Task: ReachCube-v0")
        print(f"  Action space: {env.action_space}")
        print("  Environment created successfully!")
    except Exception as e:
        print(f"  Error: {e}")
        return False

    # Test 3: Run random episode
    print("\n[3/4] Running random episode...")
    try:
        from src.policy_interface import RandomPolicy
        from src.task_evaluator import TaskEvaluator, TaskStatus

        # Get action dimension from environment
        action_dim = env.action_space.shape[0]
        print(f"  Action dimension: {action_dim}")
        policy = RandomPolicy(action_dim=action_dim)
        evaluator = TaskEvaluator("ReachCube-v0")

        obs, info = env.reset(seed=42)
        evaluator.reset()

        print(f"  Observation keys: {list(obs.keys())}")
        print(f"  Image shape: {obs['observation.images.camera1'].shape}")
        print(f"  State shape: {obs['observation.state'].shape}")

        total_reward = 0
        for step in range(50):
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            status = evaluator.step(reward=reward, info=info)
            if terminated or truncated:
                break

        result = evaluator.get_result(status)
        print(f"  Steps taken: {result.steps_taken}")
        print(f"  Total reward: {total_reward:.3f}")
        print("  Random episode completed!")

    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Metrics collection
    print("\n[4/4] Testing metrics collection...")
    try:
        from src.metrics import MetricsCollector

        metrics = MetricsCollector(
            output_dir="results",
            policy_name="random",
            task_name="ReachCube-v0",
        )
        metrics.record_episode(0, result)

        summary = metrics.get_aggregated_metrics()
        print(f"  Episodes recorded: {summary.total_episodes}")
        print(f"  Success rate: {summary.success_rate:.1%}")
        print("  Metrics collection working!")

    except Exception as e:
        print(f"  Error: {e}")
        return False

    # Cleanup
    env.close()

    print("\n" + "=" * 60)
    print("All tests passed! Simulation is ready to use.")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Run a policy evaluation:")
    print("     python scripts/run_evaluation.py --policy random --task PickPlaceCube-v0 --episodes 10")
    print()
    print("  2. Visualize a policy:")
    print("     python scripts/visualize.py --policy random --task PickPlaceCube-v0")
    print()
    print("  3. Run full benchmark:")
    print("     python scripts/benchmark_runner.py --config configs/benchmark_config.yaml")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
