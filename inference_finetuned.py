#!/usr/bin/env python3
"""
Run inference with a fine-tuned Pi0 model.

Usage:
    python inference_finetuned.py --model-path ./outputs/pi0_finetuned --task "pick and place the cube"

This script runs the fine-tuned Pi0 model on your SO100 robot.
"""

import argparse
import torch
import json
import time
import sys
from pathlib import Path
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned Pi0 model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory (e.g., ./outputs/pi0_finetuned)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description (e.g., 'pick and place the cube')"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=100,
        help="Maximum steps per episode (default: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Pi0 Inference with Fine-tuned Model")
    print("=" * 70)
    print()
    print(f"Model: {args.model_path}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.steps_per_episode}")
    print(f"Device: {args.device}")
    print()

    # Check if CUDA is available
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    # Load hardware configuration
    config = load_config()

    # Camera configuration
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"]
        )
        for name, cam in config["cameras"].items()
    }

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    try:
        model = PI0Policy.from_pretrained(args.model_path)
        model = model.to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nMake sure you've run training first:")
        print(f"  python train_pi0.py --dataset-repo-id your/dataset --output-dir {args.model_path}")
        return 1

    # Remove right_wrist camera if not available (Pi0 will pad it)
    if "observation.images.right_wrist_0_rgb" in model.config.input_features:
        del model.config.input_features["observation.images.right_wrist_0_rgb"]

    # Create preprocessor and postprocessor
    print("Setting up preprocessors...")
    preprocess, postprocess = make_pre_post_processors(
        model.config,
        args.model_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("✓ Preprocessors ready")

    # Connect to robot
    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]
    robot_cfg = SO100FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_config
    )
    robot = SO100FollowerSTS3250(robot_cfg)

    print(f"\nConnecting to robot at {follower_port}...")
    try:
        robot.connect()
        print("✓ Robot connected")
    except Exception as e:
        print(f"✗ Failed to connect to robot: {e}")
        return 1

    # Warmup cameras
    print("\nWarming up cameras...")
    for i in range(5):
        try:
            _ = robot.get_observation()
            print(f"  Warmup frame {i+1}/5")
            time.sleep(0.1)
        except Exception as e:
            print(f"  Warning: Warmup frame {i+1}/5 failed: {e}")
            time.sleep(0.2)
    print("✓ Cameras ready")

    # Prepare dataset features
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    robot_type = "so100_follower"

    print("\n" + "=" * 70)
    print("Starting Inference")
    print("=" * 70)
    print("\nPress Ctrl+C to stop at any time")
    print()

    try:
        for ep in range(args.num_episodes):
            print(f"\n[Episode {ep + 1}/{args.num_episodes}]")
            print("Press ENTER to start episode (or Ctrl+C to quit)...")
            input()

            print(f"🤖 Running episode {ep + 1}...")

            for step in range(args.steps_per_episode):
                step_start = time.time()

                # Get observation from robot
                obs = robot.get_observation()

                # Build frame for inference
                obs_frame = build_inference_frame(
                    observation=obs,
                    ds_features=dataset_features,
                    device=device,
                    task=args.task,
                    robot_type=robot_type
                )

                # Preprocess observation
                obs_preprocessed = preprocess(obs_frame)

                # Get action from policy
                with torch.no_grad():
                    action = model.select_action(obs_preprocessed)

                # Postprocess action
                action = postprocess(action)
                action = make_robot_action(action, dataset_features)

                # Send action to robot
                robot.send_action(action)

                # Print progress
                if step % 10 == 0:
                    print(f"  Step {step + 1}/{args.steps_per_episode}")

                # Maintain reasonable fps
                elapsed = time.time() - step_start
                sleep_time = max(0, 0.033 - elapsed)  # ~30 Hz
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print(f"✓ Episode {ep + 1} complete!")

        print("\n" + "=" * 70)
        print("Inference Complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\n⏸️  Inference stopped by user")

    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        print("\nDisconnecting robot...")
        robot.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
