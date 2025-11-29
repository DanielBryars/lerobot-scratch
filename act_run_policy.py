#!/usr/bin/env python3
"""
Run trained ACT policy on SO-100 robot.

This script loads the trained policy and runs it interactively.
Since ACT is a behavior cloning model, it replicates learned behaviors
based on visual observations - it doesn't understand language commands.

Usage:
    python run_policy.py

Controls:
    - Press Enter to start/pause policy execution
    - Press 'q' then Enter to quit
"""

import os
import sys
import json
import time
import threading
from pathlib import Path

# Fix Windows console encoding
os.environ['PYTHONUTF8'] = '1'

import torch
import numpy as np


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("ACT Policy Runner for SO-100")
    print("=" * 70)
    print()

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("Using CPU (no CUDA available)")
        device = "cpu"
    print()

    # Load robot configuration
    config = load_config()

    # Import robot and camera modules
    print("Loading modules...")
    import cv2
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from SO100FollowerSTS3250 import SO100FollowerSTS3250

    # Load the trained policy
    print("Loading trained ACT policy from HuggingFace...")
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy = ACTPolicy.from_pretrained("danbhf/act_so100_trained")
    policy.to(device)
    policy.eval()
    print("[OK] Policy loaded successfully!")
    print()

    # Setup cameras using OpenCV directly (more reliable on Windows)
    print("Setting up cameras...")
    cameras = {}
    for cam_name, cam_cfg in config["cameras"].items():
        print(f"  Connecting to {cam_name} (index {cam_cfg['index_or_path']})...")
        cap = cv2.VideoCapture(cam_cfg["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])
        if not cap.isOpened():
            print(f"  [WARNING] Failed to open camera {cam_cfg['index_or_path']}")
            continue
        cameras[cam_name] = cap
        print(f"  [OK] {cam_name} connected")
    print()

    # Setup follower robot (without cameras - we handle them separately)
    print(f"Connecting to follower robot on {config['follower']['port']}...")

    follower_cfg = SO100FollowerConfig(
        port=config["follower"]["port"],
        id=config["follower"]["id"],
        cameras={}  # No cameras - we handle them separately with OpenCV
    )
    robot = SO100FollowerSTS3250(follower_cfg)
    robot.connect()
    print("[OK] Robot connected!")
    print()

    # Joint names for the policy
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("=" * 70)
    print("Ready to run policy!")
    print("=" * 70)
    print()
    print("Controls:")
    print("  - Press Enter to START/PAUSE policy execution")
    print("  - Type 'q' and press Enter to QUIT")
    print()
    print("The policy will execute the pick-and-place behavior it learned")
    print("from your demonstrations when you press Enter.")
    print()

    running = False
    should_quit = False

    def input_thread():
        nonlocal running, should_quit
        while not should_quit:
            try:
                cmd = input()
                if cmd.lower() == 'q':
                    should_quit = True
                    print("\nQuitting...")
                else:
                    running = not running
                    if running:
                        print("\n>>> Policy STARTED - Robot is now autonomous <<<")
                    else:
                        print("\n>>> Policy PAUSED <<<")
            except EOFError:
                break

    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()

    step = 0
    try:
        while not should_quit:
            if running:
                # Get current robot state
                obs = robot.get_observation()

                # Get camera images
                images = {}
                for cam_name, cap in cameras.items():
                    ret, frame = cap.read()
                    if not ret:
                        print(f"Warning: Failed to read from {cam_name}")
                        continue
                    # Convert BGR to RGB and to tensor format [C, H, W]
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                    images[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0).to(device)

                # Get robot state
                state = []
                for joint in joint_names:
                    key = f"{joint}.pos"
                    if key in obs:
                        state.append(obs[key])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                # Build observation dict for policy
                policy_obs = {
                    "observation.state": state_tensor,
                    **images
                }

                # Run inference
                with torch.no_grad():
                    action = policy.select_action(policy_obs)

                # Convert action to robot command
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]  # Take first action from chunk

                # Build action dict
                action_dict = {}
                for i, joint in enumerate(joint_names):
                    action_dict[f"{joint}.pos"] = float(action[i])

                # Send to robot
                robot.send_action(action_dict)

                # Print status every 50 steps
                if step % 50 == 0:
                    gripper_pos = action_dict.get("gripper.pos", 0)
                    print(f"Step {step}: gripper={gripper_pos:.2f}")

                step += 1
                time.sleep(0.02)  # 50Hz control loop
            else:
                time.sleep(0.1)  # Idle when paused

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        print("\nCleaning up...")
        for cam_name, cap in cameras.items():
            cap.release()
        robot.disconnect()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
