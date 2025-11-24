#!/usr/bin/env python3
"""
OpenVLA Fine-tuned Inference for SO-100 robot.

This script loads YOUR fine-tuned OpenVLA model and runs inference
with the tasks it was trained on.

Usage:
    python openvla_inference_finetuned.py
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
    print("OpenVLA Fine-tuned Inference for SO-100")
    print("=" * 70)
    print()

    # Path to your fine-tuned model
    model_path = "./outputs/openvla_finetuned/final"

    if not Path(model_path).exists():
        print(f"ERROR: Fine-tuned model not found at {model_path}")
        print("Available checkpoints:")
        checkpoint_dir = Path("./outputs/openvla_finetuned")
        if checkpoint_dir.exists():
            for item in checkpoint_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint"):
                    print(f"  - {item.name}")
        print("\nYou can use a checkpoint like:")
        print("  model_path = './outputs/openvla_finetuned/checkpoint-2500'")
        return 1

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        device = "cuda"
    else:
        print("WARNING: No CUDA available, will be very slow!")
        device = "cpu"
    print()

    # Load robot configuration
    config = load_config()

    # Import modules
    print("Loading modules...")
    import cv2
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from so100_sts3250 import SO100FollowerSTS3250

    # Load fine-tuned OpenVLA model
    print(f"Loading fine-tuned model from {model_path}...")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Model dtype:", next(model.parameters()).dtype)

    model.eval()
    print("[OK] Fine-tuned model loaded successfully!")

    # Load action normalization statistics
    norm_stats_path = Path(model_path) / "action_norm_stats.json"
    if norm_stats_path.exists():
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
        action_mins = np.array(norm_stats['action_mins'])
        action_maxs = np.array(norm_stats['action_maxs'])
        print(f"[OK] Loaded action normalization stats")
        print(f"  Min: {action_mins}")
        print(f"  Max: {action_maxs}")
    else:
        print("[WARNING] No normalization stats found, using defaults")
        action_mins = np.array([-47.34, -56.31, -71.87, -58.15, -36.35, 0.0])
        action_maxs = np.array([85.93, 64.92, 57.89, 96.48, 33.54, 58.57])
    print()

    # Setup cameras using OpenCV directly
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
        cameras={}
    )
    robot = SO100FollowerSTS3250(follower_cfg)
    robot.connect()
    print("[OK] Robot connected!")
    print()

    # Joint names for the robot
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Load task descriptions from your training data
    import pyarrow.parquet as pq
    tasks_path = Path("./datasets/merged_training_set/meta/tasks.parquet")
    tasks_table = pq.read_table(tasks_path)
    tasks_df = tasks_table.to_pandas()

    tasks = {}
    for task_desc, row in tasks_df.iterrows():
        task_idx = int(row['task_index'])
        tasks[task_idx] = str(task_desc)

    print("=" * 70)
    print("Available Tasks (from your training data):")
    print("=" * 70)
    for idx, desc in tasks.items():
        print(f"  {idx}: {desc}")
    print()

    # Default instruction
    current_instruction = tasks[0]  # Start with first task

    print("=" * 70)
    print("OpenVLA Fine-tuned Ready!")
    print("=" * 70)
    print()
    print("Commands:")
    print("  - Type a task number (0-9) to select a trained task")
    print("  - Type custom text for a new instruction")
    print("  - Press Enter with no text to START/PAUSE execution")
    print("  - Type 'q' and press Enter to QUIT")
    print()
    print(f"Current instruction: \"{current_instruction}\"")
    print()

    running = False
    should_quit = False

    def input_thread():
        nonlocal running, should_quit, current_instruction
        while not should_quit:
            try:
                cmd = input()
                if cmd.lower() == 'q':
                    should_quit = True
                    print("\nQuitting...")
                elif cmd.strip() == '':
                    running = not running
                    if running:
                        print(f"\n>>> STARTED with: \"{current_instruction}\" <<<")
                    else:
                        print("\n>>> PAUSED <<<")
                elif cmd.isdigit() and int(cmd) in tasks:
                    task_idx = int(cmd)
                    current_instruction = tasks[task_idx]
                    print(f"\n>>> Selected Task {task_idx}: \"{current_instruction}\" <<<")
                    print(">>> Press Enter to start execution <<<")
                else:
                    current_instruction = cmd.strip()
                    print(f"\n>>> Custom instruction: \"{current_instruction}\" <<<")
                    print(">>> Press Enter to start execution <<<")
            except EOFError:
                break

    # Start input thread
    input_handler = threading.Thread(target=input_thread, daemon=True)
    input_handler.start()

    step = 0
    try:
        while not should_quit:
            if running:
                # Get images from ALL cameras and combine them (same as training)
                frames = []
                for cam_name in cameras.keys():
                    ret, frame = cameras[cam_name].read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    else:
                        print(f"Warning: Failed to read from {cam_name}")

                if not frames:
                    time.sleep(0.1)
                    continue

                # Combine frames horizontally (same as training)
                if len(frames) > 1:
                    frame_rgb = np.concatenate(frames, axis=1)
                else:
                    frame_rgb = frames[0]

                # Get current robot state
                obs = robot.get_observation()
                state = []
                for joint in joint_names:
                    key = f"{joint}.pos"
                    if key in obs:
                        state.append(obs[key])

                # Prepare input for OpenVLA
                from PIL import Image
                pil_image = Image.fromarray(frame_rgb)

                # Format prompt for OpenVLA (same as training)
                prompt = f"In: What action should the robot take to {current_instruction}?\nOut:"

                # Process inputs
                inputs = processor(prompt, pil_image, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                # Convert pixel values to bfloat16 to match model dtype
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

                # Generate action
                with torch.no_grad():
                    action = model.predict_action(
                        **inputs,
                        unnorm_key="bridge_orig",  # Use bridge normalization
                        do_sample=False,
                    )

                # OpenVLA outputs 7D action: [x, y, z, roll, pitch, yaw, gripper]
                # We need to map this to our 6-DOF robot
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]

                # Denormalize actions from [-1, 1] to SO-100 joint ranges
                # Model predicts in normalized space [-1, 1], convert back to actual joint angles
                normalized_action = np.clip(action[:6], -1.0, 1.0)
                denorm_action = (normalized_action + 1.0) / 2.0  # [-1, 1] -> [0, 1]
                denorm_action = denorm_action * (action_maxs - action_mins) + action_mins  # [0, 1] -> [min, max]

                # DEBUG: Print what the model predicted
                if step % 10 == 0:
                    print(f"\nStep {step}:")
                    print(f"  Current state: {[f'{s:.1f}' for s in state]}")
                    print(f"  Raw action (normalized): {action[:6]}")
                    print(f"  Denormalized action: {denorm_action}")
                    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")

                # Map denormalized actions to robot joints
                action_dict = {}
                action_dict["shoulder_pan.pos"] = float(denorm_action[0])
                action_dict["shoulder_lift.pos"] = float(denorm_action[1])
                action_dict["elbow_flex.pos"] = float(denorm_action[2])
                action_dict["wrist_flex.pos"] = float(denorm_action[3])
                action_dict["wrist_roll.pos"] = float(denorm_action[4])
                action_dict["gripper.pos"] = float(denorm_action[5])

                # Send to robot
                robot.send_action(action_dict)

                step += 1
                time.sleep(0.1)  # ~10Hz for VLA models
            else:
                time.sleep(0.1)

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
