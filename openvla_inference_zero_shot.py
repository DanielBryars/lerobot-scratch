#!/usr/bin/env python3
"""
OpenVLA Zero-Shot Inference for SO-100 robot.

This script loads the pre-trained OpenVLA model and runs inference
with natural language commands.

OpenVLA is a 7B parameter Vision-Language-Action model trained on
970K robot manipulation episodes from Open X-Embodiment.

Usage:
    python openvla_inference_zero_shot.py

Requirements:
    pip install transformers accelerate
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
    print("OpenVLA Zero-Shot Inference for SO-100")
    print("=" * 70)
    print()

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

    # Load OpenVLA model
    print("Loading OpenVLA model (this may take a few minutes)...")
    print("  Model: openvla/openvla-7b (~14GB)")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_name = "openvla/openvla-7b"

    # Load processor and model
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Model dtype:", next(model.parameters()).dtype)

    model.eval()
    print("[OK] OpenVLA loaded successfully!")
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

    # Default instruction
    current_instruction = "pick up the white block and place it on the red cross"

    print("=" * 70)
    print("OpenVLA Ready!")
    print("=" * 70)
    print()
    print("Commands:")
    print("  - Type a natural language instruction and press Enter")
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
                else:
                    current_instruction = cmd.strip()
                    print(f"\n>>> Instruction updated: \"{current_instruction}\" <<<")
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
                # Get images from ALL cameras and combine them
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

                # Format prompt for OpenVLA
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

                # Map OpenVLA action to robot joints
                # This is a simplified mapping - may need calibration
                action_dict = {}
                if len(action) >= 6:
                    # Scale actions appropriately for SO-100
                    scale = 50.0  # Adjust based on robot range
                    action_dict["shoulder_pan.pos"] = state[0] + action[0] * scale
                    action_dict["shoulder_lift.pos"] = state[1] + action[1] * scale
                    action_dict["elbow_flex.pos"] = state[2] + action[2] * scale
                    action_dict["wrist_flex.pos"] = state[3] + action[3] * scale
                    action_dict["wrist_roll.pos"] = state[4] + action[4] * scale
                    # Gripper: OpenVLA uses -1 to 1, map to robot range
                    gripper_val = action[6] if len(action) > 6 else action[5]
                    action_dict["gripper.pos"] = 50 + gripper_val * 50  # 0-100 range

                # Send to robot
                robot.send_action(action_dict)

                # Print status every 10 steps
                if step % 10 == 0:
                    print(f"Step {step}: action={action[:3]}...")

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
