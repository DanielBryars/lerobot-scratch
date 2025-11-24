#!/usr/bin/env python3
"""
SmolVLA Zero-Shot Inference for SO-100 robot.

SmolVLA is a 450M parameter Vision-Language-Action model from HuggingFace,
specifically trained on LeRobot community data including SO100/SO101 datasets.

It's much smaller and faster than OpenVLA while still supporting language commands.

Usage:
    python smolvla_inference_zero_shot.py

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
    print("SmolVLA Zero-Shot Inference for SO-100")
    print("=" * 70)
    print()

    # Check for CUDA
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        device = "cuda"
    else:
        print("WARNING: No CUDA available, will be slow!")
        device = "cpu"
    print()

    # Load robot configuration
    config = load_config()

    # Import modules
    print("Loading modules...")
    import cv2
    from PIL import Image
    from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
    from so100_sts3250 import SO100FollowerSTS3250

    # Load SmolVLA model
    print("Loading SmolVLA model...")
    print("  Model: HuggingFaceTB/SmolVLA-base (~450M params)")

    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_name = "HuggingFaceTB/SmolVLA-base"

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()
        print("[OK] SmolVLA loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load SmolVLA: {e}")
        print("\nTrying alternative model name...")
        # Try alternative model names
        alt_names = [
            "lerobot/smolvla-base",
            "HuggingFaceTB/smolvla-base",
            "huggingface/SmolVLA-base"
        ]
        model = None
        for alt_name in alt_names:
            try:
                print(f"  Trying: {alt_name}")
                processor = AutoProcessor.from_pretrained(alt_name, trust_remote_code=True)
                model = AutoModelForVision2Seq.from_pretrained(
                    alt_name,
                    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    trust_remote_code=True,
                ).to(device)
                model.eval()
                print(f"[OK] Loaded {alt_name}")
                break
            except:
                continue

        if model is None:
            print("\n[ERROR] Could not load SmolVLA. Please check HuggingFace for the correct model name.")
            return 1
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

    # Setup follower robot
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
    current_instruction = "pick up the object"

    print("=" * 70)
    print("SmolVLA Ready!")
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
                # Get camera image (use base camera)
                if "base_0_rgb" in cameras:
                    ret, frame = cameras["base_0_rgb"].read()
                else:
                    cam_name = list(cameras.keys())[0]
                    ret, frame = cameras[cam_name].read()

                if not ret:
                    print("Warning: Failed to read from camera")
                    time.sleep(0.1)
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                # Get current robot state
                obs = robot.get_observation()
                state = []
                for joint in joint_names:
                    key = f"{joint}.pos"
                    if key in obs:
                        state.append(obs[key])
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

                # Prepare input for SmolVLA
                # SmolVLA uses a specific prompt format
                prompt = f"What action should the robot take to {current_instruction}?"

                try:
                    # Process inputs
                    inputs = processor(
                        text=prompt,
                        images=pil_image,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # Generate action
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False,
                        )

                    # Decode the action from model output
                    action_text = processor.decode(outputs[0], skip_special_tokens=True)

                    # Parse action from text (SmolVLA outputs action tokens)
                    # This depends on the exact output format
                    # For now, try to extract numerical values
                    import re
                    numbers = re.findall(r'[-+]?\d*\.?\d+', action_text)

                    if len(numbers) >= 6:
                        action = [float(n) for n in numbers[:7]]
                    else:
                        # Use small random movements if parsing fails
                        print(f"  Parse issue, got: {action_text[:50]}...")
                        action = [0.0] * 7

                except Exception as e:
                    print(f"  Inference error: {e}")
                    action = [0.0] * 7

                # Map action to robot joints
                action_dict = {}
                scale = 10.0  # Adjust based on action magnitude

                action_dict["shoulder_pan.pos"] = state[0] + action[0] * scale
                action_dict["shoulder_lift.pos"] = state[1] + action[1] * scale
                action_dict["elbow_flex.pos"] = state[2] + action[2] * scale
                action_dict["wrist_flex.pos"] = state[3] + action[3] * scale
                action_dict["wrist_roll.pos"] = state[4] + action[4] * scale
                # Gripper
                gripper_val = action[6] if len(action) > 6 else action[5]
                action_dict["gripper.pos"] = max(0, min(100, state[5] + gripper_val * scale))

                # Send to robot
                robot.send_action(action_dict)

                # Print status
                if step % 10 == 0:
                    print(f"Step {step}: action={action[:3]}")

                step += 1
                time.sleep(0.1)  # ~10Hz
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
