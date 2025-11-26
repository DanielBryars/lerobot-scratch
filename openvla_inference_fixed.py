#!/usr/bin/env python3
"""
OpenVLA Inference Script - FIXED VERSION

Matches the fixed training script with:
- q01/q99 normalization (not min/max)
- Proper action token decoding
- Actually controls the robot!

Usage:
    python openvla_inference_fixed.py
"""

import os
import sys
import json
import time
from pathlib import Path

os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import numpy as np
from PIL import Image
import cv2
import pyarrow.parquet as pq

# Import robot control
import fix_camera_backend
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from so100_sts3250 import SO100FollowerSTS3250


class ActionTokenizer:
    """
    Official OpenVLA action tokenizer for inference.
    Decodes token IDs back to continuous actions.
    """
    def __init__(self, tokenizer, bins: int = 256, min_action: int = -1, max_action: int = 1):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Create uniform bins
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Action tokens are mapped to the LAST n_bins tokens of vocabulary
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs back to continuous actions."""
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]


def main():
    # Configuration
    # model_path = "./outputs/openvla_fixed/best_checkpoint"  # 2500-step model
    model_path = "./outputs/openvla_fixed_5k/best_checkpoint"  # 5000-step model (overnight run)
    robot_config_path = "./config.json"
    dataset_path = "./datasets/20251124_233735_5hz"  # To get exact task description

    print("=" * 70)
    print("OpenVLA Inference (FIXED VERSION)")
    print("=" * 70)
    print()

    # Check if model exists
    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first with: python openvla_finetune_fixed.py")
        return 1

    # Load normalization stats (q01/q99)
    norm_stats_path = Path(model_path) / "action_norm_stats.json"
    if not norm_stats_path.exists():
        print(f"ERROR: Normalization stats not found at {norm_stats_path}")
        return 1

    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    action_q01 = np.array(norm_stats['action_q01'])
    action_q99 = np.array(norm_stats['action_q99'])

    print(f"Action normalization (q01/q99):")
    print(f"  q01: {action_q01}")
    print(f"  q99: {action_q99}")
    print()

    # Load task description from dataset (same as training)
    tasks_path = Path(dataset_path) / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_table = pq.read_table(tasks_path)
        task_desc = str(tasks_table.to_pandas().index[0])
    else:
        task_desc = "move the block from left to right"
    print(f"Task (from dataset): {task_desc}")
    print()

    # Load robot config
    with open(robot_config_path) as f:
        robot_config = json.load(f)

    cameras_config = robot_config["cameras"]
    print(f"Cameras: {list(cameras_config.keys())}")
    print()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Load base model then LoRA weights
    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.to(device)
    model.eval()

    print("[OK] Model loaded")
    print()

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Connect to robot
    print("Connecting to robot...")
    follower_config = SO100FollowerConfig(
        port=robot_config["follower"]["port"],
        id=robot_config["follower"]["id"],
    )
    robot = SO100FollowerSTS3250(follower_config)
    robot.connect()
    print("[OK] Robot connected")

    # Save starting position for graceful exit
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    start_obs = robot.get_observation()
    start_position = np.array([start_obs[f"{name}.pos"] for name in joint_names])
    print(f"Starting position saved: {[f'{v:.0f}' for v in start_position]}")
    print()

    # Open cameras
    print("Opening cameras...")
    caps = {}
    for cam_name, cam_cfg in cameras_config.items():
        cap = cv2.VideoCapture(cam_cfg["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        if cap.isOpened():
            caps[cam_name] = cap
            print(f"  [OK] {cam_name}")
        else:
            print(f"  [ERROR] Failed to open {cam_name}")

    if not caps:
        print("ERROR: No cameras available!")
        robot.disconnect()
        return 1
    print()

    # Build prompt (EXACTLY as in training)
    prompt = f"In: What action should the robot take to {task_desc}?\nOut:"
    print(f"Prompt: {prompt}")
    print()

    print("=" * 70)
    print("Ready to run! Press ENTER to start inference loop...")
    print("Press 'q' to stop gracefully (returns robot to start position)")
    print("=" * 70)
    input()

    # Smoothing parameters - exponential moving average
    # Lower alpha = smoother but more lag (0.3 = 30% new prediction, 70% previous)
    SMOOTHING_ALPHA = 0.2  # Reduced from 0.3 for smoother motion
    smoothed_action = None
    current_action = start_position.copy()  # Track current position for graceful exit

    # Velocity limiting - max degrees per step for each joint
    # This prevents sudden large movements even if model predicts them
    MAX_VELOCITY = np.array([
        15.0,  # shoulder_pan
        15.0,  # shoulder_lift
        15.0,  # elbow_flex
        20.0,  # wrist_flex
        20.0,  # wrist_roll
        30.0,  # gripper (can move faster)
    ])

    print(f"Using action smoothing (alpha={SMOOTHING_ALPHA})")
    print(f"Max velocity per step: {MAX_VELOCITY[0]:.0f}° for arm joints")
    print()

    # For non-blocking keyboard input on Windows
    import msvcrt

    def check_for_quit():
        """Check if 'q' was pressed (non-blocking)."""
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            return key == 'q'
        return False

    def return_to_start(current_pos, target_pos, steps=50, delay=0.05):
        """Smoothly interpolate from current position back to start."""
        print("\nReturning to start position...")
        for i in range(steps + 1):
            t = i / steps  # 0 to 1
            # Smooth easing (ease-in-out)
            t = t * t * (3 - 2 * t)
            interp_pos = current_pos + t * (target_pos - current_pos)
            action_dict = {f"{name}.pos": float(val) for name, val in zip(joint_names, interp_pos)}
            robot.send_action(action_dict)
            time.sleep(delay)
            if i % 10 == 0:
                print(f"  {int(t*100)}%...")
        print("  Done!")

    try:
        step = 0
        running = True
        while running:
            step += 1
            start_time = time.time()

            # Check for quit key
            if check_for_quit():
                print("\n'q' pressed - stopping gracefully...")
                running = False
                break

            # Capture frames from all cameras
            frames = []
            for cam_name, cap in caps.items():
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

            # Combine frames horizontally
            combined_bgr = np.concatenate(frames, axis=1)
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(combined_rgb)

            # Process inputs
            inputs = processor(prompt, image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

            # Generate action tokens
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=8,  # 6 action dims + padding
                        do_sample=False,
                    )

            # Extract action tokens (6 tokens BEFORE the EOS token)
            # The model outputs: [action1, action2, action3, action4, action5, action6, EOS]
            # We need tokens -7:-1 (excludes EOS at position -1)
            generated_ids = output_ids[0, -7:-1].cpu().numpy()

            # Decode to continuous normalized actions
            normalized_action = action_tokenizer.decode_token_ids_to_actions(generated_ids)

            # Denormalize using q01/q99
            denorm_action = (normalized_action + 1.0) / 2.0 * (action_q99 - action_q01) + action_q01

            # Apply exponential moving average smoothing to reduce jitter
            if smoothed_action is None:
                smoothed_action = current_action.copy()  # Start from current robot position, not model prediction
            else:
                smoothed_action = SMOOTHING_ALPHA * denorm_action + (1 - SMOOTHING_ALPHA) * smoothed_action

            # Apply velocity limiting - clamp changes to max velocity per step
            delta = smoothed_action - current_action
            delta = np.clip(delta, -MAX_VELOCITY, MAX_VELOCITY)
            smoothed_action = current_action + delta

            # Track current position for graceful exit
            current_action = smoothed_action.copy()

            inference_time = time.time() - start_time

            # Send SMOOTHED action to robot (keys must end with .pos)
            action_dict = {f"{name}.pos": float(val) for name, val in zip(joint_names, smoothed_action)}
            robot.send_action(action_dict)

            # Print results
            print(f"Step {step} ({inference_time*1000:.0f}ms): ", end="")
            print(" | ".join([f"{name[:3]}:{val:.0f}" for name, val in zip(joint_names, smoothed_action)]))

    except KeyboardInterrupt:
        print("\n\nCtrl+C - stopping...")

    # Graceful exit: return to start position
    try:
        return_to_start(current_action, start_position, steps=50, delay=0.05)
    except Exception as e:
        print(f"Error returning to start: {e}")

    # Clean up
    print("Disconnecting robot...")
    robot.disconnect()
    for cap in caps.values():
        cap.release()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
