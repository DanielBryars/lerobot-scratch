#!/usr/bin/env python3
"""
Run inference with trained Diffusion Policy on SO-100 robot.

Usage:
    python inference_diffusion.py

The model is loaded from outputs/diffusion_so100/final/
"""

# IMPORTANT: Import camera backend fix BEFORE any lerobot imports
import fix_camera_backend

import argparse
import torch
import json
import time
import sys
from pathlib import Path
import numpy as np
import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.utils.constants import OBS_STATE
from SO100FollowerSTS3250 import SO100FollowerSTS3250


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return json.load(f)


def check_cameras(cameras_config):
    """
    Check all cameras are working and show preview.
    Returns dict of cv2.VideoCapture objects if successful, None otherwise.
    """
    print("\n" + "=" * 70)
    print("Camera Pre-flight Check")
    print("=" * 70)

    caps = {}
    print("\nOpening cameras...")

    for cam_name, cam_cfg in cameras_config.items():
        cap = cv2.VideoCapture(cam_cfg["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        if cap.isOpened():
            # Try to read a frame
            ret, frame = cap.read()
            if ret and frame is not None:
                caps[cam_name] = cap
                print(f"  [OK] {cam_name}: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print(f"  [ERROR] {cam_name}: Opened but can't read frames")
                cap.release()
        else:
            print(f"  [ERROR] {cam_name}: Failed to open")

    if len(caps) != len(cameras_config):
        print(f"\n[ERROR] Not all cameras available ({len(caps)}/{len(cameras_config)})")
        for cap in caps.values():
            cap.release()
        return None

    # Capture preview frames
    print("\nCapturing preview images...")
    frames = []
    for cam_name, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            # Save individual camera image
            cv2.imwrite(f"camera_test_{cam_name}.jpg", frame)
            print(f"  Saved: camera_test_{cam_name}.jpg")

    # Combine frames horizontally for preview
    if len(frames) > 1:
        combined = np.concatenate(frames, axis=1)
    else:
        combined = frames[0]

    preview_path = "camera_preview_combined.jpg"
    cv2.imwrite(preview_path, combined)
    print(f"\n[OK] Saved combined preview: {preview_path}")
    print(f"     Resolution: {combined.shape[1]}x{combined.shape[0]}")

    # Try to open preview
    import subprocess
    try:
        subprocess.Popen(['start', preview_path], shell=True)
        print("\n[INFO] Opening preview in default image viewer...")
    except Exception:
        pass

    print("\nPlease verify:")
    print("  - Both cameras are working")
    print("  - Camera angles are correct")
    print("  - Workspace is visible")
    print()

    choice = input("Continue? (y/n): ").strip().lower()
    if choice != 'y':
        print("\n[QUIT] Camera check failed")
        for cap in caps.values():
            cap.release()
        return None

    print("[OK] Cameras validated!")
    return caps


def check_robot_connection(config):
    """
    Check robot connection and read initial state.
    Returns (robot, start_position) tuple if successful, (None, None) otherwise.
    """
    print("\n" + "=" * 70)
    print("Robot Pre-flight Check")
    print("=" * 70)

    follower_port = config["follower"]["port"]
    follower_id = config["follower"]["id"]

    # Camera config (but we won't use robot's cameras - we use our own cv2 caps)
    camera_config = {
        name: OpenCVCameraConfig(
            index_or_path=cam["index_or_path"],
            width=cam["width"],
            height=cam["height"],
            fps=cam["fps"]
        )
        for name, cam in config["cameras"].items()
    }

    robot_cfg = SO100FollowerConfig(
        port=follower_port,
        id=follower_id,
        cameras=camera_config
    )
    robot = SO100FollowerSTS3250(robot_cfg)

    print(f"\nConnecting to robot at {follower_port}...")
    try:
        robot.connect()
        print("[OK] Robot connected")
    except Exception as e:
        print(f"[ERROR] Failed to connect: {e}")
        return None, None

    # Read current position using get_observation (same as main loop)
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\nReading joint positions...")
    try:
        obs = robot.get_observation()
        positions = []
        for name in joint_names:
            key = f"{name}.pos"
            if key in obs:
                pos = obs[key]
                positions.append(pos)
                print(f"  {name}: {pos:.1f}")
            else:
                print(f"  {name}: NOT FOUND in observation")
                robot.disconnect()
                return None, None

        start_position = np.array(positions, dtype=np.float32)
        print("\n[OK] Robot state read successfully")

    except Exception as e:
        print(f"[ERROR] Failed to read robot state: {e}")
        robot.disconnect()
        return None, None

    return robot, start_position


def return_to_start(robot, current_pos, target_pos, steps=50, delay=0.05):
    """Smoothly interpolate from current position back to start."""
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    print("\nReturning to start position...")
    for i in range(steps + 1):
        t = i / steps  # 0 to 1
        # Smooth easing (ease-in-out)
        t = t * t * (3 - 2 * t)
        interp_pos = current_pos + t * (target_pos - current_pos)

        # Send to robot
        action_dict = {f"{name}.pos": float(val) for name, val in zip(joint_names, interp_pos)}
        robot.send_action(action_dict)
        time.sleep(delay)

        if i % 10 == 0:
            print(f"  {int(t*100)}%...")

    print("  Done!")


def main():
    parser = argparse.ArgumentParser(description="Run inference with Diffusion Policy")
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/diffusion_so100/final",
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)"
    )
    parser.add_argument(
        "--steps-per-episode",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (default: cuda)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Control frequency in Hz (default: 5.0, matching training data)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Diffusion Policy Inference")
    print("=" * 70)
    print()
    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.steps_per_episode}")
    print(f"Device: {args.device}")
    print(f"Control FPS: {args.fps}")
    print()

    # Check model exists
    if not Path(args.model_path).exists():
        print(f"[ERROR] Model not found at: {args.model_path}")
        print("Please train the model first with: python train_diffusion.py")
        return 1

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = "cpu"

    device = torch.device(args.device)

    # Load hardware configuration
    config = load_config()

    # ========================================
    # PRE-FLIGHT CHECKS
    # ========================================

    # 1. Check cameras
    caps = check_cameras(config["cameras"])
    if caps is None:
        return 1

    # 2. Check robot connection
    robot, start_position = check_robot_connection(config)
    if robot is None:
        for cap in caps.values():
            cap.release()
        return 1

    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    current_position = start_position.copy()

    # ========================================
    # LOAD MODEL
    # ========================================
    print("\n" + "=" * 70)
    print("Loading Model")
    print("=" * 70)

    print("\nLoading Diffusion Policy...")
    try:
        policy = DiffusionPolicy.from_pretrained(args.model_path)
        policy = policy.to(device)
        policy.eval()
        print(f"[OK] Model loaded from {args.model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        robot.disconnect()
        for cap in caps.values():
            cap.release()
        return 1

    # Get config values
    n_obs_steps = policy.config.n_obs_steps  # 2
    n_action_steps = policy.config.n_action_steps  # 8

    print(f"  n_obs_steps: {n_obs_steps}")
    print(f"  n_action_steps: {n_action_steps}")
    print(f"  horizon: {policy.config.horizon}")
    print(f"  crop_shape: {policy.config.crop_shape}")

    # Create preprocessor and postprocessor
    print("\nLoading preprocessors...")
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=args.model_path,
        )
        print("[OK] Preprocessors loaded")
    except Exception as e:
        print(f"[ERROR] Failed to load preprocessors: {e}")
        robot.disconnect()
        for cap in caps.values():
            cap.release()
        return 1

    # Control period
    control_period = 1.0 / args.fps

    # For non-blocking keyboard input on Windows
    import msvcrt

    def check_for_quit():
        """Check if 'q' was pressed (non-blocking)."""
        if msvcrt.kbhit():
            key = msvcrt.getch().decode('utf-8', errors='ignore').lower()
            return key == 'q'
        return False

    print("\n" + "=" * 70)
    print("Ready to Run!")
    print("=" * 70)
    print("\nControls:")
    print("  ENTER - Start next episode")
    print("  q     - Stop and return to start position")
    print("  Ctrl+C - Emergency stop")
    print()

    try:
        for ep in range(args.num_episodes):
            print(f"\n[Episode {ep + 1}/{args.num_episodes}]")
            print("Press ENTER to start episode (or 'q' to quit)...")

            # Wait for user input
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\r' or key == b'\n':  # Enter
                        break
                    elif key.lower() == b'q':
                        print("\n'q' pressed - stopping...")
                        raise KeyboardInterrupt
                time.sleep(0.05)

            print(f"Running episode {ep + 1}...")

            # Reset policy state for new episode
            # This clears internal observation queues
            policy.reset()

            # ========================================
            # WARMUP: Fill observation queue before acting
            # ========================================
            # The policy needs n_obs_steps (2) frames of history.
            # We capture observations without sending actions to fill the queue.
            print(f"  Warming up observation queue ({n_obs_steps} frames)...")
            for warmup_step in range(n_obs_steps):
                # Capture frames
                frames_dict = {}
                for cam_name, cap in caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frames_dict[cam_name] = frame
                    else:
                        frames_dict[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

                # Read robot state
                try:
                    obs = robot.get_observation()
                    state = [obs[f"{name}.pos"] for name in joint_names]
                    state = np.array(state, dtype=np.float32)
                except Exception:
                    state = current_position.copy()

                # Build observation batch
                batch = {}
                batch[OBS_STATE] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                crop_h, crop_w = policy.config.crop_shape
                for cam_name in ["base_0_rgb", "left_wrist_0_rgb"]:
                    if cam_name in frames_dict:
                        img = frames_dict[cam_name]
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0)

                # Apply preprocessor
                batch = preprocessor(batch)
                batch = {k: v for k, v in batch.items() if v is not None and isinstance(v, torch.Tensor)}

                # Queue observation (but don't act on output)
                with torch.no_grad():
                    _ = policy.select_action(batch)

                time.sleep(control_period)

            print(f"  Warmup complete - observation queue filled")

            step = 0
            episode_running = True

            while step < args.steps_per_episode and episode_running:
                step_start = time.time()

                # Check for quit key
                if check_for_quit():
                    print("\n'q' pressed - stopping episode...")
                    episode_running = False
                    break

                # Capture frames from cameras
                frames_dict = {}
                for cam_name, cap in caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frames_dict[cam_name] = frame
                    else:
                        print(f"  Warning: Failed to read from {cam_name}")
                        frames_dict[cam_name] = np.zeros((480, 640, 3), dtype=np.uint8)

                # Read robot state using proper API
                try:
                    obs = robot.get_observation()
                    state = []
                    for name in joint_names:
                        key = f"{name}.pos"
                        if key in obs:
                            state.append(obs[key])
                        else:
                            print(f"  Warning: {key} not in observation")
                            state.append(current_position[joint_names.index(name)])
                    state = np.array(state, dtype=np.float32)
                except Exception as e:
                    print(f"  Warning: Failed to read robot state: {e}")
                    state = current_position.copy()

                # Build observation batch
                # Images at FULL resolution - policy handles cropping internally
                # State in degrees - preprocessor will normalize
                batch = {}

                # State: [batch, state_dim] = [1, 6]
                batch[OBS_STATE] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                # Images: resize full 480x640 to crop_shape (224x224) to capture entire scene
                # This matches training where we resize instead of crop
                # DEBUG: Set to True to test if model uses vision (if behavior is same, model ignores images)
                TEST_BLIND_MODE = False  # Set True to mask images with mean values

                crop_h, crop_w = policy.config.crop_shape
                for cam_name in ["base_0_rgb", "left_wrist_0_rgb"]:
                    if cam_name in frames_dict:
                        img = frames_dict[cam_name]
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # Resize to match training (224x224 captures entire scene)
                        img_resized = cv2.resize(img_rgb, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                        # Convert to tensor [C, H, W] and normalize to [0, 1]
                        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

                        # DEBUG: Replace with ImageNet mean if testing blind mode
                        if TEST_BLIND_MODE:
                            img_tensor = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).expand(3, crop_h, crop_w)

                        # Add batch dimension: [1, C, H, W]
                        batch[f"observation.images.{cam_name}"] = img_tensor.unsqueeze(0)

                # Debug: print batch format and save images for first few steps
                if step <= 2:
                    print(f"  [DEBUG] Step {step} batch (before preprocess):")
                    for k, v in batch.items():
                        print(f"    {k}: shape={v.shape}")
                    # Save camera images to see what model receives
                    for cam_name, frame in frames_dict.items():
                        # Save original
                        cv2.imwrite(f"debug_step{step}_{cam_name}_original.jpg", frame)
                        # Save resized (what model actually sees)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (crop_w, crop_h))
                        frame_resized_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"debug_step{step}_{cam_name}_resized_{crop_w}x{crop_h}.jpg", frame_resized_bgr)
                    print(f"  [DEBUG] Saved camera images: debug_step{step}_*")

                # Apply preprocessor (normalizes state with MIN_MAX, images with MEAN_STD)
                batch = preprocessor(batch)

                # Filter out None values and non-tensor values (preprocessor may add floats like next.reward)
                batch = {k: v for k, v in batch.items() if v is not None and isinstance(v, torch.Tensor)}

                if step <= 2:
                    print(f"  [DEBUG] Step {step} batch (after preprocess):")
                    for k, v in batch.items():
                        print(f"    {k}: shape={v.shape}, device={v.device}")

                # Run inference
                # Policy handles:
                # - Internal observation history via queues (n_obs_steps=2)
                # - Combining separate image keys into observation.images tensor
                # - Action chunking (returns single action from n_action_steps chunk)
                with torch.no_grad():
                    action = policy.select_action(batch)

                # Debug: print raw action
                if step <= 2:
                    print(f"  [DEBUG] Raw action: shape={action.shape}")

                # Apply postprocessor (unnormalization)
                action = postprocessor(action)

                # Handle different action output formats
                if hasattr(action, 'action'):
                    # PolicyAction object
                    action_np = action.action.cpu().numpy()
                elif hasattr(action, 'cpu'):
                    action_np = action.cpu().numpy()
                else:
                    action_np = np.array(action)

                # Debug: print final action (ALL values, in degrees)
                if step <= 5:
                    print(f"  [DEBUG] Step {step} - Action values (degrees):")
                    for jn, av in zip(joint_names, action_np.flatten()[:6]):
                        print(f"    {jn}: {av:.1f}")

                # Remove batch dimension if present
                if len(action_np.shape) > 1:
                    action_np = action_np.squeeze(0)

                # Send action to robot
                action_dict = {f"{name}.pos": float(val) for name, val in zip(joint_names, action_np)}
                robot.send_action(action_dict)

                # Track current position for safe return
                current_position = action_np.copy()

                step += 1

                # Print progress
                if step % 20 == 0:
                    print(f"  Step {step}/{args.steps_per_episode}")

                # Maintain control frequency
                elapsed = time.time() - step_start
                sleep_time = max(0, control_period - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            if episode_running:
                print(f"[OK] Episode {ep + 1} complete! ({step} steps)")
            else:
                # User quit during episode
                break

        print("\n" + "=" * 70)
        print("Inference Complete!")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nCtrl+C - Emergency stop!")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ========================================
        # GRACEFUL SHUTDOWN
        # ========================================
        print("\n" + "=" * 70)
        print("Shutting Down")
        print("=" * 70)

        # Return robot to start position
        try:
            return_to_start(robot, current_position, start_position, steps=50, delay=0.05)
        except Exception as e:
            print(f"Warning: Could not return to start: {e}")

        # Disconnect robot
        print("\nDisconnecting robot...")
        try:
            robot.disconnect()
            print("[OK] Robot disconnected")
        except Exception as e:
            print(f"Warning: {e}")

        # Release cameras
        print("Releasing cameras...")
        for cap in caps.values():
            cap.release()
        print("[OK] Cameras released")

        print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
