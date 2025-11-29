#!/usr/bin/env python
"""
Run trained ACT Policy on SO-100 robot.

This script loads a trained ACT policy and runs it with live camera input.
"""

from pathlib import Path
import argparse
import torch
import time
import cv2
import numpy as np

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so100_follower import SO100Follower, SO100FollowerConfig


def main():
    parser = argparse.ArgumentParser(description="Run ACT Policy on Robot")
    parser.add_argument("--checkpoint", type=str, default="outputs/act_so100/final",
                       help="Path to checkpoint directory")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--fps", type=int, default=5,
                       help="Control frequency (should match training)")
    parser.add_argument("--base-camera", type=int, default=0,
                       help="Base camera device index")
    parser.add_argument("--wrist-camera", type=int, default=1,
                       help="Wrist camera device index")
    parser.add_argument("--image-size", type=int, default=224,
                       help="Image size (must match training)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run without robot (just cameras)")
    parser.add_argument("--no-display", action="store_true",
                       help="Run without GUI display")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy
    print(f"\nLoading ACT policy from: {checkpoint_path}")
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.eval()
    policy.to(device)

    # Load preprocessor/postprocessor
    print("Loading preprocessors...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )

    # Get action chunk size from config
    chunk_size = policy.config.chunk_size
    n_action_steps = policy.config.n_action_steps
    print(f"  chunk_size: {chunk_size}")
    print(f"  n_action_steps: {n_action_steps}")

    # Initialize cameras
    print(f"\nInitializing cameras...")
    base_cam = cv2.VideoCapture(args.base_camera)
    wrist_cam = cv2.VideoCapture(args.wrist_camera)

    if not base_cam.isOpened():
        print(f"ERROR: Could not open base camera {args.base_camera}")
        return
    if not wrist_cam.isOpened():
        print(f"ERROR: Could not open wrist camera {args.wrist_camera}")
        return

    # Set camera resolution
    for cam in [base_cam, wrist_cam]:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"  Base camera: {args.base_camera}")
    print(f"  Wrist camera: {args.wrist_camera}")

    # Initialize robot (unless dry run)
    robot = None
    if not args.dry_run:
        print("\nInitializing robot...")
        robot_config = SO100FollowerConfig(
            port="/dev/ttyACM0",  # Adjust for your setup
        )
        robot = SO100Follower(robot_config)
        robot.connect()
        print("  Robot connected!")
    else:
        print("\nDry run mode - robot disabled")

    # Control loop timing
    frame_duration = 1.0 / args.fps
    print(f"\nControl frequency: {args.fps} Hz ({frame_duration:.3f}s per step)")

    try:
        for episode in range(args.episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{args.episodes}")
            print(f"{'='*60}")
            input("Press Enter to start episode...")

            # Reset policy's internal action queue at start of episode
            policy.reset()

            for step in range(args.max_steps):
                step_start = time.time()

                # Capture images
                ret1, base_frame = base_cam.read()
                ret2, wrist_frame = wrist_cam.read()

                if not ret1 or not ret2:
                    print("Camera read failed!")
                    break

                # Preprocess images: BGR -> RGB, resize, normalize to [0, 1]
                base_img = cv2.cvtColor(base_frame, cv2.COLOR_BGR2RGB)
                wrist_img = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)

                base_img = cv2.resize(base_img, (args.image_size, args.image_size))
                wrist_img = cv2.resize(wrist_img, (args.image_size, args.image_size))

                # Convert to tensor: [H, W, C] -> [C, H, W], float32, [0, 1]
                base_tensor = torch.from_numpy(base_img).permute(2, 0, 1).float() / 255.0
                wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float() / 255.0

                # Get current robot state
                if robot is not None:
                    state = robot.get_observation()
                    state_tensor = torch.tensor(state["observation.state"], dtype=torch.float32)
                else:
                    # Dummy state for dry run
                    state_tensor = torch.zeros(6, dtype=torch.float32)

                # Build observation dict (no temporal dimension for ACT)
                obs = {
                    "observation.images.base_0_rgb": base_tensor.unsqueeze(0),  # [1, C, H, W]
                    "observation.images.left_wrist_0_rgb": wrist_tensor.unsqueeze(0),
                    "observation.state": state_tensor.unsqueeze(0),  # [1, 6]
                }

                # Apply preprocessor
                obs = preprocessor(obs)

                # Move to device
                obs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

                # Get action from policy (handles chunking internally)
                with torch.no_grad():
                    action = policy.select_action(obs)  # [action_dim] - single action

                # Apply postprocessor to unnormalize (expects tensor, returns tensor)
                action = postprocessor(action)

                # Convert to numpy
                action = action.cpu().numpy()

                # Execute action on robot
                if robot is not None:
                    robot.send_action({"action": action})

                # Display (optional)
                if not args.no_display:
                    display = np.hstack([base_frame, wrist_frame])
                    cv2.putText(display, f"Episode {episode+1} Step {step}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(display, f"Action: {action[:3]}...", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.imshow("ACT Policy", display)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("\nQuitting...")
                        return
                    if key == ord('n'):
                        print("\nSkipping to next episode...")
                        break
                else:
                    # Print progress without GUI
                    if step % 10 == 0:
                        print(f"  Step {step}, action: {action[:3]}...")

                # Maintain control frequency
                elapsed = time.time() - step_start
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)

            print(f"Episode {episode + 1} complete")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        # Cleanup
        base_cam.release()
        wrist_cam.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # GUI not available
        if robot is not None:
            robot.disconnect()
        print("\nDone!")


if __name__ == "__main__":
    main()
