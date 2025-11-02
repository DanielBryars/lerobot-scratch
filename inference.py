import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from so100_sts3250 import SO100FollowerSTS3250  # Custom class for STS3250 motors

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20
# Device configuration
device = torch.device("cuda")  # or "cpu" or "mps" on Mac
# Load pretrained Pi0 model
model_id = "lerobot/pi0_base"  # or "lerobot/pi0_libero"
model = PI0Policy.from_pretrained(model_id)

# Remove right_wrist camera from expected inputs (we only have 2 cameras)
# Pi0 will automatically pad the missing camera
if "observation.images.right_wrist_0_rgb" in model.config.input_features:
    del model.config.input_features["observation.images.right_wrist_0_rgb"]
# Create preprocessor and postprocessor
preprocess, postprocess = make_pre_post_processors(
    model.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

# Configure your SO100 robot
follower_port = "/dev/ttyACM1"
follower_id = "follower_so100"   # Your robot ID
# IMPORTANT: Configure cameras to match Pi0's expectations
# Check your actual camera indices with lerobot-find-cameras

camera_config = {
      "base_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video4", width=640, height=360, fps=30),  # Nuroum V11 (overhead) - reduced resolution for bandwidth
      "left_wrist_0_rgb": OpenCVCameraConfig(index_or_path="/dev/video0", width=640, height=480, fps=30),  # USB2.0_CAM1 (wrist)
      # Note: Pi0 can handle missing cameras - it will pad the missing cameras automatically
  }

robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
robot = SO100FollowerSTS3250(robot_cfg)  # Use custom class for STS3250 motors
robot.connect()

# Warmup cameras - read a few frames to stabilize the stream
import time
print("Warming up cameras...")
for i in range(10):
    try:
        _ = robot.get_observation()
        print(f"  Warmup frame {i+1}/10")
        time.sleep(0.1)
    except Exception as e:
        print(f"  Warmup frame {i+1}/10 failed: {e}")
        time.sleep(0.2)
print("Cameras ready!\n")
# Define your task (natural language instruction)
task = "pick up the white lego and move to the other outlined square"  # Describe what you want the robot to do
robot_type = "so100_follower"   # For multi-embodiment datasets
# Match observation keys to policy expectations
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}
# Run inference loop
for ep in range(MAX_EPISODES):
    print(f"Starting episode {ep + 1}/{MAX_EPISODES}")
    for step in range(MAX_STEPS_PER_EPISODE):
        # Get observation from robot
        obs = robot.get_observation()
        # Build frame for inference
        obs_frame = build_inference_frame(
            observation=obs,
            ds_features=dataset_features,
            device=device,
            task=task,
            robot_type=robot_type
        )
        # Preprocess observation
        obs = preprocess(obs_frame)
        # Get action from policy
        action = model.select_action(obs)
        # Postprocess action
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
        # Send action to robot
        robot.send_action(action)
        print(f"  Step {step + 1}/{MAX_STEPS_PER_EPISODE}")
    print("Episode finished!\n")
robot.disconnect()