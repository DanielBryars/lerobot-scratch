"""
Record a dataset using SO100 with STS3250 motors.
"""

import json
import sys
from pathlib import Path

# Register STS3250 classes with lerobot
sys.path.insert(0, str(Path(__file__).parent.parent))
import sts3250_plugin  # noqa: F401

from lerobot.scripts.lerobot_record import record, RecordConfig, DatasetRecordConfig
from lerobot.cameras.opencv import OpenCVCameraConfig

from SO100FollowerSTS3250 import SO100FollowerSTS3250Config
from SO100LeaderSTS3250 import SO100LeaderSTS3250Config

# Load configuration
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Build camera configs
cameras = {}
for name, cam_cfg in config.get("cameras", {}).items():
    cameras[name] = OpenCVCameraConfig(
        index_or_path=cam_cfg["index_or_path"],
        fps=cam_cfg.get("fps", 30),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
    )

# === CONFIGURE YOUR RECORDING HERE ===
HF_USERNAME = "your_username"  # Change this to your HuggingFace username
DATASET_NAME = "so100_pick_cube"
TASK_DESCRIPTION = "Pick up the red cube and place it in the box"
NUM_EPISODES = 50
# =====================================

robot_config = SO100FollowerSTS3250Config(
    port=config["follower"]["port"],
    id=config["follower"]["id"],
    cameras=cameras,
)

teleop_config = SO100LeaderSTS3250Config(
    port=config["leader"]["port"],
    id=config["leader"]["id"],
)

dataset_config = DatasetRecordConfig(
    repo_id=f"{HF_USERNAME}/{DATASET_NAME}",
    single_task=TASK_DESCRIPTION,
    num_episodes=NUM_EPISODES,
    fps=30,
    push_to_hub=False,  # Set to True to upload to HuggingFace
)

record_config = RecordConfig(
    robot=robot_config,
    teleop=teleop_config,
    dataset=dataset_config,
    display_data=False,  # Set to True if you have opencv with GUI support
)

if __name__ == "__main__":
    print(f"Recording {NUM_EPISODES} episodes for: {TASK_DESCRIPTION}")
    print(f"Dataset will be saved to: {HF_USERNAME}/{DATASET_NAME}")
    print("\nControls during recording:")
    print("  - Press Enter to stop current episode early")
    print("  - Press 'r' to re-record current episode")
    print("  - Press 'q' to quit recording")
    print()

    dataset = record(record_config)
    print(f"\nRecording complete! Dataset saved with {dataset.num_episodes} episodes")
