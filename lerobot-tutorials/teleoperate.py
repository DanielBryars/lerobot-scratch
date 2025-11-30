"""
Teleoperate SO100 arms with STS3250 motors.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path and register STS3250 classes
sys.path.insert(0, str(Path(__file__).parent.parent))
import sts3250_plugin  # noqa: F401 - registers classes with lerobot

from SO100FollowerSTS3250 import SO100FollowerSTS3250, SO100FollowerSTS3250Config
from SO100LeaderSTS3250 import SO100LeaderSTS3250, SO100LeaderSTS3250Config
from lerobot.cameras.opencv import OpenCVCameraConfig

# Load configuration from config.json
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Convert camera configs from JSON to OpenCVCameraConfig objects
cameras = {}
for name, cam_cfg in config.get("cameras", {}).items():
    cameras[name] = OpenCVCameraConfig(
        index_or_path=cam_cfg["index_or_path"],
        fps=cam_cfg.get("fps", 30),
        width=cam_cfg.get("width", 640),
        height=cam_cfg.get("height", 480),
    )

robot_config = SO100FollowerSTS3250Config(
    port=config["follower"]["port"],
    id=config["follower"]["id"],
    cameras=cameras,
)

teleop_config = SO100LeaderSTS3250Config(
    port=config["leader"]["port"],
    id=config["leader"]["id"],
)

robot = SO100FollowerSTS3250(robot_config)
teleop_device = SO100LeaderSTS3250(teleop_config)

print("Connecting follower...")
robot.connect()
print("Connecting leader...")
teleop_device.connect()

print("Teleoperation active! Move the leader arm. Press Ctrl+C to stop.")

try:
    while True:
        action = teleop_device.get_action()
        robot.send_action(action)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    robot.disconnect()
    teleop_device.disconnect()
    print("Disconnected.")
