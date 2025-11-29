import json
import sys
from pathlib import Path

# Add parent directory to path for custom STS3250 modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.teleoperators.so100_leader import SO100LeaderConfig
from lerobot.robots.so100_follower import SO100FollowerConfig
from SO100LeaderSTS3250 import SO100LeaderSTS3250
from SO100FollowerSTS3250 import SO100FollowerSTS3250

# Load configuration from config.json
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)

robot_config = SO100FollowerConfig(
    port=config["follower"]["port"],
    id=config["follower"]["id"],
)

teleop_config = SO100LeaderConfig(
    port=config["leader"]["port"],
    id=config["leader"]["id"],
)

robot = SO100FollowerSTS3250(robot_config)
teleop_device = SO100LeaderSTS3250(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    action = teleop_device.get_action()
    robot.send_action(action)