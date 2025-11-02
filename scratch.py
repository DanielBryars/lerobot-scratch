from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower

from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/svla_so101_pickplace")


sample = dataset[100]

print("Sample keys:", sample.keys())