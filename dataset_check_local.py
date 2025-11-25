
from lerobot.datasets.lerobot_dataset import LeRobotDataset

#root = "/mnt/d/git/lerobot-scratch/datasets/20251124_233735"
root = "D:\\git\\lerobot-scratch\\datasets\\20251124_233735"

ds = LeRobotDataset(repo_id="local-debug", root=root)
print("Frames:", len(ds))
print("Episodes:", len(ds.meta.episodes))
print(ds.meta.episodes[:3])
