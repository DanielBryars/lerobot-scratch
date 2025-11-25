from lerobot.datasets.lerobot_dataset import LeRobotDataset

repo_id = "danbhf/move_the_block_from_the_left_o_20251124_233735"

ds = LeRobotDataset(repo_id)  # this will pull from the Hub
print("Total frames:", len(ds))

# Inspect metadata if it loads
meta = ds.meta
print("Info keys:", meta.info.keys())

# Try to load the episodes table for file-000
from datasets import load_dataset

episodes = load_dataset(
    repo_id,
    data_files="meta/episodes/file-000.parquet",
    split="train",
)
print(episodes.column_names)
print(episodes[:5])


