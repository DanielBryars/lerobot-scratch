"""
Record a multi-task dataset with proper task indexing.

Each episode is labeled with a specific task, and the user selects
which task they're demonstrating before each episode.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_record import (
    record_loop,
    init_keyboard_listener,
    log_say,
)
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras.configs import OpenCVCameraConfig
from lerobot.utils.control_utils import init_keyboard_listener

# Import our custom STS3250 plugin
import lerobot_robot_sts3250
from lerobot_robot_sts3250 import (
    SO100FollowerSTS3250Config,
    SO100FollowerSTS3250,
    SO100LeaderSTS3250Config,
    SO100LeaderSTS3250,
)

# Configuration
HF_USER = "danbhf"
FOLLOWER_PORT = "COM7"
LEADER_PORT = "COM8"
FPS = 30
NUM_EPISODES = 50

# Define your tasks
TASKS = {
    "1": "Pick up the white lego cube and place it in the LEFT orange square",
    "2": "Pick up the white lego cube and place it in the RIGHT orange square",
    "3": "Pick up the white lego cube and place it in the blue triangle",
}


def create_robot():
    """Create and connect the follower robot."""
    config = SO100FollowerSTS3250Config(
        port=FOLLOWER_PORT,
        cameras={
            "camera1": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=FPS),
            "camera2": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
        },
    )
    robot = SO100FollowerSTS3250(config)
    robot.connect()
    return robot


def create_teleop():
    """Create and connect the leader teleoperator."""
    config = SO100LeaderSTS3250Config(port=LEADER_PORT)
    teleop = SO100LeaderSTS3250(config)
    teleop.connect()
    return teleop


def select_task():
    """Prompt user to select a task."""
    print("\n" + "=" * 60)
    print("SELECT TASK FOR NEXT EPISODE")
    print("=" * 60)
    for key, task in TASKS.items():
        print(f"  [{key}] {task}")
    print("  [q] Quit recording")
    print()

    while True:
        choice = input("Enter task number (1-3) or 'q' to quit: ").strip().lower()
        if choice == 'q':
            return None
        if choice in TASKS:
            return TASKS[choice]
        print("Invalid choice. Please enter 1, 2, 3, or q.")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"so100_multitask_{timestamp}"
    repo_id = f"{HF_USER}/{dataset_name}"

    print("=" * 60)
    print("MULTI-TASK RECORDING")
    print("=" * 60)
    print(f"\nDataset: {repo_id}")
    print(f"Episodes to record: {NUM_EPISODES}")
    print(f"\nTasks:")
    for key, task in TASKS.items():
        print(f"  [{key}] {task}")
    print("\nControls during recording:")
    print("  Right Arrow -> Save episode, move to next")
    print("  Left Arrow  -> Discard and re-record episode")
    print("  Escape      -> Stop recording")
    print()
    input("Press ENTER to start...")

    # Connect hardware
    print("\nConnecting robot...")
    robot = create_robot()
    print("Connecting teleoperator...")
    teleop = create_teleop()
    print("Connected!\n")

    # Create dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot=robot,
        use_videos=True,
    )

    # Initialize keyboard listener
    listener, events = init_keyboard_listener()

    try:
        recorded_episodes = 0

        while recorded_episodes < NUM_EPISODES and not events.get("stop_recording", False):
            # Select task for this episode
            task = select_task()
            if task is None:
                print("\nQuitting...")
                break

            print(f"\nRecording episode {recorded_episodes + 1}/{NUM_EPISODES}")
            print(f"Task: {task}")
            print("Use leader arm to demonstrate. Press Right Arrow when done.")

            # Reset events
            events["exit_early"] = False
            events["rerecord_episode"] = False

            # Record the episode
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop=teleop,
                dataset=dataset,
                single_task=task,
            )

            if events.get("rerecord_episode", False):
                print("Re-recording episode...")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Save the episode
            dataset.save_episode()
            recorded_episodes += 1
            print(f"Episode {recorded_episodes} saved!")

            # Reset for next episode
            if recorded_episodes < NUM_EPISODES:
                print("\nReset the environment, then select the next task.")
                # Use teleop to reset
                events["exit_early"] = False
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    teleop=teleop,
                    dataset=None,  # Don't record during reset
                    single_task=None,
                )

    finally:
        if listener is not None:
            listener.stop()

        robot.disconnect()
        teleop.disconnect()

    print("\n" + "=" * 60)
    print("RECORDING COMPLETE")
    print("=" * 60)
    print(f"\nRecorded {recorded_episodes} episodes")
    print(f"Dataset: {repo_id}")

    # Show task distribution
    if dataset.meta.tasks is not None:
        print("\nTask distribution:")
        print(dataset.meta.tasks)

    # Ask about pushing to hub
    push = input("\nPush to HuggingFace Hub? (y/n): ").strip().lower()
    if push == 'y':
        print("Pushing to hub...")
        dataset.push_to_hub()
        print("Done!")
    else:
        print(f"Dataset saved locally at: {dataset.root}")


if __name__ == "__main__":
    main()
