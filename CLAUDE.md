# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a scratch repository for experimenting with the LeRobot library (https://github.com/huggingface/lerobot). LeRobot is a robotics library focused on teleoperation and robot learning, particularly for the SO-100 robot platform.

## Environment Setup

This project uses a Python virtual environment located at `venv/`. The environment is already configured with:
- Python 3.12.3
- lerobot package (version 0.3.3)
- lerobot[pi0] extras for Raspberry Pi Zero support

### Initial Setup
```bash
# Activate the virtual environment
source venv/bin/activate

# Install LeRobot (if needed)
pip install 'lerobot'
pip install 'lerobot[pi0]'
```

### Running Code
The main experimental code is in `scratch.py`. Run it with:
```bash
python scratch.py
```

## USB Device Sharing (Windows WSL)

This project includes utilities for sharing USB devices from Windows to WSL, which is necessary for connecting robot hardware:

1. **Windows side** (`win-share-usb.cmd`):
   - Starts the usbipd service
   - Binds specific USB bus IDs (12-3, 12-4, 13-3, 13-4, 6-7)
   - Attaches them to WSL
   - Run this script in Windows PowerShell/Command Prompt as Administrator

2. **WSL side** (`wsl-share-usb.sh`):
   - Lists USB devices visible in WSL
   - Run this after the Windows script to verify devices are attached

Reference: https://learn.microsoft.com/en-us/windows/wsl/connect-usb

## Code Architecture

### LeRobot Components Used

The code in `scratch.py` demonstrates the primary LeRobot components:

1. **Teleoperation**: `SO100LeaderConfig` and `SO100Leader` from `lerobot.teleoperators.so100_leader`
   - Used for controlling the leader robot arm for teleoperation

2. **Robot Control**: `SO100FollowerConfig` and `SO100Follower` from `lerobot.robots.so100_follower`
   - Used for the follower robot that mimics leader movements

3. **Datasets**: `LeRobotDataset` from `lerobot.datasets.lerobot_dataset`
   - Currently loads the "lerobot/svla_so101_pickplace" dataset
   - Dataset samples are dictionaries with various keys (printed in scratch.py)

### Development Pattern

This is a scratch/experimental repository, so the typical workflow is:
1. Modify `scratch.py` to test different LeRobot features
2. Run the script to observe behavior
3. Iterate on experiments
