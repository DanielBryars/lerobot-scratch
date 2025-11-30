@echo off
REM Record a dataset with SO100 STS3250 arms
REM Edit the variables below before running

REM Use E drive for HuggingFace cache (models, datasets, etc)
SET HF_HOME=E:\huggingface_cache

SET HF_USER=danbhf
SET DATASET_BASE=so100_pick_and_place_white_lego
SET TASK="Pick up the white lego cube and place it within the orange square on the right"
SET NUM_EPISODES=20

REM Generate simple timestamp using PowerShell
for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set TIMESTAMP=%%i
SET DATASET_NAME=%DATASET_BASE%_%TIMESTAMP%

echo Dataset: %HF_USER%/%DATASET_NAME%

REM Use venv Python directly
SET PYTHON=venv\Scripts\python.exe

REM Install plugin if not already installed
%PYTHON% -m pip show lerobot_robot_sts3250 >nul 2>&1 || %PYTHON% -m pip install -e . --no-deps

REM Record dataset
%PYTHON% -m lerobot.scripts.lerobot_record ^
    --robot.type=so100_follower_sts3250 ^
    --robot.port=COM7 ^
    --robot.id=follower_so100 ^
    --robot.cameras="{'base_0_rgb': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}, 'left_wrist_0_rgb': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30}}" ^
    --teleop.type=so100_leader_sts3250 ^
    --teleop.port=COM8 ^
    --teleop.id=leader_so100 ^
    --dataset.repo_id=%HF_USER%/%DATASET_NAME% ^
    --dataset.single_task=%TASK% ^
    --dataset.num_episodes=%NUM_EPISODES% ^
    --dataset.fps=30 ^
    --dataset.push_to_hub=true

echo.
echo Recording complete!
pause
