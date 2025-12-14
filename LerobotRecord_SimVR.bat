@echo off
REM Record a dataset using SO100 simulation with VR display
REM Uses leader arm for teleoperation, simulation as follower

REM Use E drive for HuggingFace cache
SET HF_HOME=E:\huggingface_cache

SET HF_USER=danbhf
SET DATASET_BASE=sim_vr_pick_place
SET TASK="Pick up the cube and place it in the target area"
SET NUM_EPISODES=10

REM Generate timestamp
for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set TIMESTAMP=%%i
SET DATASET_NAME=%DATASET_BASE%_%TIMESTAMP%

echo.
echo ========================================
echo   Simulation VR Recording
echo ========================================
echo Dataset: %HF_USER%/%DATASET_NAME%
echo Task: %TASK%
echo Episodes: %NUM_EPISODES%
echo.

REM Use venv Python
SET PYTHON=venv\Scripts\python.exe

REM Install plugins if not already installed
%PYTHON% -m pip show lerobot_robot_sts3250 >nul 2>&1 || %PYTHON% -m pip install -e lerobot_robot_sts3250 --no-deps
%PYTHON% -m pip show lerobot_robot_sim >nul 2>&1 || %PYTHON% -m pip install -e lerobot_robot_sim --no-deps

REM Record using standard lerobot CLI with sim robot
%PYTHON% -m lerobot.scripts.lerobot_record ^
    --robot.type=so100_sim ^
    --robot.id=sim_follower ^
    --robot.enable_vr=true ^
    --robot.sim_cameras="['wrist_cam']" ^
    --teleop.type=so100_leader_sts3250 ^
    --teleop.port=COM8 ^
    --teleop.id=leader_so100 ^
    --dataset.repo_id=%HF_USER%/%DATASET_NAME% ^
    --dataset.single_task=%TASK% ^
    --dataset.num_episodes=%NUM_EPISODES% ^
    --dataset.fps=30 ^
    --dataset.push_to_hub=false

echo.
echo Recording complete!
pause
