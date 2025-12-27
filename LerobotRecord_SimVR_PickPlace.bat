@echo off
REM Record pick-and-place dataset: Pick up Duplo block and place in bowl
REM Uses leader arm for teleoperation, simulation as follower with VR

REM Use E drive for HuggingFace cache
SET HF_HOME=E:\huggingface_cache

SET HF_USER=danbhf
SET DATASET_BASE=sim_pick_place_duplo
SET TASK="Pick up the Duplo block and place it in the bowl"
SET NUM_EPISODES=10
SET POS_RANGE=2
SET ROT_RANGE=180

REM Generate timestamp
for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set TIMESTAMP=%%i
SET DATASET_NAME=%DATASET_BASE%_%TIMESTAMP%

echo.
echo ========================================
echo   Simulation VR Recording - Pick Place
echo ========================================
echo Dataset: %HF_USER%/%DATASET_NAME%
echo Task: %TASK%
echo Episodes: %NUM_EPISODES%
echo Randomization: +/-%POS_RANGE%cm position, +/-%ROT_RANGE% deg rotation
echo.
echo Task is complete when Duplo lands in the bowl!
echo.

REM Use venv Python
SET PYTHON=venv\Scripts\python.exe

REM Install plugins if not already installed
%PYTHON% -m pip show lerobot_robot_sts3250 >nul 2>&1 || %PYTHON% -m pip install -e lerobot_robot_sts3250 --no-deps
%PYTHON% -m pip show lerobot_robot_sim >nul 2>&1 || %PYTHON% -m pip install -e lerobot_robot_sim --no-deps

REM Record using custom script with task completion detection
%PYTHON% record_sim_vr_pickplace.py ^
    --task %TASK% ^
    --num_episodes %NUM_EPISODES% ^
    --repo_id %HF_USER%/%DATASET_NAME% ^
    --fps 30 ^
    --pos_range %POS_RANGE% ^
    --rot_range %ROT_RANGE%

echo.
echo Recording complete!
pause
