@echo off
REM https://huggingface.co/docs/lerobot/en/using_dataset_tools

REM Use E drive for HuggingFace cache

SET HF_HOME=E:\huggingface_cache
SET PYTHON=venv\Scripts\python.exe
SET HF_USER=danbhf
REM Original merged dataset too long for wandb tags, use shorter name
SET POLICY_NAME=smolVla_so100_pick_place

for /f %%i in ('powershell -command "Get-Date -Format yyyyMMdd_HHmmss"') do set EVAL_TIMESTAMP=%%i
%PYTHON% -m lerobot.scripts.lerobot_record ^
    --robot.type=so100_follower_sts3250 ^
    --robot.port=COM7 ^
    --robot.id=follower_so100 ^
    --robot.cameras="{'camera1': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}, 'camera2': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30}}" ^
    --teleop.type=so100_leader_sts3250 ^
    --teleop.port=COM8 ^
    --dataset.repo_id=%HF_USER%/eval_%POLICY_NAME%_%EVAL_TIMESTAMP% ^
    --dataset.num_episodes=10 ^
    --dataset.single_task="Pick up the white lego cube and place it within the orange square on the right" ^
    --policy.path="%HF_USER%/%POLICY_NAME%"
