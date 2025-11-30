@echo off
REM https://huggingface.co/docs/lerobot/en/using_dataset_tools

REM Use E drive for HuggingFace cache

SET HF_HOME=E:\huggingface_cache
SET PYTHON=venv\Scripts\python.exe
SET HF_USER=danbhf
REM Original merged dataset too long for wandb tags, use shorter name
SET DATASET=so100_pick_and_place_white_lego_210630_212409
SET POLICY_NAME=act_so100_pick_place

REM First run: merge the datasets (comment out goto after merge is done)
REM goto skip_merge
REM Merge two datasets into a new dataset
goto skip_merge
%PYTHON% -m lerobot.scripts.lerobot_edit_dataset ^
    --repo_id %HF_USER%/%DATASET% ^
    --operation.type merge ^
    --operation.repo_ids "['danbhf/so100_pick_and_place_white_lego_20251129_210630', 'danbhf/so100_pick_and_place_white_lego_20251129_212409']" ^
    --push_to_hub=true
:skip_merge


goto skip_training
REM Fresh training (skip this if resuming)
%PYTHON% -m lerobot.scripts.lerobot_train ^
     --dataset.repo_id=%HF_USER%/%DATASET% ^
     --policy.type=act ^
     --output_dir=outputs/train/%POLICY_NAME% ^
     --job_name=%POLICY_NAME% ^
     --policy.device=cuda ^
     --wandb.enable=true ^
     --wandb.project=lerobot-so100-act ^
     --wandb.notes="Merged datasets: so100_pick_and_place_white_lego_20251129_210630 + so100_pick_and_place_white_lego_20251129_212409" ^
     --policy.repo_id=%HF_USER%/%POLICY_NAME%
:skip_training

goto skip_resume
REM Resume from last checkpoint (060000 steps)
SET CHECKPOINT_PATH=outputs/train/%POLICY_NAME%/checkpoints/060000/pretrained_model
%PYTHON% -m lerobot.scripts.lerobot_train ^
     --config_path=%CHECKPOINT_PATH%/train_config.json ^
     --resume=true
:skip_resume

REM goto skip_inference
REM Run inference with trained policy
%PYTHON% -m lerobot.scripts.lerobot_record ^
    --robot.type=so100_follower_sts3250 ^
    --robot.port=COM7 ^
    --robot.id=follower_so100 ^
    --robot.cameras="{'base_0_rgb': {'type': 'opencv', 'index_or_path': 2, 'width': 640, 'height': 480, 'fps': 30}, 'left_wrist_0_rgb': {'type': 'opencv', 'index_or_path': 0, 'width': 640, 'height': 480, 'fps': 30}}" ^
    --dataset.repo_id=%HF_USER%/eval_%POLICY_NAME% ^
    --dataset.num_episodes=10 ^
    --dataset.single_task="Pick up the white lego cube and place it within the orange square on the right" ^
    --policy.path=%HF_USER%/%POLICY_NAME%
:skip_inference

pause