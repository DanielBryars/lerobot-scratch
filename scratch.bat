@echo off
REM https://huggingface.co/docs/lerobot/en/using_dataset_tools

REM Use E drive for HuggingFace cache

SET HF_HOME=E:\huggingface_cache
SET PYTHON=venv\Scripts\python.exe
SET HF_USER=danbhf
SET DATASET=so100_pick_and_place_white_lego_20251129_210630_20251129_212409
SET POLICY_NAME=act_so100_pick_place


goto commented_out      
REM Merge two datasets into a new dataset
%PYTHON% -m lerobot.scripts.lerobot_edit_dataset ^
    --repo_id %HF_USER%/%DATASET% ^
    --operation.type merge ^
    --operation.repo_ids "['danbhf/so100_pick_and_place_white_lego_20251129_210630', 'danbhf/so100_pick_and_place_white_lego_20251129_212409']" ^
    --push_to_hub=true
:commented_out

%PYTHON% -m lerobot.scripts.lerobot_train ^
     --dataset.repo_id=%HF_USER%/%DATASET% ^
     --policy.type=act ^
     --output_dir=outputs/train/%POLICY_NAME% ^
     --job_name=%POLICY_NAME% ^
     --policy.device=cuda ^
     --wandb.enable=true ^
     --wandb.project=lerobot-so100 ^
     --policy.repo_id=%HF_USER%/%POLICY_NAME%