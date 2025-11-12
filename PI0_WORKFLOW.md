# Pi0 Fine-tuning and Inference Workflow

Complete guide for fine-tuning the Pi0 model on your SO100 robot and running inference.

## Overview

1. **Record demonstrations** using teleoperation
2. **Fine-tune Pi0 model** on your demonstrations
3. **Run inference** with the fine-tuned model

## Step 1: Record Demonstrations

Record teleoperation demonstrations to create a training dataset:

```bash
python record_demonstrations.py \
    --repo-id YOUR_USERNAME/so100_pickplace \
    --num-episodes 50 \
    --fps 30 \
    --task "pick and place the cube"
```

**Parameters:**
- `--repo-id`: Unique identifier for your dataset (e.g., "danb/so100_cube_task")
- `--num-episodes`: Number of demonstrations to record (recommended: 50-100)
- `--fps`: Recording frequency in Hz (default: 30)
- `--task`: Natural language description of the task
- `--root`: Data storage directory (default: ./data)

**Recording Process:**
1. Script starts, connects to robots
2. Press ENTER to start recording an episode
3. Perform the task by moving the leader arm
4. Press Ctrl+C when done
5. Choose:
   - 's' to save the episode
   - 'd' to discard the episode
   - 'q' to quit
6. Repeat until you have enough episodes

**Tips:**
- Record from multiple starting positions
- Include both successes and near-successes
- Be consistent in how you perform the task
- 50 episodes is a good starting point (more is better)

## Step 2: Fine-tune the Model

Fine-tune Pi0 on your recorded demonstrations:

```bash
python train_pi0.py \
    --dataset-repo-id YOUR_USERNAME/so100_pickplace \
    --output-dir ./outputs/pi0_cube_task \
    --batch-size 8 \
    --steps 3000 \
    --learning-rate 1e-5
```

**Parameters:**
- `--dataset-repo-id`: Same as used in recording (e.g., "danb/so100_cube_task")
- `--output-dir`: Where to save the fine-tuned model
- `--pretrained-path`: Base model to start from (default: "lerobot/pi0_base")
- `--batch-size`: Batch size (reduce if out of memory, default: 8)
- `--steps`: Number of training steps (default: 3000)
- `--learning-rate`: Learning rate (default: 1e-5)
- `--eval-freq`: How often to evaluate (default: 500 steps)
- `--save-freq`: How often to save checkpoints (default: 1000 steps)

**Training Configuration:**
- Model compilation: Enabled (faster training)
- Gradient checkpointing: Enabled (reduced memory usage)
- Mixed precision: bfloat16 (faster training)
- Device: CUDA (GPU)

**Expected Training Time:**
- With 50 episodes, ~30-60 minutes on a modern GPU
- More episodes or steps = longer training time

**Monitoring:**
- Watch for decreasing loss
- Check evaluation metrics every 500 steps
- Model checkpoints saved every 1000 steps

## Step 3: Run Inference

Test your fine-tuned model on the robot:

```bash
python inference_finetuned.py \
    --model-path ./outputs/pi0_cube_task \
    --task "pick and place the cube" \
    --num-episodes 5 \
    --steps-per-episode 100
```

**Parameters:**
- `--model-path`: Path to your fine-tuned model directory
- `--task`: Task description (same or similar to training)
- `--num-episodes`: How many episodes to run (default: 5)
- `--steps-per-episode`: Max steps per episode (default: 100)
- `--device`: cuda or cpu (default: cuda)

**Inference Process:**
1. Model loads and connects to robot
2. Press ENTER to start an episode
3. Robot executes the task autonomously
4. Episode ends after max steps or when done
5. Repeat for remaining episodes

## Example Full Workflow

```bash
# 1. Record 50 demonstrations
python record_demonstrations.py \
    --repo-id danb/so100_cube_pickplace \
    --num-episodes 50 \
    --task "pick up the blue cube and place it in the red box"

# 2. Fine-tune for 3000 steps
python train_pi0.py \
    --dataset-repo-id danb/so100_cube_pickplace \
    --output-dir ./outputs/pi0_cube_model \
    --steps 3000

# 3. Run inference
python inference_finetuned.py \
    --model-path ./outputs/pi0_cube_model \
    --task "pick up the blue cube and place it in the red box" \
    --num-episodes 3
```

## Troubleshooting

### Recording Issues

**Problem:** Cameras not detected
- Run `python autoconfigure.py` to detect cameras
- Check `config.json` has correct camera paths
- Verify cameras with `ls /dev/video*`

**Problem:** Robot not responding during recording
- Check gripper calibration is correct
- Verify both leader and follower are connected
- Test with `teleoperate_so100.py` first

### Training Issues

**Problem:** Out of memory during training
- Reduce `--batch-size` (try 4 or 2)
- Ensure gradient checkpointing is enabled (it is by default)
- Close other GPU programs

**Problem:** Training is very slow
- Make sure you're using GPU (`--device cuda`)
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Ensure model compilation is enabled (it is by default)

**Problem:** Loss not decreasing
- Check you have enough demonstrations (50+ recommended)
- Try lower learning rate (1e-6)
- Train for more steps (5000-10000)
- Verify demonstrations are good quality

### Inference Issues

**Problem:** Robot behavior is erratic
- Check task description matches training
- Verify model trained for enough steps
- Test environment matches training environment
- Record more diverse demonstrations

**Problem:** Robot doesn't complete task
- Increase `--steps-per-episode`
- Adjust starting position
- Check gripper calibration
- May need more training data

## Dataset Management

### View Dataset Info

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset.from_preloaded(
    repo_id="danb/so100_cube_pickplace",
    root="./data"
)

print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {dataset.num_frames}")
print(f"FPS: {dataset.fps}")
```

### Upload Dataset to HuggingFace (Optional)

```bash
# First, authenticate with HuggingFace
huggingface-cli login

# Then push your dataset
python -m lerobot.scripts.push_dataset_to_hub \
    --repo-id YOUR_USERNAME/so100_pickplace \
    --local-dir ./data/YOUR_USERNAME/so100_pickplace
```

## Hardware Requirements

### Recording
- SO100 leader and follower arms
- 2 USB cameras (overhead + wrist)
- USB connections via WSL (see README.md)

### Training
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 16GB+ system RAM
- 50GB+ free disk space

### Inference
- GPU recommended but CPU works (slower)
- Same camera and robot setup as recording

## Performance Tips

1. **Quality over Quantity**: 50 good demonstrations > 200 poor ones
2. **Consistent Environment**: Keep lighting, object positions similar
3. **Diverse Starting Positions**: Record from multiple angles/positions
4. **Clear Task Description**: Use specific, descriptive language
5. **Regular Checkpoints**: Save model every 1000 steps
6. **Monitor Training**: Watch loss and eval metrics
7. **Test Early**: Run inference after 1000 steps to check progress
8. **Iterate**: If results poor, record more demos or adjust training

## Next Steps

After successful inference:
- Record more demonstrations for better performance
- Try different tasks
- Experiment with learning rates and training steps
- Share your dataset on HuggingFace Hub
- Fine-tune from your own checkpoint for related tasks
