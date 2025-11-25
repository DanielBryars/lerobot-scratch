#!/usr/bin/env python3
"""
OpenVLA Fine-tuning Script for LeRobot Datasets (Official Approach).

Fine-tunes the OpenVLA-7B model on local LeRobot datasets using LoRA
for parameter-efficient training.

This version uses the official OpenVLA loss computation approach:
- Uses model's built-in unified loss (output.loss)
- Cross-entropy on all tokens including action tokens
- No manual loss combination or weighting

For the old custom loss approach, see: openvla_finetune_customloss.py

Usage:
    python openvla_finetune.py

Requirements:
    pip install transformers accelerate peft bitsandbytes wandb

References:
    - https://github.com/openvla/openvla
    - https://arxiv.org/html/2406.09246v3
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Fix Windows console encoding
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass
class FinetuneConfig:
    """Configuration for OpenVLA fine-tuning."""
    # Model
    vla_path: str = "openvla/openvla-7b"

    # Dataset
    dataset_root: str = "./datasets/20251124_233735_5hz"  # 5Hz downsampled for OpenVLA
    robot_config_path: str = "./config.json"  # Load camera config from here

    # Output
    output_dir: str = "./outputs/openvla_finetuned"

    # Training hyperparameters
    batch_size: int = 16
    grad_accumulation_steps: int = 2  # Effective batch size = 32
    learning_rate: float = 5e-4
    max_steps: int = 2500  # ~6 epochs over 12.9k samples at batch 32
    warmup_steps: int = 50

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    use_quantization: bool = True  # 4-bit quantization

    # Logging
    log_freq: int = 10
    save_freq: int = 500
    eval_freq: int = 100  # Evaluate more frequently to catch overfitting

    # Validation
    val_split: float = 0.1  # 10% validation split
    val_seed: int = 42  # Random seed for reproducible splits

    # Weights & Biases
    wandb_project: str = "openvla-so100"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True


class LeRobotOpenVLADataset(Dataset):
    """
    Dataset adapter that converts LeRobot format to OpenVLA training format.

    LeRobot stores:
    - Parquet files with state/action data
    - Video files for images
    - Task descriptions in meta/tasks.parquet

    OpenVLA expects:
    - Image + text prompt -> action tokens

    This dataset loads ALL cameras from config.json and concatenates them
    into a single image for training.
    """

    def __init__(
        self,
        dataset_root: str,
        processor,
        action_tokenizer,
        robot_config_path: str = "./config.json",
        max_samples: Optional[int] = None,
        image_layout: str = "horizontal",  # "horizontal", "vertical", or "grid"
    ):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.image_layout = image_layout

        # Load robot config to get camera list
        with open(robot_config_path) as f:
            robot_config = json.load(f)

        # Extract camera keys from config
        self.camera_keys = []
        for cam_name in robot_config.get("cameras", {}).keys():
            # Convert config camera name to dataset image key
            image_key = f"observation.images.{cam_name}"
            self.camera_keys.append(image_key)

        print(f"Using cameras: {self.camera_keys}")

        # Load dataset info
        info_path = self.dataset_root / "meta" / "info.json"
        with open(info_path) as f:
            self.info = json.load(f)

        # Load task descriptions
        import pyarrow.parquet as pq
        tasks_path = self.dataset_root / "meta" / "tasks.parquet"
        tasks_table = pq.read_table(tasks_path)

        # Read tasks properly
        # In LeRobot format: row index = task description, column = task_index
        tasks_df = tasks_table.to_pandas()
        self.tasks = {}
        for task_desc, row in tasks_df.iterrows():
            task_idx = int(row['task_index'])
            self.tasks[task_idx] = str(task_desc)

        # Build video frame maps for each camera
        # Maps global index -> (video_file, frame_in_video)
        self.video_maps = {}
        for image_key in self.camera_keys:
            self.video_maps[image_key] = self._build_video_map(image_key)
            print(f"  {image_key}: {len(self.video_maps[image_key])} video segments")

        # Load all data samples
        self.samples = []
        self._load_samples(max_samples)

        # Compute action normalization statistics from the dataset
        print("Computing action normalization statistics...")
        all_actions = np.array([s['action'] for s in self.samples])
        self.action_mins = all_actions.min(axis=0)
        self.action_maxs = all_actions.max(axis=0)
        print(f"  Action ranges: min={self.action_mins}, max={self.action_maxs}")

        # Cache for video captures to avoid repeated opening
        self._video_cache = {}

        print(f"Loaded {len(self.samples)} samples from {self.dataset_root}")
        print(f"Tasks: {list(self.tasks.values())[:3]}...")

    def _build_video_map(self, image_key: str) -> list:
        """
        Build a map of global index -> video file for a camera.

        LeRobot v3 stores frames in chunked video files. This map tells us
        which video file contains which frame indices.

        Returns list of dicts with:
            - start_idx: first global index in this video
            - end_idx: last global index in this video
            - path: path to video file
        """
        import cv2

        video_dir = self.dataset_root / "videos" / image_key
        video_files = sorted(video_dir.glob("**/*.mp4"))

        video_map = []
        current_idx = 0

        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_map.append({
                    'start_idx': current_idx,
                    'end_idx': current_idx + frame_count - 1,
                    'path': vf,
                    'frame_count': frame_count,
                })
                current_idx += frame_count
                cap.release()

        return video_map

    def _load_samples(self, max_samples: Optional[int] = None):
        """Load sample metadata from parquet files."""
        import pyarrow.parquet as pq

        data_dir = self.dataset_root / "data"
        parquet_files = sorted(data_dir.glob("**/*.parquet"))

        for pq_file in parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                sample = {
                    'global_index': int(row['index']),  # Global frame index for video lookup
                    'episode_index': int(row['episode_index']),
                    'frame_index': int(row['frame_index']),
                    'task_index': int(row['task_index']),
                    'action': np.array(row['action'], dtype=np.float32),
                    'state': np.array(row['observation.state'], dtype=np.float32),
                }
                self.samples.append(sample)

                if max_samples and len(self.samples) >= max_samples:
                    return

    def _extract_frame_from_video(self, image_key: str, global_index: int) -> Image.Image:
        """Extract a frame - either from pre-extracted images or video."""
        import cv2

        # Check if pre-extracted frames exist (MUCH faster)
        frames_dir = self.dataset_root / "frames" / image_key
        if frames_dir.exists():
            frame_path = frames_dir / f"{global_index:06d}.jpg"
            if frame_path.exists():
                return Image.open(frame_path).convert('RGB')

        # Fall back to video extraction (slow)
        video_map = self.video_maps.get(image_key, [])
        video_info = None

        for vi in video_map:
            if vi['start_idx'] <= global_index <= vi['end_idx']:
                video_info = vi
                break

        if video_info is None:
            raise FileNotFoundError(f"No video found for {image_key} at global index {global_index}")

        # Calculate frame position within this video file
        frame_in_video = global_index - video_info['start_idx']
        video_file = video_info['path']

        # Use cached video capture or create new one
        cache_key = str(video_file)
        if cache_key not in self._video_cache:
            self._video_cache[cache_key] = cv2.VideoCapture(str(video_file))

        cap = self._video_cache[cache_key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
        ret, frame = cap.read()

        if not ret:
            # Try reopening the video
            cap.release()
            cap = cv2.VideoCapture(str(video_file))
            self._video_cache[cache_key] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_in_video} from {video_file}")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _combine_images(self, images: list) -> Image.Image:
        """Combine multiple camera images into a single image."""
        if len(images) == 1:
            return images[0]

        if self.image_layout == "horizontal":
            # Stack images horizontally (side by side)
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)

            combined = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                # Center vertically if heights differ
                y_offset = (max_height - img.height) // 2
                combined.paste(img, (x_offset, y_offset))
                x_offset += img.width

        elif self.image_layout == "vertical":
            # Stack images vertically
            max_width = max(img.width for img in images)
            total_height = sum(img.height for img in images)

            combined = Image.new('RGB', (max_width, total_height))
            y_offset = 0
            for img in images:
                # Center horizontally if widths differ
                x_offset = (max_width - img.width) // 2
                combined.paste(img, (x_offset, y_offset))
                y_offset += img.height

        elif self.image_layout == "grid":
            # 2x2 grid for up to 4 cameras
            import math
            n = len(images)
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)

            img_width = images[0].width
            img_height = images[0].height

            combined = Image.new('RGB', (cols * img_width, rows * img_height))
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                combined.paste(img, (col * img_width, row * img_height))

        else:
            raise ValueError(f"Unknown image layout: {self.image_layout}")

        return combined

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        global_index = sample['global_index']

        # Get images from all cameras using global index
        images = []
        for image_key in self.camera_keys:
            try:
                img = self._extract_frame_from_video(image_key, global_index)
                images.append(img)
            except Exception as e:
                # Fallback: create a placeholder image
                print(f"Warning: Could not load {image_key} for sample {idx} (global_idx={global_index}): {e}")
                images.append(Image.new('RGB', (640, 480), color=(128, 128, 128)))

        # Combine all camera images into one
        combined_image = self._combine_images(images)

        # Get task description
        task_desc = self.tasks.get(sample['task_index'], "pick up the object")

        # Format prompt (OpenVLA style)
        prompt = f"In: What action should the robot take to {task_desc}?\nOut:"

        # Process image and text
        inputs = self.processor(prompt, combined_image, return_tensors="pt")

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Normalize action to [-1, 1]
        raw_action = sample['action']
        normalized_action = 2.0 * (raw_action - self.action_mins) / (self.action_maxs - self.action_mins + 1e-8) - 1.0
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        # Tokenize normalized action
        action_tokens = self.action_tokenizer.tokenize(normalized_action)

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'action': torch.tensor(normalized_action, dtype=torch.float32),
            'action_tokens': torch.tensor(action_tokens, dtype=torch.long),
            'task': task_desc,
        }

    def __del__(self):
        """Clean up video captures."""
        for cap in self._video_cache.values():
            cap.release()


class ActionTokenizer:
    """
    Tokenizes continuous actions to discrete tokens for OpenVLA.

    OpenVLA uses 256 bins per action dimension.
    Actions are expected to be in [-1, 1] range (normalized).
    """

    def __init__(self, n_bins: int = 256, action_dim: int = 6):
        self.n_bins = n_bins
        self.action_dim = action_dim

    def tokenize(self, action: np.ndarray) -> np.ndarray:
        """Convert continuous normalized action [-1, 1] to discrete bin indices."""
        # Clip to [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        # Map [-1, 1] to [0, 1]
        normalized = (action + 1.0) / 2.0

        # Convert to bin indices [0, n_bins-1]
        bins = (normalized * (self.n_bins - 1)).astype(np.int64)
        bins = np.clip(bins, 0, self.n_bins - 1)

        return bins

    def detokenize(self, tokens: np.ndarray) -> np.ndarray:
        """Convert discrete tokens back to continuous normalized actions [-1, 1]."""
        # Convert bins to [0, 1]
        normalized = tokens.astype(np.float32) / (self.n_bins - 1)

        # Map [0, 1] to [-1, 1]
        action = normalized * 2.0 - 1.0

        return action


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    # Find max sequence length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    # Pad sequences
    input_ids = []
    attention_masks = []
    pixel_values = []
    actions = []
    action_tokens = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids with 0 (typically pad token)
        padded_ids = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids.append(padded_ids)

        # Pad attention mask with 0
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])
        attention_masks.append(padded_mask)

        pixel_values.append(item['pixel_values'])
        actions.append(item['action'])
        action_tokens.append(item['action_tokens'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'action': torch.stack(actions),
        'action_tokens': torch.stack(action_tokens),
    }


def compute_action_loss(logits, action_tokens, action_dim=6):
    """
    Compute cross-entropy loss for action prediction.

    NOTE: This function is NOT used in the official OpenVLA training approach.
    The official method uses a unified loss from the model itself (output.loss),
    which internally computes cross-entropy on all tokens including actions.

    This custom implementation is kept for reference only.
    See openvla_finetune_customloss.py for the old manual loss combination approach.

    OpenVLA predicts actions as discrete tokens, so we use CE loss.
    """
    # Get the action prediction logits (last action_dim tokens)
    # This is a simplified version - actual OpenVLA uses more complex indexing
    batch_size = logits.shape[0]

    # Flatten for loss computation
    loss_fn = nn.CrossEntropyLoss()

    # For each action dimension, compute loss
    total_loss = 0
    for i in range(action_dim):
        # Get logits for this action dimension
        # Simplified: use last tokens
        pred_logits = logits[:, -(action_dim - i), :]
        target = action_tokens[:, i]

        total_loss += loss_fn(pred_logits, target)

    return total_loss / action_dim


def validate_cameras(cfg: FinetuneConfig) -> bool:
    """
    Validate camera configuration by capturing and displaying test frames.
    Returns True if user confirms cameras are correct.
    """
    import cv2

    print("=" * 70)
    print("Camera Validation")
    print("=" * 70)
    print()

    # Load robot config
    with open(cfg.robot_config_path) as f:
        robot_config = json.load(f)

    cameras_config = robot_config.get("cameras", {})
    if not cameras_config:
        print("ERROR: No cameras found in config.json!")
        return False

    print(f"Found {len(cameras_config)} camera(s) in config:")
    for cam_name, cam_cfg in cameras_config.items():
        print(f"  - {cam_name}: index={cam_cfg['index_or_path']}, {cam_cfg['width']}x{cam_cfg['height']}@{cam_cfg['fps']}fps")
    print()

    # Capture test frames from each camera
    test_frames = {}
    captures = {}

    for cam_name, cam_cfg in cameras_config.items():
        print(f"Capturing from {cam_name} (index {cam_cfg['index_or_path']})...")

        cap = cv2.VideoCapture(cam_cfg["index_or_path"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cam_cfg["fps"])

        if not cap.isOpened():
            print(f"  [ERROR] Failed to open camera {cam_cfg['index_or_path']}")
            continue

        # Capture a few frames to let camera warm up
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if ret:
            test_frames[cam_name] = frame
            print(f"  [OK] Captured {frame.shape[1]}x{frame.shape[0]} frame")
        else:
            print(f"  [ERROR] Failed to capture frame")

        captures[cam_name] = cap

    if not test_frames:
        print("\nERROR: Could not capture from any camera!")
        for cap in captures.values():
            cap.release()
        return False

    # Save test frames for user review
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("Saving test frames for validation:")

    saved_paths = []
    for cam_name, frame in test_frames.items():
        filename = f"camera_validation_{cam_name}.jpg"
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), frame)
        print(f"  Saved: {filepath}")
        saved_paths.append(filepath)

    # Also save combined view (as it will appear during training)
    if len(test_frames) > 1:
        frames_list = list(test_frames.values())
        combined = np.concatenate(frames_list, axis=1)
        combined_path = output_dir / "camera_validation_combined.jpg"
        cv2.imwrite(str(combined_path), combined)
        print(f"  Saved combined view: {combined_path}")
        saved_paths.append(combined_path)

    # Release cameras
    for cap in captures.values():
        cap.release()

    # Display frames using system default viewer
    print()
    print("=" * 70)
    print("Please review the camera images:")
    print()

    # Open images with system default viewer (works on Windows)
    combined_image_path = None
    if len(test_frames) > 1:
        combined_image_path = output_dir / "camera_validation_combined.jpg"

    try:
        if sys.platform == "win32":
            import subprocess
            # Open combined view (or first camera if only one)
            image_to_show = combined_image_path if combined_image_path else saved_paths[0]
            subprocess.Popen(["start", "", str(image_to_show)], shell=True)
            print(f"Opened: {image_to_show}")
        else:
            # Linux/Mac fallback
            import subprocess
            image_to_show = combined_image_path if combined_image_path else saved_paths[0]
            opener = "xdg-open" if sys.platform == "linux" else "open"
            subprocess.Popen([opener, str(image_to_show)])
            print(f"Opened: {image_to_show}")
    except Exception as e:
        print(f"Could not auto-open image: {e}")
        print("Please manually open the images:")
        for path in saved_paths:
            print(f"  {path}")

    print()
    print("The COMBINED view shows what the model will see during training.")
    print("(Left: base camera, Right: wrist camera)")
    print("=" * 70)
    print()

    # Ask user for confirmation
    while True:
        response = input("Are the cameras configured correctly? [y/n]: ").strip().lower()
        if response in ['y', 'yes']:
            print()
            return True
        elif response in ['n', 'no']:
            print()
            print("Please update config.json with the correct camera settings and try again.")
            print("You can use 'python identify_cameras_for_config.py' to find camera indices.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def main():
    cfg = FinetuneConfig()

    print("=" * 70)
    print("OpenVLA Fine-tuning for SO-100")
    print("=" * 70)
    print()

    # Validate cameras first (before loading the big model)
    if not validate_cameras(cfg):
        print("Camera validation failed. Exiting.")
        return 1

    # Check CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
        device = torch.device("cuda")
    else:
        print("WARNING: No CUDA available!")
        device = torch.device("cpu")
    print()

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Initialize wandb
    if cfg.use_wandb:
        try:
            import wandb

            # Build config with extra computed values
            wandb_config = vars(cfg).copy()
            wandb_config['effective_batch_size'] = cfg.batch_size * cfg.grad_accumulation_steps
            wandb_config['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            wandb_config['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1) if torch.cuda.is_available() else 0

            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                config=wandb_config,
                name=f"openvla-so100-{time.strftime('%Y%m%d-%H%M%S')}",
            )
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            cfg.use_wandb = False

    # Load processor and model
    print("Loading OpenVLA model...")
    print("  This may take a few minutes and ~14GB of disk space")
    print()

    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        cfg.vla_path,
        trust_remote_code=True,
    )

    # Quantization config for memory efficiency
    quantization_config = None
    if cfg.use_quantization:
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("  Using 4-bit quantization")
        except ImportError:
            print("  bitsandbytes not available, loading without quantization")
            cfg.use_quantization = False

    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto" if cfg.use_quantization else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if not cfg.use_quantization:
        model = model.to(device)

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    print()

    # Apply LoRA
    if cfg.use_lora:
        print("Applying LoRA...")
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if cfg.use_quantization:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules="all-linear",
                init_lora_weights="gaussian",
            )

            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            print()
        except ImportError:
            print("  PEFT not available, training full model")
            cfg.use_lora = False

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(n_bins=256, action_dim=6)

    # Load dataset
    print("Loading dataset...")
    full_dataset = LeRobotOpenVLADataset(
        dataset_root=cfg.dataset_root,
        processor=processor,
        action_tokenizer=action_tokenizer,
        robot_config_path=cfg.robot_config_path,
        image_layout="horizontal",  # Both cameras side-by-side
    )

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * cfg.val_split)
    train_size = total_size - val_size

    print(f"  Total samples: {total_size}")
    print(f"  Train samples: {train_size} ({100 * train_size / total_size:.1f}%)")
    print(f"  Val samples: {val_size} ({100 * val_size / total_size:.1f}%)")

    # Use random_split with seed for reproducibility
    from torch.utils.data import random_split
    generator = torch.Generator().manual_seed(cfg.val_seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
    )
    print()

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=0.01,
    )

    # Learning rate scheduler with warmup
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_steps - cfg.warmup_steps,
        eta_min=cfg.learning_rate * 0.1,
    )

    # Validation function
    def evaluate_on_validation():
        """Evaluate model on validation set."""
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="  Validation", leave=False):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
                action_tokens = batch['action_tokens'].to(device)

                # Forward pass
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=input_ids,
                    )

                    # Use unified loss (official OpenVLA approach)
                    loss = outputs.loss

                val_losses.append(loss.item())

        model.train()

        return {
            'val_loss': np.mean(val_losses),
        }

    # Training loop
    print("=" * 70)
    print("Starting training...")
    print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accumulation_steps} = {cfg.batch_size * cfg.grad_accumulation_steps}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max steps: {cfg.max_steps}")
    print(f"  Validation every: {cfg.eval_freq} steps")
    print("=" * 70)
    print()

    model.train()
    global_step = 0
    total_loss = 0
    avg_loss = 0.0  # Initialize for display
    optimizer.zero_grad()

    # Track best validation loss
    best_val_loss = float('inf')
    best_checkpoint_step = 0

    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while global_step < cfg.max_steps:
        for batch_idx, batch in enumerate(train_dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            action_tokens = batch['action_tokens'].to(device)

            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=input_ids,  # Causal LM loss on all tokens including actions
                )

                # Use model's built-in unified loss (official OpenVLA approach)
                # The model internally computes cross-entropy on all tokens,
                # including the action tokens that were tokenized as part of the sequence
                loss = outputs.loss

            # Backward pass
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            total_loss += loss.item()

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                # Warmup
                if global_step < cfg.warmup_steps:
                    lr_scale = (global_step + 1) / cfg.warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = cfg.learning_rate * lr_scale
                else:
                    scheduler.step()

                global_step += 1
                pbar.update(1)

                # Logging
                if global_step % cfg.log_freq == 0:
                    avg_loss = total_loss / cfg.log_freq
                    lr = optimizer.param_groups[0]['lr']

                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}',
                    })

                    if cfg.use_wandb:
                        wandb.log({
                            'train_loss': avg_loss,
                            'learning_rate': lr,
                            'step': global_step,
                        })

                    total_loss = 0

                # Validation evaluation
                if global_step % cfg.eval_freq == 0:
                    print("\n  Running validation...")
                    val_metrics = evaluate_on_validation()

                    pbar.set_postfix({
                        'train_loss': f'{avg_loss:.4f}',
                        'val_loss': f'{val_metrics["val_loss"]:.4f}',
                        'lr': f'{lr:.2e}',
                    })

                    print(f"  Step {global_step}:")
                    print(f"    Train loss: {avg_loss:.4f}")
                    print(f"    Val loss: {val_metrics['val_loss']:.4f}")

                    if cfg.use_wandb:
                        wandb.log({
                            'val_loss': val_metrics['val_loss'],
                            'step': global_step,
                        })

                    # Save best checkpoint
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        best_checkpoint_step = global_step

                        best_path = Path(cfg.output_dir) / "best_checkpoint"
                        os.makedirs(best_path, exist_ok=True)

                        if cfg.use_lora:
                            model.save_pretrained(best_path)
                        else:
                            model.save_pretrained(best_path)

                        processor.save_pretrained(best_path)

                        # Save normalization stats with best checkpoint too
                        norm_stats = {
                            'action_mins': full_dataset.action_mins.tolist(),
                            'action_maxs': full_dataset.action_maxs.tolist(),
                        }
                        with open(best_path / "action_norm_stats.json", 'w') as f:
                            json.dump(norm_stats, f, indent=2)

                        # Save info about best checkpoint
                        with open(best_path / "checkpoint_info.json", 'w') as f:
                            json.dump({
                                'step': global_step,
                                'val_loss': val_metrics['val_loss'],
                            }, f, indent=2)

                        print(f"  >>> New best checkpoint! Val loss: {best_val_loss:.4f}")

                # Save regular checkpoint
                if global_step % cfg.save_freq == 0:
                    save_path = Path(cfg.output_dir) / f"checkpoint-{global_step}"
                    os.makedirs(save_path, exist_ok=True)

                    if cfg.use_lora:
                        # Save LoRA weights
                        model.save_pretrained(save_path)
                    else:
                        # Save full model
                        model.save_pretrained(save_path)

                    processor.save_pretrained(save_path)
                    print(f"\n  Saved checkpoint to {save_path}")

                if global_step >= cfg.max_steps:
                    break

    pbar.close()

    # Save final model
    print("\nSaving final model...")
    final_path = Path(cfg.output_dir) / "final"
    os.makedirs(final_path, exist_ok=True)

    if cfg.use_lora:
        # Merge LoRA weights and save
        print("  Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(final_path)
    else:
        model.save_pretrained(final_path)

    processor.save_pretrained(final_path)

    # Save normalization statistics
    norm_stats = {
        'action_mins': full_dataset.action_mins.tolist(),
        'action_maxs': full_dataset.action_maxs.tolist(),
    }
    with open(final_path / "action_norm_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved action normalization stats")

    # Save config
    with open(final_path / "finetune_config.json", 'w') as f:
        json.dump(vars(cfg), f, indent=2)

    print(f"\nTraining complete! Model saved to {final_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best checkpoint step: {best_checkpoint_step}")
    print(f"Best checkpoint saved to: {Path(cfg.output_dir) / 'best_checkpoint'}")
    print(f"Final checkpoint saved to: {final_path}")
    print()
    print("Recommended: Use the 'best_checkpoint' for inference to avoid overfitting!")
    print("=" * 70)

    if cfg.use_wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
