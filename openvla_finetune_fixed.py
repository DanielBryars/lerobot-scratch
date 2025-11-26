#!/usr/bin/env python3
"""
OpenVLA Fine-tuning Script - FIXED VERSION

This version correctly implements the official OpenVLA training approach:
1. Actions converted to token STRINGS (not just bin indices)
2. Action strings included in the input sequence as GPT responses
3. Labels properly masked (loss only on action tokens)
4. q01/q99 normalization (not min/max)
5. Quantization disabled (hurts performance)

Fixes the critical bug where action tokens were never part of the training sequence.

Usage:
    python openvla_finetune_fixed.py
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

# Fix Windows console encoding
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

# IGNORE_INDEX for labels (standard HuggingFace convention)
IGNORE_INDEX = -100


class PurePromptBuilder:
    """
    Simplified prompt builder matching official OpenVLA implementation.

    Format:
        In: <human message>
        Out: <gpt response></s>
    """
    def __init__(self):
        self.bos = "<s>"
        self.eos = "</s>"
        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        """Add a turn to the conversation."""
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if self.turn_count % 2 == 0:  # Human turn
            wrapped = f"In: {message}\nOut: "
        else:  # GPT turn
            wrapped = f"{message if message != '' else ' '}{self.eos}"

        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        """Get the full prompt (removes leading BOS)."""
        return self.prompt.removeprefix(self.bos).rstrip()


class ActionTokenizer:
    """
    Official OpenVLA action tokenizer.

    Discretizes continuous actions into 256 bins and maps to vocabulary tokens.
    Returns DECODED TOKEN STRINGS that can be concatenated with instruction.
    """
    def __init__(self, tokenizer, bins: int = 256, min_action: int = -1, max_action: int = 1):
        self.tokenizer = tokenizer
        self.n_bins = bins
        self.min_action = min_action
        self.max_action = max_action

        # Create uniform bins
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Action tokens are mapped to the LAST n_bins tokens of vocabulary
        self.action_token_begin_idx = int(self.tokenizer.vocab_size - (self.n_bins + 1))

    def __call__(self, action: np.ndarray) -> str:
        """
        Convert action to TOKEN STRING.

        This string gets concatenated with the instruction, then everything is tokenized together.
        """
        # Clip to valid range
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))

        # Discretize into bins
        discretized_action = np.digitize(action, self.bins)

        # Map to vocabulary token IDs (from the end of vocab)
        token_ids = self.tokenizer.vocab_size - discretized_action

        # Decode to STRING
        return self.tokenizer.decode(list(token_ids))

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs back to continuous actions."""
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        return self.bin_centers[discretized_actions]


@dataclass
class FinetuneConfig:
    """Configuration for OpenVLA fine-tuning."""
    # Model
    vla_path: str = "openvla/openvla-7b"

    # Dataset
    dataset_root: str = "./datasets/20251124_233735_5hz"
    robot_config_path: str = "./config.json"

    # Output
    output_dir: str = "./outputs/openvla_fixed_5k"  # New directory for 5000-step run

    # Training hyperparameters
    batch_size: int = 8  # Reduced for 32GB GPU with quantization
    grad_accumulation_steps: int = 4  # Effective batch size = 32
    learning_rate: float = 5e-4  # Official default
    max_steps: int = 5000  # Extended overnight run
    warmup_steps: int = 50

    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 16  # Official uses min(rank, 16)
    lora_dropout: float = 0.0
    use_quantization: bool = True  # 4-bit quantization for 32GB GPU (required to fit in VRAM)

    # Logging
    log_freq: int = 10
    save_freq: int = 1000  # Save every 1000 steps for longer run
    eval_freq: int = 100

    # Validation
    val_split: float = 0.1
    val_seed: int = 42

    # Weights & Biases
    wandb_project: str = "openvla-so100-fixed"
    wandb_entity: Optional[str] = None
    use_wandb: bool = True


class LeRobotOpenVLADataset(Dataset):
    """
    FIXED Dataset implementation following official OpenVLA approach.

    Key fixes:
    1. Action tokens are converted to STRINGS and included in sequence
    2. Labels properly masked (loss only on action tokens)
    3. q01/q99 normalization (not min/max)
    """

    def __init__(
        self,
        dataset_root: str,
        robot_config_path: str,
        processor,
        tokenizer,
        split: str = "train",
        image_layout: str = "horizontal",
    ):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.tokenizer = tokenizer
        self.split = split
        self.image_layout = image_layout

        # Load robot config for cameras
        with open(robot_config_path, 'r') as f:
            robot_config = json.load(f)

        # Get camera keys
        self.camera_keys = [f"observation.images.{name}" for name in robot_config["cameras"].keys()]
        print(f"  Using {len(self.camera_keys)} cameras: {list(robot_config['cameras'].keys())}")

        # Load task descriptions
        import pyarrow.parquet as pq
        tasks_path = self.dataset_root / "meta" / "tasks.parquet"
        tasks_table = pq.read_table(tasks_path)
        self.tasks = {i: str(task) for i, task in enumerate(tasks_table.to_pandas().index)}
        print(f"  Loaded {len(self.tasks)} task(s)")

        # Build video maps
        self.video_maps = {}
        for cam_key in self.camera_keys:
            self.video_maps[cam_key] = self._build_video_map(cam_key)

        # Load samples from parquet
        self.samples = []
        self._load_samples()

        # Compute action statistics (q01/q99, NOT min/max)
        all_actions = np.array([s['action'] for s in self.samples])
        self.action_q01 = np.percentile(all_actions, 1, axis=0)  # 1st percentile
        self.action_q99 = np.percentile(all_actions, 99, axis=0)  # 99th percentile

        print(f"  Action normalization (q01/q99):")
        print(f"    q01: {self.action_q01}")
        print(f"    q99: {self.action_q99}")

        # Create action tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer, bins=256, min_action=-1, max_action=1)

        # Video cache
        self._video_cache = {}

    def _build_video_map(self, image_key: str) -> List[dict]:
        """Build mapping from global index to video files."""
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

    def _load_samples(self):
        """Load sample metadata from parquet files."""
        import pyarrow.parquet as pq

        data_dir = self.dataset_root / "data"
        parquet_files = sorted(data_dir.glob("**/*.parquet"))

        for pq_file in parquet_files:
            table = pq.read_table(pq_file)
            df = table.to_pandas()

            for idx, row in df.iterrows():
                sample = {
                    'global_index': int(row['index']),
                    'episode_index': int(row['episode_index']),
                    'frame_index': int(row['frame_index']),
                    'task_index': int(row['task_index']),
                    'action': np.array(row['action'], dtype=np.float32),
                    'state': np.array(row['observation.state'], dtype=np.float32),
                }
                self.samples.append(sample)

    def _extract_frame_from_video(self, image_key: str, global_index: int) -> Image.Image:
        """Extract a frame from video."""
        import cv2

        # Check pre-extracted frames first
        frames_dir = self.dataset_root / "frames" / image_key
        if frames_dir.exists():
            frame_path = frames_dir / f"{global_index:06d}.jpg"
            if frame_path.exists():
                return Image.open(frame_path).convert('RGB')

        # Fall back to video extraction
        video_map = self.video_maps.get(image_key, [])
        video_info = None

        for vi in video_map:
            if vi['start_idx'] <= global_index <= vi['end_idx']:
                video_info = vi
                break

        if video_info is None:
            raise FileNotFoundError(f"No video found for {image_key} at index {global_index}")

        frame_in_video = global_index - video_info['start_idx']
        video_file = video_info['path']

        # Use cached video capture
        cache_key = str(video_file)
        if cache_key not in self._video_cache:
            self._video_cache[cache_key] = cv2.VideoCapture(str(video_file))

        cap = self._video_cache[cache_key]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cap = cv2.VideoCapture(str(video_file))
            self._video_cache[cache_key] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_video)
            ret, frame = cap.read()

        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_in_video} from {video_file}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _combine_images(self, images: list) -> Image.Image:
        """Combine multiple camera images."""
        if len(images) == 1:
            return images[0]

        if self.image_layout == "horizontal":
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            combined = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for img in images:
                y_offset = (max_height - img.height) // 2
                combined.paste(img, (x_offset, y_offset))
                x_offset += img.width
            return combined

        raise ValueError(f"Unsupported layout: {self.image_layout}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        CRITICAL FIX: This now properly includes action tokens in the sequence.
        """
        sample = self.samples[idx]
        global_index = sample['global_index']

        # Get images from all cameras
        images = []
        for image_key in self.camera_keys:
            try:
                img = self._extract_frame_from_video(image_key, global_index)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load {image_key} for sample {idx}: {e}")
                images.append(Image.new('RGB', (640, 480), color=(128, 128, 128)))

        combined_image = self._combine_images(images)

        # Get task description
        task_desc = self.tasks.get(sample['task_index'], "pick up the object")

        # Normalize action to [-1, 1] using q01/q99
        raw_action = sample['action']
        normalized_action = 2.0 * (raw_action - self.action_q01) / (self.action_q99 - self.action_q01 + 1e-8) - 1.0
        normalized_action = np.clip(normalized_action, -1.0, 1.0)

        # CRITICAL FIX: Convert action to TOKEN STRING
        action_string = self.action_tokenizer(normalized_action)

        # CRITICAL FIX: Build conversation with action string as GPT response
        prompt_builder = PurePromptBuilder()
        prompt_builder.add_turn("human", f"What action should the robot take to {task_desc}?")
        prompt_builder.add_turn("gpt", action_string)

        # Get full prompt (instruction + action tokens as string)
        full_prompt = prompt_builder.get_prompt()

        # Tokenize EVERYTHING together (instruction + action tokens)
        input_ids = self.tokenizer(full_prompt, add_special_tokens=True).input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()

        # Process image
        # Note: We need to process the image separately since the processor expects (text, image)
        # But we already have the full text with actions, so we process image only
        pixel_values = self.processor.image_processor(combined_image, return_tensors="pt")["pixel_values"][0]

        # CRITICAL FIX: Mask labels - only compute loss on action tokens + EOS
        # The action tokens are the LAST (len(action) + 1) tokens (action + EOS)
        action_dim = len(raw_action)
        labels[: -(action_dim + 1)] = IGNORE_INDEX

        # Create attention mask (all 1s)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': labels,
        }

    def __del__(self):
        """Clean up video captures."""
        for cap in self._video_cache.values():
            cap.release()


def collate_fn(batch):
    """
    Custom collate function with proper padding.
    """
    # Find max sequence length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    # Pad sequences
    input_ids_list = []
    attention_masks_list = []
    pixel_values_list = []
    labels_list = []

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids with pad_token_id (typically 0)
        padded_ids = torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ])

        # Pad attention_mask with 0
        padded_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_len, dtype=torch.long)
        ])

        # Pad labels with IGNORE_INDEX
        padded_labels = torch.cat([
            item['labels'],
            torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)
        ])

        input_ids_list.append(padded_ids)
        attention_masks_list.append(padded_mask)
        labels_list.append(padded_labels)
        pixel_values_list.append(item['pixel_values'])

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_masks_list),
        'pixel_values': torch.stack(pixel_values_list),
        'labels': torch.stack(labels_list),
    }


def validate_cameras(cfg: FinetuneConfig) -> bool:
    """
    Validate camera configuration by capturing and displaying test frames.
    Returns True if user confirms cameras are correct.
    """
    import cv2
    import subprocess

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

    # Open images with system default viewer
    combined_image_path = None
    if len(test_frames) > 1:
        combined_image_path = output_dir / "camera_validation_combined.jpg"

    try:
        if sys.platform == "win32":
            image_to_show = combined_image_path if combined_image_path else saved_paths[0]
            subprocess.Popen(["start", "", str(image_to_show)], shell=True)
            print(f"Opened: {image_to_show}")
        else:
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

    # Ask user for confirmation (default to yes)
    while True:
        response = input("Are the cameras configured correctly? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            print()
            return True
        elif response in ['n', 'no']:
            print()
            print("Please update config.json with the correct camera settings and try again.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no (or just press Enter for yes).")


def main():
    cfg = FinetuneConfig()

    print("=" * 70)
    print("OpenVLA Fine-tuning for SO-100 (FIXED VERSION)")
    print("=" * 70)
    print()
    print("Key Fixes:")
    print("  - Action tokens included in input sequence")
    print("  - Labels properly masked (loss only on actions)")
    print("  - q01/q99 normalization (not min/max)")
    print("  - 4-bit quantization (fits in 32GB VRAM)")
    print()

    # Validate cameras first
    if not validate_cameras(cfg):
        print("Camera validation failed. Exiting.")
        return 1

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Training on CPU will be very slow!")
    print()

    # Load model and processor
    print("Loading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    # 4-bit quantization (required to fit in 32GB VRAM)
    quantization_config = None
    if cfg.use_quantization:
        print("  Using 4-bit quantization (NF4) to fit in VRAM")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if cfg.use_quantization:
        model = prepare_model_for_kbit_training(model)
    else:
        model = model.to(device)

    print("[OK] Model loaded")
    print()

    # Apply LoRA
    if cfg.use_lora:
        print("Applying LoRA...")
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

    # Load dataset
    print("Loading dataset...")
    full_dataset = LeRobotOpenVLADataset(
        dataset_root=cfg.dataset_root,
        robot_config_path=cfg.robot_config_path,
        processor=processor,
        tokenizer=processor.tokenizer,
        split="train",
    )

    print(f"  Total samples: {len(full_dataset)}")

    # Split into train/val
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.seed(cfg.val_seed)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - cfg.val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print()

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Save normalization stats
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = {
        'action_q01': full_dataset.action_q01.tolist(),
        'action_q99': full_dataset.action_q99.tolist(),
    }
    with open(output_dir / "action_norm_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)

    # Optimizer and scheduler
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.max_steps - cfg.warmup_steps)

    # Initialize W&B
    if cfg.use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=vars(cfg),
            name=f"openvla-fixed-{time.strftime('%Y%m%d-%H%M%S')}",
        )

    # Validation function
    def evaluate_on_validation():
        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
                labels = batch['labels'].to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                    )
                    loss = outputs.loss

                val_losses.append(loss.item())

        model.train()
        return {'val_loss': np.mean(val_losses)}

    # Training loop
    print("=" * 70)
    print("Starting training...")
    print(f"  Batch size: {cfg.batch_size} x {cfg.grad_accumulation_steps} = {cfg.batch_size * cfg.grad_accumulation_steps}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Max steps: {cfg.max_steps}")
    print("=" * 70)
    print()

    model.train()
    global_step = 0
    total_loss = 0
    optimizer.zero_grad()

    best_val_loss = float('inf')
    best_checkpoint_step = 0

    pbar = tqdm(total=cfg.max_steps, desc="Training")

    while global_step < cfg.max_steps:
        for batch_idx, batch in enumerate(train_dataloader):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            labels = batch['labels'].to(device)

            # Forward pass
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss

            # Backward pass
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()
            total_loss += loss.item()

            # Gradient accumulation step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
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

                # Validation
                if global_step % cfg.eval_freq == 0:
                    print("\n  Running validation...")
                    val_metrics = evaluate_on_validation()

                    print(f"  Step {global_step}:")
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

                        best_path = output_dir / "best_checkpoint"
                        best_path.mkdir(parents=True, exist_ok=True)

                        model.save_pretrained(best_path)
                        processor.save_pretrained(best_path)

                        with open(best_path / "action_norm_stats.json", 'w') as f:
                            json.dump(norm_stats, f, indent=2)

                        with open(best_path / "checkpoint_info.json", 'w') as f:
                            json.dump({
                                'step': global_step,
                                'val_loss': val_metrics['val_loss'],
                            }, f, indent=2)

                        print(f"    ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")

                # Save periodic checkpoint
                if global_step % cfg.save_freq == 0:
                    checkpoint_path = output_dir / f"checkpoint_{global_step}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)

                    model.save_pretrained(checkpoint_path)
                    processor.save_pretrained(checkpoint_path)

                    with open(checkpoint_path / "action_norm_stats.json", 'w') as f:
                        json.dump(norm_stats, f, indent=2)

                    print(f"\n  Saved checkpoint at step {global_step}")

                if global_step >= cfg.max_steps:
                    break

    pbar.close()

    # Save final checkpoint
    final_path = output_dir / "final_checkpoint"
    final_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    with open(final_path / "action_norm_stats.json", 'w') as f:
        json.dump(norm_stats, f, indent=2)

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest checkpoint: step {best_checkpoint_step}, val_loss={best_val_loss:.4f}")
    print(f"Saved to: {output_dir / 'best_checkpoint'}")
    print()

    if cfg.use_wandb:
        wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
