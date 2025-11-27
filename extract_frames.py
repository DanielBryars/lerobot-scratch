#!/usr/bin/env python
"""
Extract video frames to images for faster training data loading.
This bypasses the slow CPU-based video decoding during training.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import av
from tqdm import tqdm
import shutil


def extract_frames(dataset_path: Path, output_path: Path):
    """Extract all video frames to images."""

    # Read dataset info
    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)

    with open(dataset_path / "meta" / "features.json") as f:
        features = json.load(f)

    print(f"Dataset info:")
    print(f"  Total episodes: {info['total_episodes']}")
    print(f"  Total frames: {info['total_frames']}")
    print(f"  FPS: {info['fps']}")

    # Find video features
    video_features = {}
    for key, feature_info in features.items():
        if feature_info.get("dtype") == "video":
            video_features[key] = feature_info
            print(f"  Video feature: {key}")

    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    meta_output = output_path / "meta"
    if meta_output.exists():
        shutil.rmtree(meta_output)
    shutil.copytree(dataset_path / "meta", meta_output)

    # Copy data directory (parquet files)
    data_output = output_path / "data"
    if data_output.exists():
        shutil.rmtree(data_output)
    shutil.copytree(dataset_path / "data", data_output)

    # Process each video feature
    for feature_key, feature_info in video_features.items():
        print(f"\nExtracting frames for: {feature_key}")

        # Get video paths pattern
        video_path = feature_info.get("video_path", feature_info.get("path", ""))

        # Create images directory
        # Convert path like "videos/observation.images.base_0_rgb/episode_{episode_index:06d}.mp4"
        # to "images/observation.images.base_0_rgb/frame_{frame_index:06d}.png"
        img_dir_name = feature_key.replace(".", "_")
        images_dir = output_path / "images" / img_dir_name
        images_dir.mkdir(parents=True, exist_ok=True)

        # Process each episode
        videos_dir = dataset_path / "videos" / feature_key.replace("observation.images.", "").replace("_rgb", "") + "_rgb"

        # Find all video files
        video_dir = dataset_path / "videos" / feature_key.replace("observation.images.", "")
        if not video_dir.exists():
            # Try alternate path structure
            parts = feature_key.split(".")
            if len(parts) >= 3:
                video_dir = dataset_path / "videos" / parts[-1]

        if not video_dir.exists():
            # Search for the video directory
            videos_root = dataset_path / "videos"
            if videos_root.exists():
                for subdir in videos_root.iterdir():
                    if subdir.is_dir():
                        video_files = list(subdir.glob("*.mp4"))
                        if video_files:
                            video_dir = subdir
                            print(f"  Found video directory: {video_dir}")
                            break

        if not video_dir.exists():
            print(f"  Warning: Could not find video directory for {feature_key}")
            continue

        video_files = sorted(video_dir.glob("*.mp4"))
        print(f"  Found {len(video_files)} video files in {video_dir}")

        frame_index = 0
        for video_file in tqdm(video_files, desc=f"Processing {feature_key}"):
            # Open video
            container = av.open(str(video_file))
            stream = container.streams.video[0]

            for frame in container.decode(stream):
                # Convert to PIL Image
                img = frame.to_image()

                # Save as PNG (lossless) or JPG (faster to load)
                img_path = images_dir / f"frame_{frame_index:06d}.png"
                img.save(img_path)

                frame_index += 1

            container.close()

        print(f"  Extracted {frame_index} frames to {images_dir}")

        # Update features.json to use images instead of video
        new_feature_info = {
            "dtype": "image",
            "shape": feature_info["shape"],
            "names": feature_info.get("names", ["channel", "height", "width"]),
            "image_path": f"images/{img_dir_name}/frame_{{frame_index:06d}}.png",
        }
        features[feature_key] = new_feature_info

    # Update info.json to indicate images instead of videos
    info["video_backend"] = None  # Disable video backend

    # Save updated metadata
    with open(meta_output / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(meta_output / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"\nDone! Image dataset saved to: {output_path}")
    print("You can now use this dataset for faster training.")


if __name__ == "__main__":
    dataset_path = Path("datasets/20251124_233735_5hz")
    output_path = Path("datasets/20251124_233735_5hz_images")

    extract_frames(dataset_path, output_path)
