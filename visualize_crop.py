#!/usr/bin/env python3
"""
Visualize what the 84x84 center crop looks like on camera images.
This shows exactly what visual information the diffusion policy receives.
"""

import cv2
import numpy as np
from pathlib import Path

def draw_crop_region(img, crop_h=84, crop_w=84):
    """Draw the center crop region on the image."""
    h, w = img.shape[:2]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2

    # Draw rectangle showing crop region
    img_with_box = img.copy()
    cv2.rectangle(img_with_box,
                  (start_w, start_h),
                  (start_w + crop_w, start_h + crop_h),
                  (0, 255, 0), 2)

    # Extract the crop
    crop = img[start_h:start_h + crop_h, start_w:start_w + crop_w]

    return img_with_box, crop

def main():
    # Check for debug images from inference
    debug_files = list(Path(".").glob("debug_step*_*.jpg"))

    if debug_files:
        print(f"Found {len(debug_files)} debug images from inference")
        for f in sorted(debug_files):
            img = cv2.imread(str(f))
            if img is not None:
                img_with_box, crop = draw_crop_region(img)

                # Save visualization
                base_name = f.stem
                cv2.imwrite(f"{base_name}_with_crop_box.jpg", img_with_box)
                cv2.imwrite(f"{base_name}_cropped_84x84.jpg", crop)
                print(f"  {f.name} -> {base_name}_with_crop_box.jpg, {base_name}_cropped_84x84.jpg")
    else:
        print("No debug images found. Run inference first, or testing with cameras...")

        # Try to capture from cameras
        import json
        config_path = Path("config.json")
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            print("\nCapturing from cameras...")
            for cam_name, cam_cfg in config["cameras"].items():
                cap = cv2.VideoCapture(cam_cfg["index_or_path"])
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

                ret, frame = cap.read()
                if ret:
                    img_with_box, crop = draw_crop_region(frame)
                    cv2.imwrite(f"camera_{cam_name}_full.jpg", frame)
                    cv2.imwrite(f"camera_{cam_name}_with_crop_box.jpg", img_with_box)
                    cv2.imwrite(f"camera_{cam_name}_cropped_84x84.jpg", crop)
                    print(f"  {cam_name}: saved full, with_crop_box, and cropped_84x84")
                else:
                    print(f"  {cam_name}: failed to capture")
                cap.release()

    print("\nDone! Check the generated images to see:")
    print("  - Green box shows what region gets cropped")
    print("  - *_cropped_84x84.jpg shows exactly what the model sees")
    print("\nIs the block visible in the 84x84 crop?")

if __name__ == "__main__":
    main()
