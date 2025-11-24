"""
Simple script to test and identify camera indices.
Captures a frame from each camera and saves it as an image file.
Then prompts user to identify cameras and automatically updates config.json.
"""

import cv2
import time
import json
from pathlib import Path

def test_camera(index):
    """Open a camera and capture a test frame. Returns camera info dict or None."""
    print(f"\nTesting Camera #{index}...", end=" ")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {index}")
        return None

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Wait a moment for camera to initialize
    time.sleep(0.5)

    # Read a few frames to let camera warm up
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Could not read frame from camera {index}")
            cap.release()
            return None
        time.sleep(0.1)

    # Capture the final frame
    ret, frame = cap.read()
    if ret:
        filename = f"camera_{index}_test.jpg"
        cv2.imwrite(filename, frame)

        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30  # Default to 30 if FPS detection fails

        print(f"OK - Captured image saved as '{filename}' ({width}x{height} @ {fps}fps)")
        cap.release()

        return {
            'index': index,
            'width': width,
            'height': height,
            'fps': fps,
            'filename': filename
        }
    else:
        print(f"ERROR: Could not capture frame from camera {index}")
        cap.release()
        return None

def get_user_input(prompt, valid_options):
    """Get validated user input."""
    while True:
        response = input(prompt).strip()
        if response in valid_options:
            return response
        print(f"Invalid input. Please enter one of: {', '.join(valid_options)}")

def update_config(camera_mapping):
    """Update config.json with camera mappings."""
    config_path = Path(__file__).parent / "config.json"

    # Load existing config or create default structure
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"\nLoaded existing config from {config_path}")
    else:
        config = {
            "leader": {
                "port": "COM3",
                "id": "leader_so100"
            },
            "follower": {
                "port": "COM4",
                "id": "follower_so100"
            },
            "cameras": {}
        }
        print(f"\nNo existing config found. Creating new one at {config_path}")

    # Update cameras section
    config['cameras'] = camera_mapping

    # Write back to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Successfully updated {config_path}")
    print("\nCamera configuration:")
    print(json.dumps(config['cameras'], indent=2))

def main():
    print("\n" + "="*60)
    print("CAMERA IDENTIFICATION TOOL")
    print("="*60)
    print("\nThis will capture frames from cameras 0, 1, and 2.")
    print("Images will be saved as camera_0_test.jpg, camera_1_test.jpg, etc.")
    print("="*60)

    camera_info = {}

    # Test each camera
    for camera_index in [0, 1, 2]:
        result = test_camera(camera_index)
        if result:
            camera_info[camera_index] = result

    print("\n" + "="*60)
    if not camera_info:
        print("No cameras could be accessed.")
        print("Make sure:")
        print("  - Cameras are connected")
        print("  - No other application is using the cameras")
        print("  - USB sharing is configured (if using WSL)")
        print("="*60 + "\n")
        return

    print(f"Successfully captured from {len(camera_info)} camera(s)")
    print(f"Camera indices: {list(camera_info.keys())}")

    print("\n" + "="*60)
    print("IDENTIFY CAMERAS")
    print("="*60)
    print("\nPlease open the captured images and identify your cameras.")
    print("\nLeRobot expects these camera names:")
    print("  - base_0_rgb: Overhead/base camera")
    print("  - left_wrist_0_rgb: Left wrist camera")
    print("="*60)

    # Prompt user for camera identification
    valid_indices = [str(i) for i in camera_info.keys()] + ['none', 'skip']

    base_index = None
    wrist_index = None

    print("\nWhich camera index is the OVERHEAD/BASE camera?")
    print(f"Available indices: {', '.join([str(i) for i in camera_info.keys()])}")
    print("Enter 'none' or 'skip' if you don't have this camera")
    base_input = get_user_input("> ", valid_indices)
    if base_input not in ['none', 'skip']:
        base_index = int(base_input)

    print("\nWhich camera index is the LEFT WRIST camera?")
    print(f"Available indices: {', '.join([str(i) for i in camera_info.keys()])}")
    print("Enter 'none' or 'skip' if you don't have this camera")
    wrist_input = get_user_input("> ", valid_indices)
    if wrist_input not in ['none', 'skip']:
        wrist_index = int(wrist_input)

    # Build camera mapping
    camera_mapping = {}

    if base_index is not None:
        info = camera_info[base_index]
        camera_mapping['base_0_rgb'] = {
            'index_or_path': base_index,
            'width': info['width'],
            'height': info['height'],
            'fps': info['fps'],
            'description': f"Camera {base_index} (overhead/base - {info['width']}x{info['height']})"
        }

    if wrist_index is not None:
        info = camera_info[wrist_index]
        camera_mapping['left_wrist_0_rgb'] = {
            'index_or_path': wrist_index,
            'width': info['width'],
            'height': info['height'],
            'fps': info['fps'],
            'description': f"Camera {wrist_index} (left wrist - {info['width']}x{info['height']})"
        }

    if not camera_mapping:
        print("\nNo cameras were mapped. Skipping config update.")
        print("="*60 + "\n")
        return

    # Update config.json
    update_config(camera_mapping)

    print("\n" + "="*60)
    print("DONE! Your cameras are now configured in config.json")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
