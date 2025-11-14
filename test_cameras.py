"""
Simple script to test and identify camera indices.
Captures a frame from each camera and saves it as an image file.
"""

import cv2
import time

def test_camera(index):
    """Open a camera and capture a test frame."""
    print(f"\nTesting Camera #{index}...", end=" ")

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera {index}")
        return False

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
            return False
        time.sleep(0.1)

    # Capture the final frame
    ret, frame = cap.read()
    if ret:
        filename = f"camera_{index}_test.jpg"
        cv2.imwrite(filename, frame)
        print(f"OK - Captured image saved as '{filename}'")
        cap.release()
        return True
    else:
        print(f"ERROR: Could not capture frame from camera {index}")
        cap.release()
        return False

def main():
    print("\n" + "="*60)
    print("CAMERA IDENTIFICATION TOOL")
    print("="*60)
    print("\nThis will capture frames from cameras 0, 1, and 2.")
    print("Images will be saved as camera_0_test.jpg, camera_1_test.jpg, etc.")
    print("\nOpen the images to identify:")
    print("  - Which camera is OVERHEAD?")
    print("  - Which camera is WRIST?")
    print("="*60)

    successful_cameras = []

    # Test each camera
    for camera_index in [0, 1, 2]:
        if test_camera(camera_index):
            successful_cameras.append(camera_index)

    print("\n" + "="*60)
    if successful_cameras:
        print(f"Successfully captured from {len(successful_cameras)} camera(s)")
        print(f"Camera indices: {successful_cameras}")
        print("\nNext steps:")
        print("1. Open the captured image files")
        print("2. Identify which index is overhead and which is wrist")
        print("3. Update your config.json with the correct mappings:")
        print('   "overhead": {"index_or_path": X, ...}')
        print('   "wrist": {"index_or_path": Y, ...}')
    else:
        print("No cameras could be accessed.")
        print("Make sure:")
        print("  - Cameras are connected")
        print("  - No other application is using the cameras")
        print("  - USB sharing is configured (if using WSL)")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
