#!/usr/bin/env python3
"""Test camera access with different backends."""

import cv2
import time

def test_camera(index, backend_name, backend):
    """Test opening a camera with a specific backend."""
    print(f"\nTesting Camera {index} with {backend_name}...")
    try:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            print(f"  [FAILED] Could not open camera {index}")
            return False

        # Try to read a frame
        ret, frame = cap.read()
        if not ret:
            print(f"  [FAILED] Could not read frame from camera {index}")
            cap.release()
            return False

        print(f"  [OK] Camera {index} opened successfully!")
        print(f"       Frame shape: {frame.shape}")
        print(f"       Backend: {cap.getBackendName()}")

        cap.release()
        return True

    except Exception as e:
        print(f"  [ERROR] Exception: {e}")
        return False

# Test different backends
backends = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),      # DirectShow (default on Windows)
    ("CAP_MSMF", cv2.CAP_MSMF),        # Microsoft Media Foundation
    ("CAP_ANY", cv2.CAP_ANY),          # Auto-select
]

print("=" * 70)
print("Camera Access Test")
print("=" * 70)

for cam_idx in [0, 1, 2]:
    print(f"\n{'='*70}")
    print(f"Testing Camera Index {cam_idx}")
    print(f"{'='*70}")

    for backend_name, backend in backends:
        success = test_camera(cam_idx, backend_name, backend)
        if success:
            print(f"\n  --> Camera {cam_idx} works with {backend_name}")
            break
        time.sleep(0.5)  # Brief delay between attempts
    else:
        print(f"\n  --> Camera {cam_idx} FAILED with all backends")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
