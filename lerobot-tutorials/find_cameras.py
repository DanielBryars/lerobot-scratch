"""Find available cameras on the system."""
import cv2

backends = [
    ("MSMF (Windows)", cv2.CAP_MSMF),
    ("DSHOW (DirectShow)", cv2.CAP_DSHOW),
    ("ANY (auto)", cv2.CAP_ANY),
]

for backend_name, backend in backends:
    print(f"\nScanning with {backend_name}...")
    available = []

    for i in range(10):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  Camera {i}: {w}x{h} - AVAILABLE")
                available.append(i)
            else:
                print(f"  Camera {i}: opened but no frame")
            cap.release()

    if available:
        print(f"  Found: {available}")
    else:
        print(f"  No cameras found")

print("\n" + "="*50)
print("If no cameras found, check:")
print("  1. Cameras are plugged in")
print("  2. No other app is using them")
print("  3. Windows Device Manager shows them")
