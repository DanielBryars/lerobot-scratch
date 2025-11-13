#!/usr/bin/env python3
"""
Discovery script for SO100 robots and USB cameras (Windows version).
Use this to identify devices before configuring inference.py
"""

import sys
from pathlib import Path


def find_serial_ports():
    """Find all USB serial ports that could be SO100 robots."""
    print("=" * 60)
    print("SEARCHING FOR SO100 ROBOTS (USB Serial Ports)")
    print("=" * 60)

    serial_devices = []

    try:
        import serial.tools.list_ports

        ports = serial.tools.list_ports.comports()

        if ports:
            print(f"\nFound {len(ports)} serial port(s):")
            for i, port in enumerate(ports, 1):
                serial_devices.append(port.device)
                print(f"  {i}. {port.device}")
                if port.description:
                    print(f"     Description: {port.description}")
                if port.manufacturer:
                    print(f"     Manufacturer: {port.manufacturer}")
                if port.serial_number:
                    print(f"     Serial: {port.serial_number}")
                if port.vid and port.pid:
                    print(f"     VID:PID: {port.vid:04X}:{port.pid:04X}")
                print()

            print("Configuration for inference.py:")
            print(f'  follower_port = "{serial_devices[0]}"  # First detected port')
        else:
            print("\nNo serial ports found.")
            print("Make sure:")
            print("  1. SO100 robot is connected via USB")
            print("  2. Drivers are installed")
            print("  3. Device appears in Device Manager under 'Ports (COM & LPT)'")

    except ImportError:
        print("\nERROR: pyserial not installed.")
        print("Install with: pip install pyserial")
        return []
    except Exception as e:
        print(f"\nError detecting serial ports: {e}")
        return []

    print()
    return serial_devices


def find_cameras():
    """Find all USB cameras."""
    print("=" * 60)
    print("SEARCHING FOR USB CAMERAS")
    print("=" * 60)

    cameras = []

    try:
        import cv2

        # Test up to 10 camera indices
        max_cameras = 10
        print(f"\nScanning camera indices 0-{max_cameras-1}...")

        camera_info = {}

        for i in range(max_cameras):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend for Windows
            if cap.isOpened():
                # Try to read a frame to verify it's actually working
                ret, frame = cap.read()
                if ret:
                    cameras.append(i)

                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))

                    camera_info[i] = {
                        'width': width,
                        'height': height,
                        'fps': fps if fps > 0 else 30
                    }

                    print(f"  Camera {i}: {width}x{height} @ {fps if fps > 0 else 'unknown'} fps")

                cap.release()

        if cameras:
            print(f"\nFound {len(cameras)} working camera(s)")

            print("\n" + "=" * 60)
            print("Configuration for inference.py:")
            print("=" * 60)
            print("camera_config = {")

            # Generate config for up to 3 cameras (Pi0 expects base, left_wrist, right_wrist)
            camera_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]

            for idx, name in enumerate(camera_names):
                if idx < len(cameras):
                    cam_idx = cameras[idx]
                    info = camera_info.get(cam_idx, {})
                    width = info.get('width', 640)
                    height = info.get('height', 480)
                    fps = info.get('fps', 30)

                    print(f'    "{name}": OpenCVCameraConfig(index_or_path={cam_idx}, width={width}, height={height}, fps={fps}),')
                else:
                    print(f'    # "{name}": OpenCVCameraConfig(index_or_path=?, width=640, height=480, fps=30),  # Not detected')

            print("}")
            print("\nNOTE: Camera indices may change if you plug/unplug cameras or restart.")
            print("For consistent identification, keep cameras plugged into the same USB ports.")

        else:
            print("\nNo working cameras found.")
            print("Make sure:")
            print("  1. USB cameras are connected")
            print("  2. Cameras appear in Device Manager under 'Cameras' or 'Imaging devices'")
            print("  3. No other application is using the cameras")
            print("  4. Camera drivers are installed")

    except ImportError:
        print("\nERROR: opencv-python not installed.")
        print("Install with: pip install opencv-python")
        return []
    except Exception as e:
        print(f"\nError detecting cameras: {e}")
        import traceback
        traceback.print_exc()
        return []

    print()
    return cameras


def check_dependencies():
    """Check if required Python packages are installed."""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)

    dependencies = {
        'serial': 'pyserial',
        'cv2': 'opencv-python'
    }

    missing = []

    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f" {package}: installed")
        except ImportError:
            print(f" {package}: NOT INSTALLED")
            missing.append(package)

    if missing:
        print(f"\nTo install missing packages:")
        print(f"  pip install {' '.join(missing)}")

    print()
    return len(missing) == 0


def main():
    """Main discovery routine."""
    print("\n" + "=" * 60)
    print("LeRobot Device Discovery (Windows)")
    print("=" * 60)
    print()

    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies and run again.")
        return 1

    # Find devices
    serial_ports = find_serial_ports()
    cameras = find_cameras()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Serial ports found: {len(serial_ports)}")
    print(f"Cameras found: {len(cameras)}")
    print()

    if not serial_ports and not cameras:
        print("No devices detected. See troubleshooting tips above.")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
