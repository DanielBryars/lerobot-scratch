#!/usr/bin/env python3
"""
Auto-configure SO100 robots and USB cameras (Windows version).
Automatically detects devices and creates config.json

Hardware mapping:
- Follower arm: Serial ending with 764
- Leader arm: Serial ending with 835
- Overhead camera: Nuroum (base_0_rgb)
- Wrist camera: ARC (left_wrist_0_rgb)
"""

import sys
import json
from pathlib import Path


def find_serial_ports():
    """Find SO100 robots by serial number."""
    ports = {}

    try:
        import serial.tools.list_ports

        print("Scanning serial ports...")
        detected_ports = serial.tools.list_ports.comports()

        for port in detected_ports:
            serial_num = port.serial_number
            if serial_num:
                if serial_num.endswith('764'):
                    ports['follower'] = {'port': port.device, 'serial': serial_num}
                    print(f"✓ Found follower arm: {port.device} (serial: {serial_num})")
                elif serial_num.endswith('835'):
                    ports['leader'] = {'port': port.device, 'serial': serial_num}
                    print(f"✓ Found leader arm: {port.device} (serial: {serial_num})")

    except ImportError:
        print("⚠ Warning: pyserial not installed. Run: pip install pyserial")
    except Exception as e:
        print(f"⚠ Error scanning serial ports: {e}")

    return ports


def find_cameras():
    """Find cameras by name (Nuroum = overhead, ARC = wrist)."""
    cameras = {}

    try:
        import cv2

        print("Scanning cameras...")

        # We need to use Windows Media Foundation to get camera names
        # However, OpenCV doesn't provide an easy way to get camera names on Windows
        # So we'll test each index and try to identify by resolution/properties

        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    if fps <= 0:
                        fps = 30

                    # Try to identify camera by resolution
                    # Nuroum typically uses 640x360, ARC/USB2.0 uses 640x480
                    if height == 360 and 'base_0_rgb' not in cameras:
                        cameras['base_0_rgb'] = {
                            'index_or_path': i,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'description': f"Camera {i} (likely Nuroum overhead - {width}x{height})"
                        }
                        print(f"✓ Found overhead camera: Camera {i} ({width}x{height} @ {fps} fps)")
                    elif height == 480 and 'left_wrist_0_rgb' not in cameras:
                        cameras['left_wrist_0_rgb'] = {
                            'index_or_path': i,
                            'width': width,
                            'height': height,
                            'fps': fps,
                            'description': f"Camera {i} (likely ARC wrist - {width}x{height})"
                        }
                        print(f"✓ Found wrist camera: Camera {i} ({width}x{height} @ {fps} fps)")

                cap.release()

        # If we only found one camera and it's 480p, it might be the overhead
        # User will need to verify the config
        if len(cameras) == 1 and 'left_wrist_0_rgb' in cameras:
            print("\n⚠ Note: Only found one 480p camera. Please verify camera assignment in config.json")

    except ImportError:
        print("⚠ Warning: opencv-python not installed. Run: pip install opencv-python")
    except Exception as e:
        print(f"⚠ Error scanning cameras: {e}")
        import traceback
        traceback.print_exc()

    return cameras


def get_camera_names_wmf():
    """Try to get actual camera names using Windows Management Framework."""
    try:
        import wmi
        c = wmi.WMI()
        cameras = []
        for camera in c.Win32_PnPEntity():
            if camera.Caption and ('camera' in camera.Caption.lower() or 'webcam' in camera.Caption.lower()):
                cameras.append(camera.Caption)
        return cameras
    except ImportError:
        # WMI not available, that's okay
        return None
    except:
        return None


def create_config(ports, cameras):
    """Create config.json from detected devices."""
    config = {
        "leader": {
            "port": ports.get('leader', {}).get('port', 'COM3'),
            "id": "leader_so100"
        },
        "follower": {
            "port": ports.get('follower', {}).get('port', 'COM4'),
            "id": "follower_so100"
        },
        "cameras": cameras
    }

    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Created {config_path}")
    return config


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import serial.tools.list_ports
    except ImportError:
        missing.append('pyserial')

    try:
        import cv2
    except ImportError:
        missing.append('opencv-python')

    if missing:
        print("⚠ Missing dependencies:")
        for pkg in missing:
            print(f"  - {pkg}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        print()
        return False

    return True


def main():
    """Main auto-configuration routine."""
    print("=" * 60)
    print("Auto-configuring SO100 LeRobot setup (Windows)")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return 1

    # Find devices
    ports = find_serial_ports()
    cameras = find_cameras()

    # Check what we found
    print()
    if not ports.get('follower'):
        print("⚠ Warning: Follower arm (serial ending 764) not found")
    if not ports.get('leader'):
        print("⚠ Warning: Leader arm (serial ending 835) not found")
    if 'base_0_rgb' not in cameras:
        print("⚠ Warning: Overhead camera (360p) not found")
    if 'left_wrist_0_rgb' not in cameras:
        print("⚠ Warning: Wrist camera (480p) not found")

    # Show available WMF camera names if possible
    wmf_cameras = get_camera_names_wmf()
    if wmf_cameras:
        print("\nDetected camera devices in Windows:")
        for cam in wmf_cameras:
            print(f"  - {cam}")

    # Create config
    if ports or cameras:
        print()
        config = create_config(ports, cameras)
        print("\nConfiguration summary:")
        print(json.dumps(config, indent=2))
        print("\n⚠ Note: Camera assignments are based on resolution.")
        print("Please verify the camera mapping is correct for your setup.")
        return 0
    else:
        print("\n✗ No devices found")
        print("\nTroubleshooting:")
        print("  1. Ensure USB devices are connected")
        print("  2. Check Device Manager for COM ports and cameras")
        print("  3. Install drivers if needed")
        print("  4. Close other applications using the cameras")
        return 1


if __name__ == "__main__":
    sys.exit(main())
