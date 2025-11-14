#!/usr/bin/env python3
"""
List all available USB cameras on Windows.
Shows camera indices, resolutions, and frame rates.
"""

import sys


def parse_usb_info(device_id):
    """Parse USB VID/PID from device ID string."""
    import re
    if not device_id:
        return None, None

    # Look for VID and PID in device ID (e.g., USB\VID_046D&PID_0825)
    vid_match = re.search(r'VID_([0-9A-F]{4})', device_id, re.IGNORECASE)
    pid_match = re.search(r'PID_([0-9A-F]{4})', device_id, re.IGNORECASE)

    vid = vid_match.group(1) if vid_match else None
    pid = pid_match.group(1) if pid_match else None

    return vid, pid


def get_usb_location(device_id):
    """Extract USB location/address from device ID."""
    import re
    if not device_id:
        return None

    # Look for USB location info (e.g., USB\VID_046D&PID_0825\5&2a8f0c2b&0&3)
    # The last part (after the last \) often contains location info
    parts = device_id.split('\\')
    if len(parts) >= 3:
        return parts[-1]  # Return the instance ID which contains location
    return None


def identify_camera_type(caption, vid, pid, device_id):
    """Identify camera type based on device info."""
    caption_lower = caption.lower() if caption else ""

    # Known camera identification
    if 'nuroum' in caption_lower or 'v11' in caption_lower:
        return 'Overhead (Nuroum V11)'
    elif 'usb2.0' in caption_lower and 'cam' in caption_lower:
        return 'Wrist (USB2.0_CAM1)'
    elif 'dell' in caption_lower and 'webcam' in caption_lower:
        return 'Built-in Laptop (Dell - not used)'
    elif 'remote desktop' in caption_lower:
        return 'Virtual (Remote Desktop - not used)'

    # Try VID/PID matching for known devices
    if vid and pid:
        vid_pid = f"{vid}:{pid}".upper()
        # Add known VID:PID mappings here as we discover them
        # For example: if vid_pid == "XXXX:YYYY": return "Known Camera Type"

    return None


def get_camera_names_wmi():
    """Try to get camera names from Windows Management Instrumentation."""
    try:
        import wmi
        c = wmi.WMI()
        cameras = {}

        for camera in c.Win32_PnPEntity():
            if camera.Caption and ('camera' in camera.Caption.lower() or 'webcam' in camera.Caption.lower() or
                                   'video' in camera.Caption.lower() or 'cam' in camera.Caption.lower() or
                                   'usb2.0' in camera.Caption.lower()):
                device_id = camera.DeviceID if hasattr(camera, 'DeviceID') else None
                vid, pid = parse_usb_info(device_id)
                usb_location = get_usb_location(device_id)
                camera_type = identify_camera_type(camera.Caption, vid, pid, device_id)

                # Try to extract some useful info
                cameras[camera.Caption] = {
                    'caption': camera.Caption,
                    'device_id': device_id,
                    'usb_location': usb_location,
                    'status': camera.Status if hasattr(camera, 'Status') else None,
                    'manufacturer': camera.Manufacturer if hasattr(camera, 'Manufacturer') else None,
                    'vid': vid,
                    'pid': pid,
                    'description': camera.Description if hasattr(camera, 'Description') else None,
                    'camera_type': camera_type,
                }
        return cameras
    except ImportError:
        return None
    except Exception as e:
        print(f"Note: Could not query WMI: {e}")
        return None


def test_camera_capabilities(index):
    """Test a camera and get its capabilities."""
    import cv2

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None

    # Try to read a frame to verify it's working
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Try to get backend name
    backend = cap.getBackendName()

    info = {
        'index': index,
        'width': width,
        'height': height,
        'fps': fps if fps > 0 else 'Unknown',
        'backend': backend,
        'frame_shape': frame.shape if frame is not None else None
    }

    cap.release()
    return info


def list_cameras():
    """List all available cameras."""
    print("=" * 70)
    print("USB Camera Detection (Windows)")
    print("=" * 70)
    print()

    try:
        import cv2
    except ImportError:
        print("ERROR: opencv-python not installed.")
        print("Install with: pip install opencv-python")
        return []

    # First, try to get camera names from WMI
    print("Detecting camera devices in Windows...")
    wmi_cameras = get_camera_names_wmi()

    if wmi_cameras is None:
        print("Note: WMI module not available. Install with: pip install wmi")
        print("      (Enhanced camera identification requires WMI)")
        print()
    elif wmi_cameras:
        print(f"Found {len(wmi_cameras)} camera device(s) in Device Manager:")
        for name, info in wmi_cameras.items():
            print(f"  • {name}")
            if info['camera_type']:
                print(f"    Type: {info['camera_type']}")
            if info['manufacturer']:
                print(f"    Manufacturer: {info['manufacturer']}")
            if info['description']:
                print(f"    Description: {info['description']}")
            if info['vid'] and info['pid']:
                print(f"    USB VID:PID: {info['vid']}:{info['pid']}")
            if info['usb_location']:
                print(f"    USB Location: {info['usb_location']}")
            if info['status']:
                print(f"    Status: {info['status']}")
        print()
    else:
        print("Note: No cameras found via WMI (Device Manager)")
        print()

    # Now test OpenCV camera indices
    print("Testing OpenCV camera indices (0-9)...")
    print()

    cameras = []
    max_index = 10

    for i in range(max_index):
        sys.stdout.write(f"\rScanning camera {i}...")
        sys.stdout.flush()

        info = test_camera_capabilities(i)
        if info:
            cameras.append(info)

    sys.stdout.write("\r" + " " * 50 + "\r")  # Clear the scanning line
    sys.stdout.flush()

    if cameras:
        print(f"Found {len(cameras)} working camera(s):\n")
        print("=" * 70)

        # Convert WMI cameras to list for easier indexing
        wmi_camera_list = list(wmi_cameras.values()) if wmi_cameras else []

        for idx, info in enumerate(cameras):
            print(f"Camera Index: {info['index']}")
            print(f"  Resolution: {info['width']}x{info['height']}")
            print(f"  Frame Rate: {info['fps']} fps" if isinstance(info['fps'], (int, float)) else f"  Frame Rate: {info['fps']}")
            print(f"  Backend: {info['backend']}")

            # Try to match with WMI info for better identification
            # Note: This assumes similar enumeration order, which may not always be accurate
            camera_type_found = False
            if wmi_camera_list and idx < len(wmi_camera_list):
                wmi_info = wmi_camera_list[idx]
                print(f"  Device Name: {wmi_info['caption']}")
                if wmi_info['camera_type']:
                    print(f"  Camera Type: {wmi_info['camera_type']}")
                if wmi_info['manufacturer']:
                    print(f"  Manufacturer: {wmi_info['manufacturer']}")
                if wmi_info['vid'] and wmi_info['pid']:
                    print(f"  USB ID: {wmi_info['vid']}:{wmi_info['pid']}")
                if wmi_info['usb_location']:
                    print(f"  USB Location: {wmi_info['usb_location']}")
                camera_type_found = True

            # Guess camera type based on resolution if not found via WMI
            if not camera_type_found:
                if info['height'] == 360:
                    print(f"  Likely Type: Overhead camera (Nuroum)")
                elif info['height'] == 480:
                    print(f"  Likely Type: Wrist camera (ARC/USB2.0)")

            print()

        print("=" * 70)
        print("OpenCV Configuration Examples:")
        print("=" * 70)
        print()

        # Generate example configs
        for i, info in enumerate(cameras):
            camera_name = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"][i] if i < 3 else f"camera_{i}"
            print(f'"{camera_name}": OpenCVCameraConfig(')
            print(f'    index_or_path={info["index"]},')
            print(f'    width={info["width"]},')
            print(f'    height={info["height"]},')
            fps_value = int(info["fps"]) if isinstance(info["fps"], (int, float)) and info["fps"] > 0 else 30
            print(f'    fps={fps_value}')
            print(f'),')
            print()

    else:
        print("No working cameras found.")
        print()
        print("Troubleshooting:")
        print("  1. Ensure USB cameras are connected")
        print("  2. Check Device Manager under 'Cameras' or 'Imaging devices'")
        print("  3. Close other applications that may be using the cameras")
        print("  4. Try unplugging and reconnecting the cameras")
        print("  5. Verify camera drivers are installed")
        print()

    return cameras


def main():
    """Main entry point."""
    cameras = list_cameras()

    if cameras:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
