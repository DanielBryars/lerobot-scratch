#!/usr/bin/env python3
"""
Discovery script for SO100 robots and USB cameras.
Use this to identify devices before configuring inference.py
"""

import subprocess
import sys
from pathlib import Path


def find_serial_ports():
    """Find all USB serial ports that could be SO100 robots."""
    print("=" * 60)
    print("SEARCHING FOR SO100 ROBOTS (USB Serial Ports)")
    print("=" * 60)

    serial_devices = []

    # Check common serial port patterns
    port_patterns = ["/dev/ttyUSB*", "/dev/ttyACM*"]

    for pattern in port_patterns:
        try:
            result = subprocess.run(
                f"ls {pattern} 2>/dev/null",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.stdout:
                devices = result.stdout.strip().split('\n')
                serial_devices.extend([d for d in devices if d])
        except Exception as e:
            print(f"Error checking {pattern}: {e}")

    if serial_devices:
        print(f"\nFound {len(serial_devices)} serial port(s):")
        for i, device in enumerate(serial_devices, 1):
            print(f"  {i}. {device}")

            # Try to get more info about the device
            try:
                result = subprocess.run(
                    f"udevadm info --name={device} 2>/dev/null | grep -E 'ID_VENDOR|ID_MODEL|ID_SERIAL'",
                    shell=True,
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    info_lines = result.stdout.strip().split('\n')
                    for line in info_lines:
                        print(f"     {line}")
            except:
                pass

        print("\nConfiguration for inference.py:")
        print(f'  follower_port = "{serial_devices[0]}"  # First detected port')

    else:
        print("\nNo serial ports found.")
        print("Make sure:")
        print("  1. SO100 robot is connected via USB")
        print("  2. USB devices are shared to WSL (run win-share-usb.cmd)")
        print("  3. You have permissions (you may need to add user to dialout group)")

    print()
    return serial_devices


def find_cameras():
    """Find all USB cameras."""
    print("=" * 60)
    print("SEARCHING FOR USB CAMERAS")
    print("=" * 60)

    cameras = []

    # Check for video devices
    try:
        result = subprocess.run(
            "ls /dev/video* 2>/dev/null",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            devices = result.stdout.strip().split('\n')
            cameras = [d for d in devices if d]
    except Exception as e:
        print(f"Error checking video devices: {e}")

    if cameras:
        print(f"\nFound {len(cameras)} video device(s):")

        # Group devices (each camera typically has multiple /dev/videoX entries)
        # We only care about the base indices
        base_indices = set()
        camera_info = {}  # Store camera details

        for device in cameras:
            try:
                # Extract index from /dev/videoX
                idx = device.replace('/dev/video', '')
                print(f"\n  {device}")

                # Get camera info with v4l2-ctl
                card_type = None
                try:
                    result = subprocess.run(
                        f"v4l2-ctl --device={device} --info 2>/dev/null",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=2
                    )
                    if result.stdout:
                        # Extract card name
                        for line in result.stdout.split('\n'):
                            if 'Card type' in line:
                                card_type = line.strip().replace('Card type', '').replace(':', '').strip()
                                print(f"     Card type: {card_type}")
                                base_indices.add(int(idx))
                                break
                except subprocess.TimeoutExpired:
                    print(f"     (timeout querying device)")
                except:
                    pass

                # Get serial number and USB path with udevadm
                try:
                    result = subprocess.run(
                        f"udevadm info {device} 2>/dev/null | grep -E 'ID_SERIAL_SHORT=|ID_PATH='",
                        shell=True,
                        capture_output=True,
                        text=True
                    )
                    if result.stdout:
                        serial = None
                        usb_path = None
                        for line in result.stdout.split('\n'):
                            if 'ID_SERIAL_SHORT=' in line:
                                serial = line.split('=')[1].strip()
                            elif 'ID_PATH=' in line and 'ID_PATH_' not in line:
                                usb_path = line.split('=')[1].strip()
                                # Extract just the USB port info
                                if 'usb-0:' in usb_path:
                                    usb_port = usb_path.split('usb-0:')[1].split(':')[0]
                                    usb_path = f"USB port {usb_port}"

                        if serial:
                            print(f"     Serial: {serial}")
                        if usb_path:
                            print(f"     {usb_path}")

                        # Store info for later
                        if card_type and int(idx) in base_indices:
                            camera_info[int(idx)] = {
                                'card': card_type,
                                'serial': serial,
                                'usb_path': usb_path
                            }
                except:
                    pass

            except:
                pass

        print("\n" + "=" * 60)
        print("Configuration for inference.py:")
        print("=" * 60)
        print("camera_config = {")

        # Generate config for up to 3 cameras (Pi0 expects base, left_wrist, right_wrist)
        camera_names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        sorted_indices = sorted(base_indices)

        for i, name in enumerate(camera_names):
            if i < len(sorted_indices):
                idx = sorted_indices[i]
                comment = ""
                if idx in camera_info:
                    info = camera_info[idx]
                    if info['serial'] and info['usb_path']:
                        comment = f"  # {info['card']} - Serial: {info['serial']} - {info['usb_path']}"
                    elif info['serial']:
                        comment = f"  # {info['card']} - Serial: {info['serial']}"
                    else:
                        comment = f"  # {info['card']}"
                print(f'    "{name}": OpenCVCameraConfig(index_or_path={idx}, width=640, height=480, fps=30),{comment}')
            else:
                print(f'    # "{name}": OpenCVCameraConfig(index_or_path=?, width=640, height=480, fps=30),  # Not detected')

        print("}")
        print("\nNOTE: Cameras with identical serial numbers can only be differentiated by USB port.")
        print("Keep them plugged into the same USB ports to maintain consistent /dev/video indices.")

    else:
        print("\nNo video devices found.")
        print("Make sure:")
        print("  1. USB cameras are connected")
        print("  2. USB devices are shared to WSL (run win-share-usb.cmd)")
        print("  3. v4l2 drivers are loaded")
        print("\nYou can try: sudo modprobe uvcvideo")

    print()
    return cameras


def check_permissions():
    """Check if user has necessary permissions."""
    print("=" * 60)
    print("CHECKING PERMISSIONS")
    print("=" * 60)

    import os
    import grp
    import getpass

    try:
        current_user = getpass.getuser()
    except:
        current_user = os.environ.get('USER', 'unknown')

    # Check dialout group (for serial ports)
    try:
        dialout_group = grp.getgrnam('dialout')
        if current_user in dialout_group.gr_mem:
            print(f" User '{current_user}' is in 'dialout' group (serial port access)")
        else:
            print(f" User '{current_user}' is NOT in 'dialout' group")
            print(f"  To fix: sudo usermod -a -G dialout {current_user}")
            print(f"  Then log out and back in")
    except KeyError:
        print("  'dialout' group not found")

    # Check video group (for cameras)
    try:
        video_group = grp.getgrnam('video')
        if current_user in video_group.gr_mem:
            print(f" User '{current_user}' is in 'video' group (camera access)")
        else:
            print(f" User '{current_user}' is NOT in 'video' group")
            print(f"  To fix: sudo usermod -a -G video {current_user}")
            print(f"  Then log out and back in")
    except KeyError:
        print("  'video' group not found")

    print()


def main():
    """Main discovery routine."""
    print("\n" + "=" * 60)
    print("LeRobot Device Discovery")
    print("=" * 60)
    print()

    # Check permissions first
    check_permissions()

    # Find devices
    serial_ports = find_serial_ports()
    cameras = find_cameras()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Serial ports found: {len(serial_ports)}")
    print(f"Video devices found: {len(cameras)}")
    print()

    if not serial_ports and not cameras:
        print("No devices detected. See troubleshooting tips above.")
        print("\nFor WSL users:")
        print("  1. Run win-share-usb.cmd in Windows (as Administrator)")
        print("  2. Run wsl-share-usb.sh in WSL to verify")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
