#!/usr/bin/env python3
"""
Auto-configure SO100 robots and USB cameras.
Automatically detects devices and creates config.json

Hardware mapping:
- Follower arm: Serial ending with 764
- Leader arm: Serial ending with 835
- Overhead camera: Nuroum (base_0_rgb)
- Wrist camera: ARC (left_wrist_0_rgb)
"""

import subprocess
import sys
import json
from pathlib import Path


def run_command(cmd):
    """Run shell command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return ""


def get_serial_info(device):
    """Get serial number from a serial port device."""
    output = run_command(f"udevadm info --name={device} 2>/dev/null | grep ID_SERIAL_SHORT")
    if output and '=' in output:
        return output.split('=')[1].strip()
    return None


def get_camera_info(device):
    """Get camera card type."""
    output = run_command(f"v4l2-ctl --device={device} --info 2>/dev/null")
    for line in output.split('\n'):
        if 'Card type' in line:
            return line.split(':', 1)[1].strip()
    return None


def find_serial_ports():
    """Find SO100 robots by serial number."""
    ports = {}

    # Find all serial devices
    for pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
        devices = run_command(f"ls {pattern} 2>/dev/null").split('\n')

        for device in devices:
            if not device:
                continue

            serial = get_serial_info(device)
            if serial:
                if serial.endswith('764'):
                    ports['follower'] = {'port': device, 'serial': serial}
                    print(f"✓ Found follower arm: {device} (serial: {serial})")
                elif serial.endswith('835'):
                    ports['leader'] = {'port': device, 'serial': serial}
                    print(f"✓ Found leader arm: {device} (serial: {serial})")

    return ports


def find_cameras():
    """Find cameras by name (Nuroum = overhead, ARC = wrist)."""
    cameras = {}

    # Get all video devices
    devices = run_command("ls /dev/video* 2>/dev/null").split('\n')

    # Track seen cameras to avoid duplicates (each camera has multiple /dev/videoX)
    seen_cameras = set()

    for device in devices:
        if not device:
            continue

        card_type = get_camera_info(device)
        if not card_type or card_type in seen_cameras:
            continue

        seen_cameras.add(card_type)
        idx = device.replace('/dev/video', '')

        # Match camera by name
        if 'Nuroum' in card_type:
            cameras['base_0_rgb'] = {
                'index_or_path': device,
                'width': 640,
                'height': 360,
                'fps': 30,
                'description': f"{card_type} (overhead)"
            }
            print(f"✓ Found overhead camera: {device} ({card_type})")
        elif 'ARC' in card_type or 'USB2.0' in card_type:
            cameras['left_wrist_0_rgb'] = {
                'index_or_path': device,
                'width': 640,
                'height': 480,
                'fps': 30,
                'description': f"{card_type} (wrist)"
            }
            print(f"✓ Found wrist camera: {device} ({card_type})")

    return cameras


def create_config(ports, cameras):
    """Create config.json from detected devices."""
    config = {
        "leader": {
            "port": ports.get('leader', {}).get('port', '/dev/ttyACM0'),
            "id": "leader_so100"
        },
        "follower": {
            "port": ports.get('follower', {}).get('port', '/dev/ttyACM1'),
            "id": "follower_so100"
        },
        "cameras": cameras
    }

    config_path = Path(__file__).parent / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Created {config_path}")
    return config


def main():
    """Main auto-configuration routine."""
    print("=" * 60)
    print("Auto-configuring SO100 LeRobot setup")
    print("=" * 60)
    print()

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
        print("⚠ Warning: Nuroum overhead camera not found")
    if 'left_wrist_0_rgb' not in cameras:
        print("⚠ Warning: ARC wrist camera not found")

    # Create config
    if ports or cameras:
        print()
        config = create_config(ports, cameras)
        print("\nConfiguration summary:")
        print(json.dumps(config, indent=2))
        return 0
    else:
        print("\n✗ No devices found")
        print("\nTroubleshooting:")
        print("  1. Ensure USB devices are connected")
        print("  2. For WSL: Run win-share-usb.cmd in Windows (as Administrator)")
        print("  3. Check permissions: sudo usermod -a -G dialout,video $USER")
        return 1


if __name__ == "__main__":
    sys.exit(main())
