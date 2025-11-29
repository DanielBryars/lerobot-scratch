"""
Diagnostic script to read and display motor positions without enabling torque.
Useful for verifying calibration is working correctly.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for custom STS3250 modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.robots.so100_follower import SO100FollowerConfig
from lerobot.teleoperators.so100_leader import SO100LeaderConfig
from SO100FollowerSTS3250 import SO100FollowerSTS3250
from SO100LeaderSTS3250 import SO100LeaderSTS3250

# Load configuration from config.json
config_path = Path(__file__).parent.parent / "config.json"
with open(config_path) as f:
    config = json.load(f)


def print_calibration(follower_bus, leader_bus):
    """Print calibration info for both robots - read directly from motor EEPROM."""
    print(f"\n{'='*80}")
    print("CALIBRATION (read from motor EEPROM)")
    print(f"{'='*80}")
    print(f"{'Motor':<15} | {'Follower':<30} | {'Leader':<30}")
    print(f"{'':<15} | {'offset / range / drive':<30} | {'offset / range / drive':<30}")
    print(f"{'-'*15}-+-{'-'*30}-+-{'-'*30}")

    # Read actual values from motor EEPROM (not the software calibration)
    f_eeprom = follower_bus.read_calibration()
    l_eeprom = leader_bus.read_calibration()

    for motor in follower_bus.motors:
        f_cal = f_eeprom[motor]
        l_cal = l_eeprom[motor]
        f_str = f"{f_cal.homing_offset:5d} / [{f_cal.range_min:4d},{f_cal.range_max:4d}] / {f_cal.drive_mode}"
        l_str = f"{l_cal.homing_offset:5d} / [{l_cal.range_min:4d},{l_cal.range_max:4d}] / {l_cal.drive_mode}"
        print(f"{motor:<15} | {f_str:<30} | {l_str:<30}")


def print_positions(follower_bus, leader_bus):
    """Print positions for both robots in a single table."""
    print(f"\n{'='*94}")
    print("POSITIONS")
    print(f"{'='*94}")
    print(f"{'Motor':<15} | {'Follower':<20} | {'Leader':<20} | {'Diff':<10} | {'Flip Diff':<10}")
    print(f"{'':<15} | {'Raw':>8} {'Norm':>10} | {'Raw':>8} {'Norm':>10} | {'Norm':>10} | {'Norm':>10}")
    print(f"{'-'*15}-+-{'-'*20}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}")

    # Read all positions
    f_norm = follower_bus.sync_read("Present_Position")
    l_norm = leader_bus.sync_read("Present_Position")

    for motor in follower_bus.motors:
        f_raw = follower_bus.read("Present_Position", motor, normalize=False)
        l_raw = leader_bus.read("Present_Position", motor, normalize=False)
        f_n = f_norm[motor]
        l_n = l_norm[motor]
        diff = f_n - l_n

        # What if we flipped the leader value? (4095 - raw)
        l_raw_flipped = 4095 - l_raw
        l_n_flipped = -l_n  # Flipping raw inverts normalized
        flip_diff = f_n - l_n_flipped

        print(f"{motor:<15} | {f_raw:>8} {f_n:>10.2f} | {l_raw:>8} {l_n:>10.2f} | {diff:>10.2f} | {flip_diff:>10.2f}")


def main():
    print("Echo Joint Angles - Reading motor positions")
    print("Press Ctrl+C to exit\n")

    # Create configs
    follower_config = SO100FollowerConfig(
        port=config["follower"]["port"],
        id=config["follower"]["id"],
    )
    leader_config = SO100LeaderConfig(
        port=config["leader"]["port"],
        id=config["leader"]["id"],
    )

    # Create robot instances
    follower = SO100FollowerSTS3250(follower_config)
    leader = SO100LeaderSTS3250(leader_config)

    try:
        # Connect (reads calibration from motors)
        print("Connecting to follower...")
        follower.connect()
        print("Connecting to leader...")
        leader.connect()

        # Show calibration once
        print_calibration(follower.bus, leader.bus)

        while True:
            print_positions(follower.bus, leader.bus)

            print(f"\nRefreshing in 1 second... (Ctrl+C to exit)")
            time.sleep(1)

            # Clear screen and reprint calibration
            os.system('cls' if os.name == 'nt' else 'clear')
            print_calibration(follower.bus, leader.bus)

    except KeyboardInterrupt:
        print("\n\nExiting...")
    finally:
        if follower.is_connected:
            follower.disconnect()
        if leader.is_connected:
            leader.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
