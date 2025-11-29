"""
Custom SO100 Follower for STS3250 motors (Phospho pre-calibrated).
Uses fixed calibration so leader and follower have matching normalization.
"""

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.utils.errors import DeviceAlreadyConnectedError


# Fixed calibration - same for both leader and follower so normalization matches
# homing_offset=0 means we use raw motor positions, range is full 0-4095
FIXED_CALIBRATION = {
    "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
    "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=0, range_min=0, range_max=4095),
}


class SO100FollowerSTS3250(SO100Follower):
    """SO100 Follower robot with STS3250 motors (Phospho pre-calibrated)."""

    def __init__(self, config: SO100FollowerConfig):
        # Call grandparent init to set up base robot properties
        super(SO100Follower, self).__init__(config)
        self.config = config

        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        # Use sts3250 motor model with fixed calibration (same as leader)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3250", norm_mode),
                "shoulder_lift": Motor(2, "sts3250", norm_mode),
                "elbow_flex": Motor(3, "sts3250", norm_mode),
                "wrist_flex": Motor(4, "sts3250", norm_mode),
                "wrist_roll": Motor(5, "sts3250", norm_mode),
                "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
            },
            calibration=FIXED_CALIBRATION,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def is_calibrated(self) -> bool:
        return True  # Using fixed calibration

    def connect(self, calibrate: bool = True) -> None:
        """Connect without modifying motor calibration."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        # Don't read or write calibration - use fixed calibration from __init__

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def configure(self) -> None:
        """Configure motors for position control."""
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                self.bus.write("Max_Torque_Limit", motor, 500)
                self.bus.write("Protection_Current", motor, 250)
                self.bus.write("Overload_Torque", motor, 25)
