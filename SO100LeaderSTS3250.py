"""
Custom SO100 Leader for STS3250 motors (Phospho pre-calibrated).
Uses fixed calibration so leader and follower have matching normalization.
"""

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
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


class SO100LeaderSTS3250(SO100Leader):
    """SO100 Leader teleoperator with STS3250 motors (Phospho pre-calibrated)."""

    def __init__(self, config: SO100LeaderConfig):
        # Call grandparent init to set up base teleoperator properties
        super(SO100Leader, self).__init__(config)
        self.config = config

        # Use sts3250 motor model with fixed calibration (same as follower)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3250", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3250", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3250", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
            },
            calibration=FIXED_CALIBRATION,
        )

    @property
    def is_calibrated(self) -> bool:
        return True  # Using fixed calibration

    def connect(self, calibrate: bool = True) -> None:
        """Connect without modifying motor calibration."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        # Don't read or write calibration - use fixed calibration from __init__

        self.configure()

    def configure(self) -> None:
        """Configure leader arm (torque disabled for manual movement)."""
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
