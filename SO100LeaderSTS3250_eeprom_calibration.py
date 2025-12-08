"""
SO100 Leader with STS3250 motors.
Registered as 'so100_leader_sts3250' for use with lerobot CLI tools.
"""

from dataclasses import dataclass

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError


@TeleoperatorConfig.register_subclass("so100_leader_sts3250")
@dataclass
class SO100LeaderSTS3250Config(SO100LeaderConfig):
    """Config for SO100 Leader with STS3250 motors."""
    pass  # Inherits all fields from SO100LeaderConfig


class SO100LeaderSTS3250(SO100Leader):
    """SO100 Leader teleoperator with STS3250 motors."""

    config_class = SO100LeaderSTS3250Config
    name = "so100_leader_sts3250"

    def __init__(self, config: SO100LeaderSTS3250Config):
        # Call grandparent init to set up base teleoperator properties
        super(SO100Leader, self).__init__(config)
        self.config = config

        # Use sts3250 motor model - calibration read from EEPROM on connect
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
        )

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Connect and read calibration from motor EEPROM."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()

        # Read calibration from EEPROM (set by align_arms.py)
        self.bus.calibration = self.bus.read_calibration()

        self.configure()

    def configure(self) -> None:
        """Configure leader arm (torque disabled for manual movement)."""
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
