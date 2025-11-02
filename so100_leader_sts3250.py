"""
Custom SO100 Leader configuration for STS3250 motors.
This overrides the default STS3215 motor configuration for the leader arm.
"""

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode, MotorCalibration
from lerobot.teleoperators.so100_leader.config_so100_leader import SO100LeaderConfig
from lerobot.teleoperators.so100_leader.so100_leader import SO100Leader


class SO100LeaderSTS3250(SO100Leader):
    """SO100 Leader teleoperator with STS3250 motors instead of STS3215."""

    def _create_motor_bus(self):
        """Override to use STS3250 motors instead of STS3215."""
        # Create a dummy calibration with proper MotorCalibration objects
        # STS3250 has 4096 steps (0-4095), center is 2048
        dummy_calibration = {
            "shoulder_pan": MotorCalibration(
                id=1,
                drive_mode=0,  # Normal mode
                homing_offset=2048,  # Center position
                range_min=0,
                range_max=4095
            ),
            "shoulder_lift": MotorCalibration(
                id=2,
                drive_mode=0,
                homing_offset=2048,
                range_min=0,
                range_max=4095
            ),
            "elbow_flex": MotorCalibration(
                id=3,
                drive_mode=0,
                homing_offset=2048,
                range_min=0,
                range_max=4095
            ),
            "wrist_flex": MotorCalibration(
                id=4,
                drive_mode=0,
                homing_offset=2048,
                range_min=0,
                range_max=4095
            ),
            "wrist_roll": MotorCalibration(
                id=5,
                drive_mode=0,
                homing_offset=2048,
                range_min=0,
                range_max=4095
            ),
            "gripper": MotorCalibration(
                id=6,
                drive_mode=0,
                homing_offset=2048,
                range_min=0,
                range_max=4095
            ),
        }

        return FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3250", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3250", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3250", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3250", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3250", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3250", MotorNormMode.RANGE_0_100),
            },
            calibration=dummy_calibration,
        )

    def __init__(self, config: SO100LeaderConfig):
        # Call parent init first
        super().__init__(config)

        # Replace the motor bus with STS3250 version
        self.bus = self._create_motor_bus()

    def connect(self):
        """Override connect to skip calibration (robot pre-calibrated by Phospho)."""
        self.bus.connect()
        # Skip the calibration step - robot was pre-calibrated
        # Mark as connected
        self._is_connected = True
