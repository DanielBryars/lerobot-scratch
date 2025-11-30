"""
LeRobot plugin for SO100 with STS3250 motors.

This package is auto-discovered by lerobot when installed.
"""

import platform
from dataclasses import dataclass

import cv2
from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.so100_leader import SO100LeaderConfig, SO100Leader
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError


def make_cameras_with_dshow(camera_configs: dict[str, CameraConfig]) -> dict:
    """Create cameras using DirectShow on Windows."""
    from lerobot.cameras.camera import Camera
    cameras: dict[str, Camera] = {}
    for key, cfg in camera_configs.items():
        if cfg.type == "opencv" and platform.system() == "Windows":
            cam = OpenCVCamera(cfg)
            cam.backend = cv2.CAP_DSHOW
            cameras[key] = cam
        else:
            cameras.update(make_cameras_from_configs({key: cfg}))
    return cameras


@RobotConfig.register_subclass("so100_follower_sts3250")
@dataclass
class SO100FollowerSTS3250Config(SO100FollowerConfig):
    """Config for SO100 Follower with STS3250 motors."""
    pass


class SO100FollowerSTS3250(SO100Follower):
    """SO100 Follower robot with STS3250 motors."""
    config_class = SO100FollowerSTS3250Config
    name = "so100_follower_sts3250"

    def __init__(self, config: SO100FollowerSTS3250Config):
        super(SO100Follower, self).__init__(config)
        self.config = config
        norm_mode = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
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
        )
        self.cameras = make_cameras_with_dshow(config.cameras)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        self.bus.calibration = self.bus.read_calibration()
        for cam in self.cameras.values():
            cam.connect()
        self.configure()

    def configure(self) -> None:
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", motor, 16)
            self.bus.write("I_Coefficient", motor, 0)
            self.bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                self.bus.write("Max_Torque_Limit", motor, 500)
                self.bus.write("Protection_Current", motor, 250)
                self.bus.write("Overload_Torque", motor, 25)


@TeleoperatorConfig.register_subclass("so100_leader_sts3250")
@dataclass
class SO100LeaderSTS3250Config(SO100LeaderConfig):
    """Config for SO100 Leader with STS3250 motors."""
    pass


class SO100LeaderSTS3250(SO100Leader):
    """SO100 Leader teleoperator with STS3250 motors."""
    config_class = SO100LeaderSTS3250Config
    name = "so100_leader_sts3250"

    def __init__(self, config: SO100LeaderSTS3250Config):
        super(SO100Leader, self).__init__(config)
        self.config = config
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
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        self.bus.connect()
        self.bus.calibration = self.bus.read_calibration()
        self.configure()

    def configure(self) -> None:
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)


print("Registered: so100_follower_sts3250, so100_leader_sts3250")
