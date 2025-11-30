"""
Install the STS3250 plugin so lerobot CLI commands auto-discover it.

Run: python setup_sts3250_plugin.py
"""

from setuptools import setup

setup(
    name="lerobot_robot_sts3250",
    version="0.1.0",
    description="LeRobot plugin for SO100 with STS3250 motors",
    py_modules=[],
    packages=["lerobot_robot_sts3250"],
    install_requires=["lerobot"],
)
