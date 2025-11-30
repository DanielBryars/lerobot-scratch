"""Check EEPROM calibration values for both arms."""
import sys
sys.path.insert(0, ".")

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

for name, port in [("Leader", "COM8"), ("Follower", "COM7")]:
    bus = FeetechMotorsBus(
        port=port,
        motors={f"motor_{i}": Motor(i, "sts3250", MotorNormMode.RANGE_M100_100) for i in range(1, 7)}
    )
    bus.connect()
    cal = bus.read_calibration()
    print(f"\n{name} ({port}) EEPROM calibration:")
    for motor, c in cal.items():
        print(f"  {motor}: homing_offset={c.homing_offset}, range=[{c.range_min}, {c.range_max}]")
    bus.disconnect()
