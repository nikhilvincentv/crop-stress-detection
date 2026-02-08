#!/usr/bin/env python3
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import time

print("🌱 SOIL SENSOR CALIBRATION")
print("=" * 50)

try:
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1015(i2c, address=0x48)
    chan = AnalogIn(ads, ADS.P0)

    print("1. Place sensor in DRY soil (or air) and press Enter")
    input()
    print("Reading...")
    time.sleep(1)
    dry_value = chan.value
    print(f"   Dry calibration value: {dry_value}")

    print("\n2. Place sensor in WET soil (or water) and press Enter")
    input()
    print("Reading...")
    time.sleep(1)
    wet_value = chan.value
    print(f"   Wet calibration value: {wet_value}")

    print("\n✅ Calibration complete! Update your config or sensors/real_soil_moisture.py with:")
    print(f"   'dry_value': {dry_value},")
    print(f"   'wet_value': {wet_value},")

except Exception as e:
    print(f"❌ Error: {e}")
