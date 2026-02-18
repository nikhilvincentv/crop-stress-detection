#!/usr/bin/env python3
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import time

print("Testing Capacitive Soil Moisture Sensor V1.2")
print("=" * 50)

try:
    # Initialize I2C and ADC
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1015(i2c, address=0x48)  # Address detected in i2cdetect
    chan = AnalogIn(ads, ADS.P0)  # Sensor connected to A0

    print("📊 Reading sensor values (dry → wet shows decreasing voltage):")
    print("-" * 50)

    for i in range(10):
        # Read 10 samples
        print(f"Sample {i+1}: {chan.value:>5} (raw) | {chan.voltage:>5.3f} V")
        time.sleep(1)

    print("\n🔍 Interpretation:")
    print("• Dry soil (in air): ~3.0V (higher voltage)")
    print("• Wet soil (in water): ~1.0V (lower voltage)")
    
    status = "DRY" if chan.voltage > 2.0 else "WET/MOIST"
    print(f"\nYour current reading suggests: {status} soil")

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if ADS1015 is powered (VCC and GND)")
    print("2. Check I2C wiring (SDA/SCL)")
    print("3. Ensure adafruit-circuitpython-ads1x15 is installed")
