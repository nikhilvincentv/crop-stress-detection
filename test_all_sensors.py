#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

print("Testing all sensors...")
print("=" * 50)

# 1. Test BME680
print("\n--- Testing BME680 ---")
try:
    from sensors.climate_sensor import ClimateSensor
    climate = ClimateSensor({'simulation': False, 'i2c_address': 0x77})
    c_data = climate.read()
    if c_data.get('temperature_c') is not None:
        print(f"✅ BME680: {c_data['temperature_c']:.1f}°C, {c_data['humidity_percent']:.1f}%")
    else:
        print("⚠️ BME680 returned None values (Hardware fail?)")
except Exception as e:
    print(f"❌ BME680 Error: {e}")

# 2. Test Soil via ADC
print("\n--- Testing Soil ADS1015 ---")
try:
    import board
    import busio
    import adafruit_ads1x15.ads1015 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    i2c = busio.I2C(board.SCL, board.SDA)
    ads = ADS.ADS1015(i2c, address=0x48)
    chan = AnalogIn(ads, ADS.P0)
    print(f"✅ Soil ADC: {chan.value} raw, {chan.voltage:.2f}V")
except Exception as e:
    print(f"❌ Soil ADC Error: {e}")

# 3. Test CO2
print("\n--- Testing CO2 MH-Z19 ---")
try:
    from sensors.co2_sensor import CO2Sensor
    co2 = CO2Sensor({'simulation': False})
    co2_data = co2.read()
    if co2_data.get('co2_ppm') is not None:
        print(f"✅ CO2: {co2_data['co2_ppm']} ppm")
    else:
        print("⚠️ CO2 returned None (Hardware/Permission issue)")
except Exception as e:
    print(f"❌ CO2 Error: {e}")

print("\n" + "=" * 50)
print("Diagnostic summary complete.")
