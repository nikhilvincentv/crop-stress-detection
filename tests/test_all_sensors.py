#!/usr/bin/env python3
import sys
import time
from pathlib import Path
import pytest

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def test_all_sensors_integration():
    """Integration test for all sensors."""
    print("Testing all sensors...")
    print("=" * 50)
    
    errors = []

    # 1. Test BME680
    print("\n--- Testing BME680 ---")
    try:
        from sensors.climate_sensor import ClimateSensor
        climate = ClimateSensor({'simulation': False, 'i2c_address': 0x77})
        c_data = climate.read()
        if c_data.get('temperature_c') is not None:
            print(f"✅ BME680: {c_data['temperature_c']:.1f}°C, {c_data['humidity_percent']:.1f}%")
        else:
            msg = "⚠️ BME680 returned None values (Hardware fail?)"
            print(msg)
            errors.append(msg)
    except Exception as e:
        msg = f"❌ BME680 Error: {e}"
        print(msg)
        errors.append(msg)

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
        msg = f"❌ Soil ADC Error: {e}"
        print(msg)
        errors.append(msg)

    # 3. Test CO2
    print("\n--- Testing CO2 MH-Z19 ---")
    try:
        from sensors.co2_sensor import CO2Sensor
        co2 = CO2Sensor({'simulation': False})
        co2_data = co2.read()
        if co2_data.get('co2_ppm') is not None:
            print(f"✅ CO2: {co2_data['co2_ppm']} ppm")
        else:
            msg = "⚠️ CO2 returned None (Hardware/Permission issue)"
            print(msg)
            errors.append(msg)
    except Exception as e:
        msg = f"❌ CO2 Error: {e}"
        print(msg)
        errors.append(msg)

    print("\n" + "=" * 50)
    print("Diagnostic summary complete.")
    
    # Assert no errors occurred
    if errors:
        pytest.fail(f"Sensor tests failed with {len(errors)} errors: {'; '.join(errors)}")

if __name__ == "__main__":
    try:
        test_all_sensors_integration()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
