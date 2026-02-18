#!/usr/bin/env python3
import sys
import time
import subprocess
import pytest

def test_hardware_diagnostics():
    """Perform hardware diagnostics."""
    print("=== HARDWARE DIAGNOSTICS ===")
    
    errors = []

    # 1. Check serial port
    print("\n1. Checking /dev/serial0 permissions...")
    try:
        result = subprocess.run(['ls', '-l', '/dev/serial0'], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            errors.append("Serial port check failed")
    except Exception as e:
        errors.append(f"Serial port check exception: {e}")

    # 2. Check i2c devices
    print("\n2. Checking I2C devices...")
    try:
        subprocess.run(['sudo', 'i2cdetect', '-y', '1'])
    except Exception as e:
        print(f"I2C check failed: {e}")
        # Not failing the test for this as it's visual, but good to know

    # 3. Test CO2 sensor
    print("\n3. Testing CO2 sensor...")
    try:
        import mh_z19
        co2 = mh_z19.read()
        print(f"   CO2: {co2}")
    except Exception as e:
        msg = f"   CO2 Error: {e}"
        print(msg)
        errors.append(msg)

    # 4. Test BME680
    print("\n4. Testing BME680...")
    try:
        import bme680
        sensor = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
        if sensor.get_sensor_data():
            print("   BME680: Connected and reading data")
        else:
            print("   BME680: Connected but no data")
    except Exception as e:
        msg = f"   BME680 Error: {e}"
        print(msg)
        errors.append(msg)

    # 5. Test camera
    print("\n5. Testing camera...")
    try:
        from picamera2 import Picamera2
        picam2 = Picamera2()
        camera_config = picam2.create_preview_configuration()
        picam2.configure(camera_config)
        print("   Camera: OK")
    except Exception as e:
        msg = f"   Camera Error: {e}"
        print(msg)
        errors.append(msg)

    print("\n=== DIAGNOSTICS COMPLETE ===")
    
    if errors:
        pytest.fail(f"Hardware diagnostics failed with {len(errors)} errors: {'; '.join(errors)}")

if __name__ == "__main__":
    try:
        test_hardware_diagnostics()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
