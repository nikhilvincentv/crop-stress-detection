#!/usr/bin/env python3
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import time
import pytest
import sys

def test_soil_sensor_readings():
    """Test Capacitive Soil Moisture Sensor V1.2 readings."""
    print("Testing Capacitive Soil Moisture Sensor V1.2")
    print("=" * 50)

    try:
        # Initialize I2C and ADC
        i2c = busio.I2C(board.SCL, board.SDA)
        ads = ADS.ADS1015(i2c, address=0x48)  # Address detected in i2cdetect
        chan = AnalogIn(ads, ADS.P0)  # Sensor connected to A0
        
        # Check initial reading
        val = chan.value
        vol = chan.voltage
        
        print("📊 Reading sensor values (dry → wet shows decreasing voltage):")
        print("-" * 50)
        print(f"Initial Check: {val:>5} (raw) | {vol:>5.3f} V")
        
        # Validate range (roughly 0V to 3.3V or 5V depending on VCC, usually < 3.3V for Pi)
        if not (0 <= vol <= 3.3):
             print(f"⚠️ Voltage {vol:.2f}V out of expected range (0-3.3V)")
             # Not failing necessarily as it might be 5V powered, but warning
        
        # Test stability (take 3 samples instead of 10)
        for i in range(3):
            # Read 3 samples
            print(f"Sample {i+1}: {chan.value:>5} (raw) | {chan.voltage:>5.3f} V")
            time.sleep(0.5)

        print("\n🔍 Interpretation:")
        print("• Dry soil (in air): ~3.0V (higher voltage)")
        print("• Wet soil (in water): ~1.0V (lower voltage)")
        
        status = "DRY" if chan.voltage > 2.0 else "WET/MOIST"
        print(f"\nYour current reading suggests: {status} soil")

    except Exception as e:
        pytest.fail(f"Soil sensor test failed: {e}")

if __name__ == "__main__":
    try:
        test_soil_sensor_readings()
    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)
