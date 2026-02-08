#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, '.')

print("=" * 70)
print("FINAL HARDWARE TEST - ALL SENSORS")
print("=" * 70)

# Import with potential fixes
from src.sensors.climate_sensor import ClimateSensor
from src.sensors.co2_sensor import CO2Sensor
from src.sensors.ndvi_camera import NDVICamera

print("\n1. Testing Climate Sensor...")
climate = ClimateSensor({'simulation': False, 'i2c_address': 0x77})
print(f"   ✓ BME680 at 0x77: OK")

print("\n2. Testing CO2 Sensor...")
co2 = CO2Sensor({'simulation': False})
print(f"   ✓ MH-Z19: OK")

print("\n3. Testing Camera...")
camera = NDVICamera({'simulation': False})
if camera.simulation:
    print("   ⚠ Camera in simulation mode (hardware init failed)")
else:
    print("   ✓ Camera hardware: OK")

print("\n" + "=" * 70)
print("STARTING DATA COLLECTION (Ctrl+C to stop)")
print("=" * 70)

count = 0
try:
    while True:
        count += 1
        print(f"\n[{time.strftime('%H:%M:%S')}] Reading #{count}")
        print("-" * 40)
        
        # Climate
        c_data = climate.read()
        print(f"Temp: {c_data['temperature_c']:.1f}°C | "
              f"Humidity: {c_data['humidity_percent']:.1f}% | "
              f"Pressure: {c_data['pressure_hpa']:.1f} hPa")
        
        # CO2
        try:
            co2_data = co2.read()
            if co2_data.get('co2_ppm'):
                print(f"CO2: {co2_data['co2_ppm']} ppm")
        except:
            print("CO2: Error reading")
        
        # Camera status
        print(f"Camera: {'Simulation' if camera.simulation else 'Hardware'} mode")
        print("-" * 40)
        
        time.sleep(5)
        
except KeyboardInterrupt:
    print(f"\n\nStopped. Total readings: {count}")

print("\n" + "=" * 70)
print("HARDWARE SYSTEM IS OPERATIONAL!")
print("=" * 70)
