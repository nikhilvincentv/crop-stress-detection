#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, '.')

print("🌱 AGRI SAFE - WITH SOIL MOISTURE")
print("=" * 50)

# Import what works
from src.sensors.climate_sensor import ClimateSensor

# Create simulated soil moisture data
class SimpleSoilSensor:
    def read(self):
        import random
        return {
            'soil_moisture_percent': 40 + random.uniform(-5, 5),
            'status': 'simulated'
        }

# Initialize
print("\nInitializing sensors...")
climate = ClimateSensor({'simulation': True})  # Use simulation for now
soil = SimpleSoilSensor()

print("✓ Climate: Simulation mode")
print("✓ Soil: Simulation mode")
print("✓ CO2: Disabled (permission issues)")
print("✓ Camera: Disabled")

print("\n" + "=" * 50)
print("STARTING DATA COLLECTION")
print("Press Ctrl+C to stop")
print("=" * 50 + "\n")

count = 0
try:
    while True:
        count += 1
        
        # Read sensors
        climate_data = climate.read()
        soil_data = soil.read()
        
        # Display
        print(f"[{time.strftime('%H:%M:%S')}] #{count}")
        print(f"  Air Temp: {climate_data['temperature_c']:.1f}°C")
        print(f"  Humidity: {climate_data['humidity_percent']:.1f}%")
        print(f"  Soil Moisture: {soil_data['soil_moisture_percent']:.1f}%")
        print("-" * 40)
        
        time.sleep(10)
        
except KeyboardInterrupt:
    print(f"\n\n✅ Collected {count} readings")
    print("=" * 50)
