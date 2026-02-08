#!/usr/bin/env python3
import sys
import time
sys.path.insert(0, '.')

print("=" * 70)
print("🌱 AGRI SAFE - FINAL WORKING VERSION")
print("=" * 70)

# Create a simple SoilMoistureSensor class inline for robustness
class SimpleSoilSensor:
    def __init__(self, config=None):
        self.name = "Soil_Moisture"
        
    def read(self):
        import random
        return {
            'soil_moisture_percent': 40.0 + random.uniform(-5, 5),
            'status': 'simulated'
        }

from src.sensors.climate_sensor import ClimateSensor
from src.sensors.co2_sensor import CO2Sensor

print("\nInitializing sensors...")

# Climate - try hardware, fallback to simulation
try:
    climate = ClimateSensor({'simulation': False, 'i2c_address': 0x77})
    print("✓ Climate: BME680 Hardware")
except Exception as e:
    climate = ClimateSensor({'simulation': True})
    print(f"✓ Climate: Simulation mode ({e})")

# Soil
soil = SimpleSoilSensor()
print("✓ Soil: Simulation mode")

# CO2 - try hardware
try:
    co2 = CO2Sensor({'simulation': False})
    print("✓ CO2: MH-Z19 Hardware")
except Exception as e:
    co2 = CO2Sensor({'simulation': True})
    print(f"✓ CO2: Simulation mode ({e})")

print("\n" + "=" * 70)
print("✅ STARTING DATA COLLECTION")
print("=" * 70 + "\n")

count = 0
try:
    while True:
        count += 1
        
        # Read all sensors
        climate_data = climate.read()
        soil_data = soil.read()
        co2_data = co2.read()
        
        # Display
        print(f"[{time.strftime('%H:%M:%S')}] #{count}")
        print(f"  🌡️  Air Temp: {climate_data['temperature_c']:.1f}°C")
        print(f"  💧 Humidity: {climate_data['humidity_percent']:.1f}%")
        print(f"  🌱 Soil Moisture: {soil_data['soil_moisture_percent']:.1f}%")
        
        if co2_data.get('co2_ppm'):
            print(f"  🌿 CO2: {co2_data['co2_ppm']} ppm")
        
        print("-" * 40)
        
        time.sleep(10)
        
except KeyboardInterrupt:
    print(f"\n\n🎉 Data collection complete!")
    print(f"📊 Total readings: {count}")
    print("=" * 70)
