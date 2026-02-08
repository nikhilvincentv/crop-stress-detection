#!/usr/bin/env python3
import time
import json
from datetime import datetime

print("🌿 AGRI SAFE - SIMPLE & WORKING")
print("=" * 40)

data = []
filename = f"agrisafe_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

print(f"\nSaving to: {filename}")
print("Collecting every 10 seconds...")
print("Press Ctrl+C to stop\n")

try:
    count = 0
    while True:
        count += 1
        timestamp = datetime.now().isoformat()
        
        # Simulate all sensor data
        point = {
            'timestamp': timestamp,
            'air_temperature_c': 22.5 + (count % 20) * 0.1,
            'air_humidity_percent': 45.0 + (count % 15) * 0.2,
            'soil_moisture_percent': 40.0 + (count % 30) * 0.3,
            'co2_ppm': 450 + (count % 50),
            'camera_status': 'simulated'
        }
        
        data.append(point)
        
        # Display
        print(f"[{timestamp[11:19]}] #{count}")
        print(f"  Air: {point['air_temperature_c']:.1f}°C, {point['air_humidity_percent']:.1f}%")
        print(f"  Soil: {point['soil_moisture_percent']:.1f}% moist")
        print(f"  CO2: {point['co2_ppm']} ppm")
        print("-" * 40)
        
        # Save every 5 readings
        if count % 5 == 0:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        
        time.sleep(10)
        
except KeyboardInterrupt:
    # Final save
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n\n✅ Saved {len(data)} readings to {filename}")
    print("=" * 40)
