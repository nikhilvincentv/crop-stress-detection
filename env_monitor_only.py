import time
import sys
import logging
from datetime import datetime

# Ensure we can find the src directory
sys.path.append('.')

try:
    from src.sensors.climate_sensor import ClimateSensor
    from src.sensors.co2_sensor import CO2Sensor
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you are running this from the 'CropStressDetector' folder.")
    sys.exit(1)

# Configure logging to show us what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    print("=== Only Environmental Sensors Monitor ===")
    print("Press CTRL+C to stop.\n")

    # --- 1. Initialize Climate Sensor (BME680) ---
    try:
        # Note: If 0x77 fails, try changing it to 0x76
        climate = ClimateSensor({'i2c_address': 0x77, 'simulation': False})
        logger.info("✅ Climate Sensor initialized")
    except Exception as e:
        logger.error(f"❌ Failed to init Climate Sensor: {e}")
        climate = None

    # --- 2. Initialize CO2 Sensor (MH-Z19) ---
    try:
        co2_sensor = CO2Sensor({'simulation': False})
        logger.info("✅ CO2 Sensor initialized")
    except Exception as e:
        logger.error(f"❌ Failed to init CO2 Sensor: {e}")
        co2_sensor = None

    if not climate and not co2_sensor:
        logger.critical("No sensors could be initialized. Exiting.")
        return

    # --- 3. Main Loop ---
    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            output_parts = [f"[{timestamp}]"]

            # Read Climate Data
            if climate:
                c_data = climate.read()
                if c_data:
                    temp = c_data.get('temperature_c', 0)
                    hum = c_data.get('humidity_percent', 0)
                    pres = c_data.get('pressure_hpa', 0)
                    output_parts.append(f"🌡️ Temp: {temp:.1f}°C")
                    output_parts.append(f"💧 Hum: {hum:.1f}%")
                    output_parts.append(f"hPa: {pres:.0f}")

            # Read CO2 Data
            if co2_sensor:
                co2_data = co2_sensor.read()
                if co2_data:
                    ppm = co2_data.get('co2_ppm', 0)
                    output_parts.append(f"☁️ CO2: {ppm} ppm")

            # Print the combined line
            print(" | ".join(output_parts))

            # Wait 2 seconds before next read
            time.sleep(2)

        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
