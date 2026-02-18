#!/usr/bin/env python3
import serial
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_co2_with_baud(baud_rate=9600):
    """Test CO2 sensor with specific baud rate."""
    try:
        print(f"\n--- Testing with baud rate: {baud_rate} ---")
        with serial.Serial('/dev/serial0', baud_rate, timeout=2.0) as ser:
            ser.flushInput()
            # Send wake-up command (some sensors need this)
            # Command: FF 01 86 00 00 00 00 00 79 (Standard MH-Z19 read)
            # The Deepseek prompt suggested a wake-up b'\xff\x01\x79\xa0\x00\x00\x00\x00\xe6'
            wake_cmd = b'\xff\x01\x79\xa0\x00\x00\x00\x00\xe6'
            ser.write(wake_cmd)
            time.sleep(0.1)
            
            # Standard read command
            read_cmd = b'\xff\x01\x86\x00\x00\x00\x00\x00\x79'
            ser.write(read_cmd)
            time.sleep(0.5)  # Critical wait for response
            
            # Try to read all bytes
            response = ser.read(9)  # Standard response is 9 bytes
            print(f"Response ({len(response)} bytes): {response.hex()}")
            
            if len(response) >= 9:
                # Checksum validation
                checksum = (255 - (sum(response[1:8]) % 256) + 1) % 256
                if checksum == response[8]:
                    co2 = response[2] * 256 + response[3]
                    temp = response[4] - 40  # Some sensors return temp
                    print(f"✅ CO2: {co2} ppm, Temp: {temp}°C")
                    return True
                else:
                    print(f"⚠️ Checksum failed: Calced {checksum:02x}, Got {response[8]:02x}")
                    # Still show the values if they look plausible
                    co2 = response[2] * 256 + response[3]
                    print(f"   Plausible CO2: {co2} ppm")
                    return True
            else:
                print(f"⚠️ Short response: {response.hex()}")
                return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("MH-Z19 CO2 Sensor Baud Rate Discovery")
    print("=" * 40)
    
    # Test multiple baud rates
    working_baud = None
    for baud in [9600, 19200, 38400, 57600, 115200]:
        if test_co2_with_baud(baud):
            working_baud = baud
            print(f"\n✨ Found working baud rate: {baud}")
            break
    
    if not working_baud:
        print("\n❌ Failed to find a working baud rate.")
        print("Please check wiring (TX/RX might be swapped) and power (needs 5V).")
