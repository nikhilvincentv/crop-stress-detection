#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("=== CAMERA FIX TEST ===")

# Test 1: Direct picamera2 access
print("\n1. Testing direct picamera2...")
try:
    from picamera2 import Picamera2
    import time
    
    picam2 = Picamera2()
    print("   Picamera2 object created")
    
    # Try to get camera info
    try:
        info = picam2.global_camera_info()
        print(f"   Camera info: {info}")
    except:
        print("   No global_camera_info available")
    
    # Try different configurations
    print("   Trying preview configuration...")
    try:
        config = picam2.create_preview_configuration()
        picam2.configure(config)
        print("   ✓ Preview config successful!")
        
        # Try to start camera
        picam2.start()
        print("   Camera started")
        time.sleep(0.5)
        picam2.stop()
        print("   Camera stopped")
        
    except Exception as e:
        print(f"   Preview config failed: {e}")
        
        # Try still configuration
        print("   Trying still configuration...")
        try:
            config = picam2.create_still_configuration()
            picam2.configure(config)
            print("   ✓ Still config successful!")
        except Exception as e2:
            print(f"   Still config failed: {e2}")
            
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Your NDVICamera class
print("\n2. Testing NDVICamera class...")
try:
    from src.sensors.ndvi_camera import NDVICamera
    camera = NDVICamera({'simulation': False})
    print(f"   Camera initialized: simulation={camera.simulation}")
    if not camera.simulation:
        print("   ✓ REAL camera is being used!")
    else:
        print("   ⚠ Camera in simulation mode")
        
except Exception as e:
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TEST COMPLETE ===")
