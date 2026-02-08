"""NDVI camera module using Raspberry Pi NoIR Camera for plant health monitoring."""

import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from .base_sensor import BaseSensor

try:
    from picamera2 import Picamera2
    from PIL import Image
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    logging.warning("picamera2 not available. Using simulation mode.")
    try:
        from PIL import Image
    except ImportError:
        logging.error("PIL not available. Install pillow for image processing.")

logger = logging.getLogger(__name__)


class NDVICamera(BaseSensor):
    """Pi NoIR Camera with blue filter for NDVI measurements."""
    
    def __init__(self, config: dict = None):
        """
        Initialize NDVI camera.
        
        Args:
            config: Configuration dictionary with:
                - resolution: Tuple of (width, height) (default: (1920, 1080))
                - save_path: Directory to save images (default: './data/images')
                - blue_filter: Whether blue filter is installed (default: True)
        """
        super().__init__("NDVI_Camera", config)
        self.resolution = config.get('resolution', (1920, 1080))
        self.save_path = Path(config.get('save_path', './data/images'))
        self.blue_filter = config.get('blue_filter', True)
        self.camera = None
        
        # Create save directory
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        if CAMERA_AVAILABLE:
            self._initialize_camera()
    
    def _initialize_camera(self):
        """Initialize the Pi NoIR camera."""
        try:
            self.camera = Picamera2()
            
            # Check for available cameras
            cameras = self.camera.available_cameras
            if not cameras:
                logger.error("No cameras detected by libcamera")
                self.camera = None
                return
            
            logger.info(f"Found {len(cameras)} camera(s): {cameras}")
            
            config = self.camera.create_still_configuration(
                main={"size": self.resolution}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  # Allow camera to warm up
            logger.info("Pi NoIR camera initialized successfully")
        except IndexError:
            logger.error("Failed to initialize camera: No camera found (IndexError)")
            self.camera = None
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            self.camera = None
    
    def read(self) -> Dict:
        """
        Capture image and calculate NDVI.
        
        Returns:
            Dictionary containing:
                - ndvi_mean: Mean NDVI value
                - ndvi_std: Standard deviation of NDVI
                - ndvi_min: Minimum NDVI value
                - ndvi_max: Maximum NDVI value
                - image_path: Path to saved image
                - ndvi_image_path: Path to saved NDVI visualization
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"capture_{timestamp}.jpg"
        ndvi_filename = f"ndvi_{timestamp}.png"
        
        image_path = self.save_path / image_filename
        ndvi_image_path = self.save_path / ndvi_filename
        
        if CAMERA_AVAILABLE and self.camera:
            try:
                # Capture image
                self.camera.capture_file(str(image_path))
                logger.info(f"Image captured: {image_path}")
                
                # Calculate NDVI
                ndvi_stats, ndvi_array = self._calculate_ndvi(str(image_path))
                
                # Save NDVI visualization
                self._save_ndvi_image(ndvi_array, str(ndvi_image_path))
                
                ndvi_stats['image_path'] = str(image_path)
                ndvi_stats['ndvi_image_path'] = str(ndvi_image_path)
                
                return ndvi_stats
            except Exception as e:
                logger.error(f"Error capturing/processing image: {e}")
                return self._get_null_reading()
        else:
            # Simulation mode
            return self._simulate_reading(str(image_path), str(ndvi_image_path))
    
    def _calculate_ndvi(self, image_path: str) -> Tuple[Dict, np.ndarray]:
        """
        Calculate NDVI from NoIR camera image with blue filter.
        
        With blue filter:
        - Red channel captures NIR (Near-Infrared)
        - Blue channel captures visible blue light
        
        NDVI = (NIR - Blue) / (NIR + Blue)
        
        Args:
            image_path: Path to the captured image
        
        Returns:
            Tuple of (statistics dict, NDVI array)
        """
        # Load image
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Extract channels
        if self.blue_filter:
            # With blue filter: Red = NIR, Blue = Blue
            nir = img_array[:, :, 0].astype(float)
            blue = img_array[:, :, 2].astype(float)
        else:
            # Without filter (standard NoIR): Red = Red+NIR, Blue = Blue
            # This is less accurate but still usable
            nir = img_array[:, :, 0].astype(float)
            blue = img_array[:, :, 2].astype(float)
        
        # Calculate NDVI
        # Avoid division by zero
        denominator = nir + blue
        denominator[denominator == 0] = 0.001
        
        ndvi = (nir - blue) / denominator
        
        # Clip to valid range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        # Calculate statistics
        stats = {
            'ndvi_mean': float(np.mean(ndvi)),
            'ndvi_std': float(np.std(ndvi)),
            'ndvi_min': float(np.min(ndvi)),
            'ndvi_max': float(np.max(ndvi)),
            'ndvi_median': float(np.median(ndvi))
        }
        
        logger.info(f"NDVI calculated: mean={stats['ndvi_mean']:.3f}, std={stats['ndvi_std']:.3f}")
        
        return stats, ndvi
    
    def _save_ndvi_image(self, ndvi_array: np.ndarray, output_path: str):
        """
        Save NDVI array as color-mapped image.
        
        Args:
            ndvi_array: NDVI values array
            output_path: Path to save visualization
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            plt.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(label='NDVI')
            plt.title('NDVI Map')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"NDVI visualization saved: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save NDVI visualization: {e}")
    
    def _simulate_reading(self, image_path: str, ndvi_path: str) -> Dict:
        """Generate simulated NDVI readings."""
        import random
        
        # Simulate healthy to stressed plant NDVI values
        # Healthy plants: NDVI ~ 0.6-0.9
        # Stressed plants: NDVI ~ 0.2-0.5
        base_ndvi = 0.7 + random.uniform(-0.3, 0.2)
        
        return {
            'ndvi_mean': base_ndvi,
            'ndvi_std': 0.1 + random.uniform(0, 0.05),
            'ndvi_min': base_ndvi - 0.2,
            'ndvi_max': base_ndvi + 0.15,
            'ndvi_median': base_ndvi + random.uniform(-0.05, 0.05),
            'image_path': image_path,
            'ndvi_image_path': ndvi_path
        }
    
    def _get_null_reading(self) -> Dict:
        """Return null reading when camera fails."""
        return {
            'ndvi_mean': None,
            'ndvi_std': None,
            'ndvi_min': None,
            'ndvi_max': None,
            'ndvi_median': None,
            'image_path': None,
            'ndvi_image_path': None
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate camera by capturing test images.
        
        Returns:
            True if calibration successful
        """
        logger.info("Calibrating NDVI camera...")
        
        if CAMERA_AVAILABLE and self.camera:
            try:
                # Capture test images
                for i in range(3):
                    test_path = self.save_path / f"calibration_test_{i}.jpg"
                    self.camera.capture_file(str(test_path))
                    logger.info(f"Calibration image {i+1}/3 captured")
                    time.sleep(1)
                
                logger.info("Camera calibration complete")
                return True
            except Exception as e:
                logger.error(f"Calibration failed: {e}")
                return False
        else:
            logger.info("Simulation mode: calibration skipped")
            return True
    
    def capture_timelapse(self, interval_seconds: int, duration_minutes: int) -> list:
        """
        Capture timelapse images.
        
        Args:
            interval_seconds: Time between captures
            duration_minutes: Total duration
        
        Returns:
            List of captured image paths
        """
        logger.info(f"Starting timelapse: {duration_minutes} min, {interval_seconds}s interval")
        
        captured_images = []
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            reading = self.read()
            if reading.get('image_path'):
                captured_images.append(reading['image_path'])
            time.sleep(interval_seconds)
        
        logger.info(f"Timelapse complete: {len(captured_images)} images captured")
        return captured_images
    
    def __del__(self):
        """Cleanup camera resources."""
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
