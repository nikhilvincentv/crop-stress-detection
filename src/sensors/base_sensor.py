"""Base sensor class for all sensor modules."""

from abc import ABC, abstractmethod
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseSensor(ABC):
    """Abstract base class for all sensors."""
    
    def __init__(self, name: str, config: dict = None):
        """
        Initialize base sensor.
        
        Args:
            name: Sensor name/identifier
            config: Configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.last_reading = None
        self.last_timestamp = None
        logger.info(f"Initialized {self.name} sensor")
    
    @abstractmethod
    def read(self) -> dict:
        """
        Read sensor data.
        
        Returns:
            Dictionary containing sensor readings and metadata
        """
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """
        Calibrate the sensor.
        
        Returns:
            True if calibration successful, False otherwise
        """
        pass
    
    def get_reading_with_timestamp(self) -> dict:
        """
        Get sensor reading with timestamp.
        
        Returns:
            Dictionary with reading data and timestamp
        """
        reading = self.read()
        timestamp = datetime.now().isoformat()
        
        self.last_reading = reading
        self.last_timestamp = timestamp
        
        return {
            'sensor': self.name,
            'timestamp': timestamp,
            'data': reading
        }
    
    def is_healthy(self) -> bool:
        """
        Check if sensor is functioning properly.
        
        Returns:
            True if sensor is healthy, False otherwise
        """
        try:
            reading = self.read()
            return reading is not None
        except Exception as e:
            logger.error(f"Sensor {self.name} health check failed: {e}")
            return False
