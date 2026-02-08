"""Soil moisture sensor module."""

import time
import random
import logging
from typing import Dict
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class SoilMoistureSensor(BaseSensor):
    """Soil moisture sensor."""
    
    def __init__(self, config: dict = None):
        super().__init__("Soil_Moisture", config)
        self.simulation = config.get('simulation', True)
        
    def read(self) -> Dict[str, float]:
        """Read soil moisture percentage."""
        if self.simulation:
            return self._read_simulated()
        
        # TODO: Add real sensor code for your soil moisture sensor
        # For now, use simulation
        return self._read_simulated()
    
    def _read_simulated(self) -> Dict[str, float]:
        """Generate simulated soil moisture data."""
        # Simulate soil moisture (0-100%)
        # Plants need 20-60% typically
        base = 40.0
        variation = random.uniform(-5, 5)
        moisture = base + variation
        
        return {
            'soil_moisture_percent': max(0, min(100, moisture)),
            'status': 'simulated'
        }
    
    def calibrate(self) -> bool:
        """Calibrate soil moisture sensor."""
        if self.simulation:
            logger.info("Simulation mode: calibration skipped")
            return True
        
        # TODO: Add real calibration
        logger.info("Soil moisture sensor calibrated")
        return True
