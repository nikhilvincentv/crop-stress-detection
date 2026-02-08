"""Soil moisture sensor module."""

import time
import random
import logging
from typing import Dict
from .base_sensor import BaseSensor

logger = logging.getLogger(__name__)

class SoilMoistureSensor(BaseSensor):
    """Soil moisture sensor simulation."""
    
    def __init__(self, config: dict = None):
        super().__init__("Soil_Moisture", config)
        self.simulation = config.get('simulation', True)
        self.pin = config.get('pin', 17)  # GPIO pin for ADC
        
    def read(self) -> Dict[str, float]:
        """Read soil moisture percentage."""
        if self.simulation:
            return self._read_simulated()
        
        # TODO: Add real sensor code here
        # For now, use simulation
        return self._read_simulated()
    
    def _read_simulated(self) -> Dict[str, float]:
        """Generate simulated soil moisture data."""
        # Simulate soil moisture (0-100%)
        moisture = 30 + random.uniform(-5, 5)
        
        return {
            'soil_moisture_percent': max(0, min(100, moisture)),
            'voltage_v': 1.5 + random.uniform(-0.1, 0.1)
        }
