"""Real soil moisture sensor using ADC."""

import time
import logging
from typing import Dict
from .base_sensor import BaseSensor

try:
    import board
    import busio
    import adafruit_ads1x15.ads1015 as ADS
    from adafruit_ads1x15.analog_in import AnalogIn
    REAL_SENSOR_AVAILABLE = True
except ImportError:
    REAL_SENSOR_AVAILABLE = False
    logging.warning("ADS1x15 libraries not available. Using simulation.")

logger = logging.getLogger(__name__)

class RealSoilMoistureSensor(BaseSensor):
    """Real soil moisture sensor with ADS1015 ADC."""
    
    def __init__(self, config: dict = None):
        super().__init__("Real_Soil_Moisture", config)
        config = config or {}
        self.simulation = config.get('simulation', not REAL_SENSOR_AVAILABLE)
        self.i2c_address = config.get('i2c_address', 0x48)
        self.adc_channel = config.get('adc_channel', 0)  # A0
        self.dry_value = config.get('dry_value', 25000)  # ADC value when dry
        self.wet_value = config.get('wet_value', 10000)  # ADC value when wet
        
        if not self.simulation and REAL_SENSOR_AVAILABLE:
            self._initialize_adc()
        else:
            self.adc = None
            self.channel = None
    
    def _initialize_adc(self):
        """Initialize I2C and ADC."""
        try:
            # Create I2C bus
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Create ADC object
            ads = ADS.ADS1015(i2c, address=self.i2c_address)
            
            # Create single-ended input on channel
            if self.adc_channel == 0:
                self.channel = AnalogIn(ads, ADS.P0)
            elif self.adc_channel == 1:
                self.channel = AnalogIn(ads, ADS.P1)
            elif self.adc_channel == 2:
                self.channel = AnalogIn(ads, ADS.P2)
            else:
                self.channel = AnalogIn(ads, ADS.P3)
            
            self.adc = ads
            logger.info(f"ADS1015 initialized on channel A{self.adc_channel}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ADC: {e}")
            self.simulation = True
    
    def read(self) -> Dict[str, float]:
        """Read soil moisture from ADC."""
        if self.simulation or self.channel is None:
            return self._read_simulated()
        
        try:
            # Read ADC value
            adc_value = self.channel.value
            
            # Convert to moisture percentage
            moisture_percent = self._adc_to_percent(adc_value)
            
            return {
                'soil_moisture_percent': float(moisture_percent),
                'adc_value': int(adc_value),
                'voltage_v': float(self.channel.voltage),
                'sensor_type': 'capacitive_v1.2'
            }
            
        except Exception as e:
            logger.error(f"Error reading soil moisture: {e}")
            return self._read_simulated()
    
    def _adc_to_percent(self, adc_value: int) -> float:
        """Convert ADC reading to moisture percentage (0-100%)."""
        # ADC values are inverted: lower value = more wet
        if adc_value >= self.dry_value:
            return 0.0  # Completely dry
        elif adc_value <= self.wet_value:
            return 100.0  # Completely wet
        
        # Linear interpolation
        moisture = 100.0 * (self.dry_value - adc_value) / (self.dry_value - self.wet_value)
        return max(0.0, min(100.0, moisture))
    
    def _read_simulated(self) -> Dict[str, float]:
        """Generate simulated soil moisture data."""
        import random
        return {
            'soil_moisture_percent': 40.0 + random.uniform(-5, 5),
            'adc_value': 15000 + random.uniform(-1000, 1000),
            'voltage_v': 1.5 + random.uniform(-0.1, 0.1),
            'sensor_type': 'simulated'
        }
    
    def calibrate(self) -> bool:
        """Calibrate soil moisture sensor."""
        if self.simulation:
            logger.info("Simulation mode: calibration skipped")
            return True
        
        logger.info("Calibrating soil moisture sensor...")
        # TODO: Add calibration procedure
        return True
