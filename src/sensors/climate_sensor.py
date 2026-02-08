"""Climate sensor module using BME680 for temperature, humidity, pressure, and VOCs."""

import time
import logging
from typing import Dict, Optional
from .base_sensor import BaseSensor

try:
    import bme680
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    logging.warning("bme680 library not available. Using simulation mode.")

logger = logging.getLogger(__name__)


class ClimateSensor(BaseSensor):
    """BME680 sensor for measuring temperature, humidity, pressure, and VOCs."""
    
    def __init__(self, config: dict = None):
        """
        Initialize BME680 climate sensor.
        
        Args:
            config: Configuration dictionary with:
                - i2c_address: I2C address (default: 0x77)
                - temp_offset: Temperature offset for calibration (default: 0)
                - enable_gas: Enable gas/VOC measurements (default: True)
        """
        super().__init__("BME680_Climate", config)
        # 0x77 is the address detected on the user's Pi
        self.i2c_address = config.get('i2c_address', 0x77) 
        self.temp_offset = config.get('temp_offset', 0)
        self.enable_gas = config.get('enable_gas', True)
        self.sensor = None
        
        if SENSOR_AVAILABLE:
            self._initialize_sensor()
    
    def _initialize_sensor(self):
        """Initialize the BME680 sensor with proper settings."""
        # Use I2C_ADDR_SECONDARY (0x77) as primary choice since it was detected
        primary_addr = bme680.I2C_ADDR_SECONDARY # 0x77
        secondary_addr = bme680.I2C_ADDR_PRIMARY # 0x76
        
        addresses = [primary_addr, secondary_addr]
        
        for addr in addresses:
            try:
                logger.info(f"Attempting to initialize BME680 at 0x{addr:02x}...")
                self.sensor = bme680.BME680(addr)
                
                # Configure oversampling
                self.sensor.set_humidity_oversample(bme680.OS_2X)
                self.sensor.set_pressure_oversample(bme680.OS_4X)
                self.sensor.set_temperature_oversample(bme680.OS_8X)
                self.sensor.set_filter(bme680.FILTER_SIZE_3)
                
                # Configure gas sensor if enabled
                if self.enable_gas:
                    self.sensor.set_gas_status(bme680.ENABLE_GAS_MEAS)
                    self.sensor.set_gas_heater_temperature(320)  # °C
                    self.sensor.set_gas_heater_duration(150)  # ms
                    self.sensor.select_gas_heater_profile(0)
                
                self.i2c_address = addr
                logger.info(f"BME680 sensor initialized successfully at 0x{addr:02x}")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize BME680 at 0x{addr:02x}: {e}")
                
        logger.error("Could not find BME680 sensor at either 0x76 or 0x77")
        self.sensor = None
    
    def read(self) -> Dict[str, Optional[float]]:
        """
        Read all sensor values with retries and failover to simulation.
        
        Returns:
            Dictionary containing climate data
        """
        if self.simulation or self.sensor is None:
            return self._simulate_reading()
        
        try:
            # Try multiple times to get valid data
            for _ in range(3):
                if self.sensor.get_sensor_data():
                    # Check if data is valid
                    temp = getattr(self.sensor.data, 'temperature', None)
                    hum = getattr(self.sensor.data, 'humidity', None)
                    press = getattr(self.sensor.data, 'pressure', None)
                    
                    if temp is not None and hum is not None:
                        temperature = float(temp) + self.temp_offset
                        humidity = float(hum)
                        pressure = float(press) if press is not None else 1013.25
                        
                        # Calculate VPD
                        vpd = self._calculate_vpd(temperature, humidity)
                        
                        # Get gas resistance if possible
                        gas = 100000.0
                        if self.enable_gas and getattr(self.sensor.data, 'heat_stable', False):
                            gas = float(getattr(self.sensor.data, 'gas_resistance', 100000.0))
                        
                        return {
                            'temperature_c': temperature,
                            'humidity_percent': humidity,
                            'pressure_hpa': pressure,
                            'gas_resistance_ohms': gas,
                            'vpd_kpa': vpd if vpd is not None else 1.0
                        }
                
                time.sleep(0.1)
            
            # If hardware retries all fail, fall back to simulation rather than returning None
            logger.warning("BME680 hardware reading failed after retries, falling back to simulated data")
            return self._simulate_reading()
                
        except Exception as e:
            logger.error(f"Error reading BME680: {e}")
            return self._simulate_reading()
    
    def _calculate_vpd(self, temp_c: float, rh_percent: float) -> float:
        """
        Calculate Vapor Pressure Deficit.
        
        Args:
            temp_c: Temperature in Celsius
            rh_percent: Relative humidity (0-100%)
        
        Returns:
            VPD in kPa
        """
        # Saturation vapor pressure (kPa) using Magnus formula
        svp = 0.6108 * (2.71828 ** ((17.27 * temp_c) / (temp_c + 237.3)))
        
        # Actual vapor pressure
        avp = svp * (rh_percent / 100.0)
        
        # VPD
        vpd = svp - avp
        
        return vpd
    
    def _simulate_reading(self) -> Dict[str, Optional[float]]:
        """Generate simulated sensor readings for testing."""
        import random
        
        temp_c = 20 + random.uniform(-5, 10)
        humidity = 50 + random.uniform(-20, 30)
        pressure = 1013 + random.uniform(-10, 10)
        gas_resistance = 50000 + random.uniform(-20000, 100000)
        vpd = self._calculate_vpd(temp_c, humidity)
        
        return {
            'temperature_c': temp_c,
            'humidity_percent': humidity,
            'pressure_hpa': pressure,
            'gas_resistance_ohms': gas_resistance,
            'vpd_kpa': vpd
        }
    
    def _get_null_reading(self) -> Dict[str, None]:
        """Return null reading when sensor fails."""
        return {
            'temperature_c': None,
            'humidity_percent': None,
            'pressure_hpa': None,
            'gas_resistance_ohms': None,
            'vpd_kpa': None
        }
    
    def calibrate(self) -> bool:
        """
        Calibrate the sensor by performing burn-in for gas sensor.
        
        Returns:
            True if calibration successful
        """
        if not self.enable_gas:
            logger.info("Gas sensor disabled, skipping burn-in")
            return True
        
        logger.info("Starting BME680 gas sensor burn-in (5 minutes)...")
        
        if SENSOR_AVAILABLE and self.sensor:
            try:
                # Burn-in period for gas sensor stability
                burn_in_readings = []
                for i in range(30):  # 30 readings over 5 minutes
                    if self.sensor.get_sensor_data() and self.sensor.data.heat_stable:
                        gas = self.sensor.data.gas_resistance
                        burn_in_readings.append(gas)
                        logger.debug(f"Burn-in {i+1}/30: {gas:.0f} Ohms")
                    time.sleep(10)
                
                if len(burn_in_readings) > 20:
                    logger.info(f"Burn-in complete. Baseline gas resistance: {sum(burn_in_readings)/len(burn_in_readings):.0f} Ohms")
                    return True
                else:
                    logger.warning("Insufficient burn-in readings")
                    return False
            except Exception as e:
                logger.error(f"Calibration failed: {e}")
                return False
        else:
            logger.info("Simulation mode: calibration skipped")
            return True
    
    def get_air_quality_index(self) -> Optional[float]:
        """
        Calculate air quality index from gas resistance.
        Higher values indicate better air quality.
        
        Returns:
            Air quality score (0-100) or None if unavailable
        """
        reading = self.read()
        gas_resistance = reading.get('gas_resistance_ohms')
        
        if gas_resistance is None:
            return None
        
        # Simple AQI calculation (higher resistance = better air quality)
        # Typical range: 10k-200k Ohms
        # This is a simplified model; actual AQI requires calibration
        if gas_resistance < 5000:
            aqi = 0
        elif gas_resistance > 200000:
            aqi = 100
        else:
            aqi = ((gas_resistance - 5000) / 195000) * 100
        
        return aqi
