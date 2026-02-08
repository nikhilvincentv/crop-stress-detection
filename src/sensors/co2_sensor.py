"""CO2 sensor module for soil respiration measurements."""

import time
import numpy as np
from scipy import stats
from typing import List, Tuple
import logging
from .base_sensor import BaseSensor

try:
    import mh_z19
    import serial
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    logging.warning("mh_z19/pyserial libraries not available. Using simulation mode.")

logger = logging.getLogger(__name__)


class CO2Sensor(BaseSensor):
    """CO2 sensor for measuring soil respiration."""
    
    def __init__(self, config: dict = None):
        """
        Initialize CO2 sensor.
        
        Args:
            config: Configuration dictionary with:
                - chamber_volume: Chamber volume in liters
                - soil_area: Soil surface area in m²
                - measurement_duration: Duration in seconds (default: 300)
                - sampling_interval: Interval between readings in seconds (default: 10)
        """
        super().__init__("CO2_Sensor", config)
        
        # Check if we should use simulation
        self.simulation = config.get('simulation', True)  # Default to True
        
        # If not simulation, test if hardware works
        if not self.simulation:
            try:
                import serial
                # Test if serial port works
                self.baud_rate = config.get('baud_rate', 9600)
                ser = serial.Serial('/dev/serial0', self.baud_rate, timeout=1)
                
                # Send wake-up command (some sensors need this)
                wake_cmd = b'\xff\x01\x79\xa0\x00\x00\x00\x00\xe6'
                ser.write(wake_cmd)
                time.sleep(0.1)
                ser.close()
                
                # Now test mh_z19
                import mh_z19
                test_result = mh_z19.read()
                if test_result and 'co2' in test_result:
                    logger.info(f"CO2 hardware works: {test_result['co2']} ppm")
                else:
                    # Try manual read if library fails
                    logger.warning("mh_z19 library returned empty data, trying manual serial read...")
                    manual_data = self._read_manual()
                    if manual_data.get('co2_ppm') is not None:
                        logger.info(f"Manual CO2 hardware read works: {manual_data['co2_ppm']} ppm")
                    else:
                        self.simulation = True
                        logger.warning("CO2 hardware returned empty or invalid data. Using simulation.")
                    
            except Exception as e:
                self.simulation = True
                logger.warning(f"CO2 hardware error: {e}. Using simulation mode.")
        
        self.chamber_volume = config.get('chamber_volume', 2.0)
        self.soil_area = config.get('soil_area', 0.01)
        self.measurement_duration = config.get('measurement_duration', 300)
        self.sampling_interval = config.get('sampling_interval', 10)
        self.baseline_co2 = None
    
    def read(self) -> dict:
        """Read current CO2 concentration."""
        if self.simulation:
            # Return simulated data
            import random
            return {
                'co2_ppm': 400 + random.uniform(-20, 20),
                'baseline_co2': self.baseline_co2
            }
            
        try:
            import mh_z19
            data = mh_z19.read()
            if data and 'co2' in data:
                return {'co2_ppm': float(data['co2']), 'baseline_co2': self.baseline_co2}
            else:
                # Fallback to manual serial read if library fails
                return self._read_manual()
        except Exception as e:
            logger.error(f"Error reading CO2 sensor via library: {e}")
            return self._read_manual()

    def _read_manual(self) -> dict:
        """Read CO2 concentration directly from serial if library fails."""
        try:
            import serial
            with serial.Serial('/dev/serial0', getattr(self, 'baud_rate', 9600), timeout=2.0) as ser:
                ser.flushInput()
                read_cmd = b'\xff\x01\x86\x00\x00\x00\x00\x00\x79'
                ser.write(read_cmd)
                time.sleep(0.5)
                response = ser.read(9)
                
                if len(response) >= 9:
                    co2 = response[2] * 256 + response[3]
                    return {'co2_ppm': float(co2), 'baseline_co2': self.baseline_co2}
                else:
                    logger.warning(f"CO2 manual read short response: {response.hex()}")
                    return {'co2_ppm': None, 'baseline_co2': self.baseline_co2}
        except Exception as e:
            logger.error(f"Manual CO2 read failure: {e}")
            return {'co2_ppm': None, 'baseline_co2': self.baseline_co2}
    
    def measure_respiration(self) -> dict:
        """
        Measure soil respiration flux.
        
        Returns:
            Dictionary containing:
                - flux: CO2 flux in µmol m⁻² s⁻¹
                - r_squared: R² value of linear fit
                - co2_data: Time series of CO2 measurements
                - slope: CO2 accumulation rate (ppm/s)
        """
        logger.info("Starting respiration measurement...")
        
        # Collect CO2 data over time
        co2_readings = []
        timestamps = []
        start_time = time.time()
        
        # Wait for chamber equilibration
        logger.info("Equilibrating chamber (30 seconds)...")
        time.sleep(30)
        
        # Collect measurements
        while (time.time() - start_time) < self.measurement_duration:
            reading = self.read()
            co2_ppm = reading.get('co2_ppm')
            
            if co2_ppm is not None:
                co2_readings.append(co2_ppm)
                timestamps.append(time.time() - start_time)
                logger.debug(f"CO2: {co2_ppm:.1f} ppm at {timestamps[-1]:.1f}s")
            
            time.sleep(self.sampling_interval)
        
        # Calculate flux from linear regression
        flux, r_squared, slope = self._calculate_flux(timestamps, co2_readings)
        
        logger.info(f"Respiration flux: {flux:.2f} µmol m⁻² s⁻¹ (R² = {r_squared:.3f})")
        
        return {
            'flux_umol_m2_s': flux,
            'r_squared': r_squared,
            'slope_ppm_s': slope,
            'co2_data': list(zip(timestamps, co2_readings)),
            'num_samples': len(co2_readings)
        }
    
    def _calculate_flux(self, timestamps: List[float], co2_ppm: List[float]) -> Tuple[float, float, float]:
        """
        Calculate CO2 flux from time series data.
        
        Args:
            timestamps: List of timestamps in seconds
            co2_ppm: List of CO2 concentrations in ppm
        
        Returns:
            Tuple of (flux in µmol m⁻² s⁻¹, R², slope in ppm/s)
        """
        if len(timestamps) < 3:
            logger.warning("Insufficient data points for flux calculation")
            return 0.0, 0.0, 0.0
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, co2_ppm)
        r_squared = r_value ** 2
        
        # Convert slope (ppm/s) to flux (µmol m⁻² s⁻¹)
        # Flux = (dC/dt) × (V / A) × (P / (R × T)) × 10^6
        # Simplified: flux ≈ slope × chamber_volume / soil_area × conversion_factor
        
        # Conversion factor (approximate at 25°C, 1 atm)
        # 1 ppm = 1 µmol/mol air
        # At STP: 1 mol gas = 22.4 L
        conversion_factor = 1000 / 22.4  # µmol/L per ppm
        
        flux = slope * (self.chamber_volume / self.soil_area) * conversion_factor
        
        return flux, r_squared, slope
    
    def calibrate(self) -> bool:
        """Calibrate CO2 sensor by measuring baseline CO2."""
        if self.simulation:
            logger.info("Simulation mode: calibration skipped")
            self.baseline_co2 = 400  # Set a simulated baseline
            return True
            
        logger.info("Calibrating CO2 sensor...")
        readings = []
        
        for _ in range(10):
            reading = self.read()
            if reading.get('co2_ppm') is not None:
                readings.append(reading['co2_ppm'])
            time.sleep(1)
        
        if readings:
            self.baseline_co2 = np.mean(readings)
            logger.info(f"Baseline CO2: {self.baseline_co2:.1f} ppm")
            return True
        else:
            logger.error("Calibration failed: no valid readings")
            return False
