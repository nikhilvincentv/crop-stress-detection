"""CO2 sensor module for soil respiration measurements."""

import time
import numpy as np
from scipy import stats
from typing import List, Tuple
import logging
from .base_sensor import BaseSensor

try:
    import mh_z19
    SENSOR_AVAILABLE = True
except ImportError:
    SENSOR_AVAILABLE = False
    logging.warning("mh_z19 library not available. Using simulation mode.")

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
                import mh_z19
                # Test if hardware works
                test_result = mh_z19.read()
                if not test_result:
                    # Hardware failed, force simulation
                    self.simulation = True
                    logger.warning("CO2 hardware failed, falling back to simulation")
            except Exception as e:
                self.simulation = True
                logger.warning(f"CO2 hardware error: {e}. Using simulation mode.")
        
        self.chamber_volume = config.get('chamber_volume', 2.0)
        self.soil_area = config.get('soil_area', 0.01)
        self.measurement_duration = config.get('measurement_duration', 300)
        self.sampling_interval = config.get('sampling_interval', 10)
        self.baseline_co2 = None
    
    def read(self) -> dict:
        """
        Read current CO2 concentration.
        
        Returns:
            Dictionary with CO2 concentration in ppm
        """
        if SENSOR_AVAILABLE:
            try:
                data = mh_z19.read()
                co2_ppm = data.get('co2', None)
                return {'co2_ppm': co2_ppm}
            except PermissionError:
                logger.error("Permission denied to /dev/serial0. Try: sudo usermod -a -G dialout $USER && sudo chmod 666 /dev/serial0")
                return {'co2_ppm': None}
            except Exception as e:
                if "Permission denied" in str(e):
                    logger.error("Permission denied to /dev/serial0. Try: sudo usermod -a -G dialout $USER && sudo chmod 666 /dev/serial0")
                else:
                    logger.error(f"Error reading CO2 sensor: {e}")
                return {'co2_ppm': None}
        else:
            # Simulation mode
            import random
            co2_ppm = 400 + random.uniform(-20, 100)
            return {'co2_ppm': co2_ppm}
    
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
        """
        Calibrate sensor by measuring baseline CO2.
        
        Returns:
            True if calibration successful
        """
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
