"""Drought stress simulation for testing without physical hardware."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class DroughtSimulator:
    """Simulate crop stress response to drought conditions."""
    
    def __init__(self, config: dict = None):
        """
        Initialize drought simulator.
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}
        
        # Plant parameters
        self.initial_ndvi = config.get('initial_ndvi', 0.75)
        self.stress_threshold = config.get('stress_threshold', 0.4)
        self.recovery_rate = config.get('recovery_rate', 0.05)
        
        # Soil parameters
        self.initial_moisture = config.get('initial_moisture', 0.6)
        self.field_capacity = config.get('field_capacity', 0.7)
        self.wilting_point = config.get('wilting_point', 0.15)
        
        # Microbial parameters
        self.max_respiration = config.get('max_respiration', 8.0)  # μmol m⁻² s⁻¹
        self.min_respiration = config.get('min_respiration', 0.5)
        
        # Time parameters
        self.time_step_hours = config.get('time_step_hours', 1)
        
        # State variables
        self.current_time = datetime.now()
        self.soil_moisture = self.initial_moisture
        self.plant_ndvi = self.initial_ndvi
        self.soil_respiration = self.max_respiration
        self.plant_stress_level = 0.0
        self.is_watered = True
        
        # History
        self.history = []
        
        logger.info("Drought simulator initialized")
    
    def step(self, watered: bool = False, 
            temperature: Optional[float] = None,
            humidity: Optional[float] = None) -> Dict:
        """
        Simulate one time step.
        
        Args:
            watered: Whether plant was watered this step
            temperature: Air temperature (°C), if None uses random
            humidity: Relative humidity (%), if None uses random
        
        Returns:
            Dictionary with current state
        """
        # Generate climate if not provided
        if temperature is None:
            temperature = 20 + np.random.normal(5, 3)
        if humidity is None:
            humidity = 50 + np.random.normal(0, 15)
        
        humidity = np.clip(humidity, 10, 95)
        
        # Calculate VPD
        vpd = self._calculate_vpd(temperature, humidity)
        
        # Update soil moisture
        self._update_soil_moisture(watered, temperature, vpd)
        
        # Update soil respiration (responds faster than plant)
        self._update_soil_respiration()
        
        # Update plant stress (lags behind soil)
        self._update_plant_stress()
        
        # Update NDVI (lags behind stress)
        self._update_ndvi()
        
        # Generate VOC signal
        voc_signal = self._generate_voc_signal()
        
        # Advance time
        self.current_time += timedelta(hours=self.time_step_hours)
        
        # Record state
        state = {
            'timestamp': self.current_time.isoformat(),
            'soil_moisture': self.soil_moisture,
            'soil_respiration_flux': self.soil_respiration,
            'plant_stress_level': self.plant_stress_level,
            'ndvi': self.plant_ndvi,
            'temperature': temperature,
            'humidity': humidity,
            'vpd': vpd,
            'voc_resistance': voc_signal,
            'watered': watered
        }
        
        self.history.append(state)
        
        return state
    
    def _update_soil_moisture(self, watered: bool, temperature: float, vpd: float):
        """
        Update soil moisture based on watering and evapotranspiration.
        
        Args:
            watered: Whether plant was watered
            temperature: Air temperature
            vpd: Vapor pressure deficit
        """
        if watered:
            # Add water (bring to field capacity)
            self.soil_moisture = min(self.field_capacity, self.soil_moisture + 0.3)
        
        # Evapotranspiration (depends on temperature and VPD)
        et_rate = 0.01 * (1 + temperature / 25) * (1 + vpd / 2)
        et_rate *= self.time_step_hours / 24  # Scale to time step
        
        # Plant water uptake (depends on plant health)
        if self.plant_ndvi > 0.3:
            uptake_rate = 0.02 * (self.plant_ndvi / 0.75) * self.time_step_hours / 24
        else:
            uptake_rate = 0.005 * self.time_step_hours / 24
        
        # Update moisture
        self.soil_moisture -= (et_rate + uptake_rate)
        self.soil_moisture = max(self.wilting_point * 0.5, self.soil_moisture)
    
    def _update_soil_respiration(self):
        """
        Update soil microbial respiration.
        
        Respiration drops quickly when soil dries (microbes respond fast).
        """
        # Moisture factor (exponential response)
        if self.soil_moisture > self.field_capacity * 0.8:
            moisture_factor = 1.0
        elif self.soil_moisture < self.wilting_point:
            moisture_factor = 0.1
        else:
            # Exponential decay
            moisture_factor = np.exp(-3 * (1 - self.soil_moisture / self.field_capacity))
        
        # Target respiration
        target_respiration = self.min_respiration + (self.max_respiration - self.min_respiration) * moisture_factor
        
        # Smooth transition (microbes respond within hours)
        response_rate = 0.3  # 30% change per time step
        self.soil_respiration += (target_respiration - self.soil_respiration) * response_rate
        
        # Add noise
        self.soil_respiration += np.random.normal(0, 0.1)
        self.soil_respiration = np.clip(self.soil_respiration, self.min_respiration, self.max_respiration)
    
    def _update_plant_stress(self):
        """
        Update plant stress level.
        
        Plant stress lags behind soil moisture by several hours to days.
        """
        # Calculate stress from soil moisture
        if self.soil_moisture > self.field_capacity * 0.6:
            target_stress = 0.0
        elif self.soil_moisture < self.wilting_point:
            target_stress = 1.0
        else:
            # Linear increase in stress as moisture drops
            target_stress = 1 - (self.soil_moisture - self.wilting_point) / (self.field_capacity * 0.6 - self.wilting_point)
        
        # Slow response (plant takes time to show stress)
        stress_response_rate = 0.05  # 5% change per time step (slower than microbes)
        self.plant_stress_level += (target_stress - self.plant_stress_level) * stress_response_rate
        
        self.plant_stress_level = np.clip(self.plant_stress_level, 0, 1)
    
    def _update_ndvi(self):
        """
        Update plant NDVI.
        
        NDVI lags behind stress level (visible symptoms appear last).
        """
        # Target NDVI based on stress
        if self.plant_stress_level < 0.2:
            target_ndvi = self.initial_ndvi
        elif self.plant_stress_level > 0.8:
            target_ndvi = 0.2
        else:
            # Linear decline
            target_ndvi = self.initial_ndvi - (self.initial_ndvi - 0.2) * (self.plant_stress_level - 0.2) / 0.6
        
        # Very slow response (NDVI changes over days)
        ndvi_response_rate = 0.02  # 2% change per time step (slowest response)
        self.plant_ndvi += (target_ndvi - self.plant_ndvi) * ndvi_response_rate
        
        # Add noise
        self.plant_ndvi += np.random.normal(0, 0.01)
        self.plant_ndvi = np.clip(self.plant_ndvi, 0.1, 1.0)
    
    def _generate_voc_signal(self) -> float:
        """
        Generate VOC signal (gas resistance).
        
        Stressed plants emit more VOCs (lower gas resistance).
        
        Returns:
            Gas resistance in Ohms
        """
        # Base resistance (healthy plant)
        base_resistance = 150000
        
        # Stress reduces resistance (more VOCs)
        stress_factor = 1 - 0.7 * self.plant_stress_level
        
        resistance = base_resistance * stress_factor
        
        # Add noise
        resistance += np.random.normal(0, 10000)
        
        return max(10000, resistance)
    
    def _calculate_vpd(self, temp_c: float, rh_percent: float) -> float:
        """
        Calculate Vapor Pressure Deficit.
        
        Args:
            temp_c: Temperature in Celsius
            rh_percent: Relative humidity (0-100%)
        
        Returns:
            VPD in kPa
        """
        svp = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        avp = svp * (rh_percent / 100.0)
        vpd = svp - avp
        return vpd
    
    def run_drought_experiment(self, 
                              baseline_days: int = 5,
                              drought_days: int = 9,
                              recovery_days: int = 3) -> pd.DataFrame:
        """
        Run complete drought experiment simulation.
        
        Args:
            baseline_days: Days of normal watering
            drought_days: Days without water
            recovery_days: Days of recovery watering
        
        Returns:
            DataFrame with simulation results
        """
        logger.info("Starting drought experiment simulation...")
        logger.info(f"Baseline: {baseline_days} days, Drought: {drought_days} days, Recovery: {recovery_days} days")
        
        total_hours = (baseline_days + drought_days + recovery_days) * 24
        
        for hour in range(total_hours):
            day = hour // 24
            
            # Determine if watered
            if day < baseline_days:
                # Baseline: water daily
                watered = (hour % 24 == 8)  # Water at 8 AM
            elif day < baseline_days + drought_days:
                # Drought: no water
                watered = False
            else:
                # Recovery: water daily
                watered = (hour % 24 == 8)
            
            # Simulate time step
            self.step(watered=watered)
            
            if hour % 24 == 0:
                logger.info(f"Day {day}: Moisture={self.soil_moisture:.2f}, NDVI={self.plant_ndvi:.2f}, Stress={self.plant_stress_level:.2f}")
        
        # Convert history to DataFrame
        df = pd.DataFrame(self.history)
        
        logger.info(f"Simulation complete: {len(df)} time steps")
        
        return df
    
    def get_lag_analysis(self) -> Dict:
        """
        Analyze lag times between different indicators.
        
        Returns:
            Dictionary with lag time analysis
        """
        if len(self.history) < 10:
            return {'error': 'Insufficient data for lag analysis'}
        
        df = pd.DataFrame(self.history)
        
        # Find when each indicator drops below threshold
        moisture_threshold = self.field_capacity * 0.5
        respiration_threshold = self.max_respiration * 0.5
        stress_threshold = 0.5
        ndvi_threshold = self.initial_ndvi * 0.8
        
        # Find first crossing times
        moisture_drop = df[df['soil_moisture'] < moisture_threshold].index.min()
        respiration_drop = df[df['soil_respiration_flux'] < respiration_threshold].index.min()
        stress_rise = df[df['plant_stress_level'] > stress_threshold].index.min()
        ndvi_drop = df[df['ndvi'] < ndvi_threshold].index.min()
        
        # Calculate lags (in time steps)
        lags = {}
        if pd.notna(moisture_drop):
            if pd.notna(respiration_drop):
                lags['respiration_lag_from_moisture'] = (respiration_drop - moisture_drop) * self.time_step_hours
            if pd.notna(stress_rise):
                lags['stress_lag_from_moisture'] = (stress_rise - moisture_drop) * self.time_step_hours
            if pd.notna(ndvi_drop):
                lags['ndvi_lag_from_moisture'] = (ndvi_drop - moisture_drop) * self.time_step_hours
        
        if pd.notna(respiration_drop) and pd.notna(ndvi_drop):
            lags['ndvi_lag_from_respiration'] = (ndvi_drop - respiration_drop) * self.time_step_hours
        
        logger.info(f"Lag analysis: {lags}")
        
        return lags
    
    def reset(self):
        """Reset simulator to initial state."""
        self.current_time = datetime.now()
        self.soil_moisture = self.initial_moisture
        self.plant_ndvi = self.initial_ndvi
        self.soil_respiration = self.max_respiration
        self.plant_stress_level = 0.0
        self.history = []
        logger.info("Simulator reset")
