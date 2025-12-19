"""Main data collection orchestrator."""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from sensors.co2_sensor import CO2Sensor
from sensors.climate_sensor import ClimateSensor
from sensors.ndvi_camera import NDVICamera
from data.data_logger import DataLogger

logger = logging.getLogger(__name__)


class DataCollector:
    """Orchestrates data collection from all sensors."""
    
    def __init__(self, config: dict = None):
        """
        Initialize data collector.
        
        Args:
            config: Configuration dictionary with sensor and logging settings
        """
        self.config = config or {}
        
        # Initialize sensors
        logger.info("Initializing sensors...")
        self.co2_sensor = CO2Sensor(config.get('co2_sensor', {}))
        self.climate_sensor = ClimateSensor(config.get('climate_sensor', {}))
        self.ndvi_camera = NDVICamera(config.get('ndvi_camera', {}))
        
        # Initialize data logger
        self.data_logger = DataLogger(config.get('data_logger', {}))
        
        # Collection settings
        self.collection_interval = config.get('collection_interval', 3600)  # 1 hour default
        self.respiration_interval = config.get('respiration_interval', 14400)  # 4 hours default
        self.photo_interval = config.get('photo_interval', 3600)  # 1 hour default
        
        self.is_running = False
        self.measurement_count = 0
        
        logger.info("Data collector initialized")
    
    def calibrate_all_sensors(self) -> bool:
        """
        Calibrate all sensors.
        
        Returns:
            True if all calibrations successful
        """
        logger.info("Starting sensor calibration...")
        
        results = {
            'co2': self.co2_sensor.calibrate(),
            'climate': self.climate_sensor.calibrate(),
            'camera': self.ndvi_camera.calibrate()
        }
        
        success = all(results.values())
        
        if success:
            logger.info("All sensors calibrated successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"Calibration failed for: {', '.join(failed)}")
        
        return success
    
    def collect_single_measurement(self) -> Dict:
        """
        Collect a single measurement from all sensors.
        
        Returns:
            Dictionary with all sensor readings
        """
        timestamp = datetime.now().isoformat()
        
        logger.info(f"Collecting measurement #{self.measurement_count + 1}")
        
        # Collect climate data
        climate_data = self.climate_sensor.read()
        
        # Collect CO2 data (quick reading, not full respiration)
        co2_data = self.co2_sensor.read()
        
        # Collect NDVI data (if it's time for a photo)
        ndvi_data = None
        if self.measurement_count % (self.photo_interval // self.collection_interval) == 0:
            ndvi_data = self.ndvi_camera.read()
        
        # Combine all data
        measurement = {
            'timestamp': timestamp,
            'measurement_id': self.measurement_count,
            'climate': climate_data,
            'co2': co2_data,
            'ndvi': ndvi_data
        }
        
        # Log the measurement
        self.data_logger.log_measurement(measurement)
        
        self.measurement_count += 1
        
        return measurement
    
    def collect_respiration_measurement(self) -> Dict:
        """
        Collect full soil respiration measurement (takes ~5 minutes).
        
        Returns:
            Dictionary with respiration flux data
        """
        logger.info("Starting soil respiration measurement...")
        
        timestamp = datetime.now().isoformat()
        
        # Get current climate conditions
        climate_data = self.climate_sensor.read()
        
        # Measure soil respiration flux
        respiration_data = self.co2_sensor.measure_respiration()
        
        # Combine data
        measurement = {
            'timestamp': timestamp,
            'measurement_type': 'respiration',
            'climate': climate_data,
            'respiration': respiration_data
        }
        
        # Log the measurement
        self.data_logger.log_measurement(measurement)
        
        logger.info(f"Respiration flux: {respiration_data.get('flux_umol_m2_s', 'N/A'):.2f} μmol m⁻² s⁻¹")
        
        return measurement
    
    def run_continuous_collection(self, duration_hours: Optional[float] = None):
        """
        Run continuous data collection.
        
        Args:
            duration_hours: Duration to run (None = indefinite)
        """
        logger.info("Starting continuous data collection...")
        logger.info(f"Collection interval: {self.collection_interval}s")
        logger.info(f"Respiration interval: {self.respiration_interval}s")
        logger.info(f"Photo interval: {self.photo_interval}s")
        
        self.is_running = True
        start_time = time.time()
        last_respiration_time = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Check if we should stop
                if duration_hours and (current_time - start_time) > (duration_hours * 3600):
                    logger.info("Collection duration reached")
                    break
                
                # Collect regular measurement
                self.collect_single_measurement()
                
                # Check if it's time for respiration measurement
                if (current_time - last_respiration_time) >= self.respiration_interval:
                    self.collect_respiration_measurement()
                    last_respiration_time = current_time
                
                # Wait for next collection
                time.sleep(self.collection_interval)
                
        except KeyboardInterrupt:
            logger.info("Collection stopped by user")
        except Exception as e:
            logger.error(f"Collection error: {e}")
        finally:
            self.stop_collection()
    
    def run_drought_experiment(self, 
                              baseline_days: int = 5,
                              drought_days: int = 9,
                              recovery_days: int = 3):
        """
        Run automated drought stress experiment.
        
        Args:
            baseline_days: Days of normal watering
            drought_days: Days without water
            recovery_days: Days of recovery watering
        """
        logger.info("=" * 60)
        logger.info("DROUGHT STRESS EXPERIMENT")
        logger.info(f"Baseline: {baseline_days} days")
        logger.info(f"Drought: {drought_days} days")
        logger.info(f"Recovery: {recovery_days} days")
        logger.info("=" * 60)
        
        phases = [
            ('baseline', baseline_days),
            ('drought', drought_days),
            ('recovery', recovery_days)
        ]
        
        for phase_name, phase_days in phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE: {phase_name.upper()} ({phase_days} days)")
            logger.info(f"{'='*60}\n")
            
            # Update experiment phase in config
            self.data_logger.config['current_phase'] = phase_name
            
            # Run collection for this phase
            self.run_continuous_collection(duration_hours=phase_days * 24)
            
            # Prompt for watering action
            if phase_name == 'baseline':
                logger.info("\n⚠️  STOP WATERING - Begin drought phase")
            elif phase_name == 'drought':
                logger.info("\n💧 RESUME WATERING - Begin recovery phase")
        
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*60)
    
    def stop_collection(self):
        """Stop data collection and cleanup."""
        logger.info("Stopping data collection...")
        self.is_running = False
        self.data_logger.close()
        logger.info(f"Total measurements collected: {self.measurement_count}")
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of collected data.
        
        Returns:
            Dictionary with summary statistics
        """
        df = self.data_logger.load_session_data()
        
        if df.empty:
            return {'error': 'No data available'}
        
        summary = {
            'total_measurements': len(df),
            'start_time': df['timestamp'].min() if 'timestamp' in df else None,
            'end_time': df['timestamp'].max() if 'timestamp' in df else None,
            'duration_hours': None
        }
        
        # Calculate duration
        if summary['start_time'] and summary['end_time']:
            start = pd.to_datetime(summary['start_time'])
            end = pd.to_datetime(summary['end_time'])
            summary['duration_hours'] = (end - start).total_seconds() / 3600
        
        return summary


if __name__ == "__main__":
    # Example configuration
    config = {
        'co2_sensor': {
            'chamber_volume': 2.0,
            'soil_area': 0.01,
            'measurement_duration': 300,
            'sampling_interval': 10
        },
        'climate_sensor': {
            'enable_gas': True
        },
        'ndvi_camera': {
            'resolution': (1920, 1080),
            'save_path': './data/images'
        },
        'data_logger': {
            'data_dir': './data',
            'experiment_name': 'drought_stress_test',
            'log_format': 'csv'
        },
        'collection_interval': 900,  # 15 minutes
        'respiration_interval': 14400,  # 4 hours
        'photo_interval': 3600  # 1 hour
    }
    
    # Initialize collector
    collector = DataCollector(config)
    
    # Calibrate sensors
    collector.calibrate_all_sensors()
    
    # Run experiment
    # collector.run_drought_experiment(baseline_days=5, drought_days=9, recovery_days=3)
    
    # Or run continuous collection
    collector.run_continuous_collection(duration_hours=24)
