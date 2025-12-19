#!/usr/bin/env python3
"""Main entry point for crop stress early warning system."""

import sys
import logging
from pathlib import Path
import argparse
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_collector import DataCollector
from simulation.drought_simulator import DroughtSimulator
from analysis.feature_engineering import FeatureEngineer
from models.stress_predictor import StressPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crop_stress_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def run_simulation_mode():
    """Run system in simulation mode for testing."""
    logger.info("="*60)
    logger.info("CROP STRESS EARLY WARNING SYSTEM - SIMULATION MODE")
    logger.info("="*60)
    
    # Initialize simulator
    simulator = DroughtSimulator({
        'initial_ndvi': 0.75,
        'initial_moisture': 0.6,
        'time_step_hours': 1
    })
    
    # Run drought experiment
    logger.info("\nRunning drought stress simulation...")
    df = simulator.run_drought_experiment(
        baseline_days=5,
        drought_days=9,
        recovery_days=3
    )
    
    # Save simulation data
    output_path = Path('data/simulated_drought_experiment.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSimulation data saved: {output_path}")
    
    # Analyze lag times
    logger.info("\nAnalyzing lag times...")
    lags = simulator.get_lag_analysis()
    for key, value in lags.items():
        logger.info(f"  {key}: {value:.1f} hours")
    
    # Feature engineering
    logger.info("\nEngineering features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df)
    
    # Prepare prediction dataset
    logger.info("\nPreparing prediction dataset...")
    X, y = engineer.create_prediction_dataset(
        features_df,
        prediction_horizon=24,  # Predict 24 hours ahead
        target_col='ndvi'
    )
    
    logger.info(f"Dataset: {len(X)} samples, {X.shape[1]} features")
    
    # Train models
    logger.info("\nTraining prediction models...")
    
    # Compare different models
    predictor = StressPredictor('random_forest')
    comparison = predictor.compare_models(X, y)
    logger.info("\nModel Comparison:")
    logger.info(comparison.to_string())
    
    # Train best model
    best_model_type = comparison.iloc[0]['model']
    logger.info(f"\nTraining best model: {best_model_type}")
    
    best_predictor = StressPredictor(best_model_type)
    metrics = best_predictor.train(X, y, validation=True)
    
    logger.info("\nModel Performance:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Evaluate lag time prediction
    logger.info("\nEvaluating prediction horizons...")
    lag_results = best_predictor.evaluate_lag_time(
        X, y,
        horizons=[1, 6, 12, 24, 48, 72]
    )
    logger.info("\nPrediction Horizon Performance:")
    logger.info(lag_results.to_string())
    
    # Save model
    model_path = Path('models/stress_predictor.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    best_predictor.save_model(str(model_path))
    logger.info(f"\nModel saved: {model_path}")
    
    # Generate plots
    logger.info("\nGenerating visualizations...")
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    best_predictor.plot_feature_importance(
        top_n=20,
        save_path=str(plots_dir / 'feature_importance.png')
    )
    
    best_predictor.plot_predictions(
        X, y,
        save_path=str(plots_dir / 'predictions.png')
    )
    
    logger.info("\n" + "="*60)
    logger.info("SIMULATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Data: {output_path}")
    logger.info(f"  - Model: {model_path}")
    logger.info(f"  - Plots: {plots_dir}/")


def run_hardware_mode():
    """Run system with actual hardware sensors."""
    logger.info("="*60)
    logger.info("CROP STRESS EARLY WARNING SYSTEM - HARDWARE MODE")
    logger.info("="*60)
    
    # Configuration
    config = {
        'co2_sensor': {
            'chamber_volume': 2.0,
            'soil_area': 0.01,
            'measurement_duration': 300,
            'sampling_interval': 10
        },
        'climate_sensor': {
            'enable_gas': True,
            'temp_offset': 0
        },
        'ndvi_camera': {
            'resolution': (1920, 1080),
            'save_path': './data/images',
            'blue_filter': True
        },
        'data_logger': {
            'data_dir': './data',
            'experiment_name': 'drought_stress_experiment',
            'log_format': 'csv'
        },
        'collection_interval': 900,  # 15 minutes
        'respiration_interval': 14400,  # 4 hours
        'photo_interval': 3600  # 1 hour
    }
    
    # Initialize data collector
    collector = DataCollector(config)
    
    # Calibrate sensors
    logger.info("\nCalibrating sensors...")
    if not collector.calibrate_all_sensors():
        logger.warning("Some sensors failed calibration. Continuing anyway...")
    
    # Run experiment
    logger.info("\nStarting data collection...")
    logger.info("Press Ctrl+C to stop\n")
    
    try:
        # Run drought experiment
        collector.run_drought_experiment(
            baseline_days=5,
            drought_days=9,
            recovery_days=3
        )
    except KeyboardInterrupt:
        logger.info("\nStopping data collection...")
    finally:
        collector.stop_collection()
        
        # Get summary
        summary = collector.get_summary_statistics()
        logger.info("\nCollection Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Crop Stress Early Warning System'
    )
    parser.add_argument(
        '--mode',
        choices=['simulation', 'hardware'],
        default='simulation',
        help='Run mode: simulation (no hardware) or hardware (with sensors)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'simulation':
        run_simulation_mode()
    else:
        run_hardware_mode()


if __name__ == "__main__":
    main()
