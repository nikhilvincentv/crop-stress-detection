import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.simulation.drought_simulator import DroughtSimulator
from src.analysis.feature_engineering import FeatureEngineer
from src.models.lstm_predictor import LSTMPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Generate Synthetic Data
    logger.info("Generating data with DroughtSimulator...")
    simulator = DroughtSimulator({'initial_ndvi': 0.75, 'time_step_hours': 1})
    df = simulator.run_drought_experiment(baseline_days=5, drought_days=9, recovery_days=3)
    
    # 2. Feature Engineering
    logger.info("Engineering features...")
    engineer = FeatureEngineer()
    features_df = engineer.create_features(df)
    
    # Select features and create prediction target (NDVI 24h ahead)
    X_raw, y_raw = engineer.create_prediction_dataset(features_df, prediction_horizon=24, target_col='ndvi')
    
    # 3. Data Normalization (VITAL for LSTMs)
    logger.info("Normalizing data...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 4. Create Sequences [Batch, TimeSteps, Features]
    # We'll use 24 hours of data to predict the next state
    window_size = 24
    X_seq, y_seq = engineer.create_sequences(pd.DataFrame(X_scaled), y_raw, window_size=window_size)
    
    logger.info(f"Sequence data shape: {X_seq.shape}") # (Samples, 24, Features)
    
    # 5. Initialize and Train LSTM
    input_dim = X_seq.shape[2]
    predictor = LSTMPredictor(input_dim=input_dim, window_size=window_size)
    
    losses = predictor.train_model(X_seq, y_seq, epochs=30, batch_size=16)
    
    # 6. Evaluation
    logger.info("Making trial predictions...")
    preds = predictor.predict(X_seq[:5])
    logger.info(f"First 5 Predictions: {preds.flatten()}")
    logger.info(f"First 5 Actuals:     {y_seq[:5]}")
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    predictor.save("models/lstm_stress_predictor.pth")
    
    logger.info("LSTM Training Flow Complete!")

if __name__ == "__main__":
    main()
