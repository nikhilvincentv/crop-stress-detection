"""Feature engineering for crop stress prediction."""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for multi-modal crop stress prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.feature_names = []
        self.scaler = None
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw sensor data.
        
        Args:
            df: DataFrame with raw sensor readings
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating features from sensor data...")
        
        features_df = df.copy()
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        # Climate features
        features_df = self._add_climate_features(features_df)
        
        # Soil respiration features
        features_df = self._add_respiration_features(features_df)
        
        # VOC features
        features_df = self._add_voc_features(features_df)
        
        # Rolling statistics
        features_df = self._add_rolling_features(features_df)
        
        # Lag features (for time series prediction)
        features_df = self._add_lag_features(features_df)
        
        # Rate of change features
        features_df = self._add_rate_features(features_df)
        
        # Diurnal pattern features (Bond-Lamberty feedback)
        features_df = self._add_diurnal_features(features_df)
        
        # Quality control features (Bond-Lamberty feedback)
        features_df = self._add_quality_control_features(features_df)
        
        logger.info(f"Created {len(features_df.columns)} features")
        
        return features_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features.
        
        Args:
            df: Input DataFrame with timestamp column
        
        Returns:
            DataFrame with time features added
        """
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_year'] = df['timestamp'].dt.dayofyear
            df['time_since_start'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
            
            # Diurnal pattern features (Bond-Lamberty feedback)
            # Time of day categories for diurnal patterns
            df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
            df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
            df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
            
            # Sine/cosine encoding for cyclical hour (captures diurnal patterns)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def _add_climate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add climate-derived features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with climate features
        """
        # VPD is already calculated, but add stress indicators
        if 'climate_vpd_kpa' in df.columns:
            # VPD stress levels
            # Low: < 0.4 kPa, Optimal: 0.4-1.2 kPa, High: > 1.2 kPa
            df['vpd_stress'] = df['climate_vpd_kpa'].apply(
                lambda x: 0 if pd.isna(x) else (1 if x > 1.2 else (0.5 if x < 0.4 else 0))
            )
        
        # Temperature stress
        if 'climate_temperature_c' in df.columns:
            # Optimal range: 20-30°C for most crops
            df['temp_stress'] = df['climate_temperature_c'].apply(
                lambda x: 0 if pd.isna(x) else (abs(x - 25) / 25 if x < 15 or x > 35 else 0)
            )
        
        # Humidity extremes
        if 'climate_humidity_percent' in df.columns:
            df['humidity_stress'] = df['climate_humidity_percent'].apply(
                lambda x: 0 if pd.isna(x) else (1 if x < 30 or x > 80 else 0)
            )
        
        # Combined climate stress index
        stress_cols = ['vpd_stress', 'temp_stress', 'humidity_stress']
        available_stress = [col for col in stress_cols if col in df.columns]
        if available_stress:
            df['climate_stress_index'] = df[available_stress].mean(axis=1)
        
        return df
    
    def _add_respiration_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add soil respiration features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with respiration features
        """
        # CO2 flux features
        if 'respiration_flux_umol_m2_s' in df.columns:
            # Normalize flux (typical range: 0-10 μmol m⁻² s⁻¹)
            df['flux_normalized'] = df['respiration_flux_umol_m2_s'] / 10.0
            
            # Flux quality indicator (based on R²)
            if 'respiration_r_squared' in df.columns:
                df['flux_quality'] = df['respiration_r_squared'].apply(
                    lambda x: 1 if pd.notna(x) and x > 0.8 else 0
                )
        
        # Instant CO2 readings
        if 'co2_co2_ppm' in df.columns:
            # Deviation from baseline (400 ppm)
            df['co2_deviation'] = df['co2_co2_ppm'] - 400
            df['co2_deviation_pct'] = (df['co2_co2_ppm'] - 400) / 400 * 100
            
            # Respiration efficiency ratios (Bond-Lamberty, Kemanian feedback)
            # Use RATE OF CHANGE rather than absolute values
            if 'climate_humidity_percent' in df.columns:
                # CO2/moisture efficiency ratio
                moisture_safe = df['climate_humidity_percent'].replace(0, np.nan)
                df['co2_moisture_ratio'] = df['co2_co2_ppm'] / moisture_safe
                
            if 'climate_temperature_c' in df.columns:
                # CO2/temperature efficiency ratio
                temp_safe = (df['climate_temperature_c'] + 273.15)  # Convert to Kelvin to avoid division issues
                df['co2_temp_ratio'] = df['co2_co2_ppm'] / temp_safe
                
                # Temperature-moisture interaction (Winterfeldt feedback - microbial activity)
                if 'climate_humidity_percent' in df.columns:
                    df['temp_moisture_interaction'] = df['climate_temperature_c'] * df['climate_humidity_percent']
        
        return df
    
    def _add_voc_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VOC (gas resistance) features.
        Liu emphasized that VOCs can indicate root exudates and ABA signaling.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with VOC features
        """
        if 'climate_gas_resistance_ohms' in df.columns:
            # Log transform (gas resistance varies exponentially)
            df['gas_resistance_log'] = np.log10(df['climate_gas_resistance_ohms'].replace(0, np.nan))
            
            # Normalize to 0-1 range (typical: 10k-200k Ohms)
            df['gas_resistance_normalized'] = (
                (df['climate_gas_resistance_ohms'] - 10000) / 190000
            ).clip(0, 1)
            
            # VOC stress indicator (lower resistance = more VOCs = more stress)
            df['voc_stress'] = 1 - df['gas_resistance_normalized']
            
            # Root exudate proxy (Liu feedback)
            # Sudden drops in gas resistance may indicate root exudate release
            df['gas_resistance_rate'] = df['climate_gas_resistance_ohms'].diff()
            df['potential_root_exudate_event'] = (df['gas_resistance_rate'] < -5000).astype(int)
            
            # ABA signaling proxy (Liu feedback)
            # Sustained low gas resistance with high temperature = potential stress hormone signaling
            if 'climate_temperature_c' in df.columns:
                df['voc_temp_stress_indicator'] = (
                    (df['voc_stress'] > 0.6) & (df['climate_temperature_c'] > 30)
                ).astype(int)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        Args:
            df: Input DataFrame
            windows: List of window sizes (in number of measurements)
        
        Returns:
            DataFrame with rolling features
        """
        # Key columns to create rolling features for
        key_columns = [
            'climate_temperature_c',
            'climate_humidity_percent',
            'climate_vpd_kpa',
            'co2_co2_ppm',
            'climate_gas_resistance_ohms'
        ]
        
        for col in key_columns:
            if col in df.columns:
                for window in windows:
                    # Rolling mean
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    
                    # Rolling std (volatility)
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 3, 6, 12, 24]) -> pd.DataFrame:
        """
        Add lagged features for time series prediction.
        
        Args:
            df: Input DataFrame
            lags: List of lag periods
        
        Returns:
            DataFrame with lag features
        """
        # Key columns to create lag features for
        key_columns = [
            'climate_temperature_c',
            'climate_vpd_kpa',
            'co2_co2_ppm',
            'climate_gas_resistance_ohms',
            'ndvi_ndvi_mean'
        ]
        
        for col in key_columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def _add_rate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rate of change features.
        IMPORTANT: Bond-Lamberty and Kemanian emphasized using RATE OF CHANGE
        rather than absolute values for early warning detection.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with rate features
        """
        # Key columns to calculate rates for
        key_columns = [
            'climate_temperature_c',
            'climate_humidity_percent',
            'climate_vpd_kpa',
            'co2_co2_ppm',
            'climate_gas_resistance_ohms',
            'ndvi_ndvi_mean'
        ]
        
        for col in key_columns:
            if col in df.columns:
                # First derivative (rate of change)
                df[f'{col}_rate'] = df[col].diff()
                
                # Second derivative (acceleration)
                df[f'{col}_acceleration'] = df[f'{col}_rate'].diff()
                
                # Normalized rate (rate relative to current value)
                # This is the "efficiency ratio" concept from Bond-Lamberty
                col_safe = df[col].replace(0, np.nan)
                df[f'{col}_rate_normalized'] = df[f'{col}_rate'] / col_safe
        
        return df
    
    def _add_diurnal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add diurnal pattern features for CO2 and other variables.
        Bond-Lamberty emphasized that diurnal patterns contain more information
        than daily averages.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with diurnal pattern features
        """
        if 'co2_co2_ppm' in df.columns and 'hour' in df.columns:
            # Calculate diurnal amplitude (daily max - min)
            df['co2_diurnal_amplitude'] = df.groupby(df['timestamp'].dt.date)['co2_co2_ppm'].transform(
                lambda x: x.max() - x.min() if len(x) > 1 else 0
            )
            
            # Deviation from daily mean (captures diurnal position)
            df['co2_deviation_from_daily_mean'] = df['co2_co2_ppm'] - df.groupby(
                df['timestamp'].dt.date
            )['co2_co2_ppm'].transform('mean')
            
            # Morning vs afternoon CO2 difference (proxy for photosynthesis/respiration balance)
            morning_co2 = df[df['is_morning'] == 1].groupby(df['timestamp'].dt.date)['co2_co2_ppm'].mean()
            afternoon_co2 = df[df['is_afternoon'] == 1].groupby(df['timestamp'].dt.date)['co2_co2_ppm'].mean()
            
        # Temperature diurnal patterns
        if 'climate_temperature_c' in df.columns and 'hour' in df.columns:
            df['temp_diurnal_amplitude'] = df.groupby(df['timestamp'].dt.date)['climate_temperature_c'].transform(
                lambda x: x.max() - x.min() if len(x) > 1 else 0
            )
        
        return df
    
    def _add_quality_control_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add quality control features for sensor data.
        Bond-Lamberty emphasized checking sensor linearity and residual homoscedasticity.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with QC features
        """
        # Check for sensor drift (large changes that might indicate malfunction)
        if 'co2_co2_ppm' in df.columns:
            # Flag suspicious jumps (>100 ppm change in one reading)
            df['co2_suspicious_jump'] = (df['co2_co2_ppm'].diff().abs() > 100).astype(int)
            
            # Rolling coefficient of variation (high CV might indicate sensor issues)
            rolling_mean = df['co2_co2_ppm'].rolling(window=10, min_periods=1).mean()
            rolling_std = df['co2_co2_ppm'].rolling(window=10, min_periods=1).std()
            df['co2_rolling_cv'] = rolling_std / rolling_mean.replace(0, np.nan)
        
        # Temperature sensor QC
        if 'climate_temperature_c' in df.columns:
            # Flag unrealistic temperatures
            df['temp_out_of_range'] = ((df['climate_temperature_c'] < -10) | 
                                       (df['climate_temperature_c'] > 50)).astype(int)
        
        # Humidity sensor QC
        if 'climate_humidity_percent' in df.columns:
            # Flag out of range humidity
            df['humidity_out_of_range'] = ((df['climate_humidity_percent'] < 0) | 
                                           (df['climate_humidity_percent'] > 100)).astype(int)
        
        # Overall data quality flag
        qc_cols = [col for col in df.columns if 'suspicious' in col or 'out_of_range' in col]
        if qc_cols:
            df['data_quality_flag'] = df[qc_cols].sum(axis=1)
        
        return df
    
    def select_features_for_prediction(self, df: pd.DataFrame, target_col: str = 'ndvi_ndvi_mean') -> Tuple[pd.DataFrame, List[str]]:
        """
        Select relevant features for prediction.
        
        Args:
            df: DataFrame with all features
            target_col: Target variable column name
        
        Returns:
            Tuple of (features DataFrame, list of feature names)
        """
        # Exclude non-feature columns
        exclude_cols = [
            'timestamp', 'session_id', 'experiment_name', 'measurement_id',
            'image_path', 'ndvi_image_path', 'measurement_type'
        ]
        
        # Also exclude target and future NDVI values
        exclude_cols.extend([col for col in df.columns if 'ndvi' in col.lower() and 'lag' not in col])
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values (>50%)
        valid_cols = []
        for col in feature_cols:
            if df[col].notna().sum() / len(df) > 0.5:
                valid_cols.append(col)
        
        logger.info(f"Selected {len(valid_cols)} features for prediction")
        
        return df[valid_cols], valid_cols
    
    def create_prediction_dataset(self, df: pd.DataFrame, 
                                 prediction_horizon: int = 24,
                                 target_col: str = 'ndvi_ndvi_mean') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create dataset for prediction with future target values.
        
        Args:
            df: DataFrame with features
            prediction_horizon: How many time steps ahead to predict
            target_col: Target variable column name
        
        Returns:
            Tuple of (X features, y target)
        """
        # Create future target
        df['target'] = df[target_col].shift(-prediction_horizon)
        
        # Remove rows without target
        df_clean = df.dropna(subset=['target'])
        
        # Select features
        X, feature_names = self.select_features_for_prediction(df_clean, target_col)
        y = df_clean['target']
        
        # Remove any remaining NaN values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"Created prediction dataset: {len(X)} samples, {len(feature_names)} features")
        logger.info(f"Prediction horizon: {prediction_horizon} time steps")
        
        return X, y
