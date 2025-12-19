"""Machine learning models for crop stress prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class StressPredictor:
    """Multi-modal crop stress prediction model."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize stress predictor.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boosting', 'ridge', 'lasso')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        self.training_history = []
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"Initialized {model_type} stress predictor")
    
    def _initialize_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=0.1)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
             test_size: float = 0.2,
             validation: bool = True) -> Dict:
        """
        Train the stress prediction model.
        
        Args:
            X: Feature DataFrame
            y: Target variable (NDVI or stress indicator)
            test_size: Proportion of data for testing
            validation: Whether to perform cross-validation
        
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training {self.model_type} model...")
        logger.info(f"Training samples: {len(X)}, Features: {X.shape[1]}")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        
        # Cross-validation
        if validation:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train,
                cv=5, scoring='r2', n_jobs=-1
            )
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Store training history
        self.training_history.append(metrics)
        
        logger.info(f"Training complete - Test R²: {metrics['test_r2']:.3f}, RMSE: {metrics['test_rmse']:.3f}")
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure features match training
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for tree-based models).
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Tuple of (predictions, standard deviations)
        """
        if not hasattr(self.model, 'estimators_'):
            # For non-ensemble models, return predictions with None for std
            predictions = self.predict(X)
            return predictions, np.zeros_like(predictions)
        
        # Get predictions from all trees
        X_scaled = self.scaler.transform(X[self.feature_names])
        tree_predictions = np.array([tree.predict(X_scaled) for tree in self.model.estimators_])
        
        # Calculate mean and std
        predictions = tree_predictions.mean(axis=0)
        std = tree_predictions.std(axis=0)
        
        return predictions, std
    
    def evaluate_lag_time(self, X: pd.DataFrame, y: pd.Series, 
                         horizons: List[int] = [1, 6, 12, 24, 48]) -> pd.DataFrame:
        """
        Evaluate prediction performance at different time horizons.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            horizons: List of prediction horizons to test
        
        Returns:
            DataFrame with performance metrics for each horizon
        """
        logger.info("Evaluating prediction lag time...")
        
        results = []
        
        for horizon in horizons:
            # Create target with this horizon
            y_shifted = y.shift(-horizon)
            
            # Remove NaN values
            valid_idx = y_shifted.notna()
            X_valid = X[valid_idx]
            y_valid = y_shifted[valid_idx]
            
            if len(y_valid) < 50:
                logger.warning(f"Insufficient data for horizon {horizon}")
                continue
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42
            )
            
            # Scale and train
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            results.append({
                'horizon': horizon,
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'samples': len(y_test)
            })
        
        results_df = pd.DataFrame(results)
        logger.info(f"Lag time evaluation complete for {len(results)} horizons")
        
        return results_df
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare different model types.
        
        Args:
            X: Feature DataFrame
            y: Target variable
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing model types...")
        
        model_types = ['random_forest', 'gradient_boosting', 'ridge', 'lasso']
        results = []
        
        for model_type in model_types:
            # Initialize model
            predictor = StressPredictor(model_type=model_type)
            
            # Train
            metrics = predictor.train(X, y, validation=True)
            
            results.append({
                'model': model_type,
                'test_r2': metrics['test_r2'],
                'test_rmse': metrics['test_rmse'],
                'cv_r2_mean': metrics.get('cv_r2_mean', None),
                'cv_r2_std': metrics.get('cv_r2_std', None)
            })
        
        results_df = pd.DataFrame(results).sort_values('test_r2', ascending=False)
        logger.info("Model comparison complete")
        
        return results_df
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save plot (optional)
        """
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model type")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_predictions(self, X: pd.DataFrame, y_true: pd.Series, 
                        save_path: Optional[str] = None):
        """
        Plot predicted vs actual values.
        
        Args:
            X: Feature DataFrame
            y_true: True target values
            save_path: Path to save plot (optional)
        """
        y_pred = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual NDVI')
        plt.ylabel('Predicted NDVI')
        plt.title('Predicted vs Actual NDVI')
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Predictions plot saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_model(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved: {path}")
    
    def load_model(self, path: str):
        """
        Load trained model from disk.
        
        Args:
            path: Path to load model from
        """
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.feature_importance = model_data.get('feature_importance')
        self.training_history = model_data.get('training_history', [])
        
        logger.info(f"Model loaded: {path}")
