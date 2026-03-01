import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class CropStressLSTM(nn.Module):
    """Refined LSTM architecture for Crop Stress Prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, output_dim: int = 1):
        super(CropStressLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True expects [Batch, Sequence, Features]
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Fully connected output
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class LSTMPredictor:
    """Wrapper for training and inference of the PyTorch LSTM model."""
    def __init__(self, input_dim: int, window_size: int = 24):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CropStressLSTM(input_dim=input_dim).to(self.device)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.window_size = window_size
        self.input_dim = input_dim
        
    def train_model(self, X_sequences: np.ndarray, y_targets: np.ndarray, 
                    epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> List[float]:
        """Train the LSTM model."""
        self.model.train()
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_sequences).to(self.device)
        y_train = torch.FloatTensor(y_targets).view(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        losses = []
        logger.info(f"Starting LSTM training on {self.device}...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
                
        return losses

    def predict(self, X_sequence: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequence).to(self.device)
            if len(X_tensor.shape) == 2: # Single sequence
                X_tensor = X_tensor.unsqueeze(0)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

    def save(self, path: str):
        """Save model and configuration."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"LSTM model loaded from {path}")
