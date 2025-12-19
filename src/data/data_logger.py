"""Data logging system for sensor measurements."""

import json
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class DataLogger:
    """Centralized data logging system for all sensors."""
    
    def __init__(self, config: dict = None):
        """
        Initialize data logger.
        
        Args:
            config: Configuration dictionary with:
                - data_dir: Directory for data storage (default: './data')
                - experiment_name: Name of current experiment
                - log_format: 'csv' or 'json' (default: 'csv')
        """
        self.config = config or {}
        self.data_dir = Path(config.get('data_dir', './data'))
        self.experiment_name = config.get('experiment_name', 'experiment')
        self.log_format = config.get('log_format', 'csv')
        
        # Create directory structure
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)
        
        # Initialize log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{self.experiment_name}_{timestamp}"
        
        if self.log_format == 'csv':
            self.log_file = self.raw_data_dir / f"{self.session_id}.csv"
            self.csv_writer = None
            self.csv_file = None
        else:
            self.log_file = self.raw_data_dir / f"{self.session_id}.json"
            self.json_data = []
        
        logger.info(f"Data logger initialized: {self.log_file}")
    
    def log_measurement(self, measurement: Dict) -> bool:
        """
        Log a single measurement.
        
        Args:
            measurement: Dictionary containing sensor data with timestamp
        
        Returns:
            True if logging successful
        """
        try:
            # Add session metadata
            measurement['session_id'] = self.session_id
            measurement['experiment_name'] = self.experiment_name
            
            if self.log_format == 'csv':
                self._log_to_csv(measurement)
            else:
                self._log_to_json(measurement)
            
            return True
        except Exception as e:
            logger.error(f"Failed to log measurement: {e}")
            return False
    
    def _log_to_csv(self, measurement: Dict):
        """Log measurement to CSV file."""
        # Flatten nested dictionaries
        flat_data = self._flatten_dict(measurement)
        
        # Initialize CSV writer if needed
        if self.csv_writer is None:
            self.csv_file = open(self.log_file, 'w', newline='')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=flat_data.keys())
            self.csv_writer.writeheader()
        
        # Write row
        self.csv_writer.writerow(flat_data)
        self.csv_file.flush()
    
    def _log_to_json(self, measurement: Dict):
        """Log measurement to JSON file."""
        self.json_data.append(measurement)
        
        # Write to file
        with open(self.log_file, 'w') as f:
            json.dump(self.json_data, f, indent=2, default=str)
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_session_data(self, session_id: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from a session.
        
        Args:
            session_id: Session ID to load (default: current session)
        
        Returns:
            DataFrame with session data
        """
        if session_id is None:
            session_id = self.session_id
        
        # Try CSV first
        csv_path = self.raw_data_dir / f"{session_id}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        
        # Try JSON
        json_path = self.raw_data_dir / f"{session_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        
        logger.warning(f"No data found for session: {session_id}")
        return pd.DataFrame()
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all session IDs."""
        sessions = []
        for file in self.raw_data_dir.glob('*.csv'):
            sessions.append(file.stem)
        for file in self.raw_data_dir.glob('*.json'):
            if file.stem not in sessions:
                sessions.append(file.stem)
        return sorted(sessions)
    
    def export_to_format(self, output_path: str, format: str = 'csv'):
        """
        Export current session data to specified format.
        
        Args:
            output_path: Path for output file
            format: Output format ('csv', 'json', 'excel')
        """
        df = self.load_session_data()
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data exported to {output_path}")
    
    def close(self):
        """Close log files and cleanup."""
        if self.csv_file:
            self.csv_file.close()
        logger.info("Data logger closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
