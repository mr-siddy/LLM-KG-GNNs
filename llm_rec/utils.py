"""
utils.py

Utility functions for the recommendation system.
"""

import os
import json
import pandas as pd
import numpy as np
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    os.makedirs(directory_path, exist_ok=True)


def format_timestamp() -> str:
    """
    Get a formatted timestamp for file naming.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file with proper serialization.
    
    Args:
        data: Data to save
        file_path: Path to the output file
        indent: Number of spaces for indentation
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    ensure_directory(log_dir)
    log_file = os.path.join(log_dir, f"recommendation_system_{format_timestamp()}.log")
    
    # Configure logger
    logger = logging.getLogger("recommendation_system")
    logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatters and add to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def calculate_time_decay_weights(dates: pd.Series, 
                               decay_rate: float = 0.005, 
                               reference_date: Optional[datetime.date] = None) -> np.ndarray:
    """
    Calculate time decay weights for timestamps.
    
    Args:
        dates: Series of date values
        decay_rate: Rate of exponential decay
        reference_date: Reference date for calculating days difference
        
    Returns:
        Array of decay weights
    """
    if reference_date is None:
        reference_date = datetime.date.today()
        
    # Convert Series to datetime if not already
    if not pd.api.types.is_datetime64_dtype(dates):
        dates = pd.to_datetime(dates)
    
    # Extract date component
    date_vals = dates.dt.date
    
    # Calculate days difference
    days_diff = np.array([(reference_date - d).days for d in date_vals])
    
    # Apply exponential decay
    weights = np.exp(-decay_rate * days_diff)
    
    return weights


def split_train_test(df: pd.DataFrame, 
                    time_col: str = 't_dat', 
                    test_days: int = 7,
                    min_train_size: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a dataframe into train and test sets based on time.
    
    Args:
        df: DataFrame to split
        time_col: Column name containing datetime values
        test_days: Number of days to include in test set
        min_train_size: Minimum proportion for training set
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if not pd.api.types.is_datetime64_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
    
    # Sort by time
    df = df.sort_values(by=time_col)
    
    # If enough data, use last N days as test
    if len(df) >= 10:  # Arbitrary threshold
        last_date = df[time_col].max()
        cutoff_date = last_date - pd.Timedelta(days=test_days)
        train_df = df[df[time_col] <= cutoff_date]
        test_df = df[df[time_col] > cutoff_date]
        
        # If test set is too small, fall back to percentage split
        if len(test_df) < 3 or len(train_df) < len(df) * min_train_size:
            split_idx = int(len(df) * min_train_size)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
    else:
        # For small datasets, use simple percentage split
        split_idx = int(len(df) * min_train_size)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
    
    return train_df, test_df


def evaluate_recommendations(recommendations: List[Dict], 
                           ground_truth: List[Dict],
                           item_id_key: str = 'article_id') -> Dict[str, float]:
    """
    Evaluate recommendation quality against ground truth.
    
    Args:
        recommendations: List of recommended items
        ground_truth: List of actual items
        item_id_key: Key for the item ID in dictionaries
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Extract item IDs
    rec_ids = [item.get(item_id_key) for item in recommendations if item.get(item_id_key)]
    gt_ids = [item.get(item_id_key) for item in ground_truth if item.get(item_id_key)]
    
    # Calculate precision, recall, F1
    if not rec_ids or not gt_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "hit_rate": 0.0
        }
    
    # Find common items (hits)
    hits = set(rec_ids).intersection(gt_ids)
    num_hits = len(hits)
    
    precision = num_hits / len(rec_ids) if rec_ids else 0
    recall = num_hits / len(gt_ids) if gt_ids else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    hit_rate = 1.0 if num_hits > 0 else 0.0  # Binary: were any recommendations correct?
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "hit_rate": hit_rate
    }


def clean_and_validate_data(df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          date_columns: List[str] = None) -> pd.DataFrame:
    """
    Clean and validate a DataFrame.
    
    Args:
        df: DataFrame to clean
        required_columns: List of columns that must be present
        date_columns: List of columns to convert to datetime
        
    Returns:
        Cleaned DataFrame
    """
    # Check required columns
    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert date columns
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values (for demonstration, just drop rows with NA in any column)
    df = df.dropna()
    
    return df


if __name__ == "__main__":
    # Example usage
    logger = setup_logging()
    logger.info("Utility module initialized")
    
    # Example of time decay calculation
    dates = pd.Series([
        '2023-01-01', 
        '2023-02-15', 
        '2023-03-30'
    ])
    weights = calculate_time_decay_weights(dates, reference_date=datetime.date(2023, 4, 1))
    print("Time decay weights:", weights)
