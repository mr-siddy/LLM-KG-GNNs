"""
id_mapper.py

Utilities for creating and managing customer ID mappings.
"""

import pandas as pd
from typing import Dict


def create_customer_id_mapping(customers_df: pd.DataFrame) -> Dict[str, int]:
    """
    Create a mapping from original customer IDs to simple numeric IDs.
    
    Args:
        customers_df: DataFrame containing customer information with 'customer_id' column
        
    Returns:
        Dictionary mapping original customer_id to simple numeric ID
    """
    # Create a mapping: long customer_id -> simple numeric ID (starting from 1)
    unique_customer_ids = customers_df['customer_id'].unique()
    customer_id_mapping = {cid: idx for idx, cid in enumerate(unique_customer_ids, start=1)}
    return customer_id_mapping


def save_customer_id_mapping(customer_id_mapping: Dict[str, int], output_path: str = 'customer_id_mapping.csv') -> None:
    """
    Save customer ID mapping to CSV for future reference.
    
    Args:
        customer_id_mapping: Dictionary mapping original customer_id to simple numeric ID
        output_path: Path to save the mapping CSV
    """
    mapping_df = pd.DataFrame({
        'original_customer_id': list(customer_id_mapping.keys()),
        'simple_customer_id': list(customer_id_mapping.values())
    })
    mapping_df.to_csv(output_path, index=False)
    print(f"Customer ID mapping saved to {output_path}")


def load_customer_id_mapping(mapping_path: str = 'customer_id_mapping.csv') -> Dict[str, int]:
    """
    Load customer ID mapping from CSV.
    
    Args:
        mapping_path: Path to the mapping CSV
        
    Returns:
        Dictionary mapping original customer_id to simple numeric ID
    """
    mapping_df = pd.read_csv(mapping_path)
    customer_id_mapping = dict(zip(mapping_df['original_customer_id'], mapping_df['simple_customer_id']))
    return customer_id_mapping


if __name__ == "__main__":
    # Example usage
    filtered_customers = pd.read_csv('filtered_customers.csv')
    mapping = create_customer_id_mapping(filtered_customers)
    save_customer_id_mapping(mapping)
    
    # Verify by loading the mapping
    loaded_mapping = load_customer_id_mapping()
    print(f"Loaded mapping has {len(loaded_mapping)} entries")
