"""
data_loader.py

Utilities for loading and filtering data for the product recommendation system.
"""

import pandas as pd
import json
from typing import Set, Dict, Tuple


def load_customer_ids_from_json(json_file: str) -> Set[str]:
    """
    Load customer IDs from a JSON file that contains recommendation examples.
    
    Args:
        json_file: Path to the JSON file containing recommendation examples
        
    Returns:
        Set of unique customer IDs
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Extract the customer ID from the 'input' field, assuming format "Customer ID: <id>"
    customer_ids = [entry['input'].replace('Customer ID: ', '').strip() for entry in data]
    return set(customer_ids)


def load_and_filter_data(train_json: str, val_json: str, 
                         transactions_path: str, customers_path: str, 
                         articles_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and filter data based on customer IDs from JSON files.
    
    Args:
        train_json: Path to training JSON file
        val_json: Path to validation JSON file
        transactions_path: Path to transactions CSV
        customers_path: Path to customers CSV
        articles_path: Path to articles CSV
        
    Returns:
        Tuple of filtered transactions, customers, and articles DataFrames
    """
    # Load customer IDs from both train and validation JSON files
    train_customers = load_customer_ids_from_json(train_json)
    val_customers = load_customer_ids_from_json(val_json)

    # Combine unique customer IDs from both files
    all_customer_ids = train_customers.union(val_customers)
    print(f"Total unique customers: {len(all_customer_ids)}")

    # Filter transactions CSV
    transactions_df = pd.read_csv(transactions_path)
    filtered_transactions = transactions_df[transactions_df['customer_id'].isin(all_customer_ids)]
    
    # Filter customers CSV
    customers_df = pd.read_csv(customers_path)
    filtered_customers = customers_df[customers_df['customer_id'].isin(all_customer_ids)]
    
    # Filter articles CSV to only those referenced in the filtered transactions
    article_ids_in_transactions = filtered_transactions['article_id'].unique()
    articles_df = pd.read_csv(articles_path)
    filtered_articles = articles_df[articles_df['article_id'].isin(article_ids_in_transactions)]
    
    return filtered_transactions, filtered_customers, filtered_articles


def save_filtered_data(transactions: pd.DataFrame, customers: pd.DataFrame, 
                       articles: pd.DataFrame, output_dir: str = "") -> None:
    """
    Save filtered DataFrames to CSV files.
    
    Args:
        transactions: Filtered transactions DataFrame
        customers: Filtered customers DataFrame
        articles: Filtered articles DataFrame
        output_dir: Directory to save files (empty string for current directory)
    """
    prefix = "" if output_dir == "" else f"{output_dir}/"
    
    transactions.to_csv(f'{prefix}filtered_transactions_train.csv', index=False)
    print(f"Filtered transactions saved to {prefix}filtered_transactions_train.csv")
    
    customers.to_csv(f'{prefix}filtered_customers.csv', index=False)
    print(f"Filtered customers saved to {prefix}filtered_customers.csv")
    
    articles.to_csv(f'{prefix}filtered_articles.csv', index=False)
    print(f"Filtered articles saved to {prefix}filtered_articles.csv")


if __name__ == "__main__":
    # Example usage
    filtered_tx, filtered_cust, filtered_art = load_and_filter_data(
        'product_recommendation_train.json',
        'product_recommendation_val.json',
        'transactions_train.csv',
        'customers.csv',
        'articles.csv'
    )
    
    save_filtered_data(filtered_tx, filtered_cust, filtered_art)
