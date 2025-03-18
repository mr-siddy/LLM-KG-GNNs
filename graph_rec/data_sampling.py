"""
dataset_sampling.py

This script samples 1000 random customers from the H&M dataset and extracts 
their associated information from the customers and transactions files.
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
random.seed(42)  # For reproducibility

def sample_dataset(
    customers_path="../input/h-and-m-personalized-fashion-recommendations/customers.csv",
    articles_path="../input/h-and-m-personalized-fashion-recommendations/articles.csv", 
    transactions_path="../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv",
    output_dir="./sampled_data",
    sample_size=1000
):
    """
    Sample a subset of customers and their associated data.
    
    Args:
        customers_path: Path to the customers CSV file
        articles_path: Path to the articles CSV file
        transactions_path: Path to the transactions CSV file
        output_dir: Directory to save the sampled data
        sample_size: Number of customers to sample
    """
    print(f"Loading original datasets...")
    
    # Load customers dataset
    customers_df = pd.read_csv(customers_path)
    print(f"Loaded {len(customers_df)} customers")
    
    # Sample random customers
    all_customer_ids = customers_df['customer_id'].unique()
    print(f"Total unique customers: {len(all_customer_ids)}")
    
    if sample_size > len(all_customer_ids):
        sample_size = len(all_customer_ids)
        print(f"Sample size reduced to {sample_size} (total available customers)")
    
    sampled_customer_ids = random.sample(list(all_customer_ids), sample_size)
    print(f"Sampled {len(sampled_customer_ids)} unique customers")
    
    # Filter customers dataframe
    sampled_customers_df = customers_df[customers_df['customer_id'].isin(sampled_customer_ids)]
    print(f"Sampled customers dataframe shape: {sampled_customers_df.shape}")
    
    # Load and filter transactions
    print(f"Loading transactions data...")
    transactions_df = pd.read_csv(transactions_path)
    print(f"Loaded {len(transactions_df)} transactions")
    
    sampled_transactions_df = transactions_df[transactions_df['customer_id'].isin(sampled_customer_ids)]
    print(f"Sampled transactions dataframe shape: {sampled_transactions_df.shape}")
    
    # Get all unique article IDs from the sampled transactions
    sampled_article_ids = sampled_transactions_df['article_id'].unique()
    print(f"Sampled transactions contain {len(sampled_article_ids)} unique articles")
    
    # Load and filter articles
    print(f"Loading articles data...")
    articles_df = pd.read_csv(articles_path)
    print(f"Loaded {len(articles_df)} articles")
    
    sampled_articles_df = articles_df[articles_df['article_id'].isin(sampled_article_ids)]
    print(f"Sampled articles dataframe shape: {sampled_articles_df.shape}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save sampled datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sampled_customers_df.to_csv(f"{output_dir}/sampled_customers_{timestamp}.csv", index=False)
    sampled_transactions_df.to_csv(f"{output_dir}/sampled_transactions_{timestamp}.csv", index=False)
    sampled_articles_df.to_csv(f"{output_dir}/sampled_articles_{timestamp}.csv", index=False)
    
    # Save customer IDs for reference
    with open(f"{output_dir}/sampled_customer_ids_{timestamp}.txt", 'w') as f:
        for customer_id in sampled_customer_ids:
            f.write(f"{customer_id}\n")
    
    print(f"Sampled datasets saved to {output_dir}/")
    print(f"Summary:")
    print(f"- {len(sampled_customers_df)} customers")
    print(f"- {len(sampled_transactions_df)} transactions")
    print(f"- {len(sampled_articles_df)} articles")
    
    return {
        'customers': sampled_customers_df,
        'transactions': sampled_transactions_df,
        'articles': sampled_articles_df,
        'customer_ids': sampled_customer_ids
    }


if __name__ == "__main__":
    sampled_data = sample_dataset(sample_size=1000)
    
    # Basic analysis of sampled data
    customers = sampled_data['customers']
    transactions = sampled_data['transactions']
    
    # Transactions per customer
    tx_per_customer = transactions.groupby('customer_id').size()
    print(f"Transactions per customer:")
    print(f"- Min: {tx_per_customer.min()}")
    print(f"- Max: {tx_per_customer.max()}")
    print(f"- Mean: {tx_per_customer.mean():.2f}")
    print(f"- Median: {tx_per_customer.median()}")