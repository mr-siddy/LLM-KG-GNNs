"""
graph_utils.py

This module contains helper functions for advanced graph construction:
- Compute co-occurrence statistics for items
- Compute time decay weights for customer-product interactions
- Compute similarity between products using available features
- Create country-based similarities between customers
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import datetime

def compute_cooccurrence(transactions_df, product_id_map, min_cooccur=5):
    """
    Computes item co-occurrence counts from transactions.
    
    Args:
        transactions_df (pd.DataFrame): Transactions data with 'InvoiceNo' and 'StockCode' columns
        product_id_map (dict): Mapping from product ID to integer index
        min_cooccur (int): Minimum number of co-occurrences to consider an edge
    
    Returns:
        List[int], List[int], List[float]: Lists of source indices, destination indices, and corresponding weights
    """
    cooccur = defaultdict(lambda: defaultdict(int))
    
    # Group by invoice to get products purchased together
    invoice_groups = transactions_df.groupby("InvoiceNo")
    
    for _, group in invoice_groups:
        items = [product_id_map[pid] for pid in group["StockCode"] if pid in product_id_map]
        
        # Count pairwise co-occurrences
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    src, dst, weights = [], [], []
    for a, nbrs in cooccur.items():
        for b, count in nbrs.items():
            if count >= min_cooccur:
                src.append(a)
                dst.append(b)
                weights.append(count)
                
    return src, dst, weights

def compute_retail_cooccurrence(transactions_df, product_id_map, min_cooccur=2):
    """
    Enhanced co-occurrence for retail data with invoice-based grouping and quantity weighting.
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions with 'InvoiceNo', 'StockCode' and 'Quantity' columns
        product_id_map (dict): Mapping from product ID to integer index
        min_cooccur (int): Minimum co-occurrence weight to consider an edge
    
    Returns:
        List[int], List[int], List[float]: Lists of source indices, destination indices, and corresponding weights
    """
    cooccur = defaultdict(lambda: defaultdict(float))
    
    # Group by invoice number to get items purchased together
    invoice_groups = transactions_df.groupby('InvoiceNo')
    
    for _, group in invoice_groups:
        # Get valid product indices in this invoice
        items = [product_id_map[pid] for pid in group["StockCode"] if pid in product_id_map]
        
        # Consider quantity for weighted co-occurrence
        item_qty = {product_id_map[row['StockCode']]: row['Quantity'] 
                   for _, row in group.iterrows() 
                   if row['StockCode'] in product_id_map}
        
        # Calculate weighted co-occurrences
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                # Weight by quantity of both items
                weight = min(item_qty[a], item_qty[b])
                cooccur[a][b] += weight
                cooccur[b][a] += weight

    # Create edges for pairs above minimum co-occurrence
    src, dst, weights = [], [], []
    for a, nbrs in cooccur.items():
        for b, weight in nbrs.items():
            if weight >= min_cooccur:
                src.append(a)
                dst.append(b)
                weights.append(float(weight))
                
    return src, dst, weights

def compute_time_decay_weights(transactions_df, current_date=None, decay_rate=0.001):
    """
    Computes a time decay weight for each transaction based on its recency.
    
    Args:
        transactions_df (pd.DataFrame): Transactions with 'InvoiceDate' column
        current_date (datetime.date, optional): Reference date. If None, uses max date
        decay_rate (float): Rate for exponential decay
    
    Returns:
        np.array: Array of weights for each transaction
    """
    if current_date is None:
        current_date = transactions_df['InvoiceDate'].max()
    
    # Calculate days difference
    days_diff = (current_date - transactions_df['InvoiceDate']).dt.days
    
    # Apply exponential decay
    weights = np.exp(-decay_rate * days_diff)
    
    return weights

def compute_retail_temporal_weights(transactions_df, time_window_days=30):
    """
    Creates time-aware weights for transactions based on recency and frequency.
    
    Args:
        transactions_df (pd.DataFrame): DataFrame with retail transactions
        time_window_days (int): Window for considering recency
        
    Returns:
        DataFrame with time-weighted transaction scores
    """
    # Group by customer and product
    grouped = transactions_df.groupby(['CustomerID', 'StockCode'])
    
    # Calculate recency (days since last purchase) and frequency (count of purchases)
    recency = grouped['InvoiceDate'].max().reset_index()
    frequency = grouped.size().reset_index(name='frequency')
    
    # Get latest date in dataset
    latest_date = transactions_df['InvoiceDate'].max()
    
    # Calculate recency in days
    recency['days_since'] = (latest_date - recency['InvoiceDate']).dt.days
    
    # Merge recency and frequency
    temporal_scores = pd.merge(recency, frequency, on=['CustomerID', 'StockCode'])
    
    # Calculate time decay weight
    temporal_scores['weight'] = temporal_scores['frequency'] * np.exp(-0.01 * temporal_scores['days_since'])
    
    return temporal_scores[['CustomerID', 'StockCode', 'weight']]

def compute_country_similarity(transactions_df, customer_id_map):
    """
    Create edges between customers from the same country.
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions with 'CustomerID' and 'Country' columns
        customer_id_map (dict): Mapping from customer ID to integer index
        
    Returns:
        List[int], List[int]: Lists of source and destination customer indices
    """
    # Get unique (customer, country) pairs
    customer_country = transactions_df[['CustomerID', 'Country']].drop_duplicates()
    
    # Group by country
    country_groups = customer_country.groupby('Country')
    
    src, dst = [], []
    for _, group in country_groups:
        customers = [customer_id_map[cid] for cid in group['CustomerID'] if cid in customer_id_map]
        
        # Limit connections to avoid too many edges
        for i in range(len(customers)):
            for j in range(i+1, min(i+10, len(customers))):
                src.append(customers[i])
                dst.append(customers[j])
                # Add reverse edge for undirected graph
                src.append(customers[j])
                dst.append(customers[i])
    
    return src, dst

def compute_price_similarity(transactions_df, product_id_map):
    """
    Create edges between products with similar price points.
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions with 'StockCode' and 'UnitPrice' columns
        product_id_map (dict): Mapping from product ID to integer index
        
    Returns:
        List[int], List[int], List[float]: Lists of source, destination, and similarity weights
    """
    # Get average price per product
    product_prices = transactions_df.groupby('StockCode')['UnitPrice'].mean()
    
    # Define price ranges
    price_min = product_prices.min()
    price_max = product_prices.max()
    
    # Create price bins
    num_bins = 10
    bins = np.linspace(price_min, price_max, num_bins + 1)
    
    # Assign products to price bins
    product_bins = {}
    for pid, price in product_prices.items():
        if pid in product_id_map:
            for i in range(len(bins) - 1):
                if bins[i] <= price <= bins[i + 1]:
                    if i not in product_bins:
                        product_bins[i] = []
                    product_bins[i].append(product_id_map[pid])
                    break
    
    # Create edges between products in the same price bin
    src, dst, weights = [], [], []
    for _, products in product_bins.items():
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                src.append(products[i])
                dst.append(products[j])
                # Add reverse edge
                src.append(products[j])
                dst.append(products[i])
                # Add weights (all 1.0 for price similarity)
                weights.append(1.0)
                weights.append(1.0)
    
    return src, dst, weights

def compute_description_similarity(transactions_df, product_id_map):
    """
    Create edges between products with similar descriptions (using first word as a simple category).
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions with 'StockCode' and 'Description' columns
        product_id_map (dict): Mapping from product ID to integer index
        
    Returns:
        List[int], List[int], List[float]: Lists of source, destination, and similarity weights
    """
    # Get unique products and descriptions
    products = transactions_df[['StockCode', 'Description']].drop_duplicates()
    
    # Group by first word of description (as a simple category)
    category_map = {}
    for _, row in products.iterrows():
        if pd.isna(row['Description']):
            continue
            
        pid = row['StockCode']
        if pid in product_id_map:
            # Use first word as category
            category = row['Description'].split()[0].lower()
            if category not in category_map:
                category_map[category] = []
            category_map[category].append(product_id_map[pid])
    
    # Create edges between products in the same category
    src, dst, weights = [], [], []
    for _, products in category_map.items():
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                src.append(products[i])
                dst.append(products[j])
                # Add reverse edge
                src.append(products[j])
                dst.append(products[i])
                # Add weights (all 1.0 for category similarity)
                weights.append(1.0)
                weights.append(1.0)
    
    return src, dst, weights

if __name__ == "__main__":
    # Test code
    print("This module provides utility functions for graph construction.")