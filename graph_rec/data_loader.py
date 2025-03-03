"""
data_loader.py

Enhanced data loader for retail transaction data:
- Loads CSV files
- Applies advanced feature engineering (including time decay weights and market basket analysis)
- Builds a heterogeneous graph:
    - "buys" edges (customer → product) with time decay weights
    - "bought_together" edges (product ↔ product) using co-occurrence
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
from graph_utils import compute_cooccurrence, compute_time_decay_weights

def create_customer_features(transactions_df, customer_ids):
    """
    Create customer features based on transaction history.
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions data
        customer_ids (list): List of customer IDs
        
    Returns:
        torch.Tensor: Tensor of customer features
    """
    # Group by customer
    customer_stats = transactions_df.groupby('CustomerID').agg({
        'InvoiceNo': 'nunique',           # Purchase frequency
        'Quantity': 'sum',                # Total items purchased
        'UnitPrice': 'mean',              # Average price point
        'TotalValue': 'sum'        # Total spend
    }).reset_index()
    
    # Normalize features
    for col in ['InvoiceNo', 'Quantity', 'UnitPrice', 'TotalValue']:
        max_val = customer_stats[col].max()
        if max_val > 0:
            customer_stats[col] = customer_stats[col] / max_val
    
    # Create feature matrix
    cust_features = []
    for cid in customer_ids:
        if cid in customer_stats['CustomerID'].values:
            cust_info = customer_stats[customer_stats['CustomerID'] == cid].iloc[0]
            features = [
                cust_info['InvoiceNo'],
                cust_info['Quantity'],
                cust_info['UnitPrice'],
                cust_info['TotalValue']
            ]
        else:
            features = [0, 0, 0, 0]  # Default for customers with no stats
        cust_features.append(features)
    
    return torch.tensor(cust_features, dtype=torch.float)

def create_product_features(transactions_df, product_ids):
    """
    Create product features based on transaction history.
    
    Args:
        transactions_df (pd.DataFrame): Retail transactions data
        product_ids (list): List of product IDs
        
    Returns:
        torch.Tensor: Tensor of product features
    """
    # Group by product
    product_stats = transactions_df.groupby('StockCode').agg({
        'Quantity': 'sum',                # Popularity (total units sold)
        'UnitPrice': 'mean',              # Price point
        'CustomerID': 'nunique',          # Customer diversity
        'Country': lambda x: x.nunique()  # Geographic diversity
    }).reset_index()
    
    # Normalize features
    for col in ['Quantity', 'UnitPrice', 'CustomerID', 'Country']:
        max_val = product_stats[col].max()
        if max_val > 0:
            product_stats[col] = product_stats[col] / max_val
    
    # Create feature matrix
    prod_features = []
    for pid in product_ids:
        if pid in product_stats['StockCode'].values:
            prod_info = product_stats[product_stats['StockCode'] == pid].iloc[0]
            features = [
                prod_info['Quantity'],
                prod_info['UnitPrice'],
                prod_info['CustomerID'],
                prod_info['Country']
            ]
        else:
            features = [0, 0, 0, 0]  # Default for products with no stats
        prod_features.append(features)
    
    return torch.tensor(prod_features, dtype=torch.float)

def load_retail_data(data_dir="data"):
    """
    Load retail transaction data and build graph.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        Data: PyTorch Geometric Data object
        dict: Metadata dictionary
    """
    # Load retail transactions CSV
    transactions_path = os.path.join(data_dir, "retail_transactions.csv")
    transactions_df = pd.read_csv(transactions_path)
    
    # Convert data types and clean
    transactions_df['InvoiceDate'] = pd.to_datetime(transactions_df['InvoiceDate'])
    transactions_df = transactions_df[transactions_df['Quantity'] > 0]  # Filter out returns/negative quantities
    transactions_df = transactions_df.dropna(subset=['CustomerID'])  # Ensure customer IDs exist
    
    # Calculate total value
    transactions_df['TotalValue'] = transactions_df['Quantity'] * transactions_df['UnitPrice']
    
    # Create ID mappings
    customer_ids = transactions_df["CustomerID"].unique()
    product_ids = transactions_df["StockCode"].unique()
    
    customer_id_map = {cid: idx for idx, cid in enumerate(customer_ids)}
    product_id_map = {pid: idx for idx, pid in enumerate(product_ids)}
    num_customers = len(customer_id_map)
    num_products = len(product_id_map)
    print(f"Loaded {num_customers} customers and {num_products} products.")
    
    # Build customer->product "buys" edges with time decay weights
    cust_src, prod_dst, time_weights = [], [], []
    user_to_items = defaultdict(list)
    
    # Compute time decay weights
    decay_weights = compute_time_decay_weights(transactions_df, decay_rate=0.005)
    for idx, row in transactions_df.iterrows():
        cid = row["CustomerID"]
        pid = row["StockCode"]
        if cid in customer_id_map and pid in product_id_map:
            u_idx = customer_id_map[cid]
            i_idx = product_id_map[pid]
            global_i_idx = num_customers + i_idx  # Global index for product
            cust_src.append(u_idx)
            prod_dst.append(global_i_idx)
            time_weights.append(decay_weights[idx])
            user_to_items[u_idx].append(i_idx)
    
    # Create edge_index tensor for customer-product edges
    edge_index_buy = torch.tensor([cust_src, prod_dst], dtype=torch.long)
    # Store edge weights (time decay)
    edge_weight_buy = torch.tensor(time_weights, dtype=torch.float)
    
    # Make undirected: add reciprocal edges
    edge_index_buy_rev = edge_index_buy.flip(0)
    edge_index_buy = torch.cat([edge_index_buy, edge_index_buy_rev], dim=1)
    edge_weight_buy = torch.cat([edge_weight_buy, edge_weight_buy], dim=0)
    
    print(f"Constructed {edge_index_buy.size(1)} customer-product 'buys' edges.")
    
    # Build product-product "bought_together" edges using co-occurrence
    src_co, dst_co, co_weights = compute_cooccurrence(transactions_df, product_id_map, min_cooccur=5)
    prod_co_src = [num_customers + a for a in src_co]
    prod_co_dst = [num_customers + b for b in dst_co]
    
    if prod_co_src:
        edge_index_co = torch.tensor([prod_co_src, prod_co_dst], dtype=torch.long)
        # For undirected graph, add reciprocal
        edge_index_co_rev = edge_index_co.flip(0)
        edge_index_co = torch.cat([edge_index_co, edge_index_co_rev], dim=1)
        co_weights_tensor = torch.tensor(co_weights, dtype=torch.float)
        co_weights_tensor = torch.cat([co_weights_tensor, co_weights_tensor], dim=0)
        print(f"Added {edge_index_co.size(1)} product-product 'bought_together' edges.")
    else:
        edge_index_co = None
        co_weights_tensor = None
        print("No product-product edges added.")
    
    # Create country-based customer edges (optional)
    customer_country = transactions_df[['CustomerID', 'Country']].drop_duplicates()
    country_groups = customer_country.groupby('Country')
    
    country_src, country_dst = [], []
    for _, group in country_groups:
        customers = [customer_id_map[cid] for cid in group['CustomerID'] if cid in customer_id_map]
        for i in range(len(customers)):
            for j in range(i+1, min(i+10, len(customers))):  # Limit connections
                country_src.append(customers[i])
                country_dst.append(customers[j])
                # Add reverse edge
                country_src.append(customers[j])
                country_dst.append(customers[i])
    
    if country_src:
        edge_index_country = torch.tensor([country_src, country_dst], dtype=torch.long)
        country_weights = torch.ones(edge_index_country.size(1))
        print(f"Added {edge_index_country.size(1)} customer-customer 'same_country' edges.")
    else:
        edge_index_country = None
        country_weights = None
        print("No country-based edges added.")
    
    # Merge all edge types
    edge_indices = [edge_index_buy]
    edge_weights = [edge_weight_buy]
    
    if edge_index_co is not None:
        edge_indices.append(edge_index_co)
        edge_weights.append(co_weights_tensor)
    
    if edge_index_country is not None:
        edge_indices.append(edge_index_country)
        edge_weights.append(country_weights)
    
    edge_index = torch.cat(edge_indices, dim=1)
    edge_weight = torch.cat(edge_weights, dim=0)
    
    # Create node features
    customer_features = create_customer_features(transactions_df, customer_ids)
    product_features = create_product_features(transactions_df, product_ids)
    
    # Ensure same feature dimension: pad with zeros if needed
    feat_dim = max(customer_features.size(1), product_features.size(1))
    if customer_features.size(1) < feat_dim:
        pad = torch.zeros((customer_features.size(0), feat_dim - customer_features.size(1)))
        customer_features = torch.cat([customer_features, pad], dim=1)
    if product_features.size(1) < feat_dim:
        pad = torch.zeros((product_features.size(0), feat_dim - product_features.size(1)))
        product_features = torch.cat([product_features, pad], dim=1)
    
    num_nodes = num_customers + num_products
    all_features = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
    all_features[:num_customers] = customer_features
    all_features[num_customers:] = product_features
    
    # Store product descriptions for recommendation display
    product_details = {}
    for _, row in transactions_df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates('StockCode').iterrows():
        product_details[row['StockCode']] = {
            'description': row['Description'],
            'price': row['UnitPrice']
        }
    
    # Build the PyG Data object, including edge weights
    data = Data(x=all_features, edge_index=edge_index, edge_attr=edge_weight)
    
    # Prepare metadata
    meta = {
        "num_customers": num_customers,
        "num_products": num_products,
        "customer_id_map": customer_id_map,
        "product_id_map": product_id_map,
        "user_to_items": user_to_items,
        "product_details": product_details
    }
    
    return data, meta

def load_hm_data(data_dir="data"):
    """
    Load H&M dataset and build graph.
    This is a separate function to handle the H&M dataset which has a different structure.
    Only implement this if you're working with the H&M dataset in addition to retail.
    
    Args:
        data_dir (str): Directory containing the data files
        
    Returns:
        Data: PyTorch Geometric Data object
        dict: Metadata dictionary
    """
    # Load H&M products CSV
    products_path = os.path.join(data_dir, "hm_products.csv")
    transactions_path = os.path.join(data_dir, "hm_transactions.csv")
    customers_path = os.path.join(data_dir, "hm_customers.csv")
    
    # Load data
    products_df = pd.read_csv(products_path)
    transactions_df = pd.read_csv(transactions_path)
    customers_df = pd.read_csv(customers_path)
    
    # Process data - implement based on H&M dataset structure
    # This is a placeholder for H&M-specific processing
    
    # Return placeholder
    print("H&M data loading is not fully implemented yet.")
    return None, None

if __name__ == "__main__":
    data, meta = load_retail_data()
    print(data)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Node feature dimension: {data.num_node_features}")