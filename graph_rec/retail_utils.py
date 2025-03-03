"""
retail_utils.py

Utility functions for processing retail transaction data and preparing it
for the graph-based recommendation engine.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import datetime
import torch
import random

def preprocess_retail_data(df):
    """
    Preprocess retail transaction data.
    
    Args:
        df (pd.DataFrame): Raw retail transaction data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed data
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert data types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Filter out returns and canceled orders (negative quantities)
    df = df[df['Quantity'] > 0]
    
    # Remove missing customer IDs for user-item interactions
    df = df.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to integer if it's not already
    df['CustomerID'] = df['CustomerID'].astype(int)
    
    # Calculate total value per transaction
    df['TotalValue'] = df['Quantity'] * df['UnitPrice']
    
    # Remove outliers (optional)
    q1 = df['UnitPrice'].quantile(0.01)
    q3 = df['UnitPrice'].quantile(0.99)
    df = df[(df['UnitPrice'] >= q1) & (df['UnitPrice'] <= q3)]
    
    return df

def create_product_id_mapping(df):
    """
    Create mapping between product IDs and sequential indices.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        
    Returns:
        dict: Mapping from product ID to index
        dict: Reverse mapping from index to product ID
        list: List of product details for each index
    """
    # Get unique products
    products = df[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()
    products['AvgPrice'] = products.groupby('StockCode')['UnitPrice'].transform('mean')
    
    # Create forward and reverse mappings
    product_ids = products['StockCode'].unique()
    product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}
    
    # Create product details dictionary
    product_details = []
    for _, row in products.drop_duplicates('StockCode').iterrows():
        product_details.append({
            'StockCode': row['StockCode'],
            'Description': row['Description'],
            'AvgPrice': row['AvgPrice']
        })
    
    return product_to_idx, idx_to_product, product_details

def create_customer_id_mapping(df):
    """
    Create mapping between customer IDs and sequential indices.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        
    Returns:
        dict: Mapping from customer ID to index
        dict: Reverse mapping from index to customer ID
        list: List of customer details for each index
    """
    # Get unique customers
    customers = df[['CustomerID', 'Country']].drop_duplicates()
    
    # Create forward and reverse mappings
    customer_ids = customers['CustomerID'].unique()
    customer_to_idx = {cid: idx for idx, cid in enumerate(customer_ids)}
    idx_to_customer = {idx: cid for cid, idx in customer_to_idx.items()}
    
    # Create customer details with country information
    customer_details = []
    for _, row in customers.iterrows():
        customer_details.append({
            'CustomerID': row['CustomerID'],
            'Country': row['Country']
        })
    
    return customer_to_idx, idx_to_customer, customer_details

def compute_rfm_features(df, customer_ids):
    """
    Compute RFM (Recency, Frequency, Monetary) features for customers.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        customer_ids (list): List of customer IDs to compute features for
        
    Returns:
        torch.Tensor: Tensor of RFM features for each customer
    """
    # Get the latest date in the dataset
    latest_date = df['InvoiceDate'].max()
    
    # Group by customer
    customer_stats = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                 # Frequency
        'TotalValue': 'sum'                                     # Monetary
    }).reset_index()
    
    # Rename columns
    customer_stats.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    # Min-max normalization for each feature
    for col in ['Recency', 'Frequency', 'Monetary']:
        min_val = customer_stats[col].min()
        max_val = customer_stats[col].max()
        if max_val > min_val:
            customer_stats[col] = (customer_stats[col] - min_val) / (max_val - min_val)
        else:
            customer_stats[col] = 0
    
    # Invert recency (lower is better)
    customer_stats['Recency'] = 1 - customer_stats['Recency']
    
    # Create feature tensor
    features = []
    for cid in customer_ids:
        if cid in customer_stats['CustomerID'].values:
            cust_info = customer_stats[customer_stats['CustomerID'] == cid].iloc[0]
            feature = [
                float(cust_info['Recency']),
                float(cust_info['Frequency']),
                float(cust_info['Monetary'])
            ]
        else:
            feature = [0.0, 0.0, 0.0]
        features.append(feature)
    
    return torch.tensor(features, dtype=torch.float)

def compute_product_features(df, product_ids):
    """
    Compute features for products based on transaction data.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        product_ids (list): List of product IDs to compute features for
        
    Returns:
        torch.Tensor: Tensor of product features
    """
    # Group by product
    product_stats = df.groupby('StockCode').agg({
        'Quantity': 'sum',                # Total quantity sold
        'UnitPrice': 'mean',              # Average price
        'CustomerID': 'nunique',          # Number of unique customers
        'Country': lambda x: x.nunique()  # Geographic diversity
    }).reset_index()
    
    # Rename columns
    product_stats.columns = ['StockCode', 'Popularity', 'Price', 'CustomerDiversity', 'GeoDiversity']
    
    # Min-max normalization for each feature
    for col in ['Popularity', 'Price', 'CustomerDiversity', 'GeoDiversity']:
        min_val = product_stats[col].min()
        max_val = product_stats[col].max()
        if max_val > min_val:
            product_stats[col] = (product_stats[col] - min_val) / (max_val - min_val)
        else:
            product_stats[col] = 0
    
    # Create feature tensor
    features = []
    for pid in product_ids:
        if pid in product_stats['StockCode'].values:
            prod_info = product_stats[product_stats['StockCode'] == pid].iloc[0]
            feature = [
                float(prod_info['Popularity']),
                float(prod_info['Price']),
                float(prod_info['CustomerDiversity']),
                float(prod_info['GeoDiversity'])
            ]
        else:
            feature = [0.0, 0.0, 0.0, 0.0]
        features.append(feature)
    
    return torch.tensor(features, dtype=torch.float)

def create_transaction_edges(df, customer_to_idx, product_to_idx, time_decay_rate=0.005):
    """
    Create transaction edges between customers and products with time decay weights.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        customer_to_idx (dict): Mapping from customer ID to index
        product_to_idx (dict): Mapping from product ID to index
        time_decay_rate (float): Rate for exponential time decay
        
    Returns:
        torch.Tensor: Edge index tensor
        torch.Tensor: Edge weight tensor
    """
    # Get the latest date in the dataset
    latest_date = df['InvoiceDate'].max()
    
    cust_src, prod_dst, weights = [], [], []
    
    for _, row in df.iterrows():
        cid = row['CustomerID']
        pid = row['StockCode']
        
        if cid in customer_to_idx and pid in product_to_idx:
            # Get indices
            cust_idx = customer_to_idx[cid]
            prod_idx = product_to_idx[pid]
            
            # Calculate time decay weight
            days_diff = (latest_date - row['InvoiceDate']).days
            time_weight = np.exp(-time_decay_rate * days_diff)
            
            # Multiply by quantity for weighted importance
            weight = time_weight * row['Quantity']
            
            cust_src.append(cust_idx)
            prod_dst.append(prod_idx)
            weights.append(float(weight))
    
    # Create edge index tensor
    edge_index = torch.tensor([cust_src, prod_dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weight

def create_market_basket_edges(df, product_to_idx, min_support=5):
    """
    Create market basket edges between products frequently bought together.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        product_to_idx (dict): Mapping from product ID to index
        min_support (int): Minimum co-occurrence count
        
    Returns:
        torch.Tensor: Edge index tensor
        torch.Tensor: Edge weight tensor
    """
    # Group transactions by invoice
    invoice_groups = df.groupby('InvoiceNo')
    
    # Count co-occurrences
    cooccur = defaultdict(lambda: defaultdict(int))
    
    for _, group in invoice_groups:
        # Get list of products in this transaction
        products = list(set(group['StockCode']))
        products = [p for p in products if p in product_to_idx]
        
        # Count pairwise co-occurrences
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                a, b = product_to_idx[products[i]], product_to_idx[products[j]]
                cooccur[a][b] += 1
                cooccur[b][a] += 1
    
    # Create edges above minimum support
    src, dst, weights = [], [], []
    for a, nbrs in cooccur.items():
        for b, count in nbrs.items():
            if count >= min_support:
                src.append(a)
                dst.append(b)
                weights.append(float(count))
    
    if not src:  # No edges found
        return None, None
    
    # Create edge index tensor
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weight

def create_country_edges(df, customer_to_idx):
    """
    Create edges between customers from the same country.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        customer_to_idx (dict): Mapping from customer ID to index
        
    Returns:
        torch.Tensor: Edge index tensor
    """
    # Get unique (customer, country) pairs
    customer_country = df[['CustomerID', 'Country']].drop_duplicates()
    
    # Group by country
    country_groups = customer_country.groupby('Country')
    
    src, dst = [], []
    for _, group in country_groups:
        customers = [customer_to_idx[cid] for cid in group['CustomerID'] if cid in customer_to_idx]
        
        # Connect customers from the same country (limit to reasonable number)
        for i in range(len(customers)):
            for j in range(i+1, min(i+10, len(customers))):
                src.append(customers[i])
                dst.append(customers[j])
                # Add reverse edge for undirected graph
                src.append(customers[j])
                dst.append(customers[i])
    
    if not src:  # No edges found
        return None
    
    # Create edge index tensor
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    return edge_index

def get_test_customers(df, customer_to_idx, product_to_idx, test_ratio=0.2, min_transactions=5):
    """
    Split data into training and testing sets by customers.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        customer_to_idx (dict): Mapping from customer ID to index
        product_to_idx (dict): Mapping from product ID to index
        test_ratio (float): Ratio of customers to use for testing
        min_transactions (int): Minimum number of transactions per customer
        
    Returns:
        dict: Dictionary mapping customer indices to sets of product indices for testing
    """
    # Filter customers with enough transactions
    customer_counts = df.groupby('CustomerID').size()
    eligible_customers = customer_counts[customer_counts >= min_transactions].index
    
    # Select test customers
    num_test = int(len(eligible_customers) * test_ratio)
    test_customers = np.random.choice(eligible_customers, size=num_test, replace=False)
    
    # Create test set
    test_user_items = {}
    for cid in test_customers:
        if cid not in customer_to_idx:
            continue
            
        user_idx = customer_to_idx[cid]
        # Get products this customer has purchased
        products = df[df['CustomerID'] == cid]['StockCode'].unique()
        product_indices = [product_to_idx[pid] for pid in products if pid in product_to_idx]
        
        if product_indices:
            test_user_items[user_idx] = set(product_indices)
    
    return test_user_items

def split_train_test_by_time(df, test_days=30):
    """
    Split data into training and testing sets by time.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        test_days (int): Number of days to use for testing
        
    Returns:
        pd.DataFrame: Training data
        pd.DataFrame: Testing data
    """
    # Sort by date
    df = df.sort_values('InvoiceDate')
    
    # Get cutoff date
    max_date = df['InvoiceDate'].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)
    
    # Split data
    train_df = df[df['InvoiceDate'] <= cutoff_date]
    test_df = df[df['InvoiceDate'] > cutoff_date]
    
    return train_df, test_df

def create_user_item_dict(df, customer_to_idx, product_to_idx):
    """
    Create a dictionary mapping user indices to sets of item indices.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        customer_to_idx (dict): Mapping from customer ID to index
        product_to_idx (dict): Mapping from product ID to index
        
    Returns:
        dict: Dictionary mapping user indices to sets of item indices
    """
    user_to_items = defaultdict(set)
    
    for _, row in df.iterrows():
        cid = row['CustomerID']
        pid = row['StockCode']
        
        if cid in customer_to_idx and pid in product_to_idx:
            user_idx = customer_to_idx[cid]
            item_idx = product_to_idx[pid]
            user_to_items[user_idx].add(item_idx)
    
    return user_to_items

def sample_negatives(user_to_items, num_items, num_neg=5):
    """
    Sample negative items for each user.
    
    Args:
        user_to_items (dict): Dictionary mapping user indices to sets of item indices
        num_items (int): Total number of items
        num_neg (int): Number of negative samples per user
        
    Returns:
        dict: Dictionary mapping user indices to lists of negative item indices
    """
    user_to_negs = {}
    all_items = set(range(num_items))
    
    for user_idx, pos_items in user_to_items.items():
        # Get items not interacted with
        neg_candidates = list(all_items - pos_items)
        
        # Sample negative items
        if len(neg_candidates) >= num_neg:
            neg_samples = random.sample(neg_candidates, num_neg)
        else:
            # If not enough candidates, use all available with replacement
            neg_samples = random.choices(neg_candidates, k=num_neg)
            
        user_to_negs[user_idx] = neg_samples
    
    return user_to_negs

def compute_price_similarity(df, product_to_idx, num_products):
    """
    Compute price similarity between products.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        product_to_idx (dict): Mapping from product ID to index
        num_products (int): Total number of products
        
    Returns:
        torch.Tensor: Edge index tensor
        torch.Tensor: Edge weight tensor
    """
    # Get average price per product
    avg_prices = df.groupby('StockCode')['UnitPrice'].mean()
    
    # Find price ranges
    price_min = avg_prices.min()
    price_max = avg_prices.max()
    
    # Create bins (adjust number of bins as needed)
    num_bins = 10
    bin_size = (price_max - price_min) / num_bins
    
    # Assign products to price bins
    price_bins = {}
    for pid, price in avg_prices.items():
        if pid in product_to_idx:
            bin_idx = min(num_bins - 1, int((price - price_min) / bin_size))
            if bin_idx not in price_bins:
                price_bins[bin_idx] = []
            price_bins[bin_idx].append(product_to_idx[pid])
    
    # Create edges between products in the same price bin
    src, dst, weights = [], [], []
    for bin_idx, products in price_bins.items():
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                src.append(products[i])
                dst.append(products[j])
                weights.append(1.0)  # Equal weight for all price similarity edges
                
                # Add reverse edge
                src.append(products[j])
                dst.append(products[i])
                weights.append(1.0)
    
    if not src:  # No edges found
        return None, None
    
    # Create edge index tensor
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weight

def compute_category_similarity(df, product_to_idx):
    """
    Compute category similarity based on product descriptions.
    This is a simple implementation that groups products by the first word of their description.
    
    Args:
        df (pd.DataFrame): Retail transaction data
        product_to_idx (dict): Mapping from product ID to index
        
    Returns:
        torch.Tensor: Edge index tensor
        torch.Tensor: Edge weight tensor
    """
    # Extract first word of description as a simple category
    product_categories = {}
    for _, row in df[['StockCode', 'Description']].drop_duplicates().iterrows():
        if pd.isna(row['Description']):
            continue
            
        pid = row['StockCode']
        if pid in product_to_idx:
            # Use first word as category
            category = row['Description'].split()[0].lower()
            if category not in product_categories:
                product_categories[category] = []
            product_categories[category].append(product_to_idx[pid])
    
    # Create edges between products in the same category
    src, dst, weights = [], [], []
    for category, products in product_categories.items():
        for i in range(len(products)):
            for j in range(i+1, len(products)):
                src.append(products[i])
                dst.append(products[j])
                weights.append(1.0)  # Equal weight for all category similarity edges
                
                # Add reverse edge
                src.append(products[j])
                dst.append(products[i])
                weights.append(1.0)
    
    if not src:  # No edges found
        return None, None
    
    # Create edge index tensor
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_weight

def prepare_retail_graph_data(df, num_customers, num_products, 
                             customer_to_idx, product_to_idx,
                             edge_types=['transaction', 'basket', 'country', 'price', 'category']):
    """
    Prepare complete graph data for the retail recommendation system.
    
    Args:
        df (pd.DataFrame): Preprocessed retail transaction data
        num_customers (int): Number of customers
        num_products (int): Number of products
        customer_to_idx (dict): Mapping from customer ID to index
        product_to_idx (dict): Mapping from product ID to index
        edge_types (list): Types of edges to include
        
    Returns:
        dict: Complete graph data and metadata
    """
    # Create node features
    customer_ids = list(customer_to_idx.keys())
    product_ids = list(product_to_idx.keys())
    
    customer_features = compute_rfm_features(df, customer_ids)
    product_features = compute_product_features(df, product_ids)
    
    # Create different edge types
    edge_data = {}
    
    if 'transaction' in edge_types:
        transaction_edges, transaction_weights = create_transaction_edges(
            df, customer_to_idx, product_to_idx)
        edge_data['transaction'] = (transaction_edges, transaction_weights)
    
    if 'basket' in edge_types:
        basket_edges, basket_weights = create_market_basket_edges(
            df, product_to_idx)
        if basket_edges is not None:
            edge_data['basket'] = (basket_edges, basket_weights)
    
    if 'country' in edge_types:
        country_edges = create_country_edges(df, customer_to_idx)
        if country_edges is not None:
            edge_data['country'] = (country_edges, torch.ones(country_edges.size(1)))
    
    if 'price' in edge_types:
        price_edges, price_weights = compute_price_similarity(
            df, product_to_idx, num_products)
        if price_edges is not None:
            edge_data['price'] = (price_edges, price_weights)
    
    if 'category' in edge_types:
        category_edges, category_weights = compute_category_similarity(
            df, product_to_idx)
        if category_edges is not None:
            edge_data['category'] = (category_edges, category_weights)
    
    # Create user-item dictionary for training
    user_to_items = create_user_item_dict(df, customer_to_idx, product_to_idx)
    
    # Prepare test data
    test_user_items = get_test_customers(df, customer_to_idx, product_to_idx)
    
    # Create combined edge index and edge weight tensors
    all_edge_indices = []
    all_edge_weights = []
    all_edge_types = []
    
    edge_type_map = {edge_type: i for i, edge_type in enumerate(edge_data.keys())}
    
    for edge_type, (edge_index, edge_weight) in edge_data.items():
        all_edge_indices.append(edge_index)
        all_edge_weights.append(edge_weight)
        all_edge_types.append(torch.full((edge_index.size(1),), edge_type_map[edge_type]))
    
    combined_edge_index = torch.cat(all_edge_indices, dim=1)
    combined_edge_weight = torch.cat(all_edge_weights)
    combined_edge_type = torch.cat(all_edge_types)
    
    # Prepare metadata
    meta = {
        "num_customers": num_customers,
        "num_products": num_products,
        "customer_id_map": customer_to_idx,
        "product_id_map": product_to_idx,
        "user_to_items": user_to_items,
        "test_user_items": test_user_items,
        "edge_type_map": edge_type_map
    }
    
    # Create PyG Data object
    return {
        "customer_features": customer_features,
        "product_features": product_features,
        "edge_index": combined_edge_index,
        "edge_weight": combined_edge_weight,
        "edge_type": combined_edge_type,
        "meta": meta
    }

def prepare_batch_for_training(user_to_items, num_products, batch_size=64, neg_ratio=4):
    """
    Prepare a batch of data for training.
    
    Args:
        user_to_items (dict): Dictionary mapping user indices to sets of item indices
        num_products (int): Total number of products
        batch_size (int): Batch size
        neg_ratio (int): Ratio of negative to positive samples
        
    Returns:
        tuple: Batch of users, positive items, and negative items
    """
    users = list(user_to_items.keys())
    
    # Randomly select users for this batch
    batch_users = random.sample(users, min(batch_size, len(users)))
    
    pos_items = []
    neg_items = []
    final_users = []
    
    for user in batch_users:
        user_pos_items = list(user_to_items[user])
        if not user_pos_items:
            continue
            
        # Select a positive item
        pos_item = random.choice(user_pos_items)
        
        # Sample negative items
        for _ in range(neg_ratio):
            neg_item = random.randint(0, num_products - 1)
            while neg_item in user_to_items[user]:
                neg_item = random.randint(0, num_products - 1)
            
            final_users.append(user)
            pos_items.append(pos_item)
            neg_items.append(neg_item)
    
    return torch.tensor(final_users), torch.tensor(pos_items), torch.tensor(neg_items)

def compute_metrics_for_recommendations(model, data, meta, k_values=[5, 10, 20]):
    """
    Compute evaluation metrics for recommendations.
    
    Args:
        model: Trained recommendation model
        data: Graph data
        meta: Metadata dictionary
        k_values: List of k values for which to compute metrics
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    test_user_items = meta["test_user_items"]
    num_customers = meta["num_customers"]
    num_products = meta["num_products"]
    
    if not test_user_items:
        return {"error": "No test users available"}
    
    model.eval()
    with torch.no_grad():
        all_emb = model(data['edge_index'], data['edge_weight'], data['edge_type'])
    
    user_embs = all_emb[:num_customers]
    item_embs = all_emb[num_customers:num_customers + num_products]
    
    # Compute metrics for each test user
    recalls = {k: [] for k in k_values}
    precisions = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    
    for user_idx, true_items in test_user_items.items():
        # Skip users with no positive items
        if not true_items:
            continue
            
        # Get recommendations
        user_emb = user_embs[user_idx].unsqueeze(0)
        scores = torch.matmul(user_emb, item_embs.t()).squeeze(0)
        
        # Remove items the user has already interacted with
        for item in user_to_items.get(user_idx, set()):
            scores[item] = -float('inf')
        
        # Get top-k recommendations
        _, topk_idx = torch.topk(scores, max(k_values))
        topk_idx = topk_idx.cpu().tolist()
        
        # Compute metrics for different k values
        for k in k_values:
            topk = topk_idx[:k]
            hits = len(set(topk) & true_items)
            
            # Recall@k = hits / |true_items|
            recall = hits / len(true_items)
            recalls[k].append(recall)
            
            # Precision@k = hits / k
            precision = hits / k
            precisions[k].append(precision)
            
            # NDCG@k
            dcg = 0.0
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            
            for i, item in enumerate(topk):
                if item in true_items:
                    dcg += 1.0 / np.log2(i + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0.0
            ndcgs[k].append(ndcg)
    
    # Calculate average metrics
    results = {}
    for k in k_values:
        results[f'Recall@{k}'] = np.mean(recalls[k]) if recalls[k] else 0.0
        results[f'Precision@{k}'] = np.mean(precisions[k]) if precisions[k] else 0.0
        results[f'NDCG@{k}'] = np.mean(ndcgs[k]) if ndcgs[k] else 0.0
    
    return results

def get_similar_products(model, data, meta, product_id, top_k=10):
    """
    Get similar products based on learned embeddings.
    
    Args:
        model: Trained recommendation model
        data: Graph data
        meta: Metadata dictionary
        product_id: Product ID to find similar items for
        top_k: Number of similar products to return
        
    Returns:
        list: Similar products with similarity scores
    """
    product_id_map = meta["product_id_map"]
    idx_to_product = {idx: pid for pid, idx in product_id_map.items()}
    num_customers = meta["num_customers"]
    
    if product_id not in product_id_map:
        return