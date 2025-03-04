"""
modified_data_loader.py

This module loads the filtered H&M dataset and builds a heterogeneous graph
for LightGCN recommendation model. It includes:
- Loading filtered CSV files
- Creating mappings between IDs and indices
- Building a heterogeneous graph with both customer-article and article-article edges
- Computing time decay weights for recency importance
- Preparing node features
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import datetime

def compute_time_decay_weights(transactions_df, current_date=None, decay_rate=0.005):
    """
    Computes time decay weights for transactions based on recency.
    
    Args:
        transactions_df (pd.DataFrame): Transactions with a 't_dat' column (date string)
        current_date (datetime.date, optional): Reference date. If None, uses max date
        decay_rate (float): Rate for exponential decay
    
    Returns:
        np.array: Array of weights for each transaction
    """
    if current_date is None:
        current_date = pd.to_datetime(transactions_df["t_dat"]).max().date()
    else:
        current_date = pd.to_datetime(current_date).date()
        
    # Convert t_dat to datetime.date
    dates = pd.to_datetime(transactions_df["t_dat"]).dt.date
    weights = []
    for d in dates:
        delta = (current_date - d).days
        weight = np.exp(-decay_rate * delta)
        weights.append(weight)
    return np.array(weights)

def compute_cooccurrence(transactions_df, article_id_map, min_cooccur=3):
    """
    Computes item co-occurrence counts from transactions.
    
    Args:
        transactions_df (pd.DataFrame): Transactions data with at least 'customer_id' and 'article_id'
        article_id_map (dict): Mapping from article id to integer index
        min_cooccur (int): Minimum number of co-occurrences to consider an edge
    
    Returns:
        List[int], List[int], List[float]: Lists of source indices, destination indices, and weights
    """
    cooccur = defaultdict(lambda: defaultdict(int))
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(transactions_df['t_dat']):
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
    
    # Group by customer and date (items bought in same transaction)
    transactions_grouped = transactions_df.groupby(['customer_id', 't_dat'])
    
    for _, group in transactions_grouped:
        items = [article_id_map[aid] for aid in group["article_id"] if aid in article_id_map]
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

def load_filtered_data(data_dir="./data", min_cooccur=3, decay_rate=0.005):
    """
    Load filtered dataset and create a heterogeneous graph for LightGCN.
    
    Args:
        data_dir (str): Directory containing the filtered CSV files
        min_cooccur (int): Minimum number of co-occurrences to consider an article-article edge
        decay_rate (float): Time decay rate for customer-article interactions
        
    Returns:
        Data, dict: PyTorch Geometric Data object and metadata dictionary
    """
    # Load CSV files
    customers_path = os.path.join(data_dir, "filtered_customers.csv")
    articles_path = os.path.join(data_dir, "filtered_articles.csv")
    transactions_path = os.path.join(data_dir, "filtered_transactions_train.csv")
    
    print(f"Loading datasets from {data_dir}...")
    customers_df = pd.read_csv(customers_path)
    articles_df = pd.read_csv(articles_path)
    transactions_df = pd.read_csv(transactions_path)
    
    # Convert transaction dates to datetime
    transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
    
    # Create ID mappings
    customer_ids = customers_df["customer_id"].unique()
    article_ids = articles_df["article_id"].unique()
    
    customer_id_map = {cid: idx for idx, cid in enumerate(customer_ids)}
    article_id_map = {aid: idx for idx, aid in enumerate(article_ids)}
    
    num_customers = len(customer_id_map)
    num_articles = len(article_id_map)
    print(f"Loaded {num_customers} customers and {num_articles} articles")
    
    # Build customer->article "buys" edges with time decay weights
    cust_src, art_dst, time_weights = [], [], []
    user_to_items = defaultdict(list)
    
    # Compute time decay weights
    decay_weights = compute_time_decay_weights(transactions_df, decay_rate=decay_rate)
    
    # Convert to torch tensors
    for idx, row in transactions_df.iterrows():
        cid = row["customer_id"]
        aid = row["article_id"]
        if cid in customer_id_map and aid in article_id_map:
            u_idx = customer_id_map[cid]
            i_idx = article_id_map[aid]
            global_i_idx = num_customers + i_idx  # Global index for item nodes
            
            cust_src.append(u_idx)
            art_dst.append(global_i_idx)
            time_weights.append(decay_weights[idx])
            user_to_items[u_idx].append(i_idx)
    
    # Create edge_index tensor for customer-article edges
    edge_index_buy = torch.tensor([cust_src, art_dst], dtype=torch.long)
    # Store edge weights (time decay)
    edge_weight_buy = torch.tensor(time_weights, dtype=torch.float)
    
    # Make undirected: add reciprocal edges
    edge_index_buy_rev = torch.stack([edge_index_buy[1], edge_index_buy[0]], dim=0)
    edge_index_buy = torch.cat([edge_index_buy, edge_index_buy_rev], dim=1)
    edge_weight_buy = torch.cat([edge_weight_buy, edge_weight_buy], dim=0)
    
    print(f"Constructed {edge_index_buy.size(1)} customer-article edges")
    
    # Build article-article "bought_together" edges using co-occurrence
    src_co, dst_co, co_weights = compute_cooccurrence(transactions_df, article_id_map, min_cooccur=min_cooccur)
    
    # Convert to global indices (after customer indices)
    art_co_src = [num_customers + a for a in src_co]
    art_co_dst = [num_customers + b for b in dst_co]
    
    if art_co_src:
        edge_index_co = torch.tensor([art_co_src, art_co_dst], dtype=torch.long)
        # For undirected graph, add reciprocal edges
        edge_index_co_rev = torch.stack([edge_index_co[1], edge_index_co[0]], dim=0)
        edge_index_co = torch.cat([edge_index_co, edge_index_co_rev], dim=1)
        co_weights_tensor = torch.tensor(co_weights, dtype=torch.float)
        co_weights_tensor = torch.cat([co_weights_tensor, co_weights_tensor], dim=0)
        print(f"Added {edge_index_co.size(1)} article-article 'bought_together' edges")
    else:
        edge_index_co = None
        co_weights_tensor = None
        print("No article-article edges added")
    
    # Merge edges: For simplicity, we combine both edge types
    if edge_index_co is not None:
        edge_index = torch.cat([edge_index_buy, edge_index_co], dim=1)
        edge_weight = torch.cat([edge_weight_buy, co_weights_tensor], dim=0)
    else:
        edge_index = edge_index_buy
        edge_weight = edge_weight_buy

    # --- Create Node Features ---
    # Customer features: normalized age, one-hot for 'fashion_news_frequency'
    cust_features = []
    news_categories = sorted(customers_df["fashion_news_frequency"].fillna("NONE").unique())
    news_to_idx = {cat: idx for idx, cat in enumerate(news_categories)}
    num_news = len(news_categories)
    
    for cid in customer_ids:
        cust_info = customers_df[customers_df["customer_id"] == cid].iloc[0]
        # Normalize age
        age = cust_info.get("age", 30)
        if pd.isna(age):
            age = 30  # Default age if missing
        age_norm = age / 100.0  # Normalize age
        
        # One-hot encode fashion news frequency
        freq = cust_info.get("fashion_news_frequency", "NONE")
        if pd.isna(freq):
            freq = "NONE"
        one_hot = [0] * num_news
        one_hot[news_to_idx[freq]] = 1
        
        # Combine features
        cust_features.append([age_norm] + one_hot)
    
    cust_features = torch.tensor(cust_features, dtype=torch.float)
    
    # Article features: one-hot for 'product_type_no'
    product_types = sorted(articles_df["product_type_no"].fillna(0).unique())
    prod_to_idx = {pt: idx for idx, pt in enumerate(product_types)}
    num_prod = len(product_types)
    
    art_features = []
    for aid in article_ids:
        art_info = articles_df[articles_df["article_id"] == aid].iloc[0]
        pt = art_info.get("product_type_no", product_types[0])
        if pd.isna(pt):
            pt = product_types[0]
        
        one_hot = [0] * num_prod
        one_hot[prod_to_idx[pt]] = 1
        
        # Could add more features here (color, department, etc.)
        art_features.append(one_hot)
    
    art_features = torch.tensor(art_features, dtype=torch.float)
    
    # Ensure same feature dimension: pad with zeros if needed
    feat_dim = max(cust_features.size(1), art_features.size(1))
    
    if cust_features.size(1) < feat_dim:
        pad = torch.zeros((cust_features.size(0), feat_dim - cust_features.size(1)))
        cust_features = torch.cat([cust_features, pad], dim=1)
    
    if art_features.size(1) < feat_dim:
        pad = torch.zeros((art_features.size(0), feat_dim - art_features.size(1)))
        art_features = torch.cat([art_features, pad], dim=1)
    
    # Combine all features
    num_nodes = num_customers + num_articles
    all_features = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
    all_features[:num_customers] = cust_features
    all_features[num_customers:] = art_features
    
    # Build the PyG Data object, including edge weights
    data = Data(x=all_features, edge_index=edge_index, edge_attr=edge_weight)
    
    # Create metadata dictionary
    meta = {
        "num_customers": num_customers,
        "num_articles": num_articles,
        "customer_id_map": customer_id_map,
        "article_id_map": article_id_map,
        "reverse_customer_map": {v: k for k, v in customer_id_map.items()},
        "reverse_article_map": {v: k for k, v in article_id_map.items()},
        "user_to_items": user_to_items,
        "news_categories": news_categories,
        "product_types": product_types,
        "feature_dim": feat_dim
    }
    
    return data, meta

def create_train_test_split(data, meta, test_ratio=0.2, by_time=True):
    """
    Create train/test split for evaluation.
    
    Args:
        data (Data): PyTorch Geometric Data object
        meta (dict): Metadata dictionary
        test_ratio (float): Ratio of data to use for testing
        by_time (bool): If True, split by time (last transactions for test)
                        If False, randomly sample
    
    Returns:
        dict, dict: Dictionaries of training and testing user-item interactions
    """
    # Use user_to_items from meta if it exists
    if "user_to_items" in meta and meta["user_to_items"]:
        print("Using cached user-item interactions from meta")
        user_items = defaultdict(list)
        for u_idx, items in meta["user_to_items"].items():
            # Add a default timestamp if not available
            for i_idx in items:
                user_items[u_idx].append((i_idx, 0))
    else:
        # Determine transactions path
        transactions_path = os.path.join(meta.get("data_dir", "./data"), "filtered_transactions_train.csv")
        if not os.path.exists(transactions_path):
            # Try alternate path
            transactions_path = os.path.join(os.path.dirname(meta.get("data_dir", "./data")), 
                                            "filtered_transactions_train.csv")
        
        print(f"Loading transactions from {transactions_path}")
        transactions_df = pd.read_csv(transactions_path)
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
        
        user_items = defaultdict(list)
        for _, row in transactions_df.iterrows():
            cid = row["customer_id"]
            aid = row["article_id"]
            if cid in meta["customer_id_map"] and aid in meta["article_id_map"]:
                u_idx = meta["customer_id_map"][cid]
                i_idx = meta["article_id_map"][aid]
                timestamp = row["t_dat"].timestamp()
                user_items[u_idx].append((i_idx, timestamp))
    
    train_user_items = defaultdict(set)
    test_user_items = defaultdict(set)
    
    if by_time:
        # Split by time - last transactions for each user go to test set
        for u_idx, items in user_items.items():
            # Sort by timestamp
            items.sort(key=lambda x: x[1])
            # Calculate split point
            split_idx = max(1, int(len(items) * (1 - test_ratio)))
            
            # Add to train/test sets
            for i_idx, _ in items[:split_idx]:
                train_user_items[u_idx].add(i_idx)
            
            for i_idx, _ in items[split_idx:]:
                test_user_items[u_idx].add(i_idx)
    else:
        # Random split
        for u_idx, items in user_items.items():
            items = [item[0] for item in items]  # Extract just the item indices
            np.random.shuffle(items)
            
            split_idx = max(1, int(len(items) * (1 - test_ratio)))
            train_user_items[u_idx] = set(items[:split_idx])
            test_user_items[u_idx] = set(items[split_idx:])
    
    # Remove users with empty test sets
    for u_idx in list(test_user_items.keys()):
        if len(test_user_items[u_idx]) == 0:
            del test_user_items[u_idx]
    
    print(f"Created train/test split with {len(train_user_items)} training users and {len(test_user_items)} testing users")
    
    return train_user_items, test_user_items

if __name__ == "__main__":
    # Example usage
    data_dir = "/Users/sidgraph/Desktop/LLM_KG_RecSys/graph_rec/data"
    data, meta = load_filtered_data(data_dir=data_dir, min_cooccur=3)
    
    print(f"Created PyG Data object with {data.num_nodes} nodes and {data.num_edges} edges")
    
    # Create train/test split
    train_user_items, test_user_items = create_train_test_split(data, meta, test_ratio=0.2)
    
    # Save data and metadata
    import pickle
    output_dir = os.path.join(data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save PyG data
    torch.save(data, os.path.join(output_dir, "lightgcn_data.pt"))
    
    # Save metadata with train/test split
    meta_with_split = meta.copy()
    meta_with_split.update({
        "train_user_items": train_user_items,
        "test_user_items": test_user_items,
        "data_dir": data_dir
    })
    
    with open(os.path.join(output_dir, "lightgcn_meta.pkl"), "wb") as f:
        pickle.dump(meta_with_split, f)
    
    # Save ID mappings separately
    mappings = {
        "customer_id_map": meta["customer_id_map"],
        "article_id_map": meta["article_id_map"],
        "reverse_customer_map": {v: k for k, v in meta["customer_id_map"].items()},
        "reverse_article_map": {v: k for k, v in meta["article_id_map"].items()}
    }
    with open(os.path.join(output_dir, "id_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    
    num_customers = meta["num_customers"]
    num_articles = meta["num_articles"]
    
    print(f"Saved processed data to {output_dir}")
    print(f"Data statistics:")
    print(f"- {num_customers} customers")
    print(f"- {num_articles} articles")
    print(f"- {data.num_edges} total edges")
    print(f"- {len(train_user_items)} users in training set")
    print(f"- {len(test_user_items)} users in test set")