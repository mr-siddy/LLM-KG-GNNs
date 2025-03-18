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
import networkx as nx

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


def compute_cooccurrence(transactions_df, article_id_map, customer_id_map, min_cooccur=3, time_window_days=30):
    """
    Computes item co-occurrence counts from transactions with time window consideration.
    
    Args:
        transactions_df (pd.DataFrame): Transactions data
        article_id_map (dict): Mapping from article id to integer index
        min_cooccur (int): Minimum number of co-occurrences to consider an edge
        time_window_days (int): Time window in days to consider co-occurrence more relevant
        
    Returns:
        List[int], List[int], List[float]: Lists of source indices, destination indices, and weights
    """
    cooccur = defaultdict(lambda: defaultdict(int))
    
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(transactions_df['t_dat']):
        transactions_df['t_dat'] = pd.to_datetime(transactions_df['t_dat'])
    
    # Group by customer and date (items bought in same transaction)
    transactions_grouped = transactions_df.groupby(['customer_id', 't_dat'])
    
    # Track time-weighted co-occurrences
    recent_cooccur = defaultdict(lambda: defaultdict(float))
    max_date = transactions_df['t_dat'].max()
    
    for _, group in transactions_grouped:
        items = [article_id_map[aid] for aid in group["article_id"] if aid in article_id_map]
        transaction_date = group['t_dat'].iloc[0]
        
        # Calculate recency weight (more recent = higher weight)
        days_from_max = (max_date - transaction_date).days
        recency_weight = np.exp(-0.01 * days_from_max)  # Exponential decay for time
        
        for i in range(len(items)):
            for j in range(i+1, len(items)):
                a, b = items[i], items[j]
                cooccur[a][b] += 1
                cooccur[b][a] += 1
                
                # Add time-weighted co-occurrence
                recent_cooccur[a][b] += recency_weight
                recent_cooccur[b][a] += recency_weight

    # Also consider items bought by the same user within the time window
    user_purchases = defaultdict(list)
    
    for _, row in transactions_df.iterrows():
        if row["customer_id"] in customer_id_map and row["article_id"] in article_id_map:
            user = customer_id_map[row["customer_id"]]
            item = article_id_map[row["article_id"]]
            date = row["t_dat"]
            user_purchases[user].append((item, date))
    
    # Find co-occurrences within time window
    for user, purchases in user_purchases.items():
        # Sort by date
        purchases.sort(key=lambda x: x[1])
        
        # Check each purchase against later ones within time window
        for i, (item_i, date_i) in enumerate(purchases):
            for j in range(i+1, len(purchases)):
                item_j, date_j = purchases[j]
                
                # Check if within time window
                days_diff = (date_j - date_i).days
                if days_diff <= time_window_days:
                    # Add a smaller weight for time-window co-occurrence (not same transaction)
                    # This creates edges between items frequently bought by the same user in sequence
                    if cooccur[item_i][item_j] == 0:  # Only if not already co-occurring in same transaction
                        cooccur[item_i][item_j] += 0.5
                        cooccur[item_j][item_i] += 0.5
                        
                        # Also add time-recency weight
                        days_from_max_i = (max_date - date_i).days
                        recency_weight_i = np.exp(-0.01 * days_from_max_i)
                        recent_cooccur[item_i][item_j] += 0.5 * recency_weight_i
                        recent_cooccur[item_j][item_i] += 0.5 * recency_weight_i

    # Combine regular and time-weighted co-occurrences
    src, dst, weights = [], [], []
    for a, nbrs in cooccur.items():
        for b, count in nbrs.items():
            if count >= min_cooccur:
                src.append(a)
                dst.append(b)
                
                # Blend count with recency weight for final edge weight
                blended_weight = count * (1 + recent_cooccur[a][b])
                weights.append(blended_weight)
                
    return src, dst, weights


def compute_customer_features(customers_df, transactions_df, customer_id_map):
    """
    Compute enhanced customer features including:
    - Demographics (age, club status)
    - Purchase behavior (frequency, recency, monetary)
    - Fashion preferences
    
    Args:
        customers_df (pd.DataFrame): Customer data
        transactions_df (pd.DataFrame): Transaction data
        customer_id_map (dict): Mapping from customer ID to index
        
    Returns:
        torch.Tensor: Customer feature tensor
    """
    print("Computing enhanced customer features...")
    
    # Initialize feature dataframe with customer IDs
    customer_ids = list(customer_id_map.keys())
    customer_features_df = pd.DataFrame({"customer_id": customer_ids})
    
    # 1. Basic demographic features
    customers_df_mapped = customers_df.set_index('customer_id')
    
    # Age (with binning for better representation)
    age_df = pd.DataFrame(index=customer_ids)
    age_df['age'] = customers_df_mapped.loc[customer_ids, 'age'].fillna(30)
    age_df['age_bin'] = pd.cut(age_df['age'], bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                              labels=[0, 1, 2, 3, 4, 5, 6])
    
    # One-hot encode age bins
    age_dummies = pd.get_dummies(age_df['age_bin'], prefix='age_bin')
    customer_features_df = pd.concat([customer_features_df, age_dummies], axis=1)
    
    # Club membership status
    club_dummies = pd.get_dummies(
        customers_df_mapped.loc[customer_ids, 'club_member_status'].fillna('MISSING'), 
        prefix='club'
    )
    customer_features_df = pd.concat([customer_features_df, club_dummies], axis=1)
    
    # Fashion news frequency
    news_dummies = pd.get_dummies(
        customers_df_mapped.loc[customer_ids, 'fashion_news_frequency'].fillna('NONE'), 
        prefix='news'
    )
    customer_features_df = pd.concat([customer_features_df, news_dummies], axis=1)
    
    # 2. Purchase behavior features (RFM analysis)
    # First, create a dataframe of customer purchase history
    purchase_data = []
    
    for customer_id in customer_ids:
        # Filter transactions for this customer
        customer_txn = transactions_df[transactions_df['customer_id'] == customer_id]
        
        if len(customer_txn) > 0:
            # Recency: days since last purchase
            last_purchase = pd.to_datetime(customer_txn['t_dat']).max()
            recency = (pd.to_datetime(transactions_df['t_dat']).max() - last_purchase).days
            
            # Frequency: number of purchases
            frequency = len(customer_txn)
            
            # Monetary: average spending per transaction
            monetary = customer_txn['price'].mean()
            
            # Variety: number of unique articles purchased
            variety = customer_txn['article_id'].nunique()
            
            # Price sensitivity: variance in price
            price_variance = customer_txn['price'].var() if len(customer_txn) > 1 else 0
            
            # Channel preference (online vs in-store)
            channel_pref = customer_txn['sales_channel_id'].mode().iloc[0] if not customer_txn['sales_channel_id'].mode().empty else 1
            
            purchase_data.append({
                'customer_id': customer_id,
                'recency': recency,
                'frequency': frequency,
                'monetary': monetary,
                'variety': variety,
                'price_variance': price_variance,
                'channel_pref': channel_pref
            })
        else:
            # Default values for customers with no transactions
            purchase_data.append({
                'customer_id': customer_id,
                'recency': 999,  # High recency (long time since purchase)
                'frequency': 0,
                'monetary': 0,
                'variety': 0,
                'price_variance': 0,
                'channel_pref': 1  # Default channel
            })
    
    purchase_df = pd.DataFrame(purchase_data)
    
    # Normalize RFM values
    scaler = StandardScaler()
    rfm_cols = ['recency', 'frequency', 'monetary', 'variety', 'price_variance']
    purchase_df[rfm_cols] = scaler.fit_transform(purchase_df[rfm_cols])
    
    # Add RFM features to customer features
    customer_features_df = pd.merge(customer_features_df, purchase_df, on='customer_id')
    
    # 3. Product category preferences
    category_preferences = []
    
    # Get product types from transactions
    transaction_with_product = pd.merge(
        transactions_df,
        pd.DataFrame({'article_id': list(article_id_map.keys())}),
        on='article_id',
        how='inner'
    )
    
    # For each customer, calculate preferences
    for customer_id in customer_ids:
        customer_txn = transaction_with_product[transaction_with_product['customer_id'] == customer_id]
        customer_pref = {'customer_id': customer_id}
        
        if len(customer_txn) > 0:
            # Add basic product preference score
            customer_pref['has_purchases'] = 1
        else:
            customer_pref['has_purchases'] = 0
            
        category_preferences.append(customer_pref)
    
    category_pref_df = pd.DataFrame(category_preferences)
    customer_features_df = pd.merge(customer_features_df, category_pref_df, on='customer_id')
    
    # 4. Convert to tensor
    # Remove customer_id column and convert to tensor
    feature_cols = [col for col in customer_features_df.columns if col != 'customer_id']
    customer_features = torch.tensor(customer_features_df[feature_cols].values, dtype=torch.float)
    
    print(f"Created customer features with shape: {customer_features.shape}")
    return customer_features

def compute_article_features(articles_df, transactions_df, article_id_map):
    """
    Compute enhanced article features including:
    - Product attributes
    - Text embeddings from product descriptions
    - Popularity and temporal patterns
    - Category hierarchy embeddings
    
    Args:
        articles_df (pd.DataFrame): Article data
        transactions_df (pd.DataFrame): Transaction data
        article_id_map (dict): Mapping from article ID to index
        
    Returns:
        torch.Tensor: Article feature tensor
    """
    print("Computing enhanced article features...")
    
    article_ids = list(article_id_map.keys())
    articles_df_mapped = articles_df.set_index('article_id')
    
    # 1. Initialize feature dataframe with article IDs
    article_features_df = pd.DataFrame({"article_id": article_ids})
    
    # 2. Product type features - use one-hot encoding with dimensionality reduction
    prod_type_dummies = pd.get_dummies(
        articles_df_mapped.loc[article_ids, 'product_type_no'].fillna(-1),
        prefix='prod_type'
    )
    
    # Apply PCA to reduce dimensionality if there are many product types
    if prod_type_dummies.shape[1] > 20:
        pca = PCA(n_components=min(20, prod_type_dummies.shape[1] - 1))
        prod_type_reduced = pca.fit_transform(prod_type_dummies)
        prod_type_df = pd.DataFrame(
            prod_type_reduced, 
            columns=[f'prod_type_pca_{i}' for i in range(prod_type_reduced.shape[1])],
            index=article_ids
        )
    else:
        prod_type_df = prod_type_dummies
        prod_type_df.index = article_ids
    
    article_features_df = pd.merge(
        article_features_df,
        prod_type_df.reset_index(),
        left_on='article_id', 
        right_on='article_id'
    )
    
    # 3. Department features (higher-level category)
    dept_dummies = pd.get_dummies(
        articles_df_mapped.loc[article_ids, 'department_no'].fillna(-1),
        prefix='dept'
    )
    
    if dept_dummies.shape[1] > 10:
        pca = PCA(n_components=min(10, dept_dummies.shape[1] - 1))
        dept_reduced = pca.fit_transform(dept_dummies)
        dept_df = pd.DataFrame(
            dept_reduced, 
            columns=[f'dept_pca_{i}' for i in range(dept_reduced.shape[1])],
            index=article_ids
        )
    else:
        dept_df = dept_dummies
        dept_df.index = article_ids
    
    article_features_df = pd.merge(
        article_features_df,
        dept_df.reset_index(),
        left_on='article_id', 
        right_on='article_id'
    )
    
    # 4. Color features
    color_dummies = pd.get_dummies(
        articles_df_mapped.loc[article_ids, 'colour_group_name'].fillna('Unknown'),
        prefix='color'
    )
    
    if color_dummies.shape[1] > 10:
        pca = PCA(n_components=min(10, color_dummies.shape[1] - 1))
        color_reduced = pca.fit_transform(color_dummies)
        color_df = pd.DataFrame(
            color_reduced, 
            columns=[f'color_pca_{i}' for i in range(color_reduced.shape[1])],
            index=article_ids
        )
    else:
        color_df = color_dummies
        color_df.index = article_ids
    
    article_features_df = pd.merge(
        article_features_df,
        color_df.reset_index(),
        left_on='article_id', 
        right_on='article_id'
    )
    
    # 5. Text features from product name and description
    # Combine product name and description
    articles_df['text_features'] = articles_df['prod_name'].fillna('') + ' ' + articles_df['detail_desc'].fillna('')
    
    # Create TF-IDF features
    tfidf = TfidfVectorizer(max_features=20, stop_words='english')
    
    # Create a dictionary to map article_id to index for lookup
    article_texts = {article_id: articles_df.loc[articles_df['article_id'] == article_id, 'text_features'].iloc[0] 
                    if not articles_df.loc[articles_df['article_id'] == article_id, 'text_features'].empty 
                    else '' 
                    for article_id in article_ids}
    
    # Build corpus in the correct order
    text_corpus = [article_texts[article_id] for article_id in article_ids]
    
    # Check if we have any non-empty text
    if any(text for text in text_corpus):
        tfidf_matrix = tfidf.fit_transform(text_corpus)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=article_ids
        )
    else:
        # If no text, create dummy features
        tfidf_df = pd.DataFrame(
            np.zeros((len(article_ids), 5)),
            columns=[f'tfidf_{i}' for i in range(5)],
            index=article_ids
        )
    
    article_features_df = pd.merge(
        article_features_df,
        tfidf_df.reset_index(),
        left_on='article_id', 
        right_on='index'
    ).drop(columns=['index'])
    
    # 6. Popularity features from transactions
    popularity_data = []
    
    for article_id in article_ids:
        article_txn = transactions_df[transactions_df['article_id'] == article_id]
        
        if len(article_txn) > 0:
            # Number of purchases
            num_purchases = len(article_txn)
            
            # Number of unique customers
            num_customers = article_txn['customer_id'].nunique()
            
            # Average price
            avg_price = article_txn['price'].mean()
            
            # Recency: days since first and last purchase
            first_purchase = pd.to_datetime(article_txn['t_dat']).min()
            last_purchase = pd.to_datetime(article_txn['t_dat']).max()
            max_date = pd.to_datetime(transactions_df['t_dat']).max()
            
            days_since_first = (max_date - first_purchase).days
            days_since_last = (max_date - last_purchase).days
            
            # Sales velocity: purchases per day since introduction
            if days_since_first > 0:
                sales_velocity = num_purchases / days_since_first
            else:
                sales_velocity = num_purchases
                
            popularity_data.append({
                'article_id': article_id,
                'num_purchases': num_purchases,
                'num_customers': num_customers,
                'avg_price': avg_price,
                'days_since_first': days_since_first,
                'days_since_last': days_since_last,
                'sales_velocity': sales_velocity
            })
        else:
            # Default values for articles with no transactions
            popularity_data.append({
                'article_id': article_id,
                'num_purchases': 0,
                'num_customers': 0,
                'avg_price': articles_df.loc[articles_df['article_id'] == article_id, 'price'].mean() 
                            if 'price' in articles_df.columns else 0,
                'days_since_first': 999,
                'days_since_last': 999,
                'sales_velocity': 0
            })
    
    popularity_df = pd.DataFrame(popularity_data)
    
    # Normalize popularity values
    scaler = StandardScaler()
    pop_cols = ['num_purchases', 'num_customers', 'avg_price', 'days_since_first', 
                'days_since_last', 'sales_velocity']
    popularity_df[pop_cols] = scaler.fit_transform(popularity_df[pop_cols].fillna(0))
    
    article_features_df = pd.merge(article_features_df, popularity_df, on='article_id')
    
    # 7. Convert to tensor
    feature_cols = [col for col in article_features_df.columns if col != 'article_id']
    article_features = torch.tensor(article_features_df[feature_cols].fillna(0).values, dtype=torch.float)
    
    print(f"Created article features with shape: {article_features.shape}")
    return article_features

def load_filtered_data(data_dir="./data", min_cooccur=3, decay_rate=0.005):
    """
    Load filtered dataset and create a heterogeneous graph for LightGCN.
    
    Args:
        data_dir (str): Directory containing the filtered CSV files
        min_cooccur (int): Minimum number of co-occurrences to consider an article-article edge
        decay_rate (float): Time decay rate for customer-article interactions
        
    Returns:
        Data, dict, nx.Graph: PyTorch Geometric Data object, metadata dictionary, and NetworkX graph
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
    src_co, dst_co, co_weights = compute_cooccurrence(
        transactions_df, 
        article_id_map,
        customer_id_map,
        min_cooccur=min_cooccur
    )
    
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

    # Create NetworkX graph for visualization
    G = create_networkx_graph_enhanced(data, meta)
    
    return data, meta, G

def create_networkx_graph_enhanced(data, meta):
    """
    Convert PyG data to NetworkX graph with enhanced node and edge attributes.
    
    Args:
        data (torch_geometric.data.Data): PyG data object
        meta (dict): Metadata dictionary
        
    Returns:
        nx.Graph: NetworkX graph with rich node and edge attributes
    """
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy() if data.edge_attr is not None else None
    edge_type = data.edge_type.cpu().numpy() if hasattr(data, 'edge_type') else None
    
    num_customers = meta["num_customers"]
    num_articles = meta["num_articles"]
    reverse_customer_map = meta["reverse_customer_map"]
    reverse_article_map = meta["reverse_article_map"]
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add customer nodes with enhanced attributes
    for i in range(num_customers):
        original_id = reverse_customer_map[i]
        G.add_node(i, 
                   id=str(original_id), 
                   type='customer', 
                   label=f'C_{original_id}')
    
    # Add article nodes with enhanced attributes
    for i in range(num_articles):
        node_idx = i + num_customers
        original_id = reverse_article_map[i]
        G.add_node(node_idx, 
                   id=str(original_id), 
                   type='article', 
                   label=f'A_{original_id}')
    
    # Add edges with enhanced attributes
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        weight = float(edge_attr[i]) if edge_attr is not None else 1.0
        
        # Determine edge type
        if edge_type is not None:
            edge_type_val = int(edge_type[i])
            if edge_type_val == 0:
                edge_type_str = 'purchase'
            elif edge_type_val == 1:
                edge_type_str = 'co_occurrence'
            elif edge_type_val == 2:
                edge_type_str = 'similarity'
            else:
                edge_type_str = 'unknown'
        else:
            # Fallback to previous logic
            if src < num_customers and dst >= num_customers:
                edge_type_str = 'buys'
            elif src >= num_customers and dst < num_customers:
                edge_type_str = 'bought_by'
            else:
                edge_type_str = 'bought_together'
        
        G.add_edge(int(src), int(dst), weight=weight, type=edge_type_str)
    
    return G

def save_graph_as_gml(G, output_path='graph.gml', create_subgraph=True, max_nodes=1000):
    """
    Save the NetworkX graph to a GML file with optional subgraph creation.
    
    Args:
        G (nx.Graph): NetworkX graph
        output_path (str): Path to save the GML file
        create_subgraph (bool): Whether to create a smaller subgraph
        max_nodes (int): Maximum number of nodes in the subgraph
    """
    # Ensure all node ids are strings (GML requirement)
    for n, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, (int, float, bool)):
                data[key] = str(value)
    
    # Ensure edge weights are properly formatted
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, (int, float, bool)):
                data[key] = str(value)
    
    # Save the complete graph
    nx.write_gml(G, output_path)
    print(f"Graph saved as GML file: {output_path}")
    
    # Save a smaller subgraph if requested
    if create_subgraph and G.number_of_nodes() > max_nodes:
        # Create a more meaningful subgraph by selecting:
        # 1. Top N/2 customers by degree (most active)
        # 2. Top N/2 articles connected to those customers
        
        # Get customers sorted by degree
        customers = [n for n, data in G.nodes(data=True) if data.get('type') == 'customer']
        customer_degrees = sorted([(n, G.degree(n)) for n in customers], key=lambda x: x[1], reverse=True)
        
        # Take top customers
        top_customers = [n for n, _ in customer_degrees[:max_nodes//2]]
        
        # Get articles connected to these customers
        connected_articles = set()
        for cust in top_customers:
            for neighbor in G.neighbors(cust):
                if G.nodes[neighbor].get('type') == 'article':
                    connected_articles.add(neighbor)
        
        # Take top articles by degree if we have too many
        if len(connected_articles) > max_nodes//2:
            article_degrees = sorted([(n, G.degree(n)) for n in connected_articles], key=lambda x: x[1], reverse=True)
            top_articles = [n for n, _ in article_degrees[:max_nodes//2]]
        else:
            top_articles = list(connected_articles)
        
        # Combine for subgraph nodes
        subgraph_nodes = top_customers + top_articles
        subgraph = G.subgraph(subgraph_nodes)
        
        # Save the subgraph
        subgraph_path = output_path.replace('.gml', '_subgraph.gml')
        nx.write_gml(subgraph, subgraph_path)
        print(f"Subgraph with {len(subgraph_nodes)} nodes saved as: {subgraph_path}")
        
        # Save article-only subgraph for product relationships
        article_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'article']
        if len(article_nodes) > max_nodes:
            article_degrees = sorted([(n, G.degree(n)) for n in article_nodes], key=lambda x: x[1], reverse=True)
            top_articles_only = [n for n, _ in article_degrees[:max_nodes]]
            article_subgraph = G.subgraph(top_articles_only)
        else:
            article_subgraph = G.subgraph(article_nodes)
            
        article_subgraph_path = output_path.replace('.gml', '_articles.gml')
        nx.write_gml(article_subgraph, article_subgraph_path)
        print(f"Article subgraph with {article_subgraph.number_of_nodes()} nodes saved as: {article_subgraph_path}")

def create_train_test_split(data, meta, test_ratio=0.2, by_time=True):
    """
    Create train/test split for evaluation with enhanced temporal considerations.
    
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
    import argparse
    import pickle
    
    parser = argparse.ArgumentParser(description='Load and process H&M dataset for graph-based recommendation')
    parser.add_argument('--data_dir', type=str, default='./data', 
                        help='Directory containing the filtered CSV files')
    parser.add_argument('--min_cooccur', type=int, default=3, 
                        help='Minimum co-occurrences to create an article-article edge')
    parser.add_argument('--decay_rate', type=float, default=0.005, 
                        help='Decay rate for time-based edge weights')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save processed data (defaults to data_dir/processed)')
    parser.add_argument('--gml_path', type=str, default='recommendation_graph.gml', 
                        help='Path to save the GML graph file')
    parser.add_argument('--max_nodes', type=int, default=1000, 
                        help='Maximum number of nodes in the subgraph')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "processed")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir} with min_cooccur={args.min_cooccur}, decay_rate={args.decay_rate}")
    data, meta, G = load_filtered_data(
        data_dir=args.data_dir, 
        min_cooccur=args.min_cooccur, 
        decay_rate=args.decay_rate
    )

    # Add this right after loading the data
    print("\nSaving graph visualization files...")
    gml_path = os.path.join(args.output_dir, args.gml_path)
    print(f"Will save GML to: {gml_path}")
    try:
        save_graph_as_gml(G, gml_path, create_subgraph=True, max_nodes=args.max_nodes)
    except Exception as e:
        print(f"ERROR saving GML: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Created PyG Data object with {data.num_nodes} nodes and {data.num_edges} edges")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Full debugging of GML save
    try:
        gml_path = os.path.join(args.output_dir, args.gml_path)
        print(f"Will save GML to: {gml_path}")
        save_graph_as_gml(G, gml_path, create_subgraph=True, max_nodes=args.max_nodes)
        print(f"GML save completed successfully")
    except Exception as e:
        print(f"ERROR saving GML: {str(e)}")
        import traceback
        traceback.print_exc()
    # Create train/test split
    train_user_items, test_user_items = create_train_test_split(data, meta, test_ratio=0.2)
    
    # Save PyG data
    torch.save(data, os.path.join(args.output_dir, "lightgcn_data.pt"))
    print(f"Saved PyG data to {os.path.join(args.output_dir, 'lightgcn_data.pt')}")
    
    # Save metadata with train/test split
    meta_with_split = meta.copy()
    meta_with_split.update({
        "train_user_items": train_user_items,
        "test_user_items": test_user_items,
        "data_dir": args.data_dir
    })
    
    with open(os.path.join(args.output_dir, "lightgcn_meta.pkl"), "wb") as f:
        pickle.dump(meta_with_split, f)
    print(f"Saved metadata to {os.path.join(args.output_dir, 'lightgcn_meta.pkl')}")
    
    # Save ID mappings separately
    mappings = {
        "customer_id_map": meta["customer_id_map"],
        "article_id_map": meta["article_id_map"],
        "reverse_customer_map": meta["reverse_customer_map"],
        "reverse_article_map": meta["reverse_article_map"]
    }
    with open(os.path.join(args.output_dir, "id_mappings.pkl"), "wb") as f:
        pickle.dump(mappings, f)
    print(f"Saved ID mappings to {os.path.join(args.output_dir, 'id_mappings.pkl')}")
    
    # Print data statistics
    num_customers = meta["num_customers"]
    num_articles = meta["num_articles"]
    
    print(f"\nData statistics:")
    print(f"- {num_customers} customers")
    print(f"- {num_articles} articles")
    print(f"- {data.num_edges} total edges")
    print(f"- {len(train_user_items)} users in training set")
    print(f"- {len(test_user_items)} users in test set")
    
    # Print graph visualization information
    print(f"\nGraph visualization files:")
    print(f"- Full graph: {gml_path}")
    print(f"- Subgraph: {gml_path.replace('.gml', '_subgraph.gml')}")
    print(f"- Article graph: {gml_path.replace('.gml', '_articles.gml')}")