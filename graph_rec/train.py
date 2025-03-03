"""
train.py

Training loop for the LightGCN model.
Includes periodic visualization of learned embeddings.
"""

import torch
import torch.optim as optim
import random
from collections import defaultdict

from data_loader import load_data
from model import LightGCN
from visualization import visualize_embeddings

# Hyperparameters
EMBED_DIM = 64
NUM_LAYERS = 3
LR = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 10      # Increase for real training
BATCH_SIZE = 1024

# Adjust negative sampling to reflect retail purchase patterns
def sample_negatives_for_retail(transactions_df, customer_id_map, product_id_map, n_samples=5):
    neg_samples = []
    
    # Group by customer
    customer_groups = transactions_df.groupby('CustomerID')
    
    for cid, group in customer_groups:
        if cid not in customer_id_map:
            continue
            
        user_idx = customer_id_map[cid]
        # Get products this customer has purchased
        purchased = set(group['StockCode'].unique())
        
        # Get products in same price range but not purchased
        avg_price = group['UnitPrice'].mean()
        price_range = (avg_price * 0.7, avg_price * 1.3)
        
        # Find candidate products in similar price range not purchased by this user
        candidates = transactions_df[
            (transactions_df['UnitPrice'] >= price_range[0]) & 
            (transactions_df['UnitPrice'] <= price_range[1]) &
            ~transactions_df['StockCode'].isin(purchased)
        ]['StockCode'].unique()
        
        # If not enough candidates, fall back to random sampling
        if len(candidates) < n_samples:
            all_products = list(product_id_map.keys())
            available = list(set(all_products) - purchased)
            if available:
                candidates = np.random.choice(available, 
                                           size=min(n_samples, len(available)), 
                                           replace=False)
        
        # Add negative samples
        for pid in candidates[:n_samples]:
            if pid in product_id_map:
                neg_samples.append((user_idx, product_id_map[pid]))
    
    return neg_samples

def train():
    data, meta = load_data()
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]
    user_to_items = meta["user_to_items"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    
    model = LightGCN(num_users, num_items, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    all_users = list(user_to_items.keys())
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        random.shuffle(all_users)
        for batch_start in range(0, len(all_users), BATCH_SIZE):
            batch_users = all_users[batch_start: batch_start + BATCH_SIZE]
            pos_items = []
            neg_items = []
            valid_users = []
            for u in batch_users:
                if len(user_to_items[u]) == 0:
                    continue
                pos = random.choice(user_to_items[u])
                neg = random.randrange(num_items)
                while neg in user_to_items[u]:
                    neg = random.randrange(num_items)
                valid_users.append(u)
                pos_items.append(pos)
                neg_items.append(neg)
            if len(valid_users) == 0:
                continue
            valid_users = torch.tensor(valid_users, dtype=torch.long, device=device)
            pos_items  = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_items  = torch.tensor(neg_items, dtype=torch.long, device=device)
            
            all_emb = model(data.edge_index, data.edge_attr)
            u_emb = all_emb[valid_users]
            pos_emb = all_emb[num_users + pos_items]
            neg_emb = all_emb[num_users + neg_items]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            reg_loss = (u_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2)) / len(valid_users)
            loss = loss + 1e-4 * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss:.4f}")
        
        # Periodically visualize embeddings.
        model.eval()
        with torch.no_grad():
            all_emb = model(data.edge_index, data.edge_attr)
        visualize_embeddings(all_emb, method="umap", title=f"Epoch {epoch+1} Embeddings (UMAP)")
    
    torch.save(model.state_dict(), "model.pth")
    print("Training complete and model saved as 'model.pth'.")

if __name__ == "__main__":
    train()
