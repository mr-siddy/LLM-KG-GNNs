"""
model.py

Updated LightGCN model that can optionally use edge weights during message passing.
If edge weights are provided (e.g., time decay or co-occurrence), they are used in message aggregation.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree

class EnhancedLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(self.num_nodes, embed_dim)
        
        # Add edge type encodings for heterogeneous edges
        self.edge_type_emb = nn.Embedding(3, embed_dim)  # 3 types: buys, co-occurrence, country
        
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.edge_type_emb.weight)
    
    def forward(self, edge_index, edge_weight=None, edge_type=None):
        x0 = self.embeddings.weight  # [N, d]
        row, col = edge_index
        deg = degree(row, x0.size(0), dtype=x0.dtype)
        deg_sqrt_inv = torch.pow(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0
        
        H_l = x0
        out = x0 / (self.num_layers + 1)
        
        for _ in range(self.num_layers):
            if edge_weight is not None:
                if edge_type is not None:
                    # Add edge type influence
                    edge_type_influence = self.edge_type_emb(edge_type)
                    msg = H_l[row] * edge_weight.view(-1, 1) + edge_type_influence
                else:
                    msg = H_l[row] * edge_weight.view(-1, 1)
            else:
                msg = H_l[row]
            
            norm = deg_sqrt_inv[row] * deg_sqrt_inv[col]
            msg = msg * norm.view(-1, 1)
            
            H_next = scatter_add(msg, col, dim=0, dim_size=x0.size(0))
            H_l = H_next
            out += H_l / (self.num_layers + 1)
            
        return out
    
    def recommend(self, user_ids, edge_index, edge_weight, top_k=10):
        all_emb = self.forward(edge_index, edge_weight)
        user_emb = all_emb[user_ids]
        item_emb = all_emb[self.num_users:]
        scores = torch.matmul(user_emb, item_emb.t())
        _, topk_indices = torch.topk(scores, top_k, dim=1)
        return topk_indices

if __name__ == "__main__":
    # Dummy test.
    num_users, num_items = 100, 200
    import torch_geometric
    user_indices = torch.randint(0, num_users, (500,))
    item_indices = torch.randint(0, num_items, (500,))
    global_item_indices = item_indices + num_users
    edge_index = torch.tensor([user_indices, global_item_indices], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_weight = torch.ones(edge_index.size(1))
    model = LightGCN(num_users, num_items)
    embeddings = model(edge_index, edge_weight)
    print("Embeddings shape:", embeddings.shape)
