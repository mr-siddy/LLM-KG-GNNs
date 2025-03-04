"""
model.py

Enhanced LightGCN model that supports edge weights, edge types, and heterogeneous graphs.
This model can handle different types of edges (purchases, co-occurrences, etc.) and
incorporates edge weights during message passing.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.utils import degree

class LightGCN(nn.Module):
    """
    Standard LightGCN model for backward compatibility with existing code.
    """
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(self.num_nodes, embed_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, edge_index, edge_weight=None):
        x0 = self.embeddings.weight  # [N, d]
        row, col = edge_index
        deg = degree(row, x0.size(0), dtype=x0.dtype)
        deg_sqrt_inv = torch.pow(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0
        
        H_l = x0
        out = x0 / (self.num_layers + 1)
        for _ in range(self.num_layers):
            # If edge_weight is provided, incorporate it.
            if edge_weight is not None:
                norm = deg_sqrt_inv[row] * deg_sqrt_inv[col] * edge_weight
            else:
                norm = deg_sqrt_inv[row] * deg_sqrt_inv[col]
            msg = H_l[row] * norm.view(-1, 1)
            H_next = scatter_add(msg, col, dim=0, dim_size=x0.size(0))
            H_l = H_next
            out += H_l / (self.num_layers + 1)
        return out
    
    def recommend(self, user_ids, edge_index, edge_weight=None, top_k=10):
        all_emb = self.forward(edge_index, edge_weight)
        user_emb = all_emb[user_ids]
        item_emb = all_emb[self.num_users:]
        scores = torch.matmul(user_emb, item_emb.t())
        _, topk_indices = torch.topk(scores, top_k, dim=1)
        return topk_indices


class EnhancedLightGCN(nn.Module):
    """
    Enhanced LightGCN model with edge type embeddings and improved message passing.
    """
    def __init__(self, num_users, num_items, embed_dim=64, num_layers=3, num_edge_types=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        
        # Node embeddings
        self.embeddings = nn.Embedding(self.num_nodes, embed_dim)
        
        # Edge type embeddings (0: purchase, 1: co-occurrence, 2: other custom edges)
        self.edge_type_emb = nn.Embedding(num_edge_types, embed_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.edge_type_emb.weight)
    
    def forward(self, edge_index, edge_weight=None, edge_type=None):
        """
        Forward pass for the enhanced LightGCN model.
        
        Args:
            edge_index: Tensor of shape [2, num_edges] with source and target nodes
            edge_weight: Optional tensor of shape [num_edges] with edge weights
            edge_type: Optional tensor of shape [num_edges] with edge type indices
            
        Returns:
            Tensor of shape [num_nodes, embed_dim] with node embeddings
        """
        x0 = self.embeddings.weight  # [N, d]
        row, col = edge_index
        deg = degree(row, x0.size(0), dtype=x0.dtype)
        deg_sqrt_inv = torch.pow(deg, -0.5)
        deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0
        
        H_l = x0
        out = x0 / (self.num_layers + 1)
        
        for _ in range(self.num_layers):
            # Prepare messages
            if edge_type is not None:
                # Include edge type influence
                edge_influence = self.edge_type_emb(edge_type)
                msg = H_l[row] + edge_influence * 0.1  # Scale down edge type influence
            else:
                msg = H_l[row]
            
            # Apply edge weights if provided
            if edge_weight is not None:
                msg = msg * edge_weight.view(-1, 1)
            
            # Apply graph normalization
            norm = deg_sqrt_inv[row] * deg_sqrt_inv[col]
            msg = msg * norm.view(-1, 1)
            
            # Aggregate messages
            H_next = scatter_add(msg, col, dim=0, dim_size=x0.size(0))
            H_l = H_next
            out += H_l / (self.num_layers + 1)
            
        return out
    
    def recommend(self, user_ids, edge_index, edge_weight=None, edge_type=None, top_k=10):
        """
        Generate top-k recommendations for given users.
        
        Args:
            user_ids: Tensor of user indices
            edge_index: Graph edge indices
            edge_weight: Optional edge weights
            edge_type: Optional edge types
            top_k: Number of recommendations to generate
            
        Returns:
            Tensor of shape [len(user_ids), top_k] with recommended item indices
        """
        all_emb = self.forward(edge_index, edge_weight, edge_type)
        user_emb = all_emb[user_ids]
        item_emb = all_emb[self.num_users:]
        scores = torch.matmul(user_emb, item_emb.t())
        _, topk_indices = torch.topk(scores, top_k, dim=1)
        return topk_indices


if __name__ == "__main__":
    # Test both models with dummy data
    num_users, num_items = 100, 200
    
    # Create dummy edges (user->item)
    user_indices = torch.randint(0, num_users, (500,))
    item_indices = torch.randint(0, num_items, (500,))
    global_item_indices = item_indices + num_users
    edge_index = torch.tensor([user_indices, global_item_indices], dtype=torch.long)
    
    # Add reverse edges to make graph undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create dummy edge weights and types
    edge_weight = torch.rand(edge_index.size(1))
    edge_type = torch.randint(0, 3, (edge_index.size(1),))
    
    # Test LightGCN
    model1 = LightGCN(num_users, num_items, embed_dim=64, num_layers=3)
    embeddings1 = model1(edge_index, edge_weight)
    print("LightGCN embeddings shape:", embeddings1.shape)
    
    # Test EnhancedLightGCN
    model2 = EnhancedLightGCN(num_users, num_items, embed_dim=64, num_layers=3)
    embeddings2 = model2(edge_index, edge_weight, edge_type)
    print("EnhancedLightGCN embeddings shape:", embeddings2.shape)
    
    # Test recommendations
    test_users = torch.tensor([0, 1, 2])
    recs1 = model1.recommend(test_users, edge_index, edge_weight)
    recs2 = model2.recommend(test_users, edge_index, edge_weight, edge_type)
    
    print("LightGCN recommendations shape:", recs1.shape)
    print("EnhancedLightGCN recommendations shape:", recs2.shape)