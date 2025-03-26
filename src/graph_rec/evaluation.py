"""
evaluation.py

Evaluation metrics and routines to assess recommendation performance.
"""

import math
import torch

def recall_at_k(recommended, ground_truth, k):
    recommended_at_k = recommended[:k]
    hit_count = len(set(recommended_at_k) & ground_truth)
    return hit_count / len(ground_truth) if ground_truth else 0

def ndcg_at_k(recommended, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)
    idcg = 1.0
    return dcg / idcg

def revenue_at_k(recommended, ground_truth, item_prices, k):
    """
    Calculates potential revenue from top-k recommendations.
    
    Args:
        recommended: List of recommended item indices
        ground_truth: Set of actually purchased items
        item_prices: Dictionary mapping item indices to prices
        k: Number of top items to consider
    
    Returns:
        float: Potential revenue from the recommendations
    """
    recommended_at_k = recommended[:k]
    hit_items = set(recommended_at_k) & ground_truth
    
    if not hit_items:
        return 0.0
        
    revenue = sum(item_prices.get(item, 0) for item in hit_items)
    return revenue

def diversity_at_k(recommended, item_categories, k):
    """
    Measures diversity of recommendations based on product categories.
    
    Args:
        recommended: List of recommended item indices
        item_categories: Dictionary mapping item indices to categories
        k: Number of top items to consider
    
    Returns:
        float: Diversity score (0-1) with 1 being most diverse
    """
    recommended_at_k = recommended[:k]
    categories = [item_categories.get(item, "unknown") for item in recommended_at_k]
    unique_categories = len(set(categories))
    
    if k == 0:
        return 0.0
        
    return unique_categories / min(k, len(set(item_categories.values())))

def evaluate_model(model, data, meta, test_user_items, top_k=10):
    num_users = meta["num_customers"]
    model.eval()
    with torch.no_grad():
        all_emb = model(data.edge_index, data.edge_attr)
    user_embs = all_emb[:num_users]
    item_embs = all_emb[num_users:]
    
    recalls = []
    ndcgs = []
    for u, true_items in test_user_items.items():
        u_emb = user_embs[u].unsqueeze(0)
        scores = torch.matmul(u_emb, item_embs.t()).squeeze(0)
        _, topk_idx = torch.topk(scores, top_k)
        recommended = topk_idx.cpu().tolist()
        recalls.append(recall_at_k(recommended, true_items, top_k))
        ndcgs.append(ndcg_at_k(recommended, true_items, top_k))
    
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    return avg_recall, avg_ndcg


