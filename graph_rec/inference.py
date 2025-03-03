"""
inference.py

Loads the trained model and produces top-K recommendations for a given user.
"""

import torch
from data_loader import load_data
from model import LightGCN

def load_model(model_path, num_users, num_items, embed_dim=64, num_layers=3, device="cpu"):
    model = LightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_recommendations_for_user(model, data, meta, user_id, top_k=10):
    device = next(model.parameters()).device
    num_users = meta["num_customers"]
    if user_id not in meta["customer_id_map"]:
        print("User ID not found.")
        return []
    u_idx = meta["customer_id_map"][user_id]
    with torch.no_grad():
        all_emb = model(data.edge_index.to(device), data.edge_attr.to(device))
    user_emb = all_emb[u_idx].unsqueeze(0)
    item_emb = all_emb[num_users:]
    scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)
    _, topk_idx = torch.topk(scores, top_k)
    topk_idx = topk_idx.cpu().tolist()
    
    article_id_map = meta["article_id_map"]
    reverse_article_map = {v: k for k, v in article_id_map.items()}
    recommendations = [reverse_article_map[i] for i in topk_idx]
    return recommendations

def get_retail_recommendations(model, data, meta, customer_id, top_k=10):
    """
    Generate retail recommendations with additional context.
    
    Args:
        model: Trained recommendation model
        data: PyG Data object
        meta: Metadata dictionary
        customer_id: ID of customer to recommend for
        top_k: Number of recommendations to generate
    
    Returns:
        list: Recommendations with context (product ID, description, price)
    """
    device = next(model.parameters()).device
    
    if customer_id not in meta["customer_id_map"]:
        print("Customer ID not found.")
        return []
        
    u_idx = meta["customer_id_map"][customer_id]
    num_users = meta["num_customers"]
    
    # Get embeddings
    with torch.no_grad():
        all_emb = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    user_emb = all_emb[u_idx].unsqueeze(0)
    item_emb = all_emb[num_users:]
    
    # Calculate scores
    scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)
    _, topk_idx = torch.topk(scores, top_k)
    topk_idx = topk_idx.cpu().tolist()
    
    # Map back to original product IDs
    product_id_map = meta["product_id_map"]
    reverse_product_map = {v: k for k, v in product_id_map.items()}
    
    # Get product details from original data
    recommendations = []
    for i in topk_idx:
        product_id = reverse_product_map[i]
        product_info = meta.get("product_details", {}).get(product_id, {})
        recommendations.append({
            "product_id": product_id,
            "description": product_info.get("description", "Unknown"),
            "price": product_info.get("price", 0.0),
            "score": scores[i].item()
        })
    
    return recommendations

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, meta = load_data()
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]
    model = load_model("model.pth", num_users, num_items, embed_dim=EMBED_DIM, num_layers=NUM_LAYERS, device=device)
    test_user_id = list(meta["customer_id_map"].keys())[0]
    recs = get_recommendations_for_user(model, data, meta, test_user_id, top_k=10)
    print(f"Recommendations for user {test_user_id}:", recs)
