"""
inference.py

Loads the trained model and produces top-K recommendations for a given user.
Supports both EnhancedLightGCN and standard LightGCN models.
"""

import os
import torch
import pickle
import argparse
from model import EnhancedLightGCN, LightGCN

def load_model(model_path, num_users, num_items, embed_dim=64, num_layers=3, model_type="enhanced", device="cpu"):
    if model_type == "enhanced":
        model = EnhancedLightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
    else:
        model = LightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
    print(f"Loading model from {model_path}")
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
    """
    device = next(model.parameters()).device
    
    if customer_id not in meta["customer_id_map"]:
        print("Customer ID not found.")
        return []
        
    u_idx = meta["customer_id_map"][customer_id]
    num_users = meta["num_customers"]
    
    with torch.no_grad():
        all_emb = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    user_emb = all_emb[u_idx].unsqueeze(0)
    item_emb = all_emb[num_users:]
    
    scores = torch.matmul(user_emb, item_emb.t()).squeeze(0)
    _, topk_idx = torch.topk(scores, top_k)
    topk_idx = topk_idx.cpu().tolist()
    
    product_id_map = meta["product_id_map"]
    reverse_product_map = {v: k for k, v in product_id_map.items()}
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for LightGCN models")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                        help="Directory containing the processed data files (lightgcn_data.pt, lightgcn_meta.pkl)")
    parser.add_argument("--model_dir", type=str, default="./output",
                        help="Directory containing the trained model")
    parser.add_argument("--model_file", type=str, default="standard_lightgcn_best.pth",
                        help="Model file name (e.g. enhanced_lightgcn_best.pth or standard_lightgcn_best.pth)")
    parser.add_argument("--model_type", type=str, default="standard", choices=["enhanced", "standard"],
                        help="Which model to use for inference (enhanced or standard)")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimension (must match trained model)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of LightGCN layers (must match trained model)")
    parser.add_argument("--user_id", type=str, default=None,
                        help="User ID for which to generate recommendations (if not provided, the first user is used)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations to generate")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load processed data from the given directory.
    data_path = os.path.join(args.data_dir, "lightgcn_data.pt")
    meta_path = os.path.join(args.data_dir, "lightgcn_meta.pkl")
    if not os.path.exists(data_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Processed data files not found. Check your data_dir argument.")
    data = torch.load(data_path, map_location=device)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]
    
    # Construct full model path.
    model_path = os.path.join(args.model_dir, args.model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path, num_users, num_items, embed_dim=args.embed_dim, num_layers=args.num_layers, model_type=args.model_type, device=device)
    
    # Select a user for inference.
    if args.user_id is None:
        test_user_id = list(meta["customer_id_map"].keys())[0]
        print(f"No user_id provided. Using the first available user: {test_user_id}")
    else:
        test_user_id = args.user_id
    
    recs = get_recommendations_for_user(model, data, meta, test_user_id, top_k=args.top_k)
    print(f"Recommendations for user {test_user_id}:", recs)

if __name__ == "__main__":
    main()
