"""
generate_recommendations.py

This script loads a trained LightGCN model and generates recommendations for users.
It can generate recommendations for:
1. A specific user by ID
2. A batch of users
3. All users (to create a recommendation file)

The script also includes functions to explain recommendations based on:
- Similar users who purchased the item
- Similar items the user has purchased
"""

import os
import torch
import pickle
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
from tqdm import tqdm

# Import our modules
from model import LightGCN
from modified_data_loader import load_filtered_data

def load_model_and_data(args):
    """
    Load trained model and data.
    
    Args:
        args: Command line arguments
        
    Returns:
        model: Trained LightGCN model
        data: PyTorch Geometric data object
        meta: Metadata dictionary
    """
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    if os.path.exists(os.path.join(args.data_dir, "processed", "lightgcn_data.pt")):
        print("Loading processed data...")
        data = torch.load(os.path.join(args.data_dir, "processed", "lightgcn_data.pt"), map_location=device)
        with open(os.path.join(args.data_dir, "processed", "lightgcn_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
    else:
        print("Processing data from CSV files...")
        data, meta = load_filtered_data(data_dir=args.data_dir)
    
    data = data.to(device)
    
    # Load model
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]
    
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers
    ).to(device)
    
    model_path = os.path.join(args.model_dir, args.model_file)
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    return model, data, meta

def get_recommendations_for_user(model, data, meta, customer_id, top_k=10, exclude_purchased=True):
    """
    Generate recommendations for a specific user.
    
    Args:
        model: Trained LightGCN model
        data: PyTorch Geometric data object
        meta: Metadata dictionary
        customer_id: Customer ID (string)
        top_k: Number of recommendations to generate
        exclude_purchased: Whether to exclude items the user has already purchased
        
    Returns:
        recommendations: List of recommended article IDs
        scores: Corresponding scores
    """
    device = next(model.parameters()).device
    num_users = meta["num_customers"]
    customer_id_map = meta["customer_id_map"]
    
    if customer_id not in customer_id_map:
        print(f"Customer ID {customer_id} not found in the dataset.")
        return [], []
    
    user_idx = customer_id_map[customer_id]
    
    with torch.no_grad():
        all_embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    user_embedding = all_embeddings[user_idx].unsqueeze(0)
    item_embeddings = all_embeddings[num_users:]
    
    scores = torch.matmul(user_embedding, item_embeddings.t()).squeeze(0)
    
    # Exclude already purchased items if requested
    if exclude_purchased:
        purchased_items = meta.get("user_to_items", {}).get(user_idx, [])
        for item_idx in purchased_items:
            scores[item_idx] = -float('inf')
    
    # Get top-k recommendations
    scores, indices = torch.topk(scores, k=min(top_k, len(scores)))
    scores = scores.cpu().numpy()
    indices = indices.cpu().numpy()
    
    # Convert to article IDs
    reverse_article_map = meta["reverse_article_map"]
    recommendations = [reverse_article_map[idx] for idx in indices]
    
    return recommendations, scores

def get_recommendations_batch(model, data, meta, customer_ids, top_k=10, exclude_purchased=True):
    """
    Generate recommendations for a batch of users.
    
    Args:
        model: Trained LightGCN model
        data: PyTorch Geometric data object
        meta: Metadata dictionary
        customer_ids: List of customer IDs
        top_k: Number of recommendations to generate
        exclude_purchased: Whether to exclude items the user has already purchased
        
    Returns:
        recommendations_dict: Dictionary mapping customer IDs to their recommendations
    """
    device = next(model.parameters()).device
    num_users = meta["num_customers"]
    customer_id_map = meta["customer_id_map"]
    reverse_article_map = meta["reverse_article_map"]
    
    # Filter valid customer IDs
    valid_customers = [cid for cid in customer_ids if cid in customer_id_map]
    if not valid_customers:
        print("No valid customer IDs provided.")
        return {}
    
    user_indices = [customer_id_map[cid] for cid in valid_customers]
    user_indices_tensor = torch.tensor(user_indices, dtype=torch.long, device=device)
    
    with torch.no_grad():
        all_embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    user_embeddings = all_embeddings[user_indices_tensor]
    item_embeddings = all_embeddings[num_users:]
    
    # Calculate scores for all users and items
    scores = torch.matmul(user_embeddings, item_embeddings.t())
    
    recommendations_dict = {}
    for i, cid in enumerate(valid_customers):
        user_scores = scores[i]
        
        # Exclude already purchased items if requested
        if exclude_purchased:
            user_idx = customer_id_map[cid]
            purchased_items = meta.get("user_to_items", {}).get(user_idx, [])
            for item_idx in purchased_items:
                user_scores[item_idx] = -float('inf')
        
        # Get top-k recommendations
        user_scores, indices = torch.topk(user_scores, k=min(top_k, len(user_scores)))
        user_scores = user_scores.cpu().numpy()
        indices = indices.cpu().numpy()
        
        # Convert to article IDs
        recommendations = [reverse_article_map[idx] for idx in indices]
        recommendations_dict[cid] = recommendations
    
    return recommendations_dict

def explain_recommendation(model, data, meta, customer_id, article_id, n_similar=5):
    """
    Explain why an item was recommended to a user.
    
    Args:
        model: Trained LightGCN model
        data: PyTorch Geometric data object
        meta: Metadata dictionary
        customer_id: Customer ID
        article_id: Article ID to explain
        n_similar: Number of similar users/items to consider
        
    Returns:
        explanation: Dictionary with explanation information
    """
    device = next(model.parameters()).device
    customer_id_map = meta["customer_id_map"]
    article_id_map = meta["article_id_map"]
    reverse_customer_map = meta["reverse_customer_map"]
    reverse_article_map = meta["reverse_article_map"]
    num_users = meta["num_customers"]
    
    if customer_id not in customer_id_map or article_id not in article_id_map:
        print("Customer ID or Article ID not found.")
        return {}
    
    user_idx = customer_id_map[customer_id]
    item_idx = article_id_map[article_id]
    
    with torch.no_grad():
        all_embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    user_embedding = all_embeddings[user_idx]
    item_embedding = all_embeddings[num_users + item_idx]
    all_user_embeddings = all_embeddings[:num_users]
    all_item_embeddings = all_embeddings[num_users:]
    
    # Find similar users who might have purchased this item
    user_similarities = torch.matmul(user_embedding.unsqueeze(0), all_user_embeddings.t()).squeeze(0)
    user_similarities[user_idx] = -float('inf')  # Exclude the user themselves
    _, similar_user_indices = torch.topk(user_similarities, k=min(n_similar*3, len(user_similarities)))
    similar_user_indices = similar_user_indices.cpu().numpy()
    
    # Find similar items the user has purchased
    item_similarities = torch.matmul(item_embedding.unsqueeze(0), all_item_embeddings.t()).squeeze(0)
    item_similarities[item_idx] = -float('inf')  # Exclude the item itself
    _, similar_item_indices = torch.topk(item_similarities, k=min(n_similar*3, len(item_similarities)))
    similar_item_indices = similar_item_indices.cpu().numpy()
    
    # Get purchased items for similar users
    purchased_by_similar_users = []
    for sim_user_idx in similar_user_indices:
        sim_user_purchases = meta.get("user_to_items", {}).get(sim_user_idx.item(), [])
        if item_idx in sim_user_purchases:
            purchased_by_similar_users.append(sim_user_idx.item())
            if len(purchased_by_similar_users) >= n_similar:
                break
    
    # Get similar items purchased by the user
    user_purchases = meta.get("user_to_items", {}).get(user_idx, [])
    similar_items_purchased = []
    for sim_item_idx in similar_item_indices:
        if sim_item_idx.item() in user_purchases:
            similar_items_purchased.append(sim_item_idx.item())
            if len(similar_items_purchased) >= n_similar:
                break
    
    explanation = {
        "customer_id": customer_id,
        "article_id": article_id,
        "similarity_score": torch.dot(user_embedding, item_embedding).item(),
        "similar_users": [reverse_customer_map[idx] for idx in purchased_by_similar_users],
        "similar_items": [reverse_article_map[idx] for idx in similar_items_purchased]
    }
    
    return explanation

def generate_all_recommendations(model, data, meta, output_file, top_k=10, batch_size=100):
    """
    Generate recommendations for all users and save to a file.
    
    Args:
        model: Trained LightGCN model
        data: PyTorch Geometric data object
        meta: Metadata dictionary
        output_file: Path to save the recommendations
        top_k: Number of recommendations per user
        batch_size: Batch size for processing
    """
    device = next(model.parameters()).device
    customer_id_map = meta["customer_id_map"]
    reverse_article_map = meta["reverse_article_map"]
    num_users = meta["num_customers"]
    
    all_customer_ids = list(customer_id_map.keys())
    total_batches = (len(all_customer_ids) + batch_size - 1) // batch_size
    
    # Prepare for batch processing
    with torch.no_grad():
        all_embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    
    item_embeddings = all_embeddings[num_users:]
    
    # Create results dataframe
    results = []
    
    for batch_idx in tqdm(range(0, len(all_customer_ids), batch_size), desc="Generating recommendations"):
        batch_customer_ids = all_customer_ids[batch_idx:batch_idx + batch_size]
        batch_user_indices = [customer_id_map[cid] for cid in batch_customer_ids]
        batch_user_indices_tensor = torch.tensor(batch_user_indices, dtype=torch.long, device=device)
        
        # Get user embeddings for this batch
        batch_user_embeddings = all_embeddings[batch_user_indices_tensor]
        
        # Calculate scores
        batch_scores = torch.matmul(batch_user_embeddings, item_embeddings.t())
        
        # Process each user in the batch
        for i, customer_id in enumerate(batch_customer_ids):
            user_idx = batch_user_indices[i]
            user_scores = batch_scores[i]
            
            # Exclude already purchased items
            purchased_items = meta.get("user_to_items", {}).get(user_idx, [])
            for item_idx in purchased_items:
                user_scores[item_idx] = -float('inf')
            
            # Get top-k recommendations
            _, indices = torch.topk(user_scores, k=min(top_k, len(user_scores)))
            indices = indices.cpu().numpy()
            
            # Convert to article IDs and add to results
            for rank, item_idx in enumerate(indices):
                article_id = reverse_article_map[item_idx]
                results.append({
                    "customer_id": customer_id,
                    "article_id": article_id,
                    "rank": rank + 1
                })
    
    # Create and save dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Recommendations saved to {output_file}")

def load_article_metadata(data_dir):
    """
    Load article metadata for better recommendation presentation.
    
    Args:
        data_dir: Directory containing the articles CSV file
        
    Returns:
        articles_df: DataFrame with article metadata
    """
    articles_path = os.path.join(data_dir, "filtered_articles.csv")
    if os.path.exists(articles_path):
        articles_df = pd.read_csv(articles_path)
        return articles_df
    else:
        print(f"Articles file not found at {articles_path}")
        return None

def format_recommendation(article_id, score, articles_df=None):
    """
    Format article recommendation with additional metadata.
    
    Args:
        article_id: Article ID
        score: Recommendation score
        articles_df: DataFrame with article metadata
        
    Returns:
        dict: Formatted recommendation
    """
    rec = {
        "article_id": article_id,
        "score": score
    }
    
    if articles_df is not None:
        article_info = articles_df[articles_df["article_id"] == article_id]
        if not article_info.empty:
            rec.update({
                "product_name": article_info["prod_name"].values[0],
                "product_type": article_info["product_type_name"].values[0],
                "product_group": article_info["product_group_name"].values[0],
                "color": article_info["colour_group_name"].values[0] if "colour_group_name" in article_info.columns else None,
                "department": article_info["department_name"].values[0] if "department_name" in article_info.columns else None
            })
    
    return rec

def main(args):
    """Main function to generate recommendations."""
    # Load model and data
    model, data, meta = load_model_and_data(args)
    
    # Load article metadata
    articles_df = load_article_metadata(args.data_dir)
    
    # Check recommendation mode
    if args.customer_id:
        # Single customer mode
        recommendations, scores = get_recommendations_for_user(
            model, data, meta, args.customer_id, 
            top_k=args.top_k, exclude_purchased=not args.include_purchased
        )
        
        if not recommendations:
            print(f"No recommendations found for customer {args.customer_id}")
            return
        
        print(f"\nTop {len(recommendations)} recommendations for customer {args.customer_id}:")
        for i, (article_id, score) in enumerate(zip(recommendations, scores)):
            rec = format_recommendation(article_id, score, articles_df)
            print(f"\n{i+1}. Article: {rec['article_id']} (Score: {score:.4f})")
            if 'product_name' in rec:
                print(f"   Product: {rec['product_name']}")
                print(f"   Type: {rec['product_type']} ({rec['product_group']})")
                if rec['color']:
                    print(f"   Color: {rec['color']}")
                if rec['department']:
                    print(f"   Department: {rec['department']}")
        
        # Provide explanation for top recommendation if requested
        if args.explain and recommendations:
            explanation = explain_recommendation(model, data, meta, args.customer_id, recommendations[0])
            print("\nWhy this item was recommended:")
            print(f"Similarity score: {explanation['similarity_score']:.4f}")
            
            if explanation['similar_users']:
                print("\nSimilar users who bought this item:")
                for idx, user_id in enumerate(explanation['similar_users'][:3]):
                    print(f"  User {idx+1}: {user_id}")
            
            if explanation['similar_items']:
                print("\nSimilar items the user has purchased:")
                for idx, item_id in enumerate(explanation['similar_items'][:3]):
                    item_info = format_recommendation(item_id, 0, articles_df)
                    print(f"  Item {idx+1}: {item_id} - {item_info.get('product_name', 'Unknown')}")
    
    elif args.customers_file:
        # Batch mode
        try:
            with open(args.customers_file, 'r') as f:
                customer_ids = [line.strip() for line in f if line.strip()]
            
            print(f"Generating recommendations for {len(customer_ids)} customers...")
            recommendations_dict = get_recommendations_batch(
                model, data, meta, customer_ids, 
                top_k=args.top_k, exclude_purchased=not args.include_purchased
            )
            
            # Save to file
            results = []
            for cid, recs in recommendations_dict.items():
                for rank, article_id in enumerate(recs):
                    results.append({
                        "customer_id": cid,
                        "article_id": article_id,
                        "rank": rank + 1
                    })
            
            output_file = args.output_file or "batch_recommendations.csv"
            pd.DataFrame(results).to_csv(output_file, index=False)
            print(f"Recommendations saved to {output_file}")
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
    
    elif args.all_customers:
        # All customers mode
        output_file = args.output_file or "all_recommendations.csv"
        generate_all_recommendations(
            model, data, meta, output_file, 
            top_k=args.top_k, batch_size=args.batch_size
        )
    
    else:
        print("Please specify a customer ID, customers file, or use --all-customers")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate recommendations using trained LightGCN model")
    
    # Model and data arguments
    parser.add_argument("--model_dir", type=str, default="./output",
                        help="Directory containing the trained model")
    parser.add_argument("--model_file", type=str, default="best_model.pth",
                        help="Model file name (best_model.pth or final_model.pth)")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory containing the dataset files")
    parser.add_argument("--embed_dim", type=int, default=64,
                        help="Embedding dimension (must match trained model)")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of LightGCN layers (must match trained model)")
    
    # Recommendation mode arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--customer_id", type=str,
                          help="Customer ID to generate recommendations for")
    mode_group.add_argument("--customers_file", type=str,
                          help="File containing customer IDs, one per line")
    mode_group.add_argument("--all_customers", action="store_true",
                          help="Generate recommendations for all customers")
    
    # Recommendation options
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of recommendations to generate")
    parser.add_argument("--include_purchased", action="store_true",
                        help="Include items the user has already purchased")
    parser.add_argument("--explain", action="store_true",
                        help="Explain recommendations (single user mode only)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str,
                        help="Output file for batch or all-customers mode")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for processing all customers")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)