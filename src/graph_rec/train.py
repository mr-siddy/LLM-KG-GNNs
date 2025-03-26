"""
train_lightgcn.py

Training script for LightGCN recommendation model on the filtered H&M dataset.
This script supports both the standard LightGCN and the EnhancedLightGCN models.

Features:
1. Loads processed data
2. Initializes the selected model variant
3. Trains using BPR loss
4. Periodically evaluates on test set
5. Saves the best model
"""

import os
import torch
import torch.optim as optim
import numpy as np
import random
import pickle
import argparse
from collections import defaultdict
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Import models and utilities
from model import LightGCN, EnhancedLightGCN
from data_loader import load_filtered_data
from evaluation import recall_at_k, ndcg_at_k

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_model(args):
    """Train LightGCN or EnhancedLightGCN model on the filtered H&M dataset."""
    set_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    if args.load_processed and os.path.exists(os.path.join(args.data_dir, "processed/lightgcn_data.pt")):
        print("Loading processed data...")
        data = torch.load(os.path.join(args.data_dir, "processed/lightgcn_data.pt"), map_location=device)
        with open(os.path.join(args.data_dir, "processed/lightgcn_meta.pkl"), "rb") as f:
            meta = pickle.load(f)
    else:
        print("Processing data from CSV files...")
        data, meta = load_filtered_data(
            data_dir=args.data_dir, 
            min_cooccur=args.min_cooccur,
            decay_rate=args.decay_rate
        )
    
    data = data.to(device)
    train_user_items = meta.get("train_user_items", {})
    test_user_items = meta.get("test_user_items", {})
    
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]
    
    print(f"Dataset statistics:")
    print(f"- Users: {num_users}")
    print(f"- Items: {num_items}")
    print(f"- Edges: {data.num_edges}")
    print(f"- Training users: {len(train_user_items)}")
    print(f"- Testing users: {len(test_user_items)}")
    
    # Generate/process edge types if using enhanced model
    edge_type = None
    if args.model_type == "enhanced":
        if hasattr(data, 'edge_type') and data.edge_type is not None:
            edge_type = data.edge_type
        else:
            # Create simple edge type: 0 for user-item, 1 for item-item
            edge_type = torch.zeros(data.edge_index.size(1), dtype=torch.long, device=device)
            row, col = data.edge_index
            
            # Check if this is an item-item edge (both indices >= num_users)
            item_item_mask = (row >= num_users) & (col >= num_users)
            edge_type[item_item_mask] = 1
            
            # Save edge type in data
            data.edge_type = edge_type
            
            print(f"Created edge types: {torch.bincount(edge_type)} edges of each type")
    
    # Initialize model
    if args.model_type == "standard":
        print("Using standard LightGCN model")
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers
        ).to(device)
    else:
        print("Using enhanced LightGCN model with edge type embeddings")
        model = EnhancedLightGCN(
            num_users=num_users,
            num_items=num_items,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_edge_types=args.num_edge_types
        ).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_recall = 0.0
    best_epoch = 0
    train_losses = []
    test_recalls = []
    test_ndcgs = []
    
    print("\nStarting training...")
    train_users = list(train_user_items.keys())
    
    for epoch in range(args.num_epochs):
        model.train()
        random.shuffle(train_users)
        total_loss = 0.0
        total_batches = 0
        
        # Training loop
        pbar = tqdm(range(0, len(train_users), args.batch_size), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch_idx in pbar:
            batch_users = train_users[batch_idx:batch_idx + args.batch_size]
            if not batch_users:
                continue
                
            pos_items, neg_items = [], []
            valid_users = []
            for i, user in enumerate(batch_users):
                if not train_user_items[user]:
                    continue
                    
                # Sample positive item
                pos_item = random.choice(list(train_user_items[user]))
                
                # Sample negative item
                neg_item = random.randint(0, num_items - 1)
                while neg_item in train_user_items[user]:
                    neg_item = random.randint(0, num_items - 1)
                
                valid_users.append(user)
                pos_items.append(pos_item)
                neg_items.append(neg_item)
            
            if not valid_users:
                continue
                
            # Convert to tensors
            users_tensor = torch.tensor(valid_users, dtype=torch.long, device=device)
            pos_items_tensor = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_items_tensor = torch.tensor(neg_items, dtype=torch.long, device=device)
            
            # Forward pass
            if args.model_type == "standard":
                all_embeddings = model(data.edge_index, data.edge_attr)
            else:
                all_embeddings = model(data.edge_index, data.edge_attr, data.edge_type)
                
            user_embeddings = all_embeddings[users_tensor]
            pos_item_embeddings = all_embeddings[num_users + pos_items_tensor]
            neg_item_embeddings = all_embeddings[num_users + neg_items_tensor]
            
            # BPR loss
            pos_scores = torch.sum(user_embeddings * pos_item_embeddings, dim=1)
            neg_scores = torch.sum(user_embeddings * neg_item_embeddings, dim=1)
            loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
            
            # L2 regularization
            reg_loss = 1/2 * (user_embeddings.norm(2).pow(2) + 
                            pos_item_embeddings.norm(2).pow(2) + 
                            neg_item_embeddings.norm(2).pow(2)) / len(users_tensor)
            loss += args.reg_weight * reg_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.num_epochs} - Avg. Loss: {avg_loss:.4f}")
        
        # Evaluate on test set every few epochs
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.num_epochs - 1:
            recall, ndcg = evaluate(model, data, meta, test_user_items, device, args.model_type, top_k=args.top_k)
            test_recalls.append(recall)
            test_ndcgs.append(ndcg)
            
            print(f"Evaluation - Recall@{args.top_k}: {recall:.4f}, NDCG@{args.top_k}: {ndcg:.4f}")
            
            # Save best model
            if recall > best_recall:
                best_recall = recall
                best_epoch = epoch + 1
                model_filename = f"{args.model_type}_lightgcn_best.pth"
                torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))
                print(f"New best model saved as {model_filename}! Recall@{args.top_k}: {recall:.4f}")
    
    # Save final model
    model_filename = f"{args.model_type}_lightgcn_final.pth"
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))
    print(f"\nTraining complete!")
    print(f"Best model (epoch {best_epoch}) - Recall@{args.top_k}: {best_recall:.4f}")
    
    # Save metadata with model type
    meta_filename = f"{args.model_type}_lightgcn_meta.pkl"
    meta["model_type"] = args.model_type
    meta["embed_dim"] = args.embed_dim
    meta["num_layers"] = args.num_layers
    meta["best_epoch"] = best_epoch
    meta["best_recall"] = best_recall
    
    with open(os.path.join(args.output_dir, meta_filename), "wb") as f:
        pickle.dump(meta, f)
    
    # Plot training curves
    plot_training_curves(train_losses, test_recalls, test_ndcgs, args)
    
    return model, meta

def evaluate(model, data, meta, test_user_items, device, model_type="standard", top_k=10):
    """Evaluate model on test set."""
    model.eval()
    num_users = meta["num_customers"]
    
    with torch.no_grad():
        if model_type == "standard":
            all_embeddings = model(data.edge_index, data.edge_attr)
        else:
            all_embeddings = model(data.edge_index, data.edge_attr, data.edge_type)
    
    user_embeddings = all_embeddings[:num_users]
    item_embeddings = all_embeddings[num_users:]
    
    recalls = []
    ndcgs = []
    
    for user, true_items in test_user_items.items():
        if not true_items:
            continue
            
        # Get user embedding
        user_embedding = user_embeddings[user].unsqueeze(0)
        
        # Calculate scores for all items
        scores = torch.matmul(user_embedding, item_embeddings.t()).squeeze(0)
        
        # Filter out training items
        train_items = meta.get("train_user_items", {}).get(user, set())
        for item in train_items:
            scores[item] = -float('inf')
        
        # Get top-k recommendations
        _, indices = torch.topk(scores, k=top_k)
        recommended_items = indices.cpu().tolist()
        
        # Calculate metrics
        recall = recall_at_k(recommended_items, true_items, top_k)
        ndcg = ndcg_at_k(recommended_items, true_items, top_k)
        
        recalls.append(recall)
        ndcgs.append(ndcg)
    
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
    
    return avg_recall, avg_ndcg

def plot_training_curves(train_losses, test_recalls, test_ndcgs, args):
    """Plot training and evaluation curves."""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'Training Loss ({args.model_type} LightGCN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot evaluation metrics
    plt.subplot(1, 2, 2)
    eval_epochs = list(range(args.eval_freq - 1, args.num_epochs, args.eval_freq))
    if len(eval_epochs) < len(test_recalls):
        eval_epochs.append(args.num_epochs - 1)
    
    if len(eval_epochs) == len(test_recalls):
        plt.plot(eval_epochs, test_recalls, label=f'Recall@{args.top_k}')
        plt.plot(eval_epochs, test_ndcgs, label=f'NDCG@{args.top_k}')
        plt.title('Evaluation Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f'{args.model_type}_training_curves.png'))
    print(f"Training curves saved to {os.path.join(args.output_dir, f'{args.model_type}_training_curves.png')}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LightGCN variants on filtered H&M dataset")
    
    # Model type
    parser.add_argument("--model_type", type=str, default="standard", choices=["standard", "enhanced"],
                        help="Model type: standard LightGCN or enhanced with edge types")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="./data", 
                        help="Directory containing the filtered CSV files")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save models and results")
    parser.add_argument("--load_processed", action="store_true", 
                        help="Load preprocessed data if available")
    parser.add_argument("--min_cooccur", type=int, default=3, 
                        help="Minimum co-occurrences for article-article edges")
    parser.add_argument("--decay_rate", type=float, default=0.005, 
                        help="Time decay rate for customer-article interactions")
    
    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=64, 
                        help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=3, 
                        help="Number of LightGCN layers")
    parser.add_argument("--num_edge_types", type=int, default=2,
                        help="Number of edge types for enhanced model (default: user-item and item-item)")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=500, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, 
                        help="Weight decay for Adam optimizer")
    parser.add_argument("--reg_weight", type=float, default=1e-4, 
                        help="Weight for L2 regularization")
    
    # Evaluation arguments
    parser.add_argument("--top_k", type=int, default=10, 
                        help="K value for evaluation metrics")
    parser.add_argument("--eval_freq", type=int, default=5, 
                        help="Evaluate every N epochs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", 
                        choices=["cuda", "cpu"], help="Device to use")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Rename function to match new name
    model, meta = train_model(args)
    
    print(f"Training complete for {args.model_type} model")
    print(f"Model saved to {os.path.join(args.output_dir, f'{args.model_type}_lightgcn_final.pth')}")
    print(f"Best model saved to {os.path.join(args.output_dir, f'{args.model_type}_lightgcn_best.pth')}")
    
    # Optional: print final statistics
    print(f"Final statistics:")
    print(f"- Best Recall@{args.top_k}: {meta['best_recall']:.4f} (Epoch {meta['best_epoch']})")