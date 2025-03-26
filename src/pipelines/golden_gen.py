import os
import re
import torch
import pickle
import pandas as pd
import dill
from torch.serialization import safe_globals
import torch_geometric.data.data as tg_data
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ========= REUSED FUNCTIONS =========
# Define dummy DataEdgeAttr if not present
if not hasattr(tg_data, "DataEdgeAttr"):
    class DummyDataEdgeAttr:
        pass
    tg_data.DataEdgeAttr = DummyDataEdgeAttr
    print("Defined dummy DataEdgeAttr.")

def load_safe_data(data_path, device):
    print(f"Loading safe data from: {data_path}")
    with safe_globals([tg_data.DataEdgeAttr]):
        data = torch.load(data_path, map_location=device, pickle_module=dill, weights_only=False)
    return data

def load_meta(meta_path):
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta

def load_trained_model(model_path, num_users, num_items, embed_dim, num_layers, model_type, device):
    from model import LightGCN, EnhancedLightGCN
    if model_type == "enhanced":
        model = EnhancedLightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
    else:
        model = LightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def get_recommendations(model, data, meta, user_id, top_k=10):
    device = next(model.parameters()).device
    if user_id not in meta["customer_id_map"]:
        print(f"User ID {user_id} not found in meta.")
        return []
    u_idx = meta["customer_id_map"][user_id]
    num_users = meta["num_customers"]
    with torch.no_grad():
        embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    user_embedding = embeddings[u_idx].unsqueeze(0)
    item_embeddings = embeddings[num_users:]
    scores = torch.matmul(user_embedding, item_embeddings.t()).squeeze(0)
    _, topk_indices = torch.topk(scores, top_k)
    topk_indices = topk_indices.cpu().tolist()
    reverse_article_map = {v: k for k, v in meta["article_id_map"].items()}
    recommendations = [reverse_article_map.get(i, f"Unknown({i})") for i in topk_indices]
    return recommendations

def enrich_product_description(article_id, articles_df):
    row = articles_df[articles_df["article_id"] == article_id]
    if row.empty:
        return f"Product with ID {article_id} (details not found)."
    row = row.iloc[0]
    description = f"{row['prod_name']} – a {row['product_type_name']} from {row['product_group_name']} in {row['colour_group_name']}. {row.get('detail_desc', '')}"
    return description

def get_customer_profile(customer_id, customers_df):
    row = customers_df[customers_df["customer_id"] == customer_id]
    if row.empty:
        return "Customer details not found."
    row = row.iloc[0]
    profile = f"Age: {row.get('age', 'N/A')}, Membership: {row.get('club_member_status', 'N/A')}"
    return profile

def parse_customer_id(query):
    match = re.search(r'customer\s*id[:\-]?\s*([0-9a-fA-F]+)', query, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def generate_golden_example():
    # Define file paths and parameters
    data_dir = "./data/processed"           # Adjust as needed
    model_dir = "./graph_rec/output"         # Adjust as needed
    safe_data_path = os.path.join(data_dir, "lightgcn_data_safe.pt")
    meta_path = os.path.join(data_dir, "lightgcn_meta.pkl")
    model_file = "standard_lightgcN_best.pth"  # Adjust as needed
    customers_csv_path = "./data/filtered_customers.csv"  # Customer profiles
    articles_csv_path = "./data/filtered_articles.csv"     # Product metadata
    transactions_csv_path = "./data/filtered_transactions_train.csv"  # Transactions data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_safe_data(safe_data_path, device)
    meta = load_meta(meta_path)
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]

    model_path = os.path.join(model_dir, model_file)
    model_type = "standard"  # or "enhanced"
    embed_dim = 64
    num_layers = 3
    model = load_trained_model(model_path, num_users, num_items, embed_dim, num_layers, model_type, device)

    customers_df = pd.read_csv(customers_csv_path)
    articles_df = pd.read_csv(articles_csv_path)
    transactions_df = pd.read_csv(transactions_csv_path)

    user_query = "Please give me product recommendations for customer id: 071ba51649f345894a944da3e9a0e3658299780f46a7fe89e03b221ac4a604e9."
    parsed_id = parse_customer_id(user_query)
    if parsed_id is None or parsed_id not in meta["customer_id_map"]:
        print("Could not parse a valid customer id from input. Using default customer.")
        customer_id = list(meta["customer_id_map"].keys())[0]
    else:
        customer_id = parsed_id
    print(f"\nUsing customer ID: {customer_id}")

    profile_text = get_customer_profile(customer_id, customers_df)
    customer_transactions = transactions_df[transactions_df["customer_id"] == customer_id]
    if customer_transactions.empty:
        purchase_history_list = ["No purchase history found."]
    else:
        purchased_article_ids = customer_transactions["article_id"].unique().tolist()
        purchase_history_list = [enrich_product_description(aid, articles_df) for aid in purchased_article_ids]

    raw_recs = get_recommendations(model, data, meta, customer_id, top_k=10)
    recommended_products_list = [enrich_product_description(aid, articles_df) for aid in raw_recs]

    golden_example_path = "golden_examples/golden_example_copy.txt"  # Ensure this file exists with your ideal output format
    with open(golden_example_path, "r") as f:
        golden_example = f.read()

    final_prompt = (
        "Customer Profile:\n" + profile_text + "\n\n" +
        "Purchase History:\n" + "\n".join([f"- {item}" for item in purchase_history_list]) + "\n\n" +
        "Product Recommendations:\n" + "\n".join([f"- {item}" for item in recommended_products_list]) + "\n\n" +
        "Below is an example of the ideal output format:\n" +
        golden_example + "\n\n" +
        "Based on the above information, please rewrite and summarize the data into a clean, human-readable format. "
        "The output should include a concise summary of the customer's purchase history and a friendly, numbered list of product recommendations." +
        "\n\n### Response:"
    )

    print("\nConstructed Informational Prompt:\n")
    print(final_prompt)
    return final_prompt

def main():
    print("Generating golden example prompt...")
    prompt = generate_golden_example()
    # Optionally, you might forward this prompt to an LLM here if desired.
    # For now, we only print the prompt.
    
if __name__ == "__main__":
    main()
