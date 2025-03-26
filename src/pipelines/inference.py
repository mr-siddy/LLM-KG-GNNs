import os
import torch
import pickle
import pandas as pd
import dill
import torch_geometric.data.data as tg_data
from torch.serialization import safe_globals
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Ensure DataEdgeAttr is defined (dummy if not present) ---
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

def inference_pipeline():
    # ----- Parameters and File Paths -----
    data_dir = "./data/processed"           # Adjust as needed
    model_dir = "./graph_rec/output"          # Adjust as needed
    safe_data_path = os.path.join(data_dir, "lightgcn_data_safe.pt")
    meta_path = os.path.join(data_dir, "lightgcn_meta.pkl")
    model_file = "standard_lightgcN_best.pth"  # Adjust as needed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load safe processed graph data and metadata
    data = load_safe_data(safe_data_path, device)
    meta = load_meta(meta_path)
    num_users = meta["num_customers"]
    num_items = meta["num_articles"]

    # Load the trained GNN model
    model_path = os.path.join(model_dir, model_file)
    model_type = "standard"  # or "enhanced"
    embed_dim = 64
    num_layers = 3
    model = load_trained_model(model_path, num_users, num_items, embed_dim, num_layers, model_type, device)

    # --- Generate Recommendations for 1 User ---
    user_id = list(meta["customer_id_map"].keys())[0]  # Selecting the first user
    print(f"\nGenerating raw recommendations for user: {user_id}")
    raw_recs = get_recommendations(model, data, meta, user_id, top_k=10)
    print("Raw Article IDs recommended:", raw_recs)

    # --- Enrich Recommendations ---
    articles_csv_path = "./data/filtered_articles.csv"  # Adjust path as needed
    articles_df = pd.read_csv(articles_csv_path)

    enriched_products = [enrich_product_description(aid, articles_df) for aid in raw_recs]
    print("\nEnriched Product Descriptions:")
    for i, desc in enumerate(enriched_products, 1):
        print(f"{i}. {desc}")

    # --- Construct the Final LLM Prompt ---
    prompt = (
        "Below are candidate products recommended by our graph-based model. Please reformat and rewrite this information into a clear, numbered list of product recommendations. For each item, write a concise pointer that includes the product name and a brief summary of its key features. Do not include product IDs or bullet points beyond a simple numbered list.\n\n"
        "Candidate Products:\n" +
        "\n".join([f"{i}. {desc}" for i, desc in enumerate(enriched_products, 1)]) +
        "\n\nPlease provide your recommendation as a numbered list."
    )
    print("\nConstructed LLM Prompt:\n", prompt)

    # --- LLM Paraphrasing ---
    model_name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device_llm = 0 if torch.cuda.is_available() else -1
    llm_pipe = pipeline("text-generation", model=model_llm, tokenizer=tokenizer, device=device_llm)

    llm_out = llm_pipe(prompt, max_new_tokens=1500, do_sample=True, top_p=0.95, temperature=0.7)
    generated_text = llm_out[0]["generated_text"]
    if "### Response:" in generated_text:
        final_output = generated_text.split("### Response:")[-1].strip()
    else:
        final_output = generated_text.strip()

    print("\nFinal Paraphrased Output from LLM:\n", final_output)

    # --- Post-Processing the LLM Output ---
    final_output_clean = final_output.replace("*", "")
    final_output_clean = "\n".join([line.strip() for line in final_output_clean.splitlines() if line.strip() != ""])

    output_file = "data/sft/SFT_data_2.txt"
    with open(output_file, "w") as f:
        f.write(final_output_clean)

    print("\nCleaned Final Output saved to", output_file)
    print("\nCleaned Final Output:\n", final_output_clean)

def main():
    inference_pipeline()

if __name__ == "__main__":
    main()
