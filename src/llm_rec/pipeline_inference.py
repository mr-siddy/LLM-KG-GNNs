# %% [markdown]
# #### Inference

# %%
# import torch
# import dill  # pip install dill
# import os

# # Define file paths (adjust as needed)
# processed_dir = "./data/processed"
# original_data_path = os.path.join(processed_dir, "lightgcn_data.pt")
# safe_data_path = os.path.join(processed_dir, "lightgcn_data_safe.pt")

# # Attempt to load the original file using dill as the pickle_module
# print("Attempting to load original data using dill...")
# data = torch.load(original_data_path, map_location="cpu", pickle_module=dill, weights_only=False)
# print("Original data loaded successfully.")

# # Check and convert edge attributes to standard float tensors if needed
# if hasattr(data, "edge_attr") and data.edge_attr is not None:
#     try:
#         # Convert custom edge_attr to a standard float tensor by calling .tolist() and re-wrapping as a tensor
#         safe_edge_attr = torch.tensor(data.edge_attr.tolist(), dtype=torch.float)
#         data.edge_attr = safe_edge_attr
#         print("Converted edge attributes to a standard float tensor.")
#     except Exception as e:
#         print("Error converting edge_attr to float tensor:", e)
# else:
#     print("No edge attributes found or already in safe format.")

# # Save the safe data file
# torch.save(data, safe_data_path)
# print(f"Safe data saved to: {safe_data_path}")


# %%
# # %% Inference Cell with Safe Data Loading Fix

# import os
# import torch
# import pickle
# import argparse
# import dill
# import torch_geometric.data.data as tg_data

# # --- Ensure DataEdgeAttr is defined ---
# if not hasattr(tg_data, "DataEdgeAttr"):
#     class DummyDataEdgeAttr:
#         pass
#     tg_data.DataEdgeAttr = DummyDataEdgeAttr
#     print("Defined dummy DataEdgeAttr.")

# # --- Function Definitions ---

# def load_safe_data(data_path, device):
#     """
#     Load processed graph data saved in a safe format using dill as the pickle module.
#     """
#     print(f"Loading safe data from: {data_path}")
#     data = torch.load(data_path, map_location=device, pickle_module=dill, weights_only=False)
#     return data

# def load_meta(meta_path):
#     """
#     Load metadata (e.g., customer and article mappings) from a pickle file.
#     """
#     with open(meta_path, "rb") as f:
#         meta = pickle.load(f)
#     return meta

# def load_trained_model(model_path, num_users, num_items, embed_dim, num_layers, model_type, device):
#     """
#     Initialize and load the trained model's state.
#     """
#     from model import LightGCN, EnhancedLightGCN  # Import here in case PATH issues arise
#     if model_type == "enhanced":
#         model = EnhancedLightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
#     else:
#         model = LightGCN(num_users, num_items, embed_dim=embed_dim, num_layers=num_layers).to(device)
#     print(f"Loading model from {model_path}")
#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model

# def get_recommendations(model, data, meta, user_id, top_k=10):
#     """
#     Generate top-k recommendations for a given user.
#     """
#     device = next(model.parameters()).device
#     if user_id not in meta["customer_id_map"]:
#         print(f"User ID {user_id} not found in meta.")
#         return []
#     u_idx = meta["customer_id_map"][user_id]
#     num_users = meta["num_customers"]
    
#     with torch.no_grad():
#         embeddings = model(data.edge_index.to(device), data.edge_attr.to(device))
    
#     user_embedding = embeddings[u_idx].unsqueeze(0)
#     item_embeddings = embeddings[num_users:]
#     scores = torch.matmul(user_embedding, item_embeddings.t()).squeeze(0)
#     _, topk_indices = torch.topk(scores, top_k)
#     topk_indices = topk_indices.cpu().tolist()
    
#     # Convert item indices back to article IDs using reverse mapping
#     reverse_article_map = {v: k for k, v in meta["article_id_map"].items()}
#     recommendations = [reverse_article_map[i] for i in topk_indices]
#     return recommendations

# # --- Set Parameters and File Paths ---
# # Adjust these paths as needed
# data_dir = "./data/processed"           # Directory for processed data files
# model_dir = "./output"                  # Directory where the trained model is saved
# model_file = "standard_lightgcN_best.pth"  # Model filename (adjust as necessary)

# model_type = "standard"   # "standard" or "enhanced"
# embed_dim = 64
# num_layers = 3
# top_k = 10                # Number of recommendations to generate

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# safe_data_path = os.path.join(data_dir, "lightgcn_data_safe.pt")
# meta_path = os.path.join(data_dir, "lightgcn_meta.pkl")

# if not os.path.exists(safe_data_path):
#     raise FileNotFoundError(f"Safe data file not found at {safe_data_path}")
# if not os.path.exists(meta_path):
#     raise FileNotFoundError(f"Meta file not found at {meta_path}")

# # --- Load Data and Model ---
# data = load_safe_data(safe_data_path, device)
# meta = load_meta(meta_path)

# num_users = meta["num_customers"]
# num_items = meta["num_articles"]

# model_path = os.path.join(model_dir, model_file)
# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"Model file not found at {model_path}")

# model = load_trained_model(model_path, num_users, num_items, embed_dim, num_layers, model_type, device)

# # --- Generate Recommendations for 10 Users ---
# user_ids = list(meta["customer_id_map"].keys())[:10]
# print("Generating recommendations for users:")
# print(user_ids)

# for uid in user_ids:
#     recs = get_recommendations(model, data, meta, uid, top_k=top_k)
#     print(f"\nRecommendations for user {uid}:")
#     print(recs)


# %% [markdown]
# #### Pipeline

# %%
# %% Full Integrated Pipeline Cell for Reformatting Recommendations via an LLM

import os
import torch
import pickle
import pandas as pd
import dill
from torch.serialization import safe_globals
import torch_geometric.data.data as tg_data
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# --- Step 1: Load Safe Data, Metadata, and Trained GNN Model ---

# Ensure DataEdgeAttr is defined (dummy if not present)
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
    
    # Convert item indices back to article IDs using reverse mapping
    reverse_article_map = {v: k for k, v in meta["article_id_map"].items()}
    recommendations = [reverse_article_map.get(i, f"Unknown({i})") for i in topk_indices]
    return recommendations

# ----- Parameters and File Paths -----
data_dir = "./data/processed"           # Adjust as needed
model_dir = "./output"                  # Adjust as needed
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

# --- Step 2: Get Raw Recommendations for 1 User ---
user_id = list(meta["customer_id_map"].keys())[0]  # Selecting the first user
print(f"\nGenerating raw recommendations for user: {user_id}")
raw_recs = get_recommendations(model, data, meta, user_id, top_k=10)
print("Raw Article IDs recommended:", raw_recs)

# --- Step 3: Mapping Function to Enrich Raw Recommendations ---
articles_csv_path = "./data/filtered_articles.csv"  # Adjust path as needed
articles_df = pd.read_csv(articles_csv_path)

def enrich_product_description(article_id, articles_df):
    row = articles_df[articles_df["article_id"] == article_id]
    if row.empty:
        return f"Product with ID {article_id} (details not found)."
    row = row.iloc[0]
    description = f"{row['prod_name']} – a {row['product_type_name']} from {row['product_group_name']} in {row['colour_group_name']}. {row.get('detail_desc', '')}"
    return description

enriched_products = [enrich_product_description(aid, articles_df) for aid in raw_recs]
print("\nEnriched Product Descriptions:")
for i, desc in enumerate(enriched_products, 1):
    print(f"{i}. {desc}")

# --- Step 4: Construct a Revised Natural Language Prompt for the LLM ---
prompt = (
    "Below are candidate products recommended by our graph-based model. Please reformat and rewrite this information into a clear, numbered list of product recommendations. For each item, write a concise pointer that includes the product name and a brief summary of its key features. Do not include product IDs or bullet points beyond a simple numbered list.\n\n"
    "Candidate Products:\n" +
    "\n".join([f"{i}. {desc}" for i, desc in enumerate(enriched_products, 1)]) +
    "\n\nPlease provide your recommendation as a numbered list."
)
print("\nConstructed LLM Prompt:\n", prompt)



# # %%
# print(prompt)

# # %%
# 1. Consider the NT Alva 2-pack for its elegant white vest top design that offers both style and practicality with soft organic cotton and adjustable straps.
# 2. The Cat Tee in white provides a timeless and comfortable option for everyday wear.
# 3. The Cat Tee in grey is a versatile choice that blends effortlessly with any casual outfit.
# 4. The Cat Tee in pink adds a splash of color, perfect for brightening up your look.
# 5. The Cassia crew sweater in green is ideal for cooler days, offering a relaxed fit and cozy long sleeves.
# 6. The Becka hoodie in grey combines sporty appeal with comfort, thanks to its soft brushed interior and practical hood design.
# 7. The Penny Wide Culotte in black delivers a modern silhouette with a high waist and streamlined design.
# 8. The Seamless cheeky brief in black is designed with minimal seams for maximum comfort.
# 9. The Babe LS T-shirt in black stands out with its contemporary ribbed jersey style.
# 10. Lastly, the Bangkok V-neck sweater in grey provides a refined look with its deep V-neck and delicate knit.


# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "unsloth/Llama-3.2-1B-Instruct"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
device = 0 if torch.cuda.is_available() else -1

# Create a text-generation pipeline from the loaded model
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Example prompt
prompt = (
"""
You are an expert product recommender. Your task is to rewrite a list of candidate product details into a concise, friendly, numbered list of recommendations. Do not copy the input verbatim; instead, use your own words to summarize each product's key features. Ensure that all products are rewritten in the same structured format.

### **Example Output Format**
Each recommendation should follow this pattern:
1. **[Product Name]** - [Short, friendly description focusing on key features and why it’s a great choice.]

### **Examples:**
Original: "1. Red Hoodie – A cozy red hoodie with a drawstring, perfect for cool days."
Rewritten: "1. **Red Hoodie** - Stay warm and stylish with this soft, cozy hoodie featuring an adjustable drawstring—perfect for chilly days."

Original: "2. Classic Blue Jeans – A timeless pair of jeans with a straight-leg fit."
Rewritten: "2. **Classic Blue Jeans** - A wardrobe staple! These straight-leg jeans offer a timeless style that pairs effortlessly with any outfit."

Original: "3. Running Shoes – Lightweight shoes with cushioned soles for comfort."
Rewritten: "3. **Running Shoes** - Designed for comfort and performance, these lightweight shoes provide excellent cushioning for all-day wear."

---

### **Now, rewrite the following candidate products in the same structured format:**

Candidate Products:
1. NT Alva 2-pack(1) – a Vest top from Garment Upper body in White. Soft nursing tops in organic cotton jersey with narrow adjustable shoulder straps. Soft integral top with an elasticated hem and functional fastening for easier nursing access.
2. Cat Tee. – a T-shirt from Garment Upper body in White. T-shirt in soft jersey.
3. Cat Tee. – a T-shirt from Garment Upper body in Grey. T-shirt in soft jersey.
4. Cat Tee. – a T-shirt from Garment Upper body in Pink. T-shirt in soft jersey.
5. Cassia crew – a Sweater from Garment Upper body in Green. Long-sleeved top in soft sweatshirt fabric made from a cotton blend with a round neckline, low dropped shoulders and ribbing around the neckline, cuffs and hem.
6. Becka hoodie – a Hoodie from Garment Upper body in Grey. Long-sleeved top in soft sweatshirt fabric with a jersey-lined, drawstring hood, kangaroo pocket and ribbing at the cuffs and hem. Soft brushed inside.
7. Penny Wide Culotte – a Trousers from Garment Lower body in Black. 3/4-length trousers in woven fabric with a high, elasticated waist, concealed pockets in the side seams, fake welt back pockets and straight, wide legs.
8. Seamless cheeky brief – an Underwear bottom from Underwear in Black. Briefs in microfibre with a high waist, lined gusset and cutaway coverage at the back. Designed with minimal seams for comfort.
9. Babe LS – a T-shirt from Garment Upper body in Black. Long top in ribbed jersey made from a viscose blend with a wide neckline and long sleeves. Straight cut with high slits in the sides and a rounded hem.
10. Bangkok V-neck sweater – a Sweater from Garment Upper body in Grey. Jumper in a soft, fine-knit viscose blend with a deep V-neck, long sleeves and ribbing around the neckline, cuffs and hem.

**Now rewrite these in the same structured format as the examples above. Ensure all descriptions are short, engaging, and highlight key product features concisely.**

"""
)

output = pipe(prompt, max_new_tokens=300, do_sample=True, top_p=0.95, temperature=0.7)
print("Generated text:\n", output[0]["generated_text"])


# %%


# %%



