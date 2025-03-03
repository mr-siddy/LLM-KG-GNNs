"""
visualization.py

This module provides visualization routines:
- Plot a subgraph (using NetworkX) of customers and articles.
- Visualize node embeddings with dimensionality reduction (t-SNE or UMAP).
"""

import matplotlib.pyplot as plt
import networkx as nx
import torch
import umap
from sklearn.manifold import TSNE

def plot_subgraph(data, num_nodes=100):
    """
    Plots a subgraph with a limited number of nodes.
    """
    edge_index = data.edge_index.cpu().numpy()
    G = nx.Graph()
    # Use only the first num_nodes nodes for visualization.
    for i in range(num_nodes):
        G.add_node(i)
    for src, dst in zip(edge_index[0], edge_index[1]):
        if src < num_nodes and dst < num_nodes:
            G.add_edge(src, dst)
    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray")
    plt.title("Subgraph Visualization")
    plt.show()

def visualize_embeddings(embeddings, method="umap", title="Embedding Visualization"):
    """
    Reduces embedding dimensions and plots them.
    
    Args:
        embeddings (Tensor or np.array): Node embeddings.
        method (str): "umap" or "tsne".
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().detach().numpy()
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        proj = reducer.fit_transform(embeddings)
    elif method == "tsne":
        proj = TSNE(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        raise ValueError("Method must be either 'umap' or 'tsne'.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

if __name__ == "__main__":
    # Dummy test: generate random embeddings.
    import torch
    dummy_embeddings = torch.rand(200, 64)
    visualize_embeddings(dummy_embeddings, method="umap", title="UMAP Projection")
    visualize_embeddings(dummy_embeddings, method="tsne", title="t-SNE Projection")
