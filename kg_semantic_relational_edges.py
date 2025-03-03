import json
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine

# Import chromadb and its settings
import chromadb
from chromadb.config import Settings

class KnowledgeGraphDataset:
    def __init__(self, json_path=None, embedding_model_name="all-MiniLM-L6-v2", semantic_threshold=0.8):
        """
        Initialize the knowledge graph dataset.
        
        Args:
            json_path (str): Path to the JSON file with enriched NER and relation data.
            embedding_model_name (str): Name of the SentenceTransformer model.
            semantic_threshold (float): Cosine similarity threshold above which to add a semantic edge.
        """
        self.json_path = json_path
        self.nodes = {}    # Key: entity name, Value: dict with id, aggregated chunk texts, and labels.
        self.relations = []  # List of edges: (src_id, tgt_id, relation)
        self.node_counter = 0
        self.semantic_threshold = semantic_threshold

        if self.json_path:
            self.load_data()

        # Initialize the embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

        # Initialize the Chroma vector database client and collection.
        self.chroma_client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db")
        )
        # Try to get an existing collection; if not, create one.
        try:
            self.collection = self.chroma_client.get_collection("kg_embeddings")
        except Exception:
            self.collection = self.chroma_client.create_collection("kg_embeddings")

    def load_data(self):
        with open(self.json_path, "r") as f:
            self.data = json.load(f)

    def add_node(self, entity, chunk_text=None, label=None, score=None):
        """
        Add or update a node. Consolidate nodes with the same entity name.
        """
        if entity not in self.nodes:
            self.nodes[entity] = {
                "id": self.node_counter,
                "entity": entity,
                "chunk_texts": [chunk_text] if chunk_text else [],
                "labels": {}  # e.g., {label: score}
            }
            if label and score:
                self.nodes[entity]["labels"][label] = score
            self.node_counter += 1
        else:
            if chunk_text and chunk_text not in self.nodes[entity]["chunk_texts"]:
                self.nodes[entity]["chunk_texts"].append(chunk_text)
            if label and score:
                self.nodes[entity]["labels"][label] = score
        return self.nodes[entity]["id"]

    def add_relation(self, source, target, relation, chunk_text=None):
        """
        Add a relation edge between source and target entities.
        """
        source_id = self.add_node(source, chunk_text)
        target_id = self.add_node(target)
        self.relations.append((source_id, target_id, relation))

    def construct_graph(self):
        """
        Build the graph from JSON data. Processes NER entities and relation triplets.
        """
        for entry in self.data:
            chunk_text = entry.get("chunk_text", "")
            # Process NER entities
            for ner in entry.get("ner_entities", []):
                entity = ner.get("entity")
                label = ner.get("label")
                score = ner.get("score")
                self.add_node(entity, chunk_text, label, score)
                if label:
                    # Add an edge representing the type relationship (belongs_to)
                    label_id = self.add_node(label)
                    self.relations.append((self.nodes[entity]["id"], self.nodes[label]["id"], "belongs_to"))
            # Process relation triplets
            for triplet in entry.get("relation_triplets", []):
                source = triplet.get("source")
                target = triplet.get("target")
                relation = triplet.get("relation")
                self.add_relation(source, target, relation, chunk_text)

    def to_networkx(self):
        """
        Convert the internal graph representation to a NetworkX DiGraph.
        """
        G = nx.DiGraph()
        for entity, node_data in self.nodes.items():
            G.add_node(
                node_data["id"],
                entity=node_data["entity"],
                chunk_texts=node_data["chunk_texts"],
                labels=node_data["labels"]
            )
        for src, tgt, relation in self.relations:
            G.add_edge(src, tgt, relation=relation)
        return G

    def compute_node_embeddings(self):
        """
        Compute a semantic embedding for each node by aggregating its chunk_texts.
        Also adds the embedding into the Chroma vector database.
        """
        self.embeddings = {}
        for entity, node_data in self.nodes.items():
            # Aggregate all chunk texts; if none available, fallback to the entity name.
            aggregated_text = " ".join(node_data["chunk_texts"]) if node_data["chunk_texts"] else entity
            embedding = self.embedder.encode(aggregated_text)
            self.embeddings[node_data["id"]] = embedding
            # Add the embedding to the Chroma vector DB
            self.collection.add(
                documents=[aggregated_text],
                metadatas=[{"entity": entity}],
                ids=[str(node_data["id"])],
                embeddings=[embedding.tolist()]
            )

    def add_semantic_edges(self):
        """
        Compute pairwise cosine similarity between node embeddings.
        If similarity exceeds the threshold, add a semantic edge.
        """
        node_ids = list(self.embeddings.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                emb1 = self.embeddings[node_ids[i]]
                emb2 = self.embeddings[node_ids[j]]
                sim = 1 - cosine(emb1, emb2)
                if sim >= self.semantic_threshold:
                    # Add semantic similarity edges in both directions.
                    self.relations.append((node_ids[i], node_ids[j], "semantic_sim"))
                    self.relations.append((node_ids[j], node_ids[i], "semantic_sim"))

    def get_graph_data(self):
        """
        Convert the nodes and relations into a PyTorch Geometric Data object.
        Uses node embeddings as feature vectors.
        """
        edge_index = torch.tensor(
            [(src, tgt) for src, tgt, _ in self.relations],
            dtype=torch.long
        ).t().contiguous()
        edge_attr = [relation for _, _, relation in self.relations]
        num_nodes = len(self.nodes)
        # Create feature matrix using computed embeddings.
        x = torch.stack([torch.tensor(self.embeddings[i], dtype=torch.float) for i in range(num_nodes)])
        metadata = {
            node_data["id"]: {
                "entity": node_data["entity"],
                "chunk_texts": node_data["chunk_texts"],
                "labels": node_data["labels"]
            }
            for node_data in self.nodes.values()
        }
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, metadata=metadata)

    def save_as_gml(self, output_path):
        """
        Save the NetworkX graph to a GML file.
        """
        G = self.to_networkx()
        nx.write_gml(G, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construct a Knowledge Graph with Semantic Edges and Chroma Vector DB")
    parser.add_argument("--json_path", type=str, default="enriched_data_ner_relation.json", help="Path to the JSON data file")
    parser.add_argument("--output_gml", type=str, default="output_graph.gml", help="Path to output GML file")
    parser.add_argument("--semantic_threshold", type=float, default=0.8, help="Threshold for semantic similarity edge creation")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer embedding model name")
    args = parser.parse_args()

    # Initialize the dataset and construct the initial graph
    kg_dataset = KnowledgeGraphDataset(
        json_path=args.json_path,
        embedding_model_name=args.embedding_model,
        semantic_threshold=args.semantic_threshold
    )
    kg_dataset.construct_graph()
    
    # Compute node embeddings and store them in the vector DB
    kg_dataset.compute_node_embeddings()
    
    # Add additional semantic edges based on embedding similarity
    kg_dataset.add_semantic_edges()
    
    # Convert to PyTorch Geometric Data (if needed for downstream ML tasks)
    graph_data = kg_dataset.get_graph_data()
    
    # Save the graph to a GML file for visualization or further analysis
    kg_dataset.save_as_gml(args.output_gml)
    
    print("Knowledge Graph construction complete. GML file saved to", args.output_gml)
