"""
FAISS-based Vector Store for Image Embeddings
Enables fast similarity search over large image collections
"""
import os
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple


class VectorStore:
    """
    FAISS-based vector store for efficient similarity search
    
    Supports:
    - L2 (Euclidean) distance search
    - Incremental index building
    - Save/load to disk
    - Metadata storage alongside embeddings
    """
    
    def __init__(self, dimension: int = 768, metric: str = "L2"):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension (768 for CLIP ViT-B/16)
            metric: Distance metric ("L2" or "IP" for inner product)
        """
        self.dimension = dimension
        self.metric = metric
        
        # Initialize FAISS index
        if metric == "L2":
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == "IP":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine if normalized)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'L2' or 'IP'")
        
        # Metadata storage (parallel to FAISS index)
        self.metadata = []
        
    def add(self, embeddings: np.ndarray, metadata: Optional[List[Dict]] = None):
        """
        Add embeddings to the index
        
        Args:
            embeddings: Array of shape (N, dimension)
            metadata: List of N metadata dicts (one per embedding)
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        
        # Normalize for inner product (makes it cosine similarity)
        if self.metric == "IP":
            faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Add metadata
        if metadata:
            assert len(metadata) == len(embeddings), "Metadata length must match embeddings"
            self.metadata.extend(metadata)
        else:
            # Add empty dicts if no metadata provided
            self.metadata.extend([{} for _ in range(len(embeddings))])
    
    def search(self, query_embeddings: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[List[Dict]]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embeddings: Array of shape (N_queries, dimension)
            k: Number of neighbors to return
        
        Returns:
            distances: Array of shape (N_queries, k)
            indices: Array of shape (N_queries, k)
            metadata: List of lists, metadata for each query's k neighbors
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        
        # Normalize for inner product
        if self.metric == "IP":
            faiss.normalize_L2(query_embeddings)
        
        # Search
        distances, indices = self.index.search(query_embeddings, k)
        
        # Retrieve metadata
        retrieved_metadata = []
        for query_indices in indices:
            query_metadata = [self.metadata[i] if i < len(self.metadata) else {} for i in query_indices]
            retrieved_metadata.append(query_metadata)
        
        return distances, indices, retrieved_metadata
    
    def save(self, index_path: str, metadata_path: Optional[str] = None):
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save FAISS index (.faiss or .index)
            metadata_path: Path to save metadata (.json). If None, uses index_path.json
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.faiss', '.json').replace('.index', '.json')
            if not metadata_path.endswith('.json'):
                metadata_path += '.json'
        
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'metric': self.metric,
                'count': len(self.metadata)
            }, f, indent=2)
        
        print(f"✅ Saved FAISS index to {index_path}")
        print(f"✅ Saved metadata to {metadata_path}")
    
    @classmethod
    def load(cls, index_path: str, metadata_path: Optional[str] = None) -> 'VectorStore':
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata. If None, uses index_path.json
        
        Returns:
            VectorStore instance with loaded data
        """
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.replace('.faiss', '.json').replace('.index', '.json')
            if not metadata_path.endswith('.json'):
                metadata_path += '.json'
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
        
        # Create instance
        store = cls(dimension=data['dimension'], metric=data['metric'])
        store.index = index
        store.metadata = data['metadata']
        
        print(f"✅ Loaded {data['count']} embeddings from {index_path}")
        
        return store
    
    def __len__(self):
        """Return number of vectors in index"""
        return self.index.ntotal


# Example usage
if __name__ == "__main__":
    # Create some random embeddings for demo
    dimension = 768
    n_vectors = 100
    embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
    
    # Create metadata
    metadata = [{"id": f"img_{i}", "description": f"Image {i}"} for i in range(n_vectors)]
    
    # Build index
    store = VectorStore(dimension=dimension)
    store.add(embeddings, metadata)
    
    # Search
    query = np.random.randn(1, dimension).astype(np.float32)
    distances, indices, retrieved_metadata = store.search(query, k=5)
    
    print(f"Top 5 matches:")
    for i, (dist, idx, meta) in enumerate(zip(distances[0], indices[0], retrieved_metadata[0])):
        print(f"  {i+1}. Index {idx}, Distance {dist:.4f}, Metadata: {meta}")
    
    # Save and load
    store.save("demo_index.faiss")
    loaded_store = VectorStore.load("demo_index.faiss")
    print(f"Loaded store has {len(loaded_store)} vectors")
