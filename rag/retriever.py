"""
CLIP-based Image Retriever for RAG System
Uses frozen CLIP ViT-B/16 from MiniMindVLM for visual similarity
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Dict, Tuple
from model.model_vlm import MiniMindVLM
from rag.vector_store import VectorStore


class CLIPRetriever:
    """
    CLIP-based retriever for visual similarity search
    
    Uses the same CLIP ViT-B/16 model as MiniMindVLM for consistency
    """
    
    def __init__(
        self,
        vision_model_path: str = "./model/vision_model/clip-vit-base-patch16",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize CLIP retriever
        
        Args:
            vision_model_path: Path to CLIP model
            device: Device to run on
        """
        self.device = device
        
        # Load CLIP model (reuse MiniMindVLM's CLIP loader)
        self.vision_model, self.processor = MiniMindVLM.get_vision_model(vision_model_path)
        
        if self.vision_model is None:
            raise ValueError(f"CLIP model not found at {vision_model_path}")
        
        self.vision_model = self.vision_model.to(device)
        self.vision_model.eval()
        
        print(f"✅ Loaded CLIP ViT-B/16 on {device}")
    
    def encode_image(self, image: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode a single image to CLIP embedding
        
        Args:
            image: PIL Image or path to image
        
        Returns:
            Embedding as numpy array of shape (768,)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image)
        
        # Convert image to tensor
        image_tensor = MiniMindVLM.image2tensor(image, self.processor).to(self.device)
        
        # Get CLIP embedding
        with torch.no_grad():
            embedding = MiniMindVLM.get_image_embeddings(image_tensor, self.vision_model)
            # Pool patch embeddings to single vector
            embedding = embedding.mean(dim=0).cpu().numpy()  # Shape: (768,)
        
        return embedding
    
    def encode_images_batch(self, images: List[Union[str, Image.Image]], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple images in batches
        
        Args:
            images: List of PIL Images or paths
            batch_size: Batch size for encoding
        
        Returns:
            Embeddings as numpy array of shape (N, 768)
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Load images
            batch_images = []
            for img in batch:
                if isinstance(img, str):
                    img = Image.open(img)
                batch_images.append(img)
            
            # Convert to tensors
            # image2tensor returns (1, C, H, W), so we concat on batch dim
            batch_tensors = torch.cat([
                MiniMindVLM.image2tensor(img, self.processor)
                for img in batch_images
            ], dim=0).to(self.device)
            
            # Encode
            with torch.no_grad():
                batch_embeddings = MiniMindVLM.get_image_embeddings(batch_tensors, self.vision_model)
                # Pool: (batch, patches, 768) -> (batch, 768)
                batch_embeddings = batch_embeddings.mean(dim=1).cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def build_index(
        self,
        images: List[Union[str, Image.Image]],
        metadata: List[Dict],
        save_path: str,
        batch_size: int = 32
    ) -> VectorStore:
        """
        Build FAISS index from images
        
        Args:
            images: List of images or paths
            metadata: List of metadata dicts (one per image)
            save_path: Where to save index
            batch_size: Encoding batch size
        
        Returns:
            VectorStore with built index
        """
        print(f"🔨 Building index from {len(images)} images...")
        
        # Encode all images
        embeddings = self.encode_images_batch(images, batch_size=batch_size)
        print(f"✅ Encoded {len(embeddings)} images to {embeddings.shape[1]}-dim vectors")
        
        # Build vector store
        store = VectorStore(dimension=embeddings.shape[1])
        store.add(embeddings, metadata)
        
        # Save
        store.save(save_path)
        
        return store
    
    def retrieve(
        self,
        query_image: Union[str, Image.Image],
        vector_store: VectorStore,
        k: int = 5
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Retrieve k most similar images from index
        
        Args:
            query_image: Query image or path
            vector_store: VectorStore to search in
            k: Number of results to return
        
        Returns:
            distances: Array of shape (k,) with distances
            metadata: List of k metadata dicts
        """
        # Encode query
        query_embedding = self.encode_image(query_image)
        query_embedding = query_embedding.reshape(1, -1)  # (1, 768)
        
        # Search
        distances, indices, metadata = vector_store.search(query_embedding, k=k)
        
        # Return flattened results (since we only have 1 query)
        return distances[0], metadata[0]


# Example usage
if __name__ == "__main__":
    # Create retriever
    retriever = CLIPRetriever()
    
    # Example: encode single image
    # img_path = "dataset/eval_images/城市车水马龙-city-traffic.jpg"
    # embedding = retriever.encode_image(img_path)
    # print(f"Image embedding shape: {embedding.shape}")
