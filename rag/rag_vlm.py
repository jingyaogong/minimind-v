"""
RAG-Enhanced Vision-Language Model
Combines retrieval with generation for few-shot reasoning
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Union, List, Dict
from PIL import Image
from model.model_vlm import MiniMindVLM
from rag.retriever import CLIPRetriever
from rag.vector_store import VectorStore


class RAG_VLM:
    """
    RAG-enhanced VLM for improved generation through retrieval
    
    Workflow:
    1. Encode query image with CLIP
    2. Retrieve k similar images from knowledge base
    3. Build few-shot prompt with retrieved examples
    4. Generate response with VLM
    """
    
    def __init__(
        self,
        vlm_model: MiniMindVLM,
        tokenizer,
        vector_store: VectorStore,
        retriever: CLIPRetriever = None,
        k: int = 3
    ):
        """
        Initialize RAG-VLM
        
        Args:
            vlm_model: MiniMindVLM instance
            tokenizer: Tokenizer for VLM
            vector_store: VectorStore with image knowledge base
            retriever: CLIPRetriever instance (creates new one if None)
            k: Number of examples to retrieve
        """
        self.model = vlm_model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.k = k
        
        # Create retriever if not provided
        if retriever is None:
            self.retriever = CLIPRetriever()
        else:
            self.retriever = retriever
    
    def build_few_shot_prompt(
        self,
        base_prompt: str,
        retrieved_metadata: List[Dict],
        format_fn=None
    ) -> str:
        """
        Build few-shot prompt from retrieved examples
        
        Args:
            base_prompt: Original user prompt
            retrieved_metadata: List of metadata dicts from retrieval
            format_fn: Optional function to format each example
        
        Returns:
            Augmented prompt with examples
        """
        if format_fn is None:
            # Default formatting
            format_fn = lambda meta: f"Example: {meta.get('description', 'No description')}\n"
        
        # Build context from retrieved examples
        context = "Here are some similar examples:\n\n"
        for i, meta in enumerate(retrieved_metadata, 1):
            context += f"{i}. {format_fn(meta)}\n"
        
        # Combine with user prompt
        augmented_prompt = f"{context}\nNow, {base_prompt}"
        
        return augmented_prompt
    
    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        use_rag: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        format_fn=None
    ) -> Dict:
        """
        Generate response with optional RAG
        
        Args:
            image: Query image or path
            prompt: User prompt
            use_rag: Whether to use retrieval-augmented generation
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            format_fn: Optional function to format retrieved examples
        
        Returns:
            Dict with 'response', 'retrieved_metadata', 'augmented_prompt'
        """
        # Load image if path
        if isinstance(image, str):
            image = Image.open(image)
        
        retrieved_metadata = []
        augmented_prompt = prompt
        
        # Retrieve if RAG enabled
        if use_rag:
            distances, retrieved_metadata = self.retriever.retrieve(
                image,
                self.vector_store,
                k=self.k
            )
            
            # Build few-shot prompt
            augmented_prompt = self.build_few_shot_prompt(
                prompt,
                retrieved_metadata,
                format_fn=format_fn
            )
            
            print(f"\n📚 Retrieved {len(retrieved_metadata)} examples")
            print(f"   Distances: {distances}")
        
        # Generate with VLM
        # Note: Actual generation code depends on MiniMindVLM's interface
        # This is a placeholder showing the concept
        
        # For now, return structure (actual generation would call model.generate())
        return {
            'response': augmented_prompt,  # Placeholder: would be actual generation
            'retrieved_metadata': retrieved_metadata,
            'augmented_prompt': augmented_prompt,
            'retrieval_distances': distances.tolist() if use_rag else []
        }
    
    def evaluate_retrieval(
        self,
        test_images: List[Union[str, Image.Image]],
        ground_truth_labels: List[str],
        metadata_label_fn=lambda meta: meta.get('label', '')
    ) -> Dict:
        """
        Evaluate retrieval quality
        
        Args:
            test_images: List of test images
            ground_truth_labels: True labels for each test image
            metadata_label_fn: Function to extract label from metadata
        
        Returns:
            Dict with Recall@K, MRR, etc.
        """
        recall_at_k = 0
        mrr = 0
        
        for image, true_label in zip(test_images, ground_truth_labels):
            distances, retrieved_metadata = self.retriever.retrieve(
                image,
                self.vector_store,
                k=self.k
            )
            
            # Check if true label in retrieved results
            retrieved_labels = [metadata_label_fn(meta) for meta in retrieved_metadata]
            
            if true_label in retrieved_labels:
                recall_at_k += 1
                # Mean reciprocal rank
                rank = retrieved_labels.index(true_label) + 1
                mrr += 1.0 / rank
        
        recall_at_k /= len(test_images)
        mrr /= len(test_images)
        
        return {
            f'Recall@{self.k}': recall_at_k,
            'MRR': mrr
        }


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Load VLM
    from model.model_vlm import VLMConfig
    config = VLMConfig()
    vlm = MiniMindVLM(config)
    tokenizer = AutoTokenizer.from_pretrained('./model')
    
    # Load vector store (assuming it exists)
    # vector_store = VectorStore.load("knowledge_base.faiss")
    
    # Create RAG-VLM
    # rag_vlm = RAG_VLM(vlm, tokenizer, vector_store, k=3)
    
    # Generate with RAG
    # result = rag_vlm.generate(
    #     image="test_image.jpg",
    #     prompt="Describe this scene in detail",
    #     use_rag=True
    # )
    
    print("RAG-VLM module ready!")
