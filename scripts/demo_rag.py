#!/usr/bin/env python3
"""
RAG System Demo
Demonstrates retrieval-augmented generation with MiniMindVLM
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from PIL import Image
from rag.retriever import CLIPRetriever
from rag.vector_store import VectorStore


def demo_retrieval(query_image_path: str, index_path: str, k: int = 3):
    """
    Demo RAG retrieval
    
    Args:
        query_image_path: Path to query image
        index_path: Path to FAISS index
        k: Number of results to retrieve
    """
    print(f"\n{'='*70}")
    print("🔍 RAG Retrieval Demo")
    print(f"{'='*70}\n")
    
    # Load query image
    query_image = Image.open(query_image_path)
    print(f"📷 Query image: {query_image_path}")
    print(f"   Size: {query_image.size}")
    print()
    
    # Load vector store
    print(f"📂 Loading knowledge base from {index_path}...")
    vector_store = VectorStore.load(index_path)
    print()
    
    # Create retriever
    print("🔧 Initializing CLIP retriever...")
    retriever = CLIPRetriever()
    print()
    
    # Retrieve similar images
    print(f"🔍 Retrieving top-{k} similar images...")
    distances, metadata = retriever.retrieve(query_image, vector_store, k=k)
    print()
    
    # Display results
    print(f"{'='*70}")
    print(f"📊 Top-{k} Most Similar Images:")
    print(f"{'='*70}\n")
    
    for i, (dist, meta) in enumerate(zip(distances, metadata), 1):
        print(f"{i}. {meta['filename']}")
        print(f"   Distance: {dist:.4f}")
        print(f"   Description: {meta.get('description', 'No description')}")
        print(f"   Path: {meta.get('path', 'Unknown')}")
        print()
    
    print(f"{'='*70}\n")
    
    # Recommendation
    if distances[0] < 5.0:
        print("✅ Retrieved very similar images (distance < 5.0)")
    elif distances[0] < 10.0:
        print("⚠️  Retrieved moderately similar images (distance < 10.0)")
    else:
        print("❌ No very similar images found (distance >= 10.0)")
    
    print(f"\n{'='*70}\n")
    
    return distances, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo RAG retrieval")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Path to query image"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="rag_demo_index.faiss",
        help="Path to FAISS index"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Number of results to retrieve"
    )
    
    args = parser.parse_args()
    
    demo_retrieval(args.query, args.index, args.k)
