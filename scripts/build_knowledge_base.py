#!/usr/bin/env python3
"""
Build RAG Knowledge Base from Images
Creates FAISS index from image directory for retrieval-augmented generation
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rag.retriever import CLIPRetriever


def build_knowledge_base(
    images_dir: str,
    output_path: str,
    description_file: str = None,
    batch_size: int = 32
):
    """
    Build knowledge base from image directory
    
    Args:
        images_dir: Directory containing images
        output_path: Where to save FAISS index
        description_file: Optional JSON file with descriptions per image
        batch_size: Batch size for encoding
    """
    print(f"\n{'='*60}")
    print("🔨 Building Knowledge Base for RAG")
    print(f"{'='*60}\n")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images_dir = Path(images_dir)
    image_files = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"📂 Found {len(image_files)} images in {images_dir}")
    
    # Load descriptions if provided
    descriptions = {}
    if description_file and os.path.exists(description_file):
        with open(description_file, 'r') as f:
            descriptions = json.load(f)
        print(f"📝 Loaded descriptions for {len(descriptions)} images")
    
    # Create metadata
    metadata = []
    image_paths = []
    
    for img_file in tqdm(image_files, desc="Preparing metadata"):
        img_name = img_file.name
        
        # Create metadata entry
        meta = {
            'id': img_file.stem,
            'filename': img_name,
            'path': str(img_file.absolute()),
            'description': descriptions.get(img_name, f"Image: {img_name}"),
        }
        
        metadata.append(meta)
        image_paths.append(str(img_file))
    
    # Create retriever
    print("\n🔧 Initializing CLIP retriever...")
    retriever = CLIPRetriever()
    
    # Build index
    print(f"\n🚀 Encoding images in batches of {batch_size}...")
    vector_store = retriever.build_index(
        images=image_paths,
        metadata=metadata,
        save_path=output_path,
        batch_size=batch_size
    )
    
    print(f"\n{'='*60}")
    print(f"✅ Knowledge Base Built Successfully!")
    print(f"{'='*60}")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Index file: {output_path}")
    print(f"  Metadata file: {output_path.replace('.faiss', '.json')}")
    print(f"  Embedding dimension: {vector_store.dimension}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build RAG knowledge base from images")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for FAISS index (.faiss)"
    )
    parser.add_argument(
        "--descriptions",
        type=str,
        default=None,
        help="Optional JSON file with image descriptions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    
    args = parser.parse_args()
    
    build_knowledge_base(
        images_dir=args.images_dir,
        output_path=args.output,
        description_file=args.descriptions,
        batch_size=args.batch_size
    )
