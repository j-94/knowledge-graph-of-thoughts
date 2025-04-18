#!/usr/bin/env python
"""
Bookmark Embedding Generation Script.

This script demonstrates loading bookmarks from a JSONL file and generating embeddings.
It serves as an example of how to use the BookmarkLoader and EmbeddingGenerator classes.
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path to import local modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_analysis.src.data_loaders.bookmark_loader import BookmarkLoader, Bookmark
from data_analysis.src.transformers.bookmark_transformer import EmbeddingGenerator
from kgot.utils.bookmark_config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate embeddings for bookmarks')
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input JSONL file containing bookmarks'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/bookmark_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to output directory for embeddings'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Name of sentence-transformer model to use'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help='Batch size for embedding generation'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit the number of bookmarks to process'
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments
    input_file = args.input or config.paths.raw_data
    output_dir = args.output or config.paths.cache_dir
    model_name = args.model or config.processing.embedding_model
    batch_size = args.batch_size or config.processing.batch_size
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize loader and generator
    logger.info(f"Loading bookmarks from {input_file}")
    loader = BookmarkLoader()
    
    # Load bookmarks
    bookmarks = loader.load_bookmarks(input_file)
    
    # Apply limit if specified
    if args.limit and args.limit < len(bookmarks):
        logger.info(f"Limiting to {args.limit} bookmarks")
        bookmarks = bookmarks[:args.limit]
    
    # Clean and normalize bookmarks
    logger.info("Cleaning and normalizing bookmarks")
    cleaned_bookmarks = loader.clean_and_normalize(bookmarks)
    
    # Initialize embedding generator
    logger.info(f"Initializing embedding generator with model {model_name}")
    generator = EmbeddingGenerator(
        model_name=model_name,
        cache_dir=output_dir,
        batch_size=batch_size
    )
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(cleaned_bookmarks)} bookmarks")
    embeddings = generator.batch_embed_bookmarks(cleaned_bookmarks)
    
    # Create embedding matrix and compute similarity
    logger.info("Creating embedding matrix")
    embedding_matrix, bookmark_ids = generator.create_embedding_matrix(embeddings)
    
    logger.info("Computing similarity matrix")
    similarity_matrix = generator.compute_similarity_matrix(embedding_matrix)
    
    # Save similarity matrix and bookmark IDs
    similarity_path = Path(output_dir) / 'similarity_matrix.npy'
    ids_path = Path(output_dir) / 'bookmark_ids.json'
    
    np.save(similarity_path, similarity_matrix)
    with open(ids_path, 'w') as f:
        import json
        json.dump(bookmark_ids, f)
    
    logger.info(f"Saved similarity matrix to {similarity_path}")
    logger.info(f"Saved bookmark IDs to {ids_path}")
    
    # Display summary statistics
    logger.info(f"Processed {len(cleaned_bookmarks)} bookmarks")
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Show top 5 most similar bookmark pairs
    if len(bookmarks) > 1:
        logger.info("Top 5 most similar bookmark pairs:")
        
        # Create mask for upper triangle to avoid duplicates and self-similarities
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]
        
        # Get indices of top 5 similarities
        if len(similarities) >= 5:
            top_indices = np.argsort(similarities)[-5:][::-1]
            
            # Convert flat indices to 2D indices
            rows, cols = np.where(mask)
            top_pairs = [(rows[idx], cols[idx]) for idx in top_indices]
            
            # Display similar pairs
            for i, (idx1, idx2) in enumerate(top_pairs):
                bm1 = cleaned_bookmarks[idx1]
                bm2 = cleaned_bookmarks[idx2]
                sim = similarity_matrix[idx1, idx2]
                logger.info(f"{i+1}. Similarity: {sim:.4f}")
                logger.info(f"   - {bm1.title} ({bm1.id})")
                logger.info(f"   - {bm2.title} ({bm2.id})")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 