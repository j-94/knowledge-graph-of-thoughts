"""
Bookmark Transformer Module.

This module provides functionality to generate embeddings and transform bookmark data
for use in knowledge graph construction.
"""

import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from data_analysis.src.data_loaders.bookmark_loader import Bookmark

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generates vector embeddings for bookmark content using sentence-transformers.
    Handles batching, caching, and efficient processing of bookmark text.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None, 
                 batch_size: int = 64, show_progress: bool = True):
        """
        Initialize the embedding generator with the specified model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
            cache_dir: Directory to store embedding cache.
            batch_size: Batch size for processing.
            show_progress: Whether to show progress bars.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.show_progress = show_progress
        
        # Setup caching
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = Path(cache_dir) / f"embedding_cache_{model_name.replace('/', '_')}.pkl"
            self.load_cache()
        else:
            self.cache = {}
    
    def load_cache(self):
        """Load embedding cache if it exists."""
        if hasattr(self, 'cache_file') and self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save embedding cache to disk."""
        if hasattr(self, 'cache_file') and self.cache:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                logger.info(f"Saved {len(self.cache)} embeddings to cache {self.cache_file}")
            except Exception as e:
                logger.warning(f"Failed to save embedding cache: {e}")
    
    def get_text_for_embedding(self, bookmark: Bookmark) -> str:
        """
        Prepare text from a bookmark for embedding generation.
        
        Args:
            bookmark: The bookmark object.
            
        Returns:
            A string containing the most relevant text for embedding.
        """
        # Prioritize content if available
        if bookmark.content and len(bookmark.content.strip()) > 10:
            text = bookmark.content
        # Fall back to title + description if content is missing or too short
        else:
            parts = [bookmark.title]
            if bookmark.description:
                parts.append(bookmark.description)
            text = " ".join(parts)
        
        # Truncate very long texts to the first ~5000 characters
        # This is a heuristic - the beginning of content often contains the most important information
        max_len = 5000
        if len(text) > max_len:
            text = text[:max_len]
            
        return text
    
    def generate_bookmark_embedding(self, bookmark: Bookmark) -> np.ndarray:
        """
        Generate an embedding for a single bookmark.
        
        Args:
            bookmark: The bookmark to generate an embedding for.
            
        Returns:
            Numpy array with the embedding vector.
        """
        text = self.get_text_for_embedding(bookmark)
        
        # Check cache first
        cache_key = f"{bookmark.id}_{hash(text)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(text, show_progress_bar=False)
        
        # Cache the result
        self.cache[cache_key] = embedding
        return embedding
    
    def batch_embed_bookmarks(self, bookmarks: List[Bookmark]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for a batch of bookmarks.
        
        Args:
            bookmarks: List of bookmarks to embed.
            
        Returns:
            Dictionary mapping bookmark IDs to embedding vectors.
        """
        result = {}
        texts_to_embed = []
        bookmark_ids = []
        cache_keys = []
        
        # First pass: check cache and collect texts that need embedding
        for bookmark in bookmarks:
            text = self.get_text_for_embedding(bookmark)
            cache_key = f"{bookmark.id}_{hash(text)}"
            
            if cache_key in self.cache:
                # Use cached embedding
                result[bookmark.id] = self.cache[cache_key]
            else:
                # Queue for embedding
                texts_to_embed.append(text)
                bookmark_ids.append(bookmark.id)
                cache_keys.append(cache_key)
        
        # If we have texts to embed
        if texts_to_embed:
            logger.info(f"Generating embeddings for {len(texts_to_embed)} bookmarks")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts_to_embed,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress
            )
            
            # Store results
            for i, (bookmark_id, cache_key) in enumerate(zip(bookmark_ids, cache_keys)):
                embedding = embeddings[i]
                self.cache[cache_key] = embedding
                result[bookmark_id] = embedding
            
            # Save cache after batch processing
            if hasattr(self, 'cache_file'):
                self.save_cache()
        
        return result
    
    def create_embedding_matrix(self, bookmark_embeddings: Dict[str, np.ndarray], 
                                bookmark_order: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create a matrix of embeddings for efficient similarity computation.
        
        Args:
            bookmark_embeddings: Dictionary mapping bookmark IDs to embeddings.
            bookmark_order: Optional list specifying the order of bookmarks in the matrix.
                           If None, uses the sorted dictionary keys.
            
        Returns:
            Tuple of (embedding_matrix, ordered_bookmark_ids)
        """
        if bookmark_order is None:
            bookmark_order = sorted(bookmark_embeddings.keys())
        
        # Extract embeddings in the specified order
        embedding_matrix = np.vstack([bookmark_embeddings[bm_id] for bm_id in bookmark_order])
        
        return embedding_matrix, bookmark_order
    
    def compute_similarity_matrix(self, embedding_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix from embeddings.
        
        Args:
            embedding_matrix: Matrix of embeddings, shape (n_bookmarks, embedding_dim).
            
        Returns:
            Similarity matrix, shape (n_bookmarks, n_bookmarks).
        """
        # Normalize the embeddings
        normalized = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        
        # Compute dot product of normalized vectors (cosine similarity)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        return similarity_matrix

# Optional helper functions

def embed_bookmarks(bookmarks: List[Bookmark], model_name: str = "all-MiniLM-L6-v2", 
                    cache_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Helper function to generate embeddings for a list of bookmarks.
    
    Args:
        bookmarks: List of bookmarks to embed.
        model_name: Name of the sentence-transformer model to use.
        cache_dir: Optional directory for caching embeddings.
        
    Returns:
        Dictionary mapping bookmark IDs to embedding vectors.
    """
    generator = EmbeddingGenerator(model_name=model_name, cache_dir=cache_dir)
    return generator.batch_embed_bookmarks(bookmarks) 