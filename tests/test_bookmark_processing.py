"""
Tests for bookmark processing modules.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest
import numpy as np
import pandas as pd

from data_analysis.src.data_loaders.bookmark_loader import BookmarkLoader, Bookmark
from data_analysis.src.transformers.bookmark_transformer import EmbeddingGenerator

# Sample bookmark data for testing
SAMPLE_BOOKMARKS = [
    {
        "id": "bm1",
        "title": "Example Bookmark 1",
        "url": "https://example.com/1",
        "tags": ["test", "example"],
        "timestamp": "2023-01-01T12:00:00Z",
        "description": "This is a test bookmark",
        "content": "This is the content of the first test bookmark.",
        "source": "manual",
        "metadata": {"category": "testing"}
    },
    {
        "id": "bm2",
        "title": "Example Bookmark 2",
        "url": "https://example.com/2",
        "tags": ["test", "sample"],
        "timestamp": "2023-01-02T12:00:00Z",
        "description": None,
        "content": "This is the content of the second test bookmark. It has more text to test different embedding lengths.",
        "source": "import",
        "metadata": None
    }
]

@pytest.fixture
def sample_jsonl_file():
    """Create a temporary JSONL file with sample bookmark data."""
    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False, mode='w') as f:
        for bookmark in SAMPLE_BOOKMARKS:
            f.write(json.dumps(bookmark) + '\n')
    
    file_path = Path(f.name)
    yield file_path
    
    # Cleanup
    if file_path.exists():
        os.unlink(file_path)

@pytest.fixture
def sample_bookmarks():
    """Create sample Bookmark objects."""
    return [Bookmark(**bm) for bm in SAMPLE_BOOKMARKS]

class TestBookmarkLoader:
    """Tests for the BookmarkLoader class."""
    
    def test_load_bookmarks(self, sample_jsonl_file):
        """Test loading bookmarks from a JSONL file."""
        loader = BookmarkLoader()
        bookmarks = loader.load_bookmarks(sample_jsonl_file)
        
        # Check that all bookmarks were loaded and validated
        assert len(bookmarks) == len(SAMPLE_BOOKMARKS)
        assert all(isinstance(b, Bookmark) for b in bookmarks)
        
        # Check specific bookmark fields
        assert bookmarks[0].id == "bm1"
        assert bookmarks[0].title == "Example Bookmark 1"
        assert bookmarks[1].url == "https://example.com/2"
        assert bookmarks[1].description is None
        
        # Check datetime parsing
        assert isinstance(bookmarks[0].timestamp, datetime)
    
    def test_clean_and_normalize(self, sample_bookmarks):
        """Test cleaning and normalizing bookmarks."""
        # Add some messy data
        messy_bookmark = Bookmark(
            id="bm3",
            title="  Messy Title  ",
            url="https://example.com/3",
            tags=["  Test  ", "test", "EXAMPLE"],
            timestamp=datetime.now(),
            content="  Multiple    spaces   in   content  ",
            source="test"
        )
        
        loader = BookmarkLoader()
        cleaned = loader.clean_and_normalize([messy_bookmark])
        
        # Check that the bookmark was cleaned
        assert cleaned[0].title == "Messy Title"
        assert " Multiple spaces in content " in cleaned[0].content
        assert len(cleaned[0].tags) == 2  # Duplicates removed and normalized
        assert all(tag in ["test", "example"] for tag in cleaned[0].tags)
    
    def test_export_to_pandas(self, sample_bookmarks):
        """Test exporting bookmarks to pandas DataFrame."""
        loader = BookmarkLoader()
        df = loader.export_to_pandas(sample_bookmarks)
        
        # Check that the DataFrame has the right structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_bookmarks)
        assert "id" in df.columns
        assert "title" in df.columns
        assert "tags" in df.columns
        
        # Check specific values
        assert df.iloc[0]["id"] == "bm1"
        assert df.iloc[1]["title"] == "Example Bookmark 2"

class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""
    
    def test_init(self):
        """Test initializing the EmbeddingGenerator."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert hasattr(generator, "model")
    
    def test_get_text_for_embedding(self, sample_bookmarks):
        """Test preparing text for embedding."""
        generator = EmbeddingGenerator()
        
        # Test with content
        text1 = generator.get_text_for_embedding(sample_bookmarks[0])
        assert text1 == sample_bookmarks[0].content
        
        # Test with missing content
        no_content_bookmark = Bookmark(
            id="bm3",
            title="No Content Bookmark",
            url="https://example.com/3",
            tags=["test"],
            timestamp=datetime.now(),
            content="",  # Empty content
            source="test"
        )
        text2 = generator.get_text_for_embedding(no_content_bookmark)
        assert text2 == "No Content Bookmark"
        
        # Test with content and description
        with_desc_bookmark = Bookmark(
            id="bm4",
            title="With Description",
            url="https://example.com/4",
            tags=["test"],
            timestamp=datetime.now(),
            content="",  # Empty content
            description="This is a description",
            source="test"
        )
        text3 = generator.get_text_for_embedding(with_desc_bookmark)
        assert "With Description" in text3
        assert "This is a description" in text3
    
    @pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS") == "1", 
                      reason="Skipping slow tests that require model download")
    def test_generate_bookmark_embedding(self, sample_bookmarks):
        """Test generating embeddings for a single bookmark."""
        generator = EmbeddingGenerator()
        
        # Generate embedding
        embedding = generator.generate_bookmark_embedding(sample_bookmarks[0])
        
        # Check embedding properties
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1  # Should be a 1D vector
        assert embedding.shape[0] == 384  # Default model produces 384-dim embeddings
    
    @pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS") == "1", 
                      reason="Skipping slow tests that require model download")
    def test_batch_embed_bookmarks(self, sample_bookmarks):
        """Test generating embeddings for multiple bookmarks."""
        generator = EmbeddingGenerator()
        
        # Generate embeddings for all bookmarks
        embeddings = generator.batch_embed_bookmarks(sample_bookmarks)
        
        # Check results
        assert len(embeddings) == len(sample_bookmarks)
        assert all(isinstance(emb, np.ndarray) for emb in embeddings.values())
        assert all(bm.id in embeddings for bm in sample_bookmarks)
        
        # Check embedding dimensions
        assert all(emb.shape[0] == 384 for emb in embeddings.values())
    
    @pytest.mark.skipif(os.environ.get("SKIP_SLOW_TESTS") == "1", 
                      reason="Skipping slow tests that require model download")
    def test_similarity_computation(self, sample_bookmarks):
        """Test computing similarity between bookmark embeddings."""
        generator = EmbeddingGenerator()
        
        # Generate embeddings
        embeddings = generator.batch_embed_bookmarks(sample_bookmarks)
        
        # Create embedding matrix
        matrix, ids = generator.create_embedding_matrix(embeddings)
        
        # Compute similarity
        sim_matrix = generator.compute_similarity_matrix(matrix)
        
        # Check similarity matrix properties
        assert sim_matrix.shape == (len(sample_bookmarks), len(sample_bookmarks))
        assert np.allclose(np.diag(sim_matrix), 1.0)  # Self-similarity should be 1.0
        assert np.all(sim_matrix >= -1.0) and np.all(sim_matrix <= 1.0)  # Cosine similarity bounds 