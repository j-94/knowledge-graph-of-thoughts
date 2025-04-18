"""
Bookmark Data Loader Module.

This module provides functionality to load and validate bookmark data from JSONL files
based on the Bookmark schema.
"""

import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, validator

from .jsonl_loader import JsonlDataLoader

# Configure logging
logger = logging.getLogger(__name__)

class Bookmark(BaseModel):
    """
    Pydantic model representing a bookmark entry.
    Provides validation and type checking for bookmark data.
    """
    id: str
    title: str
    url: str
    tags: List[str]
    timestamp: datetime
    description: Optional[str] = None
    content: str
    source: str
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('url')
    def validate_url(cls, v):
        """Validate that the URL has a proper format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate that tags are non-empty."""
        if not v:
            return []
        if any(not isinstance(tag, str) or not tag.strip() for tag in v):
            raise ValueError('Tags must be non-empty strings')
        return [tag.strip() for tag in v]

class BookmarkLoader(JsonlDataLoader):
    """
    Data loader for bookmark data stored in JSONL format.
    Extends JsonlDataLoader with bookmark-specific validation and processing.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the bookmark data loader.
        
        Args:
            verbose: Whether to display progress bars and logging information.
        """
        super().__init__(verbose=verbose)
    
    def load_bookmarks(self, file_path: Union[str, Path], 
                       validate: bool = True, 
                       encoding: str = 'utf-8',
                       **kwargs) -> List[Bookmark]:
        """
        Load bookmarks from a JSONL file and validate against the Bookmark schema.
        
        Args:
            file_path: Path to the JSONL file containing bookmark data.
            validate: Whether to validate each entry against the Bookmark schema.
            encoding: File encoding to use.
            **kwargs: Additional arguments for JsonlDataLoader.load.
            
        Returns:
            A list of validated Bookmark objects.
        """
        # Load raw data using parent class method
        df = self.load(file_path, encoding=encoding, **kwargs)
        
        if df.empty:
            logger.warning(f"No bookmark data found in {file_path}")
            return []
        
        # Convert DataFrame to list of dicts
        bookmark_dicts = df.to_dict(orient='records')
        
        if validate:
            # Validate and create Bookmark objects
            valid_bookmarks = []
            errors = []
            
            for i, bookmark_dict in enumerate(bookmark_dicts):
                try:
                    # Create a Bookmark object which will validate the data
                    bookmark = Bookmark(**bookmark_dict)
                    valid_bookmarks.append(bookmark)
                except Exception as e:
                    errors.append((i, str(e)))
            
            # Log validation errors
            if errors:
                logger.warning(f"Found {len(errors)} invalid bookmarks out of {len(bookmark_dicts)}")
                for i, error in errors[:10]:  # Show first 10 errors
                    logger.warning(f"  Row {i}: {error}")
                if len(errors) > 10:
                    logger.warning(f"  ... and {len(errors) - 10} more errors")
            
            logger.info(f"Successfully loaded and validated {len(valid_bookmarks)} bookmarks")
            return valid_bookmarks
        else:
            # Skip validation and create Bookmark objects directly
            return [Bookmark(**bd) for bd in bookmark_dicts]
    
    def clean_and_normalize(self, bookmarks: List[Bookmark]) -> List[Bookmark]:
        """
        Perform data cleaning and normalization on a list of bookmarks.
        
        Args:
            bookmarks: List of Bookmark objects to clean and normalize.
            
        Returns:
            List of cleaned and normalized Bookmark objects.
        """
        cleaned_bookmarks = []
        
        for bookmark in bookmarks:
            # Create a dictionary from the bookmark for easier manipulation
            bookmark_dict = bookmark.dict()
            
            # Clean and normalize fields
            if bookmark_dict.get('title'):
                bookmark_dict['title'] = bookmark_dict['title'].strip()
            
            if bookmark_dict.get('content'):
                # Remove excessive whitespace
                bookmark_dict['content'] = ' '.join(bookmark_dict['content'].split())
            
            if bookmark_dict.get('tags'):
                # Normalize tags: lowercase, remove duplicates
                tags = [tag.lower().strip() for tag in bookmark_dict['tags']]
                bookmark_dict['tags'] = list(set(tags))
            
            # Create a new Bookmark with cleaned data
            cleaned_bookmarks.append(Bookmark(**bookmark_dict))
        
        return cleaned_bookmarks
    
    def export_to_pandas(self, bookmarks: List[Bookmark]) -> pd.DataFrame:
        """
        Convert a list of Bookmark objects to a pandas DataFrame.
        
        Args:
            bookmarks: List of Bookmark objects.
            
        Returns:
            A pandas DataFrame containing the bookmark data.
        """
        # Convert Bookmark objects to dictionaries
        bookmark_dicts = [bookmark.dict() for bookmark in bookmarks]
        
        # Create DataFrame
        df = pd.DataFrame(bookmark_dicts)
        return df

# Optional: Create a helper function to load bookmarks more easily
def load_bookmarks(file_path: Union[str, Path], **kwargs) -> List[Bookmark]:
    """
    Helper function to load bookmarks from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing bookmark data.
        **kwargs: Additional arguments for BookmarkLoader.load_bookmarks.
        
    Returns:
        A list of validated Bookmark objects.
    """
    loader = BookmarkLoader()
    return loader.load_bookmarks(file_path, **kwargs) 