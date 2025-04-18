"""
JSON Data Loader Module.

This module provides functionality to load JSON data
into pandas DataFrames with support for various options.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Iterator, Callable

import pandas as pd
import numpy as np
import ijson

from .base_loader import BaseDataLoader

# Configure logging
logger = logging.getLogger(__name__)


class JsonDataLoader(BaseDataLoader):
    """
    Data loader for JSON files.
    
    This loader supports parsing standard JSON files into pandas DataFrames,
    with options for handling nested structures and large files.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the JSON data loader.
        
        Args:
            verbose: Whether to display progress bars and logging information.
        """
        super().__init__(verbose=verbose)
    
    def load(self, file_path: Union[str, Path],
             record_path: Optional[Union[str, List[str]]] = None,
             orient: Optional[str] = None,
             flatten_nested: bool = False,
             flatten_separator: str = '.',
             max_level: Optional[int] = None,
             encoding: str = 'utf-8',
             **kwargs) -> pd.DataFrame:
        """
        Load data from a JSON file into a pandas DataFrame.
        
        Args:
            file_path: Path to the JSON file to load.
            record_path: Path to the JSON objects in a nested JSON structure.
                Can be a string or a list of strings for nested paths.
            orient: Orient format of the JSON data, as expected by pandas.json_normalize.
                Options include: 'split', 'records', 'index', 'columns', 'values'.
            flatten_nested: Whether to flatten nested JSON structures into columns.
            flatten_separator: Separator to use when flattening nested column names.
            max_level: Maximum nesting level to flatten if flatten_nested is True.
            encoding: File encoding to use.
            **kwargs: Additional arguments for pandas.read_json or json_normalize.
            
        Returns:
            A pandas DataFrame containing the parsed JSON data.
            
        Raises:
            ValueError: If the JSON file cannot be parsed properly.
        """
        # Validate the file
        file_path = self.validate_file(file_path)
        
        try:
            logger.info(f"Loading JSON file: {file_path}")
            
            # Check file size to determine appropriate loading method
            file_size = file_path.stat().st_size
            is_large_file = file_size > 100 * 1024 * 1024  # 100 MB threshold
            
            if is_large_file:
                logger.info(f"JSON file is large ({file_size / (1024 * 1024):.2f} MB), using streaming parser")
                return self._load_large_json(
                    file_path, record_path, flatten_nested, 
                    flatten_separator, max_level, encoding
                )
            
            # Standard loading for regular-sized files
            with open(file_path, 'r', encoding=encoding) as f:
                try:
                    # Load JSON data
                    if orient:
                        # Use pandas read_json with specific orient format
                        df = pd.read_json(f, orient=orient, **kwargs)
                    else:
                        # Load JSON and convert to DataFrame
                        data = json.load(f)
                        
                        # Handle different JSON structures
                        if isinstance(data, dict):
                            # For dictionary data, check if it needs normalization
                            if record_path:
                                # Navigate to the records path
                                records = data
                                if isinstance(record_path, str):
                                    path_parts = record_path.split('.')
                                else:
                                    path_parts = record_path
                                
                                for part in path_parts:
                                    if part in records:
                                        records = records[part]
                                    else:
                                        raise ValueError(f"Record path '{record_path}' not found in JSON data")
                                
                                if flatten_nested:
                                    df = pd.json_normalize(
                                        records, sep=flatten_separator, max_level=max_level, **kwargs
                                    )
                                else:
                                    df = pd.DataFrame(records)
                            else:
                                # Convert dict to dataframe (single row)
                                df = pd.DataFrame([data])
                        
                        elif isinstance(data, list):
                            # For list data, normalize if needed
                            if flatten_nested:
                                df = pd.json_normalize(
                                    data, sep=flatten_separator, max_level=max_level, **kwargs
                                )
                            else:
                                df = pd.DataFrame(data)
                        else:
                            raise ValueError("JSON data is neither a dictionary nor a list")
            
            logger.info(f"Successfully loaded {len(df)} records from JSON file")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading JSON file {file_path}", e)
            raise
    
    def _load_large_json(self, file_path: Path,
                         record_path: Optional[Union[str, List[str]]] = None,
                         flatten_nested: bool = False,
                         flatten_separator: str = '.',
                         max_level: Optional[int] = None,
                         encoding: str = 'utf-8') -> pd.DataFrame:
        """
        Load a large JSON file using streaming to avoid memory issues.
        
        Args:
            file_path: Path to the JSON file.
            record_path: Path to the JSON objects in the nested structure.
            flatten_nested: Whether to flatten nested structures.
            flatten_separator: Separator for flattened column names.
            max_level: Maximum nesting level to flatten.
            encoding: File encoding.
            
        Returns:
            A pandas DataFrame containing the parsed data.
        """
        data = []
        ijson_path = ''
        
        if record_path:
            if isinstance(record_path, str):
                ijson_path = record_path.replace('.', '.item.')
            else:
                ijson_path = '.item.'.join(record_path)
        
        try:
            with open(file_path, 'rb') as f:
                if ijson_path:
                    # Stream specified path
                    items = ijson.items(f, f'{ijson_path}.item')
                else:
                    # Stream root array elements
                    items = ijson.items(f, 'item')
                
                # Create progress bar
                file_size = file_path.stat().st_size
                progress_bar = self.create_progress_bar(
                    file_size, desc=f"Loading large JSON file: {file_path.name}"
                )
                
                for item in items:
                    data.append(item)
                    # Update progress approximately
                    pos = f.tell()
                    self.update_progress(progress_bar, pos - progress_bar.n)
                
                self.close_progress(progress_bar)
                
                # Convert to DataFrame
                if data:
                    if flatten_nested:
                        df = pd.json_normalize(
                            data, sep=flatten_separator, max_level=max_level
                        )
                    else:
                        df = pd.DataFrame(data)
                    
                    logger.info(f"Successfully loaded {len(df)} records from large JSON file")
                    return df
                else:
                    logger.warning("No data found in JSON file")
                    return pd.DataFrame()
                    
        except Exception as e:
            self.log_error(f"Error streaming large JSON file {file_path}", e)
            raise
    
    def load_streaming(self, file_path: Union[str, Path],
                       record_path: Optional[Union[str, List[str]]] = None,
                       chunk_size: int = 1000,
                       encoding: str = 'utf-8',
                       **kwargs) -> Iterator[pd.DataFrame]:
        """
        Load a JSON file in streaming mode, yielding chunks of data.
        
        Args:
            file_path: Path to the JSON file.
            record_path: Path to the JSON objects in the nested structure.
            chunk_size: Number of records per chunk.
            encoding: File encoding.
            **kwargs: Additional arguments for processing.
            
        Yields:
            DataFrame chunks containing the parsed data.
        """
        file_path = self.validate_file(file_path)
        ijson_path = ''
        
        if record_path:
            if isinstance(record_path, str):
                ijson_path = record_path.replace('.', '.item.')
            else:
                ijson_path = '.item.'.join(record_path)
        
        logger.info(f"Streaming JSON file: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                chunk = []
                
                if ijson_path:
                    # Stream specified path
                    items = ijson.items(f, f'{ijson_path}.item')
                else:
                    # Stream root array elements
                    items = ijson.items(f, 'item')
                
                for item in items:
                    chunk.append(item)
                    
                    if len(chunk) >= chunk_size:
                        yield pd.DataFrame(chunk, **kwargs)
                        chunk = []
                
                # Yield remaining items
                if chunk:
                    yield pd.DataFrame(chunk, **kwargs)
                    
        except Exception as e:
            self.log_error(f"Error streaming JSON file {file_path}", e)
            raise
    
    def validate_json(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file contains valid JSON.
        
        Args:
            file_path: Path to the file to validate.
            
        Returns:
            True if the file contains valid JSON, False otherwise.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                json.load(f)
            logger.info(f"File contains valid JSON: {file_path}")
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in {file_path}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error validating JSON file {file_path}: {str(e)}")
            return False 