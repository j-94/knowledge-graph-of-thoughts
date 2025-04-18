"""
Data Loader Factory Module.

This module provides a factory class for creating appropriate data loaders
based on the file format or other criteria.
"""

import logging
from typing import Optional, Type, Dict

from .base_loader import BaseDataLoader
from .jsonl_loader import JsonlDataLoader
from .csv_loader import CsvDataLoader
from .json_loader import JsonDataLoader
from .text_loader import StructuredTextLoader
from .excel_loader import ExcelDataLoader

# Import database loader if available
try:
    from .database_loader import DatabaseLoader
    HAS_DATABASE_LOADER = True
except ImportError:
    HAS_DATABASE_LOADER = False

# Configure logging
logger = logging.getLogger(__name__)


class DataLoaderFactory:
    """
    Factory class for creating data loader instances.
    
    This class provides methods to get the appropriate data loader
    for a given file format or data source.
    """
    
    def __init__(self):
        """
        Initialize the data loader factory.
        """
        # Register standard loader types
        self._loaders: Dict[str, Type[BaseDataLoader]] = {
            'jsonl': JsonlDataLoader,
            'csv': CsvDataLoader,
            'json': JsonDataLoader,
            'text': StructuredTextLoader,
            'excel': ExcelDataLoader,
        }
        
        # Register database loader if available
        if HAS_DATABASE_LOADER:
            self._loaders['database'] = DatabaseLoader
    
    def register_loader(self, format_name: str, loader_class: Type[BaseDataLoader]) -> None:
        """
        Register a new loader class for a specific format.
        
        Args:
            format_name: The name of the format (e.g., 'parquet', 'hdf5').
            loader_class: The loader class to register.
            
        Raises:
            ValueError: If the loader class doesn't inherit from BaseDataLoader.
        """
        if not issubclass(loader_class, BaseDataLoader):
            raise ValueError(f"Loader class must inherit from BaseDataLoader")
        
        self._loaders[format_name.lower()] = loader_class
        logger.info(f"Registered loader for format: {format_name}")
    
    def get_loader(self, format_name: str, verbose: bool = True) -> BaseDataLoader:
        """
        Get a loader instance for the specified format.
        
        Args:
            format_name: The name of the format (e.g., 'jsonl', 'csv').
            verbose: Whether the loader should display progress bars and detailed logging.
            
        Returns:
            An instance of the appropriate data loader.
            
        Raises:
            ValueError: If no loader is registered for the specified format.
        """
        format_name = format_name.lower()
        
        if format_name not in self._loaders:
            registered_formats = ", ".join(self._loaders.keys())
            raise ValueError(
                f"No loader registered for format: {format_name}. "
                f"Registered formats: {registered_formats}"
            )
        
        loader_class = self._loaders[format_name]
        logger.debug(f"Creating loader for format: {format_name}")
        
        return loader_class(verbose=verbose)
    
    def get_available_formats(self) -> list[str]:
        """
        Get a list of all available formats that have registered loaders.
        
        Returns:
            A list of format names.
        """
        return list(self._loaders.keys()) 