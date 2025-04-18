"""
Base Data Loader Module.

This module provides a base class for data loaders with common functionality
such as file validation, error handling, and progress reporting.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable
from abc import ABC, abstractmethod

import pandas as pd
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """
    Abstract base class for all data loaders.
    
    This class defines common functionality that all data loaders should implement,
    including file validation, error handling, and progress reporting.
    
    Attributes:
        verbose: Whether to display progress bars and detailed logging.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the base data loader.
        
        Args:
            verbose: Whether to display progress bars and detailed logging.
        """
        self.verbose = verbose
    
    @abstractmethod
    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from a file into a pandas DataFrame.
        
        Args:
            file_path: Path to the file to load.
            **kwargs: Additional arguments specific to the loader implementation.
            
        Returns:
            A pandas DataFrame containing the loaded data.
        """
        pass
    
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that a file exists and is accessible.
        
        Args:
            file_path: Path to the file to validate.
            
        Returns:
            A Path object representing the validated file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read due to permissions.
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            logger.error(f"Path is not a file: {path}")
            raise FileNotFoundError(f"Path is not a file: {path}")
        
        if not os.access(path, os.R_OK):
            logger.error(f"Permission denied for file: {path}")
            raise PermissionError(f"Permission denied for file: {path}")
        
        return path
    
    def get_file_size(self, file_path: Path) -> int:
        """
        Get the size of a file in bytes.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            The size of the file in bytes.
        """
        return file_path.stat().st_size
    
    def create_progress_bar(self, total: int, desc: str = "Processing") -> tqdm:
        """
        Create a progress bar for long-running operations.
        
        Args:
            total: Total number of items to process.
            desc: Description to display next to the progress bar.
            
        Returns:
            A tqdm progress bar object.
        """
        if self.verbose:
            return tqdm(total=total, desc=desc)
        return None
    
    def update_progress(self, progress_bar: Optional[tqdm], amount: int = 1) -> None:
        """
        Update a progress bar if it exists.
        
        Args:
            progress_bar: The progress bar to update.
            amount: The amount to increment the progress bar by.
        """
        if progress_bar is not None and self.verbose:
            progress_bar.update(amount)
    
    def close_progress(self, progress_bar: Optional[tqdm]) -> None:
        """
        Close a progress bar if it exists.
        
        Args:
            progress_bar: The progress bar to close.
        """
        if progress_bar is not None and self.verbose:
            progress_bar.close()
    
    def log_error(self, message: str, exception: Exception) -> None:
        """
        Log an error message and the exception.
        
        Args:
            message: The error message.
            exception: The exception that was raised.
        """
        logger.error(f"{message}: {str(exception)}")
    
    def apply_transformations(self, df: pd.DataFrame, 
                             transformations: List[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
        """
        Apply a series of transformations to a DataFrame.
        
        Args:
            df: The DataFrame to transform.
            transformations: A list of functions that take and return a DataFrame.
            
        Returns:
            The transformed DataFrame.
        """
        result = df.copy()
        
        for i, transform in enumerate(transformations):
            try:
                result = transform(result)
                logger.debug(f"Applied transformation {i+1}")
            except Exception as e:
                self.log_error(f"Error applying transformation {i+1}", e)
                raise
        
        return result
    
    def sample_data(self, df: pd.DataFrame, n: int = 5, random: bool = False) -> pd.DataFrame:
        """
        Get a sample of rows from a DataFrame.
        
        Args:
            df: The DataFrame to sample from.
            n: The number of rows to sample.
            random: Whether to take a random sample (True) or the first n rows (False).
            
        Returns:
            A DataFrame containing the sampled rows.
        """
        if len(df) <= n:
            return df
        
        if random:
            return df.sample(n=n)
        else:
            return df.head(n)
    
    def get_dataframe_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about a DataFrame.
        
        Args:
            df: The DataFrame to get information about.
            
        Returns:
            A dictionary containing information about the DataFrame.
        """
        info = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum(),
            "missing_values": df.isna().sum().to_dict()
        }
        
        return info
    
    def convert_types(self, df: pd.DataFrame, type_dict: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert column types in a DataFrame.
        
        Args:
            df: The DataFrame to convert.
            type_dict: A dictionary mapping column names to types.
            
        Returns:
            The DataFrame with converted types.
        """
        result = df.copy()
        
        for column, dtype in type_dict.items():
            if column in result.columns:
                try:
                    result[column] = result[column].astype(dtype)
                except Exception as e:
                    self.log_error(f"Error converting column '{column}' to {dtype}", e)
                    logger.warning(f"Keeping original type for column '{column}'")
        
        return result
    
    def validate_schema(self, df: pd.DataFrame, required_columns: List[str], 
                       required_types: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that a DataFrame has the required columns and types.
        
        Args:
            df: The DataFrame to validate.
            required_columns: A list of column names that must be present.
            required_types: A dictionary mapping column names to required types.
            
        Returns:
            True if the DataFrame meets all requirements, False otherwise.
        """
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for required types if specified
        if required_types:
            type_mismatches = []
            for col, dtype in required_types.items():
                if col in df.columns:
                    # Check if types match (this is a loose check)
                    actual_type = df[col].dtype
                    if not pd.api.types.is_dtype_equal(actual_type, dtype):
                        type_mismatches.append((col, str(actual_type), str(dtype)))
            
            if type_mismatches:
                for col, actual, expected in type_mismatches:
                    logger.warning(f"Column '{col}' has type {actual}, expected {expected}")
                return False
        
        return True 