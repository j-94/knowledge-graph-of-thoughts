"""
Dataset Module.

This module provides a flexible Dataset class for loading, processing,
and managing datasets from various sources and formats.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Callable, Iterator, Tuple

import pandas as pd
import numpy as np

from data_analysis.src.data_loaders import (
    DataLoaderFactory,
    JsonlDataLoader,
    CsvDataLoader,
    JsonDataLoader,
    StructuredTextLoader,
)

# Configure logging
logger = logging.getLogger(__name__)


class Dataset:
    """
    A dataset class for loading and managing data from various sources.
    
    This class provides a unified interface for working with data loaded
    from different file formats and sources. It supports operations like
    filtering, sampling, and iterating over the data.
    
    Attributes:
        data: The underlying pandas DataFrame containing the dataset.
        metadata: Dictionary containing metadata about the dataset.
    """
    
    def __init__(self, 
                data: Optional[pd.DataFrame] = None,
                metadata: Optional[Dict[str, Any]] = None,
                verbose: bool = True):
        """
        Initialize a dataset.
        
        Args:
            data: Optional pandas DataFrame to initialize with.
            metadata: Optional dictionary with metadata about the dataset.
            verbose: Whether to display progress bars and detailed logging.
        """
        self.data = data if data is not None else pd.DataFrame()
        self.metadata = metadata if metadata is not None else {}
        self.verbose = verbose
        self.loader_factory = DataLoaderFactory()
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path], 
                 file_format: Optional[str] = None,
                 **kwargs) -> 'Dataset':
        """
        Create a dataset from a file.
        
        Args:
            file_path: Path to the data file.
            file_format: Format of the file ('jsonl', 'csv', 'json', 'txt').
                If None, inferred from file extension.
            **kwargs: Additional arguments passed to the loader.
            
        Returns:
            A new Dataset instance containing the loaded data.
            
        Raises:
            ValueError: If the file format is not supported or cannot be inferred.
            FileNotFoundError: If the file does not exist.
        """
        # Create a new dataset instance
        dataset = cls(verbose=kwargs.pop('verbose', True))
        
        # Load the data
        return dataset.load_file(file_path, file_format, **kwargs)
    
    def load_file(self, file_path: Union[str, Path],
                 file_format: Optional[str] = None,
                 **kwargs) -> 'Dataset':
        """
        Load data from a file into this dataset.
        
        Args:
            file_path: Path to the data file.
            file_format: Format of the file ('jsonl', 'csv', 'json', 'txt').
                If None, inferred from file extension.
            **kwargs: Additional arguments passed to the loader.
            
        Returns:
            Self reference for method chaining.
            
        Raises:
            ValueError: If the file format is not supported or cannot be inferred.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        
        # Infer format from file extension if not provided
        if file_format is None:
            ext = path.suffix.lower()
            if ext == '.jsonl':
                file_format = 'jsonl'
            elif ext == '.csv':
                file_format = 'csv'
            elif ext == '.json':
                file_format = 'json'
            elif ext in ['.txt', '.text']:
                file_format = 'text'
            elif ext in ['.xlsx', '.xls', '.xlsm']:
                file_format = 'excel'
            else:
                raise ValueError(
                    f"Could not infer file format from extension: {ext}. "
                    f"Please specify file_format parameter."
                )
        
        # Get the appropriate loader
        if file_format == 'jsonl':
            loader = JsonlDataLoader(verbose=self.verbose)
        elif file_format == 'csv':
            loader = CsvDataLoader(verbose=self.verbose)
        elif file_format == 'json':
            loader = JsonDataLoader(verbose=self.verbose)
        elif file_format == 'text':
            loader = StructuredTextLoader(verbose=self.verbose)
        else:
            # Try to get a loader from the factory
            try:
                loader = self.loader_factory.get_loader(file_format, verbose=self.verbose)
            except ValueError:
                raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Loading {file_format} file: {path}")
        
        # Load the data
        self.data = loader.load(path, **kwargs)
        
        # Update metadata
        self.metadata.update({
            'source_file': str(path),
            'file_format': file_format,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
        })
        
        logger.info(f"Successfully loaded dataset with {len(self.data)} rows and {len(self.data.columns)} columns")
        return self
    
    def load_streaming(self, file_path: Union[str, Path],
                      file_format: Optional[str] = None,
                      chunk_size: int = 10000,
                      **kwargs) -> 'Dataset':
        """
        Load data from a large file in streaming mode.
        
        Args:
            file_path: Path to the data file.
            file_format: Format of the file.
            chunk_size: Number of rows to read at a time.
            **kwargs: Additional arguments passed to the loader.
            
        Returns:
            Self reference for method chaining.
        """
        path = Path(file_path)
        
        # Infer format from file extension if not provided
        if file_format is None:
            ext = path.suffix.lower()
            if ext == '.jsonl':
                file_format = 'jsonl'
            elif ext == '.csv':
                file_format = 'csv'
            elif ext == '.json':
                file_format = 'json'
            else:
                raise ValueError(
                    f"Could not infer file format from extension: {ext}. "
                    f"Please specify file_format parameter."
                )
        
        logger.info(f"Loading {file_format} file in streaming mode: {path}")
        
        # Get the appropriate loader
        if file_format == 'jsonl':
            loader = JsonlDataLoader(verbose=self.verbose)
            # Use the appropriate streaming method
            if hasattr(loader, 'load_streaming'):
                self.data = loader.load_streaming(path, chunk_size=chunk_size, **kwargs)
            else:
                self.data = loader.load(path, **kwargs)
        elif file_format == 'csv':
            loader = CsvDataLoader(verbose=self.verbose)
            self.data = loader.load_streaming(path, chunk_size=chunk_size, **kwargs)
        elif file_format == 'json':
            loader = JsonDataLoader(verbose=self.verbose)
            if hasattr(loader, 'load_streaming'):
                self.data = loader.load_streaming(path, chunk_size=chunk_size, **kwargs)
            else:
                self.data = loader.load(path, **kwargs)
        else:
            # Try to get a loader from the factory
            try:
                loader = self.loader_factory.get_loader(file_format, verbose=self.verbose)
                # Try to use streaming if available
                if hasattr(loader, 'load_streaming'):
                    self.data = loader.load_streaming(path, chunk_size=chunk_size, **kwargs)
                else:
                    self.data = loader.load(path, **kwargs)
            except ValueError:
                raise ValueError(f"Unsupported file format: {file_format}")
        
        # Update metadata
        self.metadata.update({
            'source_file': str(path),
            'file_format': file_format,
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'loaded_in_streaming_mode': True,
        })
        
        logger.info(f"Successfully loaded dataset with {len(self.data)} rows and {len(self.data.columns)} columns")
        return self
    
    def get_iterator(self, file_path: Union[str, Path],
                    file_format: Optional[str] = None,
                    chunk_size: int = 10000,
                    **kwargs) -> Iterator[pd.DataFrame]:
        """
        Get an iterator for loading a file in chunks.
        
        Args:
            file_path: Path to the data file.
            file_format: Format of the file.
            chunk_size: Number of rows to read at a time.
            **kwargs: Additional arguments passed to the loader.
            
        Returns:
            An iterator yielding DataFrame chunks.
            
        Raises:
            ValueError: If the file format is not supported or cannot be inferred.
        """
        path = Path(file_path)
        
        # Infer format from file extension if not provided
        if file_format is None:
            ext = path.suffix.lower()
            if ext == '.jsonl':
                file_format = 'jsonl'
            elif ext == '.csv':
                file_format = 'csv'
            elif ext == '.json':
                file_format = 'json'
            else:
                raise ValueError(
                    f"Could not infer file format from extension: {ext}. "
                    f"Please specify file_format parameter."
                )
        
        logger.info(f"Creating iterator for {file_format} file: {path}")
        
        # Get the appropriate loader and iterator method
        if file_format == 'jsonl':
            loader = JsonlDataLoader(verbose=self.verbose)
            if hasattr(loader, 'load_iterator'):
                return loader.load_iterator(path, chunk_size=chunk_size, **kwargs)
            else:
                # Fallback
                df = loader.load(path, **kwargs)
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i+chunk_size]
        elif file_format == 'csv':
            loader = CsvDataLoader(verbose=self.verbose)
            return loader.load_iterator(path, chunk_size=chunk_size, **kwargs)
        elif file_format == 'json':
            loader = JsonDataLoader(verbose=self.verbose)
            if hasattr(loader, 'load_iterator'):
                return loader.load_iterator(path, chunk_size=chunk_size, **kwargs)
            else:
                # Fallback
                df = loader.load(path, **kwargs)
                for i in range(0, len(df), chunk_size):
                    yield df.iloc[i:i+chunk_size]
        else:
            # Try to get a loader from the factory
            try:
                loader = self.loader_factory.get_loader(file_format, verbose=self.verbose)
                if hasattr(loader, 'load_iterator'):
                    return loader.load_iterator(path, chunk_size=chunk_size, **kwargs)
                else:
                    # Fallback
                    df = loader.load(path, **kwargs)
                    for i in range(0, len(df), chunk_size):
                        yield df.iloc[i:i+chunk_size]
            except ValueError:
                raise ValueError(f"Unsupported file format: {file_format}")
    
    def sample(self, n: int = 5, random: bool = False) -> 'Dataset':
        """
        Create a new dataset with a sample of rows from this dataset.
        
        Args:
            n: Number of rows to sample.
            random: Whether to take a random sample (True) or the first n rows (False).
            
        Returns:
            A new Dataset instance containing the sampled rows.
        """
        if len(self.data) <= n:
            return Dataset(data=self.data.copy(), 
                         metadata=self.metadata.copy(),
                         verbose=self.verbose)
        
        if random:
            sampled_data = self.data.sample(n=n)
        else:
            sampled_data = self.data.head(n)
        
        # Create new metadata
        new_metadata = self.metadata.copy()
        new_metadata.update({
            'is_sample': True,
            'sample_size': n,
            'sample_method': 'random' if random else 'head',
            'original_size': len(self.data)
        })
        
        return Dataset(data=sampled_data, metadata=new_metadata, verbose=self.verbose)
    
    def filter(self, condition: Union[str, Callable[[pd.DataFrame], pd.Series]]) -> 'Dataset':
        """
        Create a new dataset with rows filtered by a condition.
        
        Args:
            condition: Either a string expression to evaluate or a callable that
                takes a DataFrame and returns a boolean Series.
            
        Returns:
            A new Dataset instance containing the filtered rows.
            
        Raises:
            ValueError: If the condition is invalid.
        """
        if isinstance(condition, str):
            try:
                mask = eval(f"self.data.{condition}")
                if not isinstance(mask, pd.Series) or mask.dtype != bool:
                    raise ValueError("Expression must evaluate to a boolean Series")
            except Exception as e:
                raise ValueError(f"Invalid filter expression: {str(e)}")
        elif callable(condition):
            try:
                mask = condition(self.data)
                if not isinstance(mask, pd.Series) or mask.dtype != bool:
                    raise ValueError("Callable must return a boolean Series")
            except Exception as e:
                raise ValueError(f"Error in filter callable: {str(e)}")
        else:
            raise ValueError("Condition must be a string expression or callable")
        
        # Apply filter
        filtered_data = self.data[mask]
        
        # Create new metadata
        new_metadata = self.metadata.copy()
        new_metadata.update({
            'is_filtered': True,
            'filter_condition': str(condition),
            'original_size': len(self.data),
            'filtered_size': len(filtered_data)
        })
        
        return Dataset(data=filtered_data, metadata=new_metadata, verbose=self.verbose)
    
    def select(self, columns: List[str]) -> 'Dataset':
        """
        Create a new dataset with only the specified columns.
        
        Args:
            columns: List of column names to select.
            
        Returns:
            A new Dataset instance containing only the specified columns.
            
        Raises:
            ValueError: If any of the specified columns don't exist.
        """
        # Check if all columns exist
        missing_columns = [col for col in columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Columns not found: {missing_columns}")
        
        # Select columns
        selected_data = self.data[columns].copy()
        
        # Create new metadata
        new_metadata = self.metadata.copy()
        new_metadata.update({
            'selected_columns': columns,
            'original_columns': list(self.data.columns)
        })
        
        return Dataset(data=selected_data, metadata=new_metadata, verbose=self.verbose)
    
    def describe(self) -> Dict[str, Any]:
        """
        Get descriptive statistics and information about the dataset.
        
        Returns:
            A dictionary containing dataset description.
        """
        description = {
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            'missing_values': self.data.isna().sum().to_dict(),
            'metadata': self.metadata,
        }
        
        # Add numeric column statistics if there are numeric columns
        numeric_columns = self.data.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            description['numeric_stats'] = self.data[numeric_columns].describe().to_dict()
        
        # Add categorical column statistics if there are categorical/object columns
        cat_columns = self.data.select_dtypes(include=['category', 'object']).columns
        if len(cat_columns) > 0:
            description['categorical_stats'] = {
                col: {
                    'unique_values': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts(dropna=False).head(5).to_dict()
                }
                for col in cat_columns
            }
        
        return description
    
    def save(self, file_path: Union[str, Path], 
            file_format: Optional[str] = None,
            **kwargs) -> Path:
        """
        Save the dataset to a file.
        
        Args:
            file_path: Path where the dataset will be saved.
            file_format: Format to save as ('jsonl', 'csv', 'json', 'parquet').
                If None, inferred from file extension.
            **kwargs: Additional arguments specific to the format.
            
        Returns:
            Path to the saved file.
            
        Raises:
            ValueError: If the file format is not supported or cannot be inferred.
        """
        path = Path(file_path)
        
        # Create directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Infer format from file extension if not provided
        if file_format is None:
            ext = path.suffix.lower()
            if ext == '.jsonl':
                file_format = 'jsonl'
            elif ext == '.csv':
                file_format = 'csv'
            elif ext == '.json':
                file_format = 'json'
            elif ext == '.parquet':
                file_format = 'parquet'
            else:
                raise ValueError(
                    f"Could not infer file format from extension: {ext}. "
                    f"Please specify file_format parameter."
                )
        
        logger.info(f"Saving dataset to {file_format} file: {path}")
        
        # Save based on format
        if file_format == 'jsonl':
            self.data.to_json(
                path,
                orient='records',
                lines=True,
                **kwargs
            )
        elif file_format == 'csv':
            self.data.to_csv(
                path,
                index=kwargs.pop('index', False),
                **kwargs
            )
        elif file_format == 'json':
            self.data.to_json(
                path,
                orient=kwargs.pop('orient', 'records'),
                **kwargs
            )
        elif file_format == 'parquet':
            self.data.to_parquet(
                path,
                index=kwargs.pop('index', False),
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported file format for saving: {file_format}")
        
        logger.info(f"Successfully saved dataset with {len(self.data)} rows to {path}")
        return path
    
    def transform(self, transformer: Callable[[pd.DataFrame], pd.DataFrame]) -> 'Dataset':
        """
        Apply a transformation to the dataset.
        
        Args:
            transformer: A function that takes a DataFrame and returns a transformed DataFrame.
            
        Returns:
            Self reference for method chaining.
            
        Raises:
            ValueError: If the transformer doesn't return a DataFrame.
        """
        try:
            transformed_data = transformer(self.data)
            
            if not isinstance(transformed_data, pd.DataFrame):
                raise ValueError("Transformer must return a pandas DataFrame")
            
            self.data = transformed_data
            
            # Update metadata
            self.metadata.update({
                'transformed': True,
                'transformer': str(transformer)
            })
            
            return self
            
        except Exception as e:
            logger.error(f"Error applying transformation: {str(e)}")
            raise
    
    def apply(self, func: Callable, axis: int = 0, **kwargs) -> 'Dataset':
        """
        Apply a function along an axis of the DataFrame.
        
        Args:
            func: Function to apply.
            axis: 0 for columns, 1 for rows.
            **kwargs: Additional arguments to pass to DataFrame.apply.
            
        Returns:
            Self reference for method chaining.
        """
        self.data = self.data.apply(func, axis=axis, **kwargs)
        return self
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return the first n rows of the dataset.
        
        Args:
            n: Number of rows to return.
            
        Returns:
            A pandas DataFrame containing the first n rows.
        """
        return self.data.head(n)
    
    def tail(self, n: int = 5) -> pd.DataFrame:
        """
        Return the last n rows of the dataset.
        
        Args:
            n: Number of rows to return.
            
        Returns:
            A pandas DataFrame containing the last n rows.
        """
        return self.data.tail(n)
    
    def to_dict(self, orient: str = 'records') -> List[Dict[str, Any]]:
        """
        Convert the dataset to a list of dictionaries.
        
        Args:
            orient: The format of the output dictionaries.
            
        Returns:
            A list of dictionaries or dictionary of lists, depending on the orient parameter.
        """
        return self.data.to_dict(orient=orient)
    
    def to_numpy(self) -> np.ndarray:
        """
        Convert the dataset to a NumPy array.
        
        Returns:
            A NumPy array containing the dataset values.
        """
        return self.data.to_numpy()
    
    def __len__(self) -> int:
        """
        Get the number of rows in the dataset.
        
        Returns:
            The number of rows.
        """
        return len(self.data)
    
    def __getitem__(self, key) -> 'Dataset':
        """
        Get a slice or column of the dataset.
        
        Args:
            key: Index, column name, or slice.
            
        Returns:
            A new Dataset containing the selected data.
        """
        if isinstance(key, (list, tuple)) and all(isinstance(k, str) for k in key):
            # List of column names
            return self.select(key)
        
        # Other types of indexing
        result = self.data[key]
        
        if isinstance(result, pd.DataFrame):
            # Return a new Dataset
            return Dataset(data=result, verbose=self.verbose)
        else:
            # Return a Series directly
            return result 