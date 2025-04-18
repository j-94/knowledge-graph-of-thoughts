"""
JSON Lines Data Loader Module.

This module provides functionality to load JSON Lines (JSONL) data,
where each line is a valid JSON object, into pandas DataFrames.
"""

import logging
import json
from pathlib import Path
from typing import Optional, Union, Dict, Any, List, Iterator, Callable

import pandas as pd
import numpy as np

from .base_loader import BaseDataLoader

# Configure logging
logger = logging.getLogger(__name__)


class JsonlDataLoader(BaseDataLoader):
    """
    Data loader for JSON Lines (JSONL) files.
    
    This loader specifically handles JSONL format where each line contains
    a complete, self-contained JSON object.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the JSONL data loader.
        
        Args:
            verbose: Whether to display progress bars and logging information.
        """
        super().__init__(verbose=verbose)
    
    def load(self, file_path: Union[str, Path],
             orient: Optional[str] = None,
             convert_dtypes: bool = True,
             dtype: Optional[Dict[str, Any]] = None,
             encoding: str = 'utf-8',
             nrows: Optional[int] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load a JSONL file into a pandas DataFrame.
        
        Args:
            file_path: Path to the JSONL file to load.
            orient: The format JSON data should be in. Generally not needed for JSONL.
            convert_dtypes: Whether to convert dtypes after reading.
            dtype: Dictionary mapping column names to data types.
            encoding: File encoding to use.
            nrows: Number of rows to read (None to read all).
            **kwargs: Additional arguments for pandas.read_json.
            
        Returns:
            A pandas DataFrame containing the parsed JSONL data.
        """
        # Validate file
        file_path = self.validate_file(file_path)
        
        try:
            logger.info(f"Loading JSONL file: {file_path}")
            
            # Check file size to determine loading method
            file_size = file_path.stat().st_size
            
            # For large files or when limiting rows, use chunking
            if file_size > 100 * 1024 * 1024 or nrows is not None:  # 100 MB threshold
                if nrows is not None:
                    logger.info(f"Loading first {nrows} rows from JSONL file")
                    return self._load_limited_rows(file_path, nrows, encoding, **kwargs)
                else:
                    logger.info(f"JSONL file is large ({file_size / (1024 * 1024):.2f} MB), using chunked loading")
                    return self._load_large_jsonl(file_path, encoding, **kwargs)
            
            # Standard loading for regular files
            df = pd.read_json(
                file_path,
                lines=True,
                orient=orient,
                dtype=dtype,
                encoding=encoding,
                **kwargs
            )
            
            # Convert data types if requested
            if convert_dtypes and not dtype:
                df = df.convert_dtypes()
            
            logger.info(f"Successfully loaded {len(df)} records from JSONL file")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading JSONL file {file_path}", e)
            raise
    
    def _load_large_jsonl(self, file_path: Path, encoding: str, **kwargs) -> pd.DataFrame:
        """
        Load a large JSONL file using chunked processing.
        
        Args:
            file_path: Path to the JSONL file.
            encoding: File encoding.
            **kwargs: Additional arguments for pandas.read_json.
            
        Returns:
            A pandas DataFrame with all data combined.
        """
        chunks = []
        total_rows = self.count_lines(file_path)
        
        # Create progress bar
        progress_bar = self.create_progress_bar(
            total_rows, desc=f"Loading large JSONL file: {file_path.name}"
        )
        
        try:
            # Use pandas chunked reading
            chunk_size = min(10000, max(1000, total_rows // 20))  # Adaptive chunk size
            
            for chunk_df in pd.read_json(
                file_path, lines=True, chunksize=chunk_size, encoding=encoding, **kwargs
            ):
                chunks.append(chunk_df)
                self.update_progress(progress_bar, len(chunk_df))
            
            # Combine all chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded {len(df)} records from large JSONL file")
                return df
            else:
                logger.warning("No data found in JSONL file")
                return pd.DataFrame()
                
        except Exception as e:
            self.log_error(f"Error loading large JSONL file {file_path}", e)
            raise
        finally:
            self.close_progress(progress_bar)
    
    def _load_limited_rows(self, file_path: Path, nrows: int, encoding: str, **kwargs) -> pd.DataFrame:
        """
        Load a limited number of rows from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file.
            nrows: Maximum number of rows to load.
            encoding: File encoding.
            **kwargs: Additional arguments for pandas.read_json.
            
        Returns:
            A pandas DataFrame with the limited rows.
        """
        lines = []
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i >= nrows:
                        break
                    if line.strip():  # Skip empty lines
                        lines.append(line)
            
            if not lines:
                logger.warning(f"No valid lines found in the first {nrows} rows of JSONL file")
                return pd.DataFrame()
            
            # Parse the collected lines
            data = '\n'.join(lines)
            df = pd.read_json(data, lines=True, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} rows from JSONL file")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading limited rows from JSONL file {file_path}", e)
            raise
    
    def load_streaming(self, file_path: Union[str, Path],
                       chunk_size: int = 1000,
                       encoding: str = 'utf-8',
                       **kwargs) -> Iterator[pd.DataFrame]:
        """
        Load a JSONL file in streaming mode, yielding chunks of data.
        
        Args:
            file_path: Path to the JSONL file.
            chunk_size: Number of lines to read in each chunk.
            encoding: File encoding.
            **kwargs: Additional arguments for pandas.read_json.
            
        Yields:
            DataFrame chunks containing the parsed JSONL data.
        """
        file_path = self.validate_file(file_path)
        logger.info(f"Streaming JSONL file: {file_path}")
        
        try:
            # Create an iterator over chunks
            for chunk_df in pd.read_json(
                file_path, lines=True, chunksize=chunk_size, encoding=encoding, **kwargs
            ):
                yield chunk_df
                
        except Exception as e:
            self.log_error(f"Error streaming JSONL file {file_path}", e)
            raise
    
    def load_sample(self, file_path: Union[str, Path],
                   n: int = 5,
                   random: bool = False,
                   encoding: str = 'utf-8',
                   **kwargs) -> pd.DataFrame:
        """
        Load a sample of rows from a JSONL file.
        
        Args:
            file_path: Path to the JSONL file.
            n: Number of rows to sample.
            random: Whether to take a random sample or the first n rows.
            encoding: File encoding.
            **kwargs: Additional arguments for pandas.read_json.
            
        Returns:
            A pandas DataFrame with the sampled rows.
        """
        file_path = self.validate_file(file_path)
        
        if not random:
            # For head sample, just load first n rows
            return self._load_limited_rows(file_path, n, encoding, **kwargs)
        
        # For random sampling
        import random as rnd
        
        try:
            # Count total lines
            total_lines = self.count_lines(file_path)
            
            if n >= total_lines:
                # If requested sample size exceeds file size, return all rows
                return self.load(file_path, encoding=encoding, **kwargs)
            
            # Choose random line numbers
            sample_indices = sorted(rnd.sample(range(total_lines), n))
            lines = []
            
            with open(file_path, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i in sample_indices:
                        if line.strip():  # Skip empty lines
                            lines.append(line)
                        # If we have enough lines, we can stop reading
                        if len(lines) == n:
                            break
            
            if not lines:
                logger.warning(f"No valid lines found in random sample from JSONL file")
                return pd.DataFrame()
            
            # Parse the sampled lines
            data = '\n'.join(lines)
            df = pd.read_json(data, lines=True, **kwargs)
            
            logger.info(f"Successfully loaded {len(df)} random rows from JSONL file")
            return df
            
        except Exception as e:
            self.log_error(f"Error sampling from JSONL file {file_path}", e)
            raise
    
    def count_lines(self, file_path: Union[str, Path]) -> int:
        """
        Count the number of non-empty lines in a JSONL file.
        
        Args:
            file_path: Path to the JSONL file.
            
        Returns:
            The number of non-empty lines in the file.
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r') as f:
                count = sum(1 for line in f if line.strip())
            return count
        except Exception as e:
            self.log_error(f"Error counting lines in JSONL file {file_path}", e)
            raise
    
    def validate_jsonl(self, file_path: Union[str, Path], 
                      max_errors: int = 10,
                      sample_size: Optional[int] = 100) -> bool:
        """
        Validate that a file contains valid JSONL data.
        
        Args:
            file_path: Path to the file to validate.
            max_errors: Maximum number of errors before failing validation.
            sample_size: Number of lines to sample for validation (None to check all).
            
        Returns:
            True if the file contains valid JSONL, False otherwise.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        error_count = 0
        line_count = 0
        
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    if sample_size is not None and i >= sample_size:
                        break
                    
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    line_count += 1
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        error_count += 1
                        logger.warning(f"Invalid JSON on line {i+1}: {str(e)}")
                        
                        if error_count >= max_errors:
                            logger.error(f"Too many JSON errors found (>{max_errors})")
                            return False
            
            if line_count == 0:
                logger.warning(f"No non-empty lines found in file: {file_path}")
                return False
                
            if error_count > 0:
                logger.warning(f"Found {error_count} JSON errors in {line_count} lines")
                return False
                
            logger.info(f"File contains valid JSONL data: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating JSONL file {file_path}: {str(e)}")
            return False
    
    def export_to_jsonl(self, df: pd.DataFrame,
                       output_path: Union[str, Path],
                       orient: str = 'records',
                       encoding: str = 'utf-8',
                       **kwargs) -> Path:
        """
        Export a DataFrame to a JSONL file.
        
        Args:
            df: The DataFrame to export.
            output_path: Path where the JSONL file will be saved.
            orient: Format of the JSON output (default is 'records').
            encoding: File encoding.
            **kwargs: Additional arguments for DataFrame.to_json.
            
        Returns:
            Path to the saved JSONL file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Exporting DataFrame to JSONL: {output_path}")
            
            df.to_json(
                output_path,
                orient=orient,
                lines=True,
                date_format='iso',
                encoding=encoding,
                **kwargs
            )
            
            logger.info(f"Successfully exported {len(df)} records to JSONL file")
            return output_path
            
        except Exception as e:
            self.log_error(f"Error exporting DataFrame to JSONL: {output_path}", e)
            raise 