"""
CSV Data Loader Module.

This module provides a data loader for CSV data with support for various
formatting options, advanced validation, and efficient loading of large files.
"""

import csv
import logging
import io
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Iterator, Callable, Tuple

import pandas as pd
import numpy as np

from .base_loader import BaseDataLoader

# Configure logging
logger = logging.getLogger(__name__)


class CsvDataLoader(BaseDataLoader):
    """
    Data loader for CSV files.
    
    This class provides methods for loading CSV data into pandas DataFrames,
    with support for various formats, validation, and streaming for large files.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the CSV data loader.
        
        Args:
            verbose: Whether to display progress bars and detailed logging.
        """
        super().__init__(verbose=verbose)
    
    def load(self, file_path: Union[str, Path], 
             delimiter: str = ',',
             header: Union[int, List[int], None] = 0,
             quotechar: str = '"',
             *,
             encoding: str = 'utf-8',
             skiprows: Optional[Union[int, List[int]]] = None,
             skip_blank_lines: bool = True,
             comment: Optional[str] = None,
             na_values: Optional[Union[str, List[str], Dict[str, List[str]]]] = None,
             convert_dtypes: bool = True,
             parse_dates: Union[bool, List[int], List[str], Dict[str, List[str]]] = False,
             dtype: Optional[Dict[str, Any]] = None,
             **kwargs) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.
        
        Args:
            file_path: Path to the CSV file to load.
            delimiter: Character used to separate values in the CSV.
            header: Row number(s) to use as the column names. Default is 0.
                If None, no header row is used.
            quotechar: Character used to denote the start and end of a quoted item.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            skiprows: Line numbers to skip (0-indexed) or number of lines to skip.
            skip_blank_lines: Whether to skip blank lines. Default is True.
            comment: Character indicating a comment; lines beginning with this character
                will be skipped. Default is None.
            na_values: Values to recognize as NA/NaN.
            convert_dtypes: Whether to convert dtypes after reading. Default is True.
            parse_dates: Whether to parse date columns. Can be boolean, list of column
                indices, list of column names, or dict mapping column names to format strings.
            dtype: Dictionary mapping column names to data types. Default is None.
            **kwargs: Additional arguments to pass to pandas.read_csv.
                
        Returns:
            A pandas DataFrame containing the loaded data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read due to permissions.
            ValueError: If the CSV format is invalid or unsupported.
        """
        try:
            # Validate file
            path = self.validate_file(file_path)
            logger.info(f"Loading CSV file: {path}")
            
            # Get file size for logging
            file_size = self.get_file_size(path)
            logger.info(f"CSV file size: {file_size / (1024*1024):.2f} MB")
            
            # Load the CSV file
            df = pd.read_csv(
                path,
                delimiter=delimiter,
                header=header,
                quotechar=quotechar,
                encoding=encoding,
                skiprows=skiprows,
                skip_blank_lines=skip_blank_lines,
                comment=comment,
                na_values=na_values,
                dtype=dtype,
                parse_dates=parse_dates,
                **kwargs
            )
            
            # Convert data types if requested
            if convert_dtypes and not dtype:
                df = df.convert_dtypes()
            
            logger.info(f"Successfully loaded CSV file with {len(df)} rows and {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.log_error(f"Error loading CSV file: {path}", e)
            raise
    
    def load_streaming(self, file_path: Union[str, Path], 
                      chunk_size: int = 10000,
                      *,
                      encoding: str = 'utf-8',
                      **kwargs) -> pd.DataFrame:
        """
        Load a large CSV file in chunks to manage memory usage.
        
        Args:
            file_path: Path to the CSV file to load.
            chunk_size: Number of rows to read in each chunk.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to pandas.read_csv.
                
        Returns:
            A pandas DataFrame containing all the loaded data combined.
        """
        path = self.validate_file(file_path)
        logger.info(f"Loading CSV file in streaming mode: {path}")
        
        # Get total rows for progress tracking
        total_rows = self._count_rows(path, encoding, kwargs.get('delimiter', ','), 
                                     kwargs.get('skiprows', None))
        
        progress_bar = self.create_progress_bar(total_rows, "Loading CSV in chunks")
        chunks = []
        loaded_rows = 0
        
        try:
            # Read file in chunks
            for chunk_df in pd.read_csv(path, chunksize=chunk_size, encoding=encoding, **kwargs):
                chunks.append(chunk_df)
                rows_in_chunk = len(chunk_df)
                loaded_rows += rows_in_chunk
                self.update_progress(progress_bar, rows_in_chunk)
            
            # Combine all chunks
            if chunks:
                result = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully loaded CSV file with {len(result)} rows in streaming mode")
                return result
            else:
                logger.warning(f"No data loaded from CSV file: {path}")
                return pd.DataFrame()
        
        except Exception as e:
            self.log_error(f"Error loading CSV file in streaming mode: {path}", e)
            raise
        
        finally:
            self.close_progress(progress_bar)
    
    def load_iterator(self, file_path: Union[str, Path],
                     chunk_size: int = 10000,
                     *,
                     encoding: str = 'utf-8',
                     **kwargs) -> Iterator[pd.DataFrame]:
        """
        Create an iterator for loading CSV files in chunks.
        
        Args:
            file_path: Path to the CSV file to load.
            chunk_size: Number of rows to read in each chunk.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to pandas.read_csv.
                
        Yields:
            Chunks of the loaded data as pandas DataFrames.
        """
        path = self.validate_file(file_path)
        logger.info(f"Creating iterator for CSV file: {path}")
        
        try:
            # Create an iterator
            chunks_iterator = pd.read_csv(
                path, chunksize=chunk_size, encoding=encoding, **kwargs
            )
            
            # Yield each chunk
            for chunk in chunks_iterator:
                yield chunk
        
        except Exception as e:
            self.log_error(f"Error creating iterator for CSV file: {path}", e)
            raise
    
    def load_sample(self, file_path: Union[str, Path],
                   n: int = 5,
                   random: bool = False,
                   *,
                   encoding: str = 'utf-8',
                   **kwargs) -> pd.DataFrame:
        """
        Load a sample of rows from a CSV file.
        
        Args:
            file_path: Path to the CSV file to load.
            n: The number of rows to sample.
            random: Whether to take a random sample (True) or the first n rows (False).
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to load() method.
                
        Returns:
            A pandas DataFrame containing the sampled rows.
        """
        if random:
            return self._sample_random_rows(file_path, n, encoding, **kwargs)
        else:
            # For head sampling, we can directly use pandas nrows parameter
            kwargs['nrows'] = n
            return self.load(file_path, encoding=encoding, **kwargs)
    
    def _sample_random_rows(self, file_path: Union[str, Path],
                           n: int,
                           encoding: str,
                           **kwargs) -> pd.DataFrame:
        """
        Take a random sample from a CSV file.
        
        Args:
            file_path: Path to the CSV file to load.
            n: The number of rows to sample.
            encoding: The encoding to use for the file.
            **kwargs: Additional arguments to pass to pandas.read_csv.
                
        Returns:
            A pandas DataFrame containing the sampled rows.
        """
        path = self.validate_file(file_path)
        
        # Count total rows
        total_rows = self._count_rows(path, encoding, kwargs.get('delimiter', ','), 
                                     kwargs.get('skiprows', None))
        
        # If n is greater than total rows, return all
        if n >= total_rows:
            return self.load(file_path, encoding=encoding, **kwargs)
        
        # Choose random row indices
        import random
        skiprows = kwargs.pop('skiprows', None)
        header = kwargs.pop('header', 0)
        
        # Determine rows to skip (accounting for header and existing skiprows)
        header_offset = 1 if header is not None else 0
        if skiprows is None:
            skiprows = []
        elif isinstance(skiprows, int):
            skiprows = list(range(skiprows))
        
        # Generate all row indices, excluding header and already skipped rows
        all_indices = set(range(total_rows + header_offset))
        skipindices = set(skiprows)
        if header is not None:
            if isinstance(header, int):
                skipindices.add(header)
            else:  # It's a list
                skipindices.update(header)
        
        # Available indices are all rows minus skipped rows and header
        available_indices = list(all_indices - skipindices)
        
        # Choose random indices from available indices
        sample_indices = random.sample(available_indices, n)
        
        # Create a new skiprows list that skips everything except our sample and header
        new_skiprows = list(all_indices - set(sample_indices) - skipindices)
        
        # Load the file with the new skiprows
        return pd.read_csv(path, skiprows=new_skiprows, encoding=encoding, **kwargs)
    
    def _count_rows(self, file_path: Path, 
                   encoding: str,
                   delimiter: str = ',',
                   skiprows: Optional[Union[int, List[int]]] = None) -> int:
        """
        Count the number of rows in a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            encoding: The encoding to use for the file.
            delimiter: Character used to separate values in the CSV.
            skiprows: Line numbers to skip (0-indexed) or number of lines to skip.
                
        Returns:
            The number of rows in the file.
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # Create a CSV reader
                reader = csv.reader(f, delimiter=delimiter)
                
                # Process skiprows
                if skiprows is not None:
                    if isinstance(skiprows, int):
                        for _ in range(skiprows):
                            next(reader, None)
                    else:  # It's a list
                        for i, row in enumerate(reader):
                            if i not in skiprows:
                                break
                
                # Count the remaining rows
                count = sum(1 for _ in reader)
                
            return count
        
        except Exception as e:
            logger.warning(f"Error counting rows in CSV file: {e}")
            # Fallback: use chunk iterator to count rows
            count = 0
            for chunk in pd.read_csv(file_path, chunksize=10000, encoding=encoding, 
                                    delimiter=delimiter, skiprows=skiprows):
                count += len(chunk)
            return count
    
    def load_with_validation(self, file_path: Union[str, Path],
                            validation_funcs: List[Callable[[pd.DataFrame], bool]],
                            *,
                            encoding: str = 'utf-8',
                            **kwargs) -> pd.DataFrame:
        """
        Load a CSV file and apply validation functions.
        
        Args:
            file_path: Path to the CSV file to load.
            validation_funcs: List of validation functions that take a DataFrame
                and return a boolean indicating whether the validation passed.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to load() method.
                
        Returns:
            A pandas DataFrame with the loaded data.
            
        Raises:
            ValueError: If any validation function returns False.
        """
        # Load the data
        df = self.load(file_path, encoding=encoding, **kwargs)
        
        # Apply validation functions
        for i, validate in enumerate(validation_funcs):
            try:
                if not validate(df):
                    logger.error(f"Validation {i+1} failed for CSV file: {file_path}")
                    raise ValueError(f"Validation {i+1} failed for CSV file: {file_path}")
            except Exception as e:
                self.log_error(f"Error during validation {i+1} for CSV file: {file_path}", e)
                raise
        
        logger.info(f"All validations passed for CSV file: {file_path}")
        return df
    
    def load_with_type_inference(self, file_path: Union[str, Path],
                               *,
                               encoding: str = 'utf-8',
                               **kwargs) -> pd.DataFrame:
        """
        Load a CSV file with automatic type inference.
        
        Args:
            file_path: Path to the CSV file to load.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to load() method.
                
        Returns:
            A pandas DataFrame with inferred types.
        """
        # First load the data without type conversion
        kwargs['convert_dtypes'] = False
        df = self.load(file_path, encoding=encoding, **kwargs)
        
        # Apply pandas type inference
        return df.convert_dtypes()
    
    def infer_delimiter(self, file_path: Union[str, Path], 
                       sample_size: int = 5,
                       *,
                       encoding: str = 'utf-8') -> str:
        """
        Attempt to automatically detect the delimiter in a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            sample_size: Number of lines to sample for detection.
            encoding: The encoding to use for the file. Default is 'utf-8'.
                
        Returns:
            The detected delimiter character.
        """
        path = self.validate_file(file_path)
        
        # Common delimiters to check
        delimiters = [',', '\t', ';', '|', ' ']
        counts = {d: 0 for d in delimiters}
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                # Read sample lines
                sample_lines = [next(f) for _ in range(sample_size) if next(f, None)]
            
            # Count occurrences of each delimiter in each line
            for line in sample_lines:
                for delimiter in delimiters:
                    # Count consistent occurrences across lines
                    counts[delimiter] += line.count(delimiter)
            
            # Get the delimiter with the highest consistent count
            most_common = max(counts.items(), key=lambda x: x[1])
            
            if most_common[1] > 0:
                logger.info(f"Detected delimiter: '{most_common[0]}'")
                return most_common[0]
            else:
                logger.warning(f"Could not detect delimiter, defaulting to comma")
                return ','
                
        except Exception as e:
            self.log_error(f"Error detecting delimiter: {path}", e)
            return ','
    
    def infer_header(self, file_path: Union[str, Path], 
                    *,
                    encoding: str = 'utf-8',
                    delimiter: Optional[str] = None) -> int:
        """
        Attempt to detect if the CSV file has a header row.
        
        Args:
            file_path: Path to the CSV file.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            delimiter: The delimiter to use. If None, will attempt to detect it.
                
        Returns:
            0 if a header is detected, None otherwise.
        """
        path = self.validate_file(file_path)
        
        if delimiter is None:
            delimiter = self.infer_delimiter(path, encoding=encoding)
        
        try:
            # Read a small sample of the file
            with open(path, 'r', encoding=encoding) as f:
                sample = [next(f) for _ in range(3) if next(f, None)]
            
            if not sample:
                return None
            
            # Parse the first two rows
            first_row = next(csv.reader([sample[0]], delimiter=delimiter))
            second_row = next(csv.reader([sample[1]], delimiter=delimiter)) if len(sample) > 1 else []
            
            # Check if first row could be a header
            # 1. Headers are often strings while data might contain numeric values
            # 2. Headers typically don't have many empty values
            
            if not first_row:
                return None
                
            # Check if first row has fewer empty values than second row
            first_row_empty = sum(1 for cell in first_row if not cell.strip())
            second_row_empty = sum(1 for cell in second_row if not cell.strip()) if second_row else 0
            
            # Check if first row has more non-numeric values than second row
            first_row_non_numeric = sum(1 for cell in first_row if not self._is_numeric(cell))
            second_row_non_numeric = sum(1 for cell in second_row if not self._is_numeric(cell)) if second_row else 0
            
            # If first row has fewer empty cells and more non-numeric values, it's likely a header
            if (first_row_empty <= second_row_empty and 
                first_row_non_numeric >= second_row_non_numeric and
                first_row_non_numeric > 0):
                logger.info("Header row detected")
                return 0
            else:
                logger.info("No header row detected")
                return None
                
        except Exception as e:
            self.log_error(f"Error detecting header: {path}", e)
            return 0  # Default to assuming there is a header
    
    def _is_numeric(self, value: str) -> bool:
        """
        Check if a string value is numeric.
        
        Args:
            value: The string value to check.
                
        Returns:
            True if the value is numeric, False otherwise.
        """
        if not value.strip():
            return False
            
        try:
            float(value.replace(',', ''))
            return True
        except ValueError:
            return False
    
    def detect_encoding(self, file_path: Union[str, Path]) -> str:
        """
        Attempt to detect the encoding of a CSV file.
        
        Args:
            file_path: Path to the CSV file.
                
        Returns:
            The detected encoding.
        """
        path = self.validate_file(file_path)
        
        try:
            import chardet
            
            # Read a sample of the file
            with open(path, 'rb') as f:
                sample = f.read(10000)  # Read first 10KB
            
            # Detect encoding
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logger.info(f"Detected encoding: {encoding} with confidence: {confidence:.2f}")
            
            # Fall back to utf-8 if detection fails or has low confidence
            if not encoding or confidence < 0.7:
                logger.warning(f"Low confidence in encoding detection, defaulting to utf-8")
                return 'utf-8'
                
            return encoding
            
        except ImportError:
            logger.warning("chardet package not installed, defaulting to utf-8")
            return 'utf-8'
        except Exception as e:
            self.log_error(f"Error detecting encoding: {path}", e)
            return 'utf-8'
    
    def export_to_csv(self, df: pd.DataFrame,
                     output_path: Union[str, Path],
                     index: bool = False,
                     header: bool = True,
                     quoting: int = csv.QUOTE_MINIMAL,
                     *,
                     encoding: str = 'utf-8',
                     **kwargs) -> Path:
        """
        Export a DataFrame to a CSV file.
        
        Args:
            df: The DataFrame to export.
            output_path: Path where the CSV file will be saved.
            index: Whether to include the index in the output. Default is False.
            header: Whether to include the column names. Default is True.
            quoting: Controls quoting behavior. Default is csv.QUOTE_MINIMAL.
            encoding: The encoding to use for the file. Default is 'utf-8'.
            **kwargs: Additional arguments to pass to pandas.DataFrame.to_csv.
                
        Returns:
            Path to the saved CSV file.
            
        Raises:
            IOError: If the file cannot be written.
        """
        # Convert to Path object
        output_path = Path(output_path)
        
        # Create directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Exporting DataFrame to CSV: {output_path}")
            
            # Export the DataFrame to CSV
            df.to_csv(
                output_path,
                index=index,
                header=header,
                quoting=quoting,
                encoding=encoding,
                **kwargs
            )
            
            logger.info(f"Successfully exported {len(df)} rows to CSV file: {output_path}")
            return output_path
            
        except Exception as e:
            self.log_error(f"Error exporting DataFrame to CSV: {output_path}", e)
            raise
    
    def process_with_column_mapping(self, df: pd.DataFrame, 
                                   column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Rename columns based on a mapping dictionary.
        
        Args:
            df: The DataFrame to process.
            column_mapping: Dictionary mapping original column names to new names.
                
        Returns:
            DataFrame with renamed columns.
        """
        try:
            # Get only the mappings for columns that exist in the DataFrame
            valid_mappings = {k: v for k, v in column_mapping.items() if k in df.columns}
            
            if not valid_mappings:
                logger.warning("No valid column mappings found")
                return df
            
            # Rename the columns
            result = df.rename(columns=valid_mappings)
            
            mapped_cols = list(valid_mappings.values())
            logger.info(f"Renamed {len(valid_mappings)} columns: {mapped_cols}")
            
            return result
            
        except Exception as e:
            self.log_error("Error during column mapping", e)
            raise
    
    def detect_date_columns(self, df: pd.DataFrame, 
                           sample_size: int = 1000) -> List[str]:
        """
        Detect columns that contain date or datetime values.
        
        Args:
            df: DataFrame to analyze.
            sample_size: Number of rows to sample for detection.
                
        Returns:
            List of column names that likely contain date values.
        """
        date_columns = []
        
        # Sample the DataFrame
        sample_df = df.head(sample_size) if len(df) > sample_size else df
        
        for col in sample_df.columns:
            # Skip if column is not string or object type
            if df[col].dtype != 'object' and not pd.api.types.is_string_dtype(df[col]):
                continue
                
            # Sample non-null values
            values = sample_df[col].dropna().astype(str)
            if len(values) < 5:  # Skip if not enough data
                continue
                
            # Try to parse as dates
            try:
                pd.to_datetime(values, errors='raise')
                date_columns.append(col)
                logger.info(f"Detected date column: {col}")
            except:
                pass
                
        return date_columns
    
    def auto_convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Automatically convert detected date columns to datetime.
        
        Args:
            df: DataFrame to process.
                
        Returns:
            DataFrame with date columns converted to datetime.
        """
        date_columns = self.detect_date_columns(df)
        
        if not date_columns:
            logger.info("No date columns detected")
            return df
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        for col in date_columns:
            try:
                result[col] = pd.to_datetime(result[col], errors='coerce')
                logger.info(f"Converted column '{col}' to datetime")
            except Exception as e:
                logger.warning(f"Failed to convert column '{col}' to datetime: {e}")
                
        return result 