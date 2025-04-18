#!/usr/bin/env python3
"""
Example script demonstrating how to load and analyze JSONL data.

This script shows how to use the Dataset class to load a JSONL file,
explore its contents, and perform basic data analysis.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the root directory to Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data_analysis.src.dataset import Dataset


def main():
    """
    Main function to demonstrate loading and analyzing JSONL data.
    """
    # Path to the JSONL file
    file_path = "/Users/imac/Desktop/webdev-setup/knowledge-graph-of-thoughts/data-bookmark.jsonl"
    
    # Load the dataset
    logger.info(f"Loading dataset from {file_path}")
    try:
        # Method 1: Using the class method (recommended)
        dataset = Dataset.from_file(file_path)
        
        # Method 2: Creating object and then loading (alternative)
        # dataset = Dataset()
        # dataset.load_file(file_path)
        
        # Method 3: For large files, use streaming
        # dataset = Dataset()
        # dataset.load_streaming(file_path, chunk_size=10000)
        
        logger.info(f"Successfully loaded dataset with {len(dataset)} rows")
        
        # Display basic information
        print("\n==== Dataset Overview ====")
        print(f"Rows: {len(dataset)}")
        print(f"Columns: {len(dataset.data.columns)}")
        print(f"Column names: {list(dataset.data.columns)}")
        
        # Display a sample of the data
        print("\n==== Data Sample ====")
        print(dataset.head())
        
        # Get and display descriptive statistics
        print("\n==== Descriptive Statistics ====")
        description = dataset.describe()
        
        # Print some key statistics
        print(f"Missing values: {description['missing_values']}")
        
        if 'numeric_stats' in description:
            print("\n==== Numeric Column Statistics ====")
            for col, stats in description['numeric_stats'].items():
                print(f"{col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}")
        
        if 'categorical_stats' in description:
            print("\n==== Top Categorical Values ====")
            for col, stats in description['categorical_stats'].items():
                print(f"\n{col} (unique values: {stats['unique_values']})")
                for value, count in list(stats['top_values'].items())[:3]:
                    print(f"  - {value}: {count}")
        
        # Example of filtering data
        if len(dataset) > 0:
            print("\n==== Data Filtering Example ====")
            # Choose a column that exists in your data for filtering
            # This is a generic example that you'll need to adapt to your actual data structure
            first_column = dataset.data.columns[0]
            
            # For demonstration, we'll filter based on non-null values in the first column
            filtered = dataset.filter(lambda df: df[first_column].notna())
            print(f"Original size: {len(dataset)}, Filtered size: {len(filtered)}")
            
            # Example of saving the dataset
            # output_path = Path("./filtered_data.jsonl")
            # filtered.save(output_path)
            # print(f"Saved filtered dataset to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise


if __name__ == "__main__":
    main() 