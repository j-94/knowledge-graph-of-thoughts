#!/usr/bin/env python3
"""
Simple test script for JsonlDataLoader.
"""

import logging
import json
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_analysis.src.data_loaders.jsonl_loader import JsonlDataLoader

def main():
    """Test the JsonlDataLoader class"""
    
    # Create a test JSONL file
    test_file = Path("test_data.jsonl")
    
    # Sample data
    test_data = [
        {"id": 1, "name": "Alice", "age": 30, "items": ["book", "laptop"]},
        {"id": 2, "name": "Bob", "age": 25, "items": ["phone"]},
        {"id": 3, "name": "Charlie", "age": 35, "items": ["tablet", "headphones", "mouse"]}
    ]
    
    # Write test data to file
    with open(test_file, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Created test file: {test_file}")
    
    try:
        # Create loader
        loader = JsonlDataLoader(verbose=True)
        
        # Test basic loading
        logger.info("Testing basic load method")
        df = loader.load(test_file)
        print("\nBasic load result:")
        print(df)
        
        # Test sample loading
        logger.info("Testing sample loading (first 2 rows)")
        df_sample = loader.load_sample(test_file, n=2)
        print("\nSample (first 2 rows):")
        print(df_sample)
        
        # Test random sample
        logger.info("Testing random sample")
        df_random = loader.load_sample(test_file, n=2, random=True)
        print("\nRandom sample (2 rows):")
        print(df_random)
        
        # Test validation
        logger.info("Testing JSONL validation")
        is_valid = loader.validate_jsonl(test_file)
        print(f"\nFile is valid JSONL: {is_valid}")
        
        # Test export
        logger.info("Testing export to JSONL")
        # Add a new row
        df.loc[len(df)] = {"id": 4, "name": "David", "age": 40, "items": ["watch"]}
        export_file = Path("exported_data.jsonl")
        loader.export_to_jsonl(df, export_file)
        
        # Verify export by loading it back
        logger.info("Verifying exported data")
        df_exported = loader.load(export_file)
        print("\nExported and reloaded data:")
        print(df_exported)
        
        logger.info("All tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        # Clean up test files
        if test_file.exists():
            test_file.unlink()
            logger.info(f"Removed test file: {test_file}")
        
        if export_file.exists():
            export_file.unlink()
            logger.info(f"Removed export file: {export_file}")

if __name__ == "__main__":
    main() 