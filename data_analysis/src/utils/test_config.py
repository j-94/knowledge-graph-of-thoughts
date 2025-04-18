#!/usr/bin/env python
"""
Test script for the configuration loading utility.

This script verifies that the configuration loader can correctly load and merge
configurations from different environments.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to sys.path to allow imports
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import config loader
try:
    from src.utils.config import ConfigLoader, BookmarkConfig
    import yaml
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you're running this script from the data_analysis directory.")
    sys.exit(1)

def print_config_section(name, config_section):
    """Print a configuration section in a readable format."""
    print(f"\n=== {name} ===")
    if hasattr(config_section, "dict"):
        config_dict = config_section.dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
    else:
        print(f"  {config_section}")

def test_config_loading(environment):
    """Test loading configuration for the specified environment."""
    try:
        print(f"Loading configuration for environment: {environment}")
        loader = ConfigLoader()
        config = loader.load_config(environment)
        
        print("\nConfiguration successfully loaded!")
        
        # Print key configuration sections
        print_config_section("Database Configuration", config.database)
        print_config_section("Bookmark Processing Configuration", config.bookmark_processing)
        print_config_section("Embedding Configuration", config.embeddings)
        print_config_section("Relationship Configuration", config.relationships)
        
        # Print merged values for key settings that differ by environment
        print("\n=== Environment-specific Settings ===")
        print(f"  Batch Size: {config.bookmark_processing.batch_size}")
        print(f"  Embedding Model: {config.embeddings.model_name}")
        print(f"  Similarity Threshold: {config.relationships.similarity_threshold}")
        print(f"  Logging Level: {config.logging.level}")
        print(f"  API Debug Mode: {config.api.debug}")
        
        # Save combined config for inspection
        output_dir = Path("data_analysis/config/output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"combined_{environment}_config.yaml"
        
        loader.save_config(config, output_path)
        print(f"\nCombined configuration saved to {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error testing configuration: {str(e)}")
        return 1

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test configuration loading")
    parser.add_argument(
        "--env", 
        default="dev", 
        choices=["dev", "test", "prod"],
        help="Environment to test (dev, test, or prod)"
    )
    
    args = parser.parse_args()
    return test_config_loading(args.env)

if __name__ == "__main__":
    sys.exit(main()) 