#!/usr/bin/env python
"""
Test script to verify that all required dependencies for bookmark processing
can be properly imported.
"""

import sys
import importlib
from typing import List, Dict, Any, Tuple

def test_import(module_name: str) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return True, f"✅ {module_name} (version: {version})"
    except ImportError as e:
        return False, f"❌ {module_name}: {str(e)}"

def main():
    """Test importing all required dependencies."""
    print("Testing required dependencies for Knowledge Graph of Thoughts bookmark processing:")
    print("-" * 80)
    
    # Core dependencies from PRD
    core_deps = [
        "pandas",               # Data processing
        "numpy",                # Numerical operations
        "networkx",             # Graph operations
        "neo4j",                # Neo4j database connection
        "fastapi",              # API layer
        "pydantic",             # Data validation
        "transformers",         # For embeddings
    ]
    
    # Additional useful dependencies
    additional_deps = [
        "bs4",                  # BeautifulSoup for HTML parsing
        "langchain",            # For LLM integration
        "dotenv",               # For environment variables
        "matplotlib",           # For visualization
        "plotly",               # For interactive visualization
        "tqdm",                 # For progress bars
    ]
    
    success_count = 0
    failure_count = 0
    
    # Test core dependencies
    print("\nCore dependencies:")
    for dep in core_deps:
        success, message = test_import(dep)
        print(message)
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Test additional dependencies
    print("\nAdditional useful dependencies:")
    for dep in additional_deps:
        success, message = test_import(dep)
        print(message)
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    # Print summary
    print("\n" + "-" * 80)
    print(f"Summary: {success_count} dependencies available, {failure_count} missing")
    
    if failure_count > 0:
        print("\nTo install missing dependencies, run:")
        print("pip install -e .")
        return 1
    else:
        print("\nAll required dependencies are available!")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 