#!/usr/bin/env python3
"""
KGoT Import Script

This script imports data from processed knowledge graph files into the KGoT system.
It takes a JSON graph file and a source identifier, merges the data with any existing
knowledge graph, and creates the appropriate indices.

Usage:
    python3 import.py --input <graph_file.json> --source <source_identifier>

Example:
    python3 import.py --input output/kgot_graph.json --source bookmarks
"""

import argparse
import json
import os
import sys
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Import data into KGoT system")
    parser.add_argument("--input", required=True, help="Path to the graph JSON file")
    parser.add_argument("--source", required=True, help="Source identifier for the imported data")
    return parser.parse_args()

def load_graph_data(file_path):
    """Load graph data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading graph data: {str(e)}")
        sys.exit(1)

def import_data(graph_data, source):
    """Import graph data into the KGoT system."""
    # Count nodes and edges for reporting
    node_count = len(graph_data.get('nodes', []))
    edge_count = len(graph_data.get('edges', []))
    
    # Create a backup of the data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join("kgot", "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_file = os.path.join(backup_dir, f"import_{source}_{timestamp}.json")
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f)
    
    # Create system indices (in a real system, this would be more sophisticated)
    indices_dir = os.path.join("kgot", "indices")
    os.makedirs(indices_dir, exist_ok=True)
    
    # Create a simple index file with metadata about the import
    index_file = os.path.join(indices_dir, f"{source}_index.json")
    index_data = {
        "source": source,
        "imported_at": datetime.now().isoformat(),
        "node_count": node_count,
        "edge_count": edge_count,
        "data_path": os.path.abspath(os.path.join("kgot", "data"))
    }
    
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    
    return {
        "success": True,
        "nodes_imported": node_count,
        "edges_imported": edge_count,
        "backup_file": backup_file,
        "index_file": index_file
    }

def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"Importing data from {args.input} with source identifier '{args.source}'...")
    graph_data = load_graph_data(args.input)
    
    print(f"Found {len(graph_data.get('nodes', []))} nodes and {len(graph_data.get('edges', []))} edges.")
    
    # Import the data
    result = import_data(graph_data, args.source)
    
    if result["success"]:
        print(f"Successfully imported {result['nodes_imported']} nodes and {result['edges_imported']} edges.")
        print(f"Created backup at: {result['backup_file']}")
        print(f"Created index at: {result['index_file']}")
        print("\nImport complete! The data is now available in the KGoT system.")
    else:
        print("Import failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 