#!/bin/bash

# Script to process bookmarks using Docker
mkdir -p output

function show_help {
  echo "Bookmark Data Processor Script"
  echo ""
  echo "Usage: $0 [option]"
  echo ""
  echo "Options:"
  echo "  standard    Process bookmarks with standard settings (organize by source)"
  echo "  markdown    Export bookmarks in markdown format"
  echo "  recent      Process only recent bookmarks (after 2023-01-01)"
  echo "  github      Process only GitHub bookmarks"
  echo "  twitter     Process only Twitter bookmarks"
  echo "  custom      Run with custom options (provide as arguments)"
  echo "  clean       Remove generated output files and Docker images"
  echo "  kgot        Process bookmarks and prepare for Knowledge Graph of Thoughts"
  echo "  kgot-export Export bookmarks in KGoT-compatible format"
  echo "  kgot-import Import processed bookmarks into KGoT system"
  echo "  custom-ontology Process bookmarks with a custom ontology definition"
  echo "  help        Show this help message"
  echo ""
  echo "Examples:"
  echo "  $0 standard"
  echo "  $0 markdown"
  echo "  $0 custom --input data-bookmark.jsonl --output output/custom.jsonl --format json --pretty"
  echo "  $0 kgot"
  echo "  $0 custom-ontology ontology.json"
  echo ""
}

case "$1" in
  standard)
    echo "Processing bookmarks with standard settings..."
    docker-compose run --rm bookmark-processor
    ;;
    
  markdown)
    echo "Exporting bookmarks to markdown..."
    docker-compose --profile markdown run --rm markdown-export
    ;;
    
  recent)
    echo "Processing recent bookmarks..."
    docker-compose --profile recent run --rm recent-bookmarks
    ;;
    
  github)
    echo "Processing GitHub bookmarks..."
    docker-compose --profile github run --rm github-bookmarks
    ;;
    
  twitter)
    echo "Processing Twitter bookmarks..."
    docker-compose build
    docker run --rm -v "$(pwd):/app" -v "$(pwd)/output:/app/output" bookmark-processor-bookmark-processor --input /app/data-bookmark.jsonl --output /app/output/twitter-bookmarks.jsonl --pretty --filter-source twitter
    ;;
    
  custom)
    shift  # Remove the "custom" argument
    echo "Processing bookmarks with custom settings: $@"
    docker-compose build
    docker run --rm -v "$(pwd):/app" -v "$(pwd)/output:/app/output" bookmark-processor-bookmark-processor $@
    ;;
    
  kgot)
    echo "Processing bookmarks for Knowledge Graph of Thoughts integration..."
    # First, process the bookmarks to JSON format
    echo "Step 1: Processing bookmarks to JSON format..."
    mkdir -p output
    
    # Use the correct Docker image name - using our existing service
    docker-compose build
    
    # Run with our service name, not the image name
    echo "Running Docker container to process bookmarks..."
    docker-compose run --rm bookmark-processor --input /app/data-bookmark.jsonl --output /app/output/kgot-bookmarks.json --format json --pretty --group-by source
    
    echo "Converting to KGoT-compatible format..."
    # Check if the output file exists before processing
    if [ ! -f "output/kgot-bookmarks.json" ]; then
      echo "Error: Failed to create kgot-bookmarks.json. Check Docker setup."
      exit 1
    fi
    
    # Check if Python is installed
    if command -v python3 &> /dev/null; then
      echo "Step 2: Converting to Knowledge Graph format using Python..."
      python3 -c "
import json
import os
import re
from datetime import datetime

# Helper function to parse dates in various formats
def parse_date(date_str):
    if not date_str:
        return datetime.now().isoformat()
        
    # Handle various date formats
    formats = [
        '%Y-%m-%d %H:%M:%S.%f%z',    # 2024-11-21 07:25:19.324000+00:00
        '%Y-%m-%dT%H:%M:%S.%f%z',    # 2024-11-21T07:25:19.324000+00:00
        '%Y-%m-%d %H:%M:%S%z',       # 2024-11-21 07:25:19+00:00
        '%Y-%m-%dT%H:%M:%S%z',       # 2024-11-21T07:25:19+00:00
        '%Y-%m-%dT%H:%M:%S.%fZ',     # 2024-11-21T07:25:19.324Z
        '%Y-%m-%dT%H:%M:%SZ',        # 2024-11-21T07:25:19Z
        '%Y-%m-%d',                  # 2024-11-21
    ]
    
    # Normalize the date string - handle Z timezone
    date_str = date_str.strip()
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+0000'
    
    # Try each format
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    
    # If all formats fail, use regex to extract date components
    try:
        # Match patterns like: 2024-11-21 07:25:19.324000+00:00
        match = re.match(r'(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:?\d{2}|Z)', date_str)
        if match:
            date_part, time_part, _, tz_part = match.groups()
            # Format with fixed microsecond precision
            parsed = f'{date_part}T{time_part}+00:00'
            return parsed
    except Exception:
        pass
    
    print(f'Warning: Could not parse date \"{date_str}\", using current date')
    return datetime.now().isoformat()

# Load processed bookmarks
try:
    with open('output/kgot-bookmarks.json', 'r') as f:
        data = json.load(f)
    
    print(f'Successfully loaded bookmark data with {sum(len(bookmarks) for bookmarks in data.values())} bookmarks')
    
    # Convert to KGoT format (nodes and edges)
    nodes = []
    edges = []
    node_ids = set()
    
    # Process each source group
    for source, bookmarks in data.items():
        print(f'Processing {len(bookmarks)} bookmarks from source: {source}')
        for bookmark in bookmarks:
            # Create node for bookmark
            node_id = f'b_{bookmark[\"id\"]}'
            
            # Parse and normalize the date
            created_at = bookmark.get('created_at', '')
            normalized_date = parse_date(created_at) if created_at else ''
            
            node = {
                'id': node_id,
                'type': 'bookmark',
                'title': bookmark.get('title', 'Untitled'),
                'url': bookmark['url'],
                'source': bookmark['source'],
                'created_at': normalized_date,
                'content': bookmark.get('content', '')
            }
            
            # Add metadata if available
            if 'metadata' in bookmark:
                # Make a clean copy of metadata to avoid invalid types
                clean_metadata = {}
                
                # Only copy valid string/number/boolean values
                if isinstance(bookmark['metadata'], dict):
                    for key, value in bookmark['metadata'].items():
                        if isinstance(value, (str, int, float, bool)) or isinstance(value, list):
                            clean_metadata[key] = value
                
                node['metadata'] = clean_metadata
            
            nodes.append(node)
            node_ids.add(node_id)
            
            # Create tags as nodes and connect with edges
            if 'metadata' in bookmark and isinstance(bookmark['metadata'], dict) and 'tags' in bookmark['metadata']:
                # Ensure tags is a list
                tags = bookmark['metadata']['tags']
                if isinstance(tags, list):
                    for tag in tags:
                        if isinstance(tag, (str, int)):
                            tag_str = str(tag)
                            tag_id = f't_{tag_str}'
                            
                            # Add tag node if not exists
                            if tag_id not in node_ids:
                                tag_node = {
                                    'id': tag_id,
                                    'type': 'tag',
                                    'name': tag_str
                                }
                                nodes.append(tag_node)
                                node_ids.add(tag_id)
                            
                            # Create edge from bookmark to tag
                            edge = {
                                'source': node_id,
                                'target': tag_id,
                                'type': 'has_tag'
                            }
                            edges.append(edge)
                elif isinstance(tags, str):
                    # Handle case where tags is a single string
                    tag_id = f't_{tags}'
                    
                    if tag_id not in node_ids:
                        tag_node = {
                            'id': tag_id,
                            'type': 'tag',
                            'name': tags
                        }
                        nodes.append(tag_node)
                        node_ids.add(tag_id)
                    
                    edge = {
                        'source': node_id,
                        'target': tag_id,
                        'type': 'has_tag'
                    }
                    edges.append(edge)
    
    # Save as separate files for nodes and edges
    with open('output/kgot_nodes.jsonl', 'w') as f:
        for node in nodes:
            f.write(json.dumps(node) + '\\n')
    
    with open('output/kgot_edges.jsonl', 'w') as f:
        for edge in edges:
            f.write(json.dumps(edge) + '\\n')
    
    # Also save as a single file for easy loading
    kgot_data = {
        'nodes': nodes,
        'edges': edges
    }
    with open('output/kgot_graph.json', 'w') as f:
        json.dump(kgot_data, f, indent=2)
    
    print(f'Created {len(nodes)} nodes and {len(edges)} edges for Knowledge Graph')
except Exception as e:
    print(f'Error processing bookmark data: {str(e)}')
    import traceback
    traceback.print_exc()
"
    else
      echo "Python3 is required for KGoT conversion. Please install Python3 and try again."
    fi
    ;;
    
  kgot-export)
    echo "Exporting bookmarks in KGoT-compatible format..."
    if [ ! -f "output/kgot-bookmarks.json" ]; then
      echo "Running KGoT processing first..."
      $0 kgot
    fi
    
    # Copy to KGoT data directory if it exists
    if [ -d "kgot/data" ]; then
      cp output/kgot_*.jsonl kgot/data/
      cp output/kgot_graph.json kgot/data/
      echo "Exported to kgot/data directory"
    else
      echo "KGoT data directory not found. Files are available in the output directory."
    fi
    ;;
    
  kgot-import)
    echo "Importing processed bookmarks into KGoT system..."
    if [ ! -f "output/kgot_graph.json" ]; then
      echo "KGoT processed files not found. Running KGoT processing first..."
      $0 kgot
    fi
    
    # Run the KGoT import command if available
    if [ -f "./kgot/import.py" ]; then
      python3 ./kgot/import.py --input output/kgot_graph.json --source bookmarks
      echo "Imported bookmark data into KGoT system"
    else
      echo "KGoT import tool not found. Please ensure the KGoT system is properly installed."
      echo "KGoT-compatible files are available in the output directory."
    fi
    ;;
    
  custom-ontology)
    echo "Processing bookmarks with custom ontology..."
    ONTOLOGY_FILE="$2"
    
    if [ -z "$ONTOLOGY_FILE" ]; then
      echo "Error: You must provide an ontology definition file"
      echo "Usage: $0 custom-ontology ontology.json"
      exit 1
    fi
    
    if [ ! -f "$ONTOLOGY_FILE" ]; then
      echo "Error: Ontology file $ONTOLOGY_FILE not found"
      exit 1
    fi
    
    echo "Using ontology definition from: $ONTOLOGY_FILE"
    
    # First, process the bookmarks to JSON format
    echo "Step 1: Processing bookmarks to JSON format..."
    mkdir -p output
    
    docker-compose build
    docker-compose run --rm bookmark-processor --input /app/data-bookmark.jsonl --output /app/output/kgot-bookmarks.json --format json --pretty --group-by source
    
    echo "Converting to KGoT-compatible format with custom ontology..."
    # Check if the output file exists before processing
    if [ ! -f "output/kgot-bookmarks.json" ]; then
      echo "Error: Failed to create kgot-bookmarks.json. Check Docker setup."
      exit 1
    fi
    
    # Check if Python is installed
    if command -v python3 &> /dev/null; then
      echo "Step 2: Converting to Knowledge Graph format using Python with custom ontology..."
      python3 -c "
import json
import os
import sys
import re
from datetime import datetime

# Helper function to parse dates in various formats
def parse_date(date_str):
    if not date_str:
        return datetime.now().isoformat()
        
    # Handle various date formats
    formats = [
        '%Y-%m-%d %H:%M:%S.%f%z',    # 2024-11-21 07:25:19.324000+00:00
        '%Y-%m-%dT%H:%M:%S.%f%z',    # 2024-11-21T07:25:19.324000+00:00
        '%Y-%m-%d %H:%M:%S%z',       # 2024-11-21 07:25:19+00:00
        '%Y-%m-%dT%H:%M:%S%z',       # 2024-11-21T07:25:19+00:00
        '%Y-%m-%dT%H:%M:%S.%fZ',     # 2024-11-21T07:25:19.324Z
        '%Y-%m-%dT%H:%M:%SZ',        # 2024-11-21T07:25:19Z
        '%Y-%m-%d',                  # 2024-11-21
    ]
    
    # Normalize the date string - handle Z timezone
    date_str = date_str.strip()
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+0000'
    
    # Try each format
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.isoformat()
        except ValueError:
            continue
    
    # If all formats fail, use regex to extract date components
    try:
        # Match patterns like: 2024-11-21 07:25:19.324000+00:00
        match = re.match(r'(\d{4}-\d{2}-\d{2})[T\s](\d{2}:\d{2}:\d{2})\.(\d+)([+-]\d{2}:?\d{2}|Z)', date_str)
        if match:
            date_part, time_part, _, tz_part = match.groups()
            # Format with fixed microsecond precision
            parsed = f'{date_part}T{time_part}+00:00'
            return parsed
    except Exception:
        pass
    
    print(f'Warning: Could not parse date \"{date_str}\", using current date')
    return datetime.now().isoformat()

# Load the ontology definition
try:
    with open('$ONTOLOGY_FILE', 'r') as f:
        ontology = json.load(f)
        
    print(f'Successfully loaded ontology definition from {\"$ONTOLOGY_FILE\"}')
    
    # Extract ontology parameters
    node_types = ontology.get('nodeTypes', [])
    relation_types = ontology.get('relationTypes', [])
    category_mappings = ontology.get('categoryMappings', {})
    entity_types = ontology.get('entityTypes', [])
    
    print(f'Ontology has {len(node_types)} node types, {len(relation_types)} relation types')
    print(f'Using {len(category_mappings)} category mappings and {len(entity_types)} entity types')
    
except Exception as e:
    print(f'Error loading ontology definition: {str(e)}')
    sys.exit(1)

# Load processed bookmarks
try:
    with open('output/kgot-bookmarks.json', 'r') as f:
        data = json.load(f)
    
    print(f'Successfully loaded bookmark data with {sum(len(bookmarks) for bookmarks in data.values())} bookmarks')
    
    # Convert to KGoT format (nodes and edges)
    nodes = []
    edges = []
    node_ids = set()
    
    # Create nodes for each node type defined in the ontology
    # (for categories, entities, etc. that will be referenced)
    for node_type in node_types:
        if 'predefined' in node_type and node_type['predefined']:
            for item in node_type['items']:
                node_id = f'{node_type[\"prefix\"]}_{item[\"id\"]}'
                if node_id not in node_ids:
                    node = {
                        'id': node_id,
                        'type': node_type['type'],
                        'name': item['name'],
                        'category': node_type.get('category', 'unknown')
                    }
                    # Add any additional properties from the ontology
                    for key, value in item.items():
                        if key not in ['id', 'name']:
                            node[key] = value
                            
                    nodes.append(node)
                    node_ids.add(node_id)
                    print(f'Created predefined node {node_id}: {item[\"name\"]}')
    
    # Process each source group
    for source, bookmarks in data.items():
        print(f'Processing {len(bookmarks)} bookmarks from source: {source}')
        for bookmark in bookmarks:
            # Create node for bookmark
            node_id = f'b_{bookmark[\"id\"]}'
            
            # Parse and normalize the date
            created_at = bookmark.get('created_at', '')
            normalized_date = parse_date(created_at) if created_at else ''
            
            node = {
                'id': node_id,
                'type': 'bookmark',
                'title': bookmark.get('title', 'Untitled'),
                'url': bookmark['url'],
                'source': bookmark['source'],
                'created_at': normalized_date,
                'content': bookmark.get('content', '')
            }
            
            # Add metadata if available
            if 'metadata' in bookmark:
                # Make a clean copy of metadata to avoid invalid types
                clean_metadata = {}
                
                # Only copy valid string/number/boolean values
                if isinstance(bookmark['metadata'], dict):
                    for key, value in bookmark['metadata'].items():
                        if isinstance(value, (str, int, float, bool)) or isinstance(value, list):
                            clean_metadata[key] = value
                
                node['metadata'] = clean_metadata
            
            nodes.append(node)
            node_ids.add(node_id)
            
            # Apply category mappings from the ontology
            # This connects bookmarks to predefined categories
            for category_type, mapping_rules in category_mappings.items():
                for rule in mapping_rules:
                    category_node_id = None
                    applies = False
                    
                    # Check if this bookmark should be connected to this category
                    if 'sourceMatch' in rule and rule['sourceMatch'] == bookmark['source']:
                        applies = True
                    
                    if 'urlPattern' in rule and rule['urlPattern'] in bookmark['url']:
                        applies = True
                        
                    if 'metadataField' in rule and 'metadata' in bookmark:
                        field = rule['metadataField']
                        if field in bookmark['metadata']:
                            applies = True
                    
                    if applies:
                        # Get the category node ID
                        if 'targetCategory' in rule:
                            category_node_id = f\"{rule['nodeTypePrefix']}_{rule['targetCategory']}\"
                        
                        # Create the edge connecting bookmark to category
                        if category_node_id and category_node_id in node_ids:
                            edge = {
                                'source': node_id,
                                'target': category_node_id,
                                'type': rule.get('relationType', 'belongs_to')
                            }
                            edges.append(edge)
                            print(f'Connected bookmark {node_id} to category {category_node_id}')
            
            # Create tags as nodes and connect with edges
            if 'metadata' in bookmark and isinstance(bookmark['metadata'], dict) and 'tags' in bookmark['metadata']:
                # Ensure tags is a list
                tags = bookmark['metadata']['tags']
                if isinstance(tags, list):
                    for tag in tags:
                        if isinstance(tag, (str, int)):
                            tag_str = str(tag)
                            tag_id = f't_{tag_str}'
                            
                            # Add tag node if not exists
                            if tag_id not in node_ids:
                                tag_node = {
                                    'id': tag_id,
                                    'type': 'tag',
                                    'name': tag_str
                                }
                                nodes.append(tag_node)
                                node_ids.add(tag_id)
                            
                            # Create edge from bookmark to tag
                            edge = {
                                'source': node_id,
                                'target': tag_id,
                                'type': 'has_tag'
                            }
                            edges.append(edge)
                elif isinstance(tags, str):
                    # Handle case where tags is a single string
                    tag_id = f't_{tags}'
                    
                    if tag_id not in node_ids:
                        tag_node = {
                            'id': tag_id,
                            'type': 'tag',
                            'name': tags
                        }
                        nodes.append(tag_node)
                        node_ids.add(tag_id)
                    
                    edge = {
                        'source': node_id,
                        'target': tag_id,
                        'type': 'has_tag'
                    }
                    edges.append(edge)
    
    # Process any additional relationship rules from the ontology
    for relation in relation_types:
        if 'rules' in relation:
            for rule in relation['rules']:
                if 'sourceNodeType' in rule and 'targetNodeType' in rule:
                    source_type = rule['sourceNodeType']
                    target_type = rule['targetNodeType']
                    
                    # Find matching source and target nodes
                    for source_node in [n for n in nodes if n['type'] == source_type]:
                        for target_node in [n for n in nodes if n['type'] == target_type]:
                            # Check if rule conditions are met
                            conditions_met = True
                            
                            if 'conditions' in rule:
                                for condition in rule['conditions']:
                                    if 'sourceProperty' in condition and 'targetProperty' in condition:
                                        source_value = source_node.get(condition['sourceProperty'])
                                        target_value = target_node.get(condition['targetProperty'])
                                        
                                        if condition.get('match') == 'exact' and source_value != target_value:
                                            conditions_met = False
                                        elif condition.get('match') == 'contains' and not (source_value and target_value and source_value in target_value):
                                            conditions_met = False
                            
                            if conditions_met:
                                edge = {
                                    'source': source_node['id'],
                                    'target': target_node['id'],
                                    'type': relation['type']
                                }
                                
                                # Add any additional properties to the edge
                                if 'properties' in rule:
                                    for key, value in rule['properties'].items():
                                        edge[key] = value
                                
                                edges.append(edge)
                                print(f'Created relationship: {source_node[\"id\"]} --[{relation[\"type\"]}]--> {target_node[\"id\"]}')
    
    # Save as separate files for nodes and edges
    with open('output/kgot_nodes.jsonl', 'w') as f:
        for node in nodes:
            f.write(json.dumps(node) + '\\n')
    
    with open('output/kgot_edges.jsonl', 'w') as f:
        for edge in edges:
            f.write(json.dumps(edge) + '\\n')
    
    # Also save as a single file for easy loading
    kgot_data = {
        'nodes': nodes,
        'edges': edges,
        'ontology': ontology  # Include the ontology in the output for reference
    }
    with open('output/kgot_graph.json', 'w') as f:
        json.dump(kgot_data, f, indent=2)
    
    print(f'Created {len(nodes)} nodes and {len(edges)} edges for Knowledge Graph')
    print(f'Using custom ontology with {len(node_types)} node types and {len(relation_types)} relation types')
except Exception as e:
    print(f'Error processing bookmark data: {str(e)}')
    import traceback
    traceback.print_exc()
"
    else
      echo "Python3 is required for KGoT conversion. Please install Python3 and try again."
    fi
    ;;
    
  clean)
    echo "Cleaning up..."
    rm -rf output/*
    docker-compose down --rmi local
    ;;
    
  help|*)
    show_help
    ;;
esac

# List output dir contents if any operation was performed
if [ "$1" != "help" ] && [ "$1" != "clean" ]; then
  echo "Output directory contents:"
  ls -la output/
fi 