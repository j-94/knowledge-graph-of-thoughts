#!/usr/bin/env python3
"""
Bookmark Data Processor

This script processes the data-bookmark.jsonl file to:
1. Make it more human-readable
2. Reorganize entries (by source, date, or custom criteria)
3. Update IDs sequentially
4. Remove unwanted entries
5. Export in a clean format

Usage:
    python process_bookmarks.py --input data-bookmark.jsonl --output organized-bookmarks.jsonl

Optional flags:
    --format [jsonl|json|csv|markdown] - Output format (default: jsonl)
    --group-by [source|date|none] - How to organize entries (default: source)
    --remove-older-than YYYY-MM-DD - Filter out entries older than specified date
    --min-id N - Remove entries with ID less than N
    --pretty - Make the output more human-readable (indented JSON, etc.)
    --filter-source SOURCE1,SOURCE2 - Keep only entries from specified sources
"""

import argparse
import json
import csv
import os
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import re

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process bookmark data for better organization and readability")
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--format", choices=["jsonl", "json", "csv", "markdown"], default="jsonl",
                        help="Output format (default: jsonl)")
    parser.add_argument("--group-by", choices=["source", "date", "none"], default="source",
                        help="How to organize entries (default: source)")
    parser.add_argument("--remove-older-than", type=str, 
                        help="Filter out entries older than YYYY-MM-DD")
    parser.add_argument("--min-id", type=int, help="Remove entries with ID less than specified number")
    parser.add_argument("--pretty", action="store_true", help="Make output more human-readable")
    parser.add_argument("--filter-source", type=str, help="Only include entries from these sources (comma-separated)")
    
    return parser.parse_args()

def parse_date(date_str):
    """Parse date string into datetime object."""
    # Handle various date formats
    formats = [
        "%Y-%m-%d %H:%M:%S%z",  # 2025-04-14T20:09:29Z
        "%Y-%m-%dT%H:%M:%S%z",   # 2025-04-14T20:09:29Z
        "%Y-%m-%dT%H:%M:%S.%f%z",  # 2025-04-14T20:09:29.123Z
        "%Y-%m-%d",                # 2025-04-14
    ]
    
    # Clean up the date string
    date_str = date_str.strip()
    if date_str.endswith('Z'):
        date_str = date_str[:-1] + '+0000'
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    print(f"Warning: Could not parse date '{date_str}', using current date")
    return datetime.now()

def clean_url(url):
    """Clean and normalize URL."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    return url

def process_bookmark(entry, new_id):
    """Process a single bookmark entry."""
    # Extract essential fields or add defaults
    bookmark = {
        "id": new_id,
        "url": clean_url(entry.get("url", "")),
        "source": entry.get("source", "unknown"),
        "content": entry.get("content", "").strip(),
        "created_at": entry.get("created_at", ""),
    }
    
    # Add a title if missing but available in metadata
    if "title" not in bookmark and "metadata" in entry and entry["metadata"]:
        metadata = entry["metadata"]
        if isinstance(metadata, dict):
            bookmark["title"] = metadata.get("title", "")
    
    # Clean up the content if needed
    if len(bookmark["content"]) > 300:
        bookmark["content"] = bookmark["content"][:297] + "..."
    
    # Organize metadata in a more readable structure
    if "metadata" in entry and entry["metadata"]:
        metadata = entry["metadata"]
        clean_metadata = {}
        
        # Extract the most useful metadata fields
        important_fields = ["language", "stars", "forks", "owner", "repo", 
                            "author", "tags", "description", "username", "user_name"]
        
        for field in important_fields:
            if field in metadata and metadata[field]:
                clean_metadata[field] = metadata[field]
        
        bookmark["metadata"] = clean_metadata
    
    return bookmark

def read_jsonl(file_path):
    """Read JSONL file and return list of dictionaries."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.strip():  # Skip empty lines
                    entry = json.loads(line)
                    entries.append(entry)
            except json.JSONDecodeError:
                print(f"Error: Couldn't parse line {line_num}, skipping")
    return entries

def filter_entries(entries, args):
    """Filter entries based on command line arguments."""
    filtered = []
    cutoff_date = None
    
    if args.remove_older_than:
        try:
            cutoff_date = datetime.strptime(args.remove_older_than, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{args.remove_older_than}', should be YYYY-MM-DD")
            sys.exit(1)
    
    sources = None
    if args.filter_source:
        sources = [s.strip().lower() for s in args.filter_source.split(',')]
    
    for entry in entries:
        # Skip if ID is less than minimum
        if args.min_id and entry.get("id", 0) < args.min_id:
            continue
        
        # Skip if from filtered source
        if sources and entry.get("source", "").lower() not in sources:
            continue
        
        # Skip if older than cutoff date
        if cutoff_date and "created_at" in entry:
            try:
                entry_date = parse_date(entry["created_at"])
                if entry_date < cutoff_date:
                    continue
            except (ValueError, TypeError):
                # If date parsing fails, keep the entry
                pass
        
        filtered.append(entry)
    
    return filtered

def group_entries(entries, group_by):
    """Group entries by specified criteria."""
    if group_by == "none":
        return {"all": entries}
    
    groups = defaultdict(list)
    
    for entry in entries:
        if group_by == "source":
            key = entry.get("source", "unknown")
        elif group_by == "date":
            try:
                date_str = entry.get("created_at", "")
                if date_str:
                    date_obj = parse_date(date_str)
                    key = date_obj.strftime("%Y-%m")  # Group by year-month
                else:
                    key = "unknown_date"
            except (ValueError, TypeError):
                key = "unknown_date"
        else:
            key = "all"
        
        groups[key].append(entry)
    
    # Sort the groups
    return dict(sorted(groups.items()))

def write_jsonl_output(grouped_entries, output_path, pretty=False):
    """Write entries to JSONL output file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        current_id = 1
        
        for group, entries in grouped_entries.items():
            # Sort entries by date within each group
            entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            
            for entry in entries:
                processed = process_bookmark(entry, current_id)
                if pretty:
                    f.write(json.dumps(processed, ensure_ascii=False, indent=2) + '\n')
                else:
                    f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                current_id += 1

def write_json_output(grouped_entries, output_path, pretty=False):
    """Write entries to JSON output file."""
    result = {}
    current_id = 1
    
    for group, entries in grouped_entries.items():
        # Sort entries by date within each group
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        
        result[group] = []
        for entry in entries:
            processed = process_bookmark(entry, current_id)
            result[group].append(processed)
            current_id += 1
    
    with open(output_path, 'w', encoding='utf-8') as f:
        indent = 2 if pretty else None
        json.dump(result, f, ensure_ascii=False, indent=indent)

def write_csv_output(grouped_entries, output_path):
    """Write entries to CSV output file."""
    fieldnames = ["id", "title", "url", "source", "content", "created_at", "metadata_summary"]
    
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        current_id = 1
        for group, entries in grouped_entries.items():
            # Sort entries by date within each group
            entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            
            # Add a group header row
            writer.writerow({
                "id": "",
                "title": f"=== GROUP: {group} ===",
                "url": "",
                "source": "",
                "content": "",
                "created_at": "",
                "metadata_summary": ""
            })
            
            for entry in entries:
                processed = process_bookmark(entry, current_id)
                
                # Prepare metadata summary
                metadata_summary = ""
                if "metadata" in processed:
                    metadata_parts = []
                    for k, v in processed["metadata"].items():
                        metadata_parts.append(f"{k}: {v}")
                    metadata_summary = "; ".join(metadata_parts)
                
                row = {
                    "id": processed["id"],
                    "title": processed.get("title", ""),
                    "url": processed["url"],
                    "source": processed["source"],
                    "content": processed["content"],
                    "created_at": processed["created_at"],
                    "metadata_summary": metadata_summary
                }
                writer.writerow(row)
                current_id += 1

def write_markdown_output(grouped_entries, output_path):
    """Write entries to Markdown output file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Organized Bookmarks\n\n")
        f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        
        current_id = 1
        
        for group, entries in grouped_entries.items():
            f.write(f"## Group: {group}\n\n")
            
            # Sort entries by date within each group
            entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
            
            for entry in entries:
                processed = process_bookmark(entry, current_id)
                
                # Write entry as markdown
                f.write(f"### [{current_id}] {processed.get('title', 'Untitled')}\n\n")
                f.write(f"- **URL**: [{processed['url']}]({processed['url']})\n")
                f.write(f"- **Source**: {processed['source']}\n")
                f.write(f"- **Created**: {processed['created_at']}\n")
                
                if processed.get("content"):
                    f.write(f"\n{processed['content']}\n\n")
                
                if "metadata" in processed and processed["metadata"]:
                    f.write("**Metadata**:\n\n")
                    for k, v in processed["metadata"].items():
                        f.write(f"- {k}: {v}\n")
                
                f.write("\n---\n\n")
                current_id += 1

def main():
    """Main function to process bookmarks."""
    args = parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Read input file
    print(f"Reading entries from {args.input}...")
    entries = read_jsonl(args.input)
    print(f"Found {len(entries)} entries")
    
    # Filter entries
    print("Applying filters...")
    filtered_entries = filter_entries(entries, args)
    print(f"Kept {len(filtered_entries)} entries after filtering")
    
    # Group entries
    print(f"Grouping entries by {args.group_by}...")
    grouped_entries = group_entries(filtered_entries, args.group_by)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write output in specified format
    print(f"Writing to {args.output} in {args.format} format...")
    if args.format == "jsonl":
        write_jsonl_output(grouped_entries, args.output, args.pretty)
    elif args.format == "json":
        write_json_output(grouped_entries, args.output, args.pretty)
    elif args.format == "csv":
        write_csv_output(grouped_entries, args.output)
    elif args.format == "markdown":
        write_markdown_output(grouped_entries, args.output)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 