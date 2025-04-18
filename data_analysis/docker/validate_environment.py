#!/usr/bin/env python
"""
Environment Validation Script for KGoT Data Analysis

This script validates that all required libraries are properly installed
and that connections to services like Neo4j are working correctly.
"""

import os
import sys
import time
from datetime import datetime

# Setup colored output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(message):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def print_section(message):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{message}{Colors.ENDC}")
    print(f"{Colors.BLUE}{'-' * len(message)}{Colors.ENDC}\n")

def print_success(message):
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")

def check_library(name, lib=None):
    try:
        if lib is None:
            exec(f"import {name}")
        else:
            exec(f"import {lib} as {name}")
        version = eval(f"{name}.__version__")
        print_success(f"{name} successfully imported (version: {version})")
        return True
    except ImportError:
        print_error(f"Failed to import {name}")
        return False
    except AttributeError:
        print_success(f"{name} successfully imported (version unknown)")
        return True
    except Exception as e:
        print_warning(f"{name} import issue: {str(e)}")
        return False

def check_neo4j_connection():
    try:
        from neo4j import GraphDatabase, exceptions

        uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "password")
        
        print(f"Connecting to Neo4j at {uri}...")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # This verifies that the connection is live
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' AS message")
            message = result.single()["message"]
            print_success(f"Neo4j connection test: {message}")
            
            # Get Neo4j version
            version_result = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions")
            record = version_result.single()
            print_success(f"Connected to {record['name']} (version: {record['versions'][0]})")
            
        driver.close()
        return True
    except exceptions.ServiceUnavailable:
        print_error("Neo4j service unavailable - is the database running?")
        return False
    except Exception as e:
        print_error(f"Neo4j connection error: {str(e)}")
        return False

def create_sample_dataframe():
    try:
        import pandas as pd
        import numpy as np
        
        # Create a simple DataFrame
        df = pd.DataFrame({
            'A': np.random.randn(10),
            'B': np.random.randn(10),
            'C': np.random.choice(['X', 'Y', 'Z'], 10),
            'D': pd.date_range(start='2023-01-01', periods=10)
        })
        
        print_success("Created sample DataFrame:")
        print(df.head())
        
        # Basic stats
        print("\nDataFrame statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print_error(f"Error creating sample DataFrame: {str(e)}")
        return None

def create_sample_visualization(df):
    if df is None:
        return False
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set(style="whitegrid")
        
        # Create a simple plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='A', y='B', hue='C')
        plt.title('Sample Visualization')
        
        # Save the plot
        plt.savefig('/app/data/validation_plot.png')
        print_success("Created and saved sample visualization to /app/data/validation_plot.png")
        
        return True
    except Exception as e:
        print_error(f"Error creating visualization: {str(e)}")
        return False

def check_graph_capabilities():
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create a simple graph
        G = nx.Graph()
        G.add_nodes_from(range(1, 6))
        G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
        
        print_success(f"Created NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Basic graph metrics
        print("\nGraph metrics:")
        print(f"Density: {nx.density(G)}")
        print(f"Average shortest path length: {nx.average_shortest_path_length(G)}")
        print(f"Diameter: {nx.diameter(G)}")
        
        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(G, pos, node_color='lightblue', with_labels=True, node_size=500)
        plt.title('Sample Graph')
        plt.axis('off')
        
        # Save the graph
        plt.savefig('/app/data/graph_validation.png')
        print_success("Created and saved graph visualization to /app/data/graph_validation.png")
        
        return True
    except Exception as e:
        print_error(f"Error testing graph capabilities: {str(e)}")
        return False

def main():
    print_header("KGoT Data Analysis Environment Validation")
    
    # System information
    print_section("System Information")
    print(f"Python version: {sys.version}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Core data libraries
    print_section("Core Data Libraries")
    check_library("pandas")
    check_library("np", "numpy")
    check_library("scipy")
    check_library("sklearn", "sklearn")
    
    # Visualization libraries
    print_section("Visualization Libraries")
    check_library("matplotlib")
    check_library("plt", "matplotlib.pyplot")
    check_library("seaborn")
    check_library("plotly")
    
    # Graph libraries
    print_section("Graph Libraries")
    check_library("networkx")
    
    # Database connections
    print_section("Database Connections")
    check_neo4j_connection()
    
    # Create sample data and visualizations
    print_section("Data Processing Verification")
    df = create_sample_dataframe()
    create_sample_visualization(df)
    check_graph_capabilities()
    
    print_header("Validation Complete")

if __name__ == "__main__":
    main() 