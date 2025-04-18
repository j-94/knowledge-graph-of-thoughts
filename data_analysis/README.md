# KGoT Data Analysis Environment

This directory contains a Docker-based development environment for data analysis within the Knowledge Graph of Thoughts (KGoT) project. It provides a consistent, reproducible environment with all necessary tools for data processing, analysis, visualization, and integration with the KGoT knowledge graph.

## Features

- Python 3.11 with pandas 2.2.2 and essential data science libraries
- Neo4j database for graph storage and analysis
- Jupyter Lab for interactive data exploration and visualization
- Tools for data validation, cleaning, and transformation
- Utilities for converting structured data to knowledge graphs

## Directory Structure

```
data_analysis/
├── data/                # Data storage (mapped to container)
├── docker/              # Docker configuration files
│   ├── Dockerfile       # Python environment definition
│   ├── requirements.txt # Python dependencies
│   └── validate_environment.py # Environment validation script
├── docker-compose.yml   # Container orchestration
├── notebooks/           # Jupyter notebooks (mapped to container)
└── README.md            # This file
```

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your system
- Git (to clone the repository)

### Getting Started

1. Create the necessary directories for data and notebooks:

```bash
mkdir -p data_analysis/data data_analysis/notebooks
```

2. Start the environment:

```bash
cd data_analysis
docker-compose up -d
```

3. Access Jupyter Lab:

Open your browser and navigate to: http://localhost:8888

The default token is: `kgot`

4. Access Neo4j Browser:

Open your browser and navigate to: http://localhost:7474

Default credentials:
- Username: `neo4j`
- Password: `password`

5. Validate the environment:

```bash
docker exec -it kgot_data_analysis python /app/docker/validate_environment.py
```

## Working with Data

### Loading Data

- Place your data files in the `data_analysis/data/` directory
- Access them from within Jupyter at `/app/data/`
- For large files like `data-bookmark.jsonl`, use chunking techniques

### Example Usage

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_json('/app/data/your_data.json', lines=True)

# Basic exploration
print(df.info())
print(df.describe())

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='feature1', y='feature2', hue='category')
plt.title('Data Visualization')
plt.savefig('/app/data/output_plot.png')
```

### Neo4j Integration

```python
from neo4j import GraphDatabase

# Connect to Neo4j
uri = "bolt://neo4j:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Example query
with driver.session() as session:
    result = session.run("MATCH (n) RETURN count(n) as count")
    print(f"Node count: {result.single()['count']}")

driver.close()
```

## Shutting Down

To stop the containers:

```bash
docker-compose down
```

To remove all containers and volumes:

```bash
docker-compose down -v
```

## Troubleshooting

- If Neo4j connection fails, ensure the Neo4j container is running
- For permission issues with data files, check directory permissions
- If Jupyter notebook is slow, consider allocating more resources in Docker settings 