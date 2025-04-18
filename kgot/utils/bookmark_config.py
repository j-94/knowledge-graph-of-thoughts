"""
Bookmark Configuration Module.

This module provides functionality to load and validate bookmark processing
configuration using Pydantic models.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, List, Any

from pydantic import BaseModel, Field, validator, root_validator
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    uri: str = Field(..., description="Neo4j connection URI")
    user: str = Field(..., description="Database username")
    password: str = Field(..., description="Database password")
    database: str = Field("neo4j", description="Database name")
    
    @validator('password')
    def check_env_password(cls, v):
        """Check for password environment variable override."""
        env_password = os.getenv('BOOKMARK_NEO4J_PASSWORD')
        if env_password:
            return env_password
        return v
    
    @validator('uri')
    def check_env_uri(cls, v):
        """Check for URI environment variable override."""
        env_uri = os.getenv('BOOKMARK_NEO4J_URI')
        if env_uri:
            return env_uri
        return v

class ProcessingConfig(BaseModel):
    """Processing settings configuration."""
    batch_size: int = Field(128, description="Batch size for processing")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Sentence transformer model name")
    similarity_threshold: float = Field(0.75, description="Threshold for similarity relationships", ge=0.0, le=1.0)
    max_content_length: int = Field(5000, description="Maximum content length for embedding")

class PathsConfig(BaseModel):
    """Path settings configuration."""
    raw_data: str = Field(..., description="Path to raw bookmark data file")
    cache_dir: str = Field("cache/embeddings", description="Directory for caching embeddings")
    graph_snapshot_dir: str = Field("logs/bookmark_graph_snapshots", description="Directory for graph snapshots")
    
    @root_validator
    def create_directories(cls, values):
        """Create directories if they don't exist."""
        for key in ['cache_dir', 'graph_snapshot_dir']:
            if key in values and values[key]:
                os.makedirs(values[key], exist_ok=True)
        return values

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    file: Optional[str] = Field(None, description="Log file path")
    
    @validator('level')
    def check_log_level(cls, v):
        """Validate logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid logging level. Must be one of {valid_levels}")
        return v_upper
    
    @root_validator
    def create_log_directory(cls, values):
        """Create log directory if it doesn't exist."""
        if 'file' in values and values['file']:
            log_dir = os.path.dirname(values['file'])
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        return values

class TaskKGConfig(BaseModel):
    """Task Knowledge Graph configuration."""
    enabled: bool = Field(True, description="Whether to enable task-KG tracking")
    neo4j_label: str = Field("Task", description="Neo4j label for task nodes")
    status_property: str = Field("status", description="Property name for task status")
    dependency_relationship: str = Field("DEPENDS_ON", description="Relationship name for task dependencies")
    subtask_relationship: str = Field("HAS_SUBTASK", description="Relationship name for subtasks")

class BookmarkConfig(BaseModel):
    """Main configuration model for bookmark processing."""
    database: DatabaseConfig
    processing: ProcessingConfig = ProcessingConfig()
    paths: PathsConfig
    logging: LoggingConfig = LoggingConfig()
    task_kg: TaskKGConfig = TaskKGConfig()

def load_config(config_path: Union[str, Path] = "config/bookmark_config.yaml") -> BookmarkConfig:
    """
    Load and validate the bookmark configuration.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Validated BookmarkConfig object.
    """
    try:
        # Resolve path
        config_path = Path(config_path)
        
        # Load YAML config
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create and validate config
        config = BookmarkConfig(**config_dict)
        return config
    
    except Exception as e:
        raise ValueError(f"Error loading bookmark configuration: {e}")

# Singleton config instance
_config_instance = None

def get_config(config_path: Optional[Union[str, Path]] = None) -> BookmarkConfig:
    """
    Get the bookmark configuration, loading it if necessary.
    
    Args:
        config_path: Optional path to the configuration file.
        
    Returns:
        Validated BookmarkConfig object.
    """
    global _config_instance
    
    if _config_instance is None or config_path is not None:
        _config_instance = load_config(config_path or "config/bookmark_config.yaml")
    
    return _config_instance 