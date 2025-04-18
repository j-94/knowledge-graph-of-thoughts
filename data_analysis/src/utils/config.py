"""
Configuration loading utility for Knowledge Graph of Thoughts bookmark processing.

This module provides functionality to load, merge, and validate configuration settings
from YAML files with environment-specific overrides and environment variable substitution.
"""

import os
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, validator

# Configure logging
logger = logging.getLogger(__name__)

# Define configuration models
class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    host: str = Field("neo4j", description="Neo4j host address")
    port: int = Field(7687, description="Neo4j port")
    user: str = Field("neo4j", description="Neo4j username")
    password: str = Field("password", description="Neo4j password")
    database_name: str = Field("neo4j", description="Neo4j database name")
    connection_pool_size: Optional[int] = Field(None, description="Connection pool size")
    connection_timeout: Optional[int] = Field(None, description="Connection timeout in seconds")
    connection_acquisition_timeout: Optional[int] = Field(None, description="Connection acquisition timeout")
    connection_liveness_check_timeout: Optional[int] = Field(None, description="Connection liveness check timeout")
    max_transaction_retry_time: Optional[int] = Field(None, description="Maximum transaction retry time")

class ValidationConfig(BaseModel):
    """Validation configuration for bookmark data."""
    required_fields: list[str] = Field(["id", "title", "url", "tags"], 
                                     description="Fields required in each bookmark")
    optional_fields: list[str] = Field(["timestamp", "description", "content", "source", "metadata"],
                                     description="Optional fields in bookmarks")
    strict_mode: bool = Field(False, description="Enforce strict validation")
    fail_fast: bool = Field(False, description="Stop processing on first validation error")

class NormalizationConfig(BaseModel):
    """Normalization configuration for bookmark data."""
    timestamp_format: str = Field("%Y-%m-%d %H:%M:%S%z", description="Timestamp format string")
    url_normalization: bool = Field(True, description="Normalize URLs")
    text_cleanup: bool = Field(True, description="Clean up text fields")

class BookmarkProcessingConfig(BaseModel):
    """Bookmark processing configuration."""
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
    batch_size: int = Field(1000, description="Number of bookmarks to process in each batch")
    max_workers: int = Field(4, description="Maximum number of worker processes/threads")
    timeout_seconds: int = Field(300, description="Processing timeout in seconds")

class EmbeddingConfig(BaseModel):
    """Embedding generation configuration."""
    model_name: str = Field("sentence-transformers/all-mpnet-base-v2", 
                           description="Name of the sentence transformer model")
    dimension: int = Field(768, description="Embedding dimension")
    batch_size: int = Field(32, description="Batch size for embedding generation")
    cache_enabled: bool = Field(True, description="Enable embedding caching")
    cache_dir: str = Field("/app/data/embedding_cache", description="Directory for embedding cache")
    use_title_fallback: bool = Field(True, description="Use title when content is unavailable")

class RelationshipType(BaseModel):
    """Configuration for a relationship type."""
    name: str = Field(..., description="Name of the relationship type")
    weight_multiplier: float = Field(1.0, description="Weight multiplier for this relationship type")

class RelationshipConfig(BaseModel):
    """Relationship generation configuration."""
    similarity_threshold: float = Field(0.7, description="Minimum similarity score to create relationship")
    tag_overlap_min: int = Field(2, description="Minimum number of shared tags for relationship")
    max_temporal_distance_hours: int = Field(24, description="Maximum time difference in hours")
    types: list[RelationshipType] = Field(
        default=[
            RelationshipType(name="SIMILAR_TO", weight_multiplier=1.0),
            RelationshipType(name="SHARES_TAGS_WITH", weight_multiplier=0.8),
            RelationshipType(name="CREATED_NEAR", weight_multiplier=0.5)
        ],
        description="List of relationship types and their properties"
    )

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                       description="Log format string")
    file: str = Field("/app/logs/bookmark_processing.log", description="Log file path")

class ApiConfig(BaseModel):
    """API configuration."""
    host: str = Field("0.0.0.0", description="API host address")
    port: int = Field(8000, description="API port")
    debug: bool = Field(False, description="Enable debug mode")
    enable_docs: bool = Field(True, description="Enable API documentation")
    workers: Optional[int] = Field(None, description="Number of worker processes")
    cors_origins: list[str] = Field(["http://localhost:3000", "http://localhost:8080"], 
                                  description="Allowed CORS origins")

class BookmarkConfig(BaseModel):
    """Complete bookmark processing configuration."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    bookmark_processing: BookmarkProcessingConfig = Field(default_factory=BookmarkProcessingConfig)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    relationships: RelationshipConfig = Field(default_factory=RelationshipConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)


class ConfigLoader:
    """
    Configuration loader for the bookmark processing system.
    
    This class handles loading configuration from YAML files, merging configuration
    from different environments, and substituting environment variables.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to the configuration directory. If None, uses default.
        """
        self.config_dir = Path(config_dir) if config_dir else Path("data_analysis/config")
        self.env_pattern = re.compile(r'\${([^}^{]+)}')
        
    def _substitute_env_vars(self, config_str: str) -> str:
        """
        Substitute environment variables in configuration strings.
        
        Args:
            config_str: Configuration string with ${VAR} placeholders
            
        Returns:
            Configuration string with environment variables substituted
        """
        def _replace_env_var(match):
            env_var = match.group(1)
            # Handle the ${VAR:-default} pattern
            if ':-' in env_var:
                env_name, default = env_var.split(':-', 1)
                return os.environ.get(env_name, default)
            # Regular ${VAR} pattern
            return os.environ.get(env_var, '')
        
        return self.env_pattern.sub(_replace_env_var, config_str)
        
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a YAML configuration file with environment variable substitution.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Dictionary with configuration values
            
        Raises:
            FileNotFoundError: If the configuration file is not found
            yaml.YAMLError: If the YAML file is invalid
        """
        try:
            with open(file_path, 'r') as f:
                # Read file content and substitute environment variables
                file_content = f.read()
                file_content = self._substitute_env_vars(file_content)
                
                # Parse YAML content
                config = yaml.safe_load(file_content)
                return config or {}
                
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {file_path}")
            raise
            
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            # If both are dictionaries, merge recursively
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                # Otherwise override the value
                result[key] = value
                
        return result
    
    def load_config(self, environment: str = "dev") -> BookmarkConfig:
        """
        Load configuration for the specified environment.
        
        Args:
            environment: Environment name ("dev", "test", "prod")
            
        Returns:
            Complete configuration object
            
        Raises:
            FileNotFoundError: If required configuration files are not found
        """
        logger.info(f"Loading configuration for environment: {environment}")
        
        # Load base configuration
        base_config_path = self.config_dir / "base_config.yaml"
        base_config = self._load_yaml_file(base_config_path)
        logger.debug(f"Loaded base configuration from {base_config_path}")
        
        # Load environment-specific configuration
        env_config_path = self.config_dir / environment / "config.yaml"
        try:
            env_config = self._load_yaml_file(env_config_path)
            logger.debug(f"Loaded {environment} configuration from {env_config_path}")
            
            # Merge configurations
            merged_config = self._deep_merge(base_config, env_config)
            
        except FileNotFoundError:
            # If environment config is not found, use base config
            logger.warning(f"Environment configuration not found: {env_config_path}")
            merged_config = base_config
            
        # Validate and create configuration object
        try:
            config = BookmarkConfig(**merged_config)
            return config
            
        except Exception as e:
            logger.error(f"Invalid configuration: {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")
            
    def save_config(self, config: BookmarkConfig, output_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration object
            output_path: Path to save the configuration
            
        Raises:
            IOError: If the configuration cannot be saved
        """
        try:
            # Convert config to dictionary
            config_dict = config.dict()
            
            # Save to YAML file
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise 