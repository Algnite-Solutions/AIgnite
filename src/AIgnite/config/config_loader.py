import os
import yaml
from typing import Dict, Any
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced configuration loading function with environment variable support.
    
    Args:
        config_path: Path to config.yaml file. If None, uses environment variable or default path.
        set_env: Whether to set configuration values as environment variables.
        
    Returns:
        Dictionary containing configuration parameters
        
    Raises:
        FileNotFoundError: If given config path but config file not found
        ValueError: If required configuration sections are missing or if loading fails
    """
    # Get config path from parameter, environment variable, or default
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            config = full_config["INDEX_SERVICE"]
        
        # Validate required sections for index service
        required_sections = ['vector_db', 'metadata_db', 'minio_db']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in config.yaml")
        
        # Validate database configurations
        if 'model_name' not in config['vector_db']:
            raise ValueError("Missing 'model_name' in vector_db configuration")
            
        if 'db_url' not in config['metadata_db']:
            raise ValueError("Missing 'db_url' in metadata_db configuration")
        
        # Validate MinIO configuration
        minio_required = ['endpoint', 'access_key', 'secret_key', 'bucket_name']
        for param in minio_required:
            if param not in config['minio_db']:
                raise ValueError(f"Missing required MinIO parameter '{param}' in config.yaml")
        
        # Set environment variables if requested
        '''
        if set_env:
            logger.debug('Setting environment variables from loaded configuration')
            set_config_environment_variables(config)
        '''
        logger.info(f"Successfully loaded configuration from file: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise ValueError(f"Error loading config: {str(e)}")

class ConfigLoader:
    @staticmethod
    def load_config(config_path: str = None) -> Dict[str, Any]:
        """Load configuration from yaml file.
        
        Args:
            config_path: Path to config.yaml file. If None, will look in default locations.
            
        Returns:
            Dictionary containing configuration parameters
        """
        if config_path is None:
            # Look for config in default locations
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'config.yaml'),  # Same directory as this file
                os.path.join(os.getcwd(), 'config.yaml'),  # Current working directory
                os.path.join(os.getcwd(), 'config', 'config.yaml'),  # config subdirectory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError("Could not find config.yaml in default locations")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['vector_db', 'metadata_db', 'minio_db']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config.yaml")
            
            # Validate MinIO configuration
            minio_required = ['endpoint', 'access_key', 'secret_key', 'bucket_name']
            for param in minio_required:
                if param not in config['minio_db']:
                    raise ValueError(f"Missing required MinIO parameter '{param}' in config.yaml")
            
            logger.debug(f"Successfully loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise 