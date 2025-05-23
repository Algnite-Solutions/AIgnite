from typing import Tuple, Optional
from .vector_db import VectorDB
from .metadata_db import MetadataDB
from .image_db import MinioImageDB
from ..config.config_loader import ConfigLoader
import logging

# Set up logging
logger = logging.getLogger(__name__)

def init_databases(config_path: Optional[str] = None) -> Tuple[VectorDB, MetadataDB, MinioImageDB]:
    """Initialize all required databases using configuration from yaml file.
    
    Args:
        config_path: Optional path to config.yaml file. If None, will look in default locations.
        
    Returns:
        Tuple of (VectorDB, MetadataDB, MinioImageDB) instances
    """
    logger.debug("Loading configuration and initializing databases...")
    
    try:
        # Load configuration
        config = ConfigLoader.load_config(config_path)
        
        # Initialize vector database
        vector_db = VectorDB(
            model_name=config['vector_db']['model_name']
        )
        logger.debug(f"Vector database initialized with model {config['vector_db']['model_name']}")
        
        # Initialize metadata database
        metadata_db = MetadataDB(
            db_path=config['metadata_db']['db_url']
        )
        logger.debug("Metadata database initialized")
        
        # Initialize MinIO image database
        image_db = MinioImageDB(
            endpoint=config['minio_db']['endpoint'],
            access_key=config['minio_db']['access_key'],
            secret_key=config['minio_db']['secret_key'],
            bucket_name=config['minio_db']['bucket_name'],
            secure=config['minio_db'].get('secure', False)
        )
        logger.debug(f"MinIO image database initialized with endpoint {config['minio_db']['endpoint']}")
        
        return vector_db, metadata_db, image_db
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {str(e)}")
        raise RuntimeError(f"Database initialization failed: {str(e)}") 