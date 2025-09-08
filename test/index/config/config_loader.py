import os
import yaml
from typing import Dict, Any, Optional
import logging

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
        
        logger.info(f"Successfully loaded configuration from file: {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        raise ValueError(f"Error loading config: {str(e)}")


class ConfigLoader:
    """可扩展的配置加载器基类
    
    支持不同服务类型的配置加载，子类可以重写 _extract_service_config 方法
    来支持不同的配置结构。
    """
    
    def __init__(self, service_name: str = "INDEX_SERVICE"):
        """初始化配置加载器
        
        Args:
            service_name: 服务名称，用于从配置文件中提取对应的配置节
        """
        self.service_name = service_name
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径。如果为 None，将在默认位置查找
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 如果找不到配置文件
            ValueError: 如果配置验证失败
        """
        # 查找配置文件路径
        if config_path is None:
            config_path = self._find_config_file()
        
        try:
            # 加载配置文件
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # 提取服务配置
            config = self._extract_service_config(full_config)
            
            # 验证配置
            self._validate_config(config)
            
            logger.info(f"Successfully loaded {self.service_name} configuration from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ValueError(f"Error loading config: {str(e)}")
    
    def _extract_service_config(self, full_config: Dict[str, Any]) -> Dict[str, Any]:
        """提取指定服务的配置
        
        Args:
            full_config: 完整的配置文件内容
            
        Returns:
            提取的服务配置
            
        Raises:
            ValueError: 如果找不到指定的服务配置
        """
        if self.service_name not in full_config:
            raise ValueError(f"Missing required section '{self.service_name}' in config.yaml")
        
        return full_config[self.service_name]
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """验证配置
        
        Args:
            config: 要验证的配置
            
        Raises:
            ValueError: 如果配置验证失败
        """
        # 验证必需的服务配置节
        required_sections = ['vector_db', 'metadata_db', 'minio_db']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section '{section}' in {self.service_name} configuration")
        
        # 验证数据库配置
        if 'model_name' not in config['vector_db']:
            raise ValueError("Missing 'model_name' in vector_db configuration")
            
        if 'db_url' not in config['metadata_db']:
            raise ValueError("Missing 'db_url' in metadata_db configuration")
        
        # 验证 MinIO 配置
        minio_required = ['endpoint', 'access_key', 'secret_key', 'bucket_name']
        for param in minio_required:
            if param not in config['minio_db']:
                raise ValueError(f"Missing required MinIO parameter '{param}' in {self.service_name} configuration")
    
    def _find_config_file(self) -> str:
        """在默认位置查找配置文件
        
        Returns:
            找到的配置文件路径
            
        Raises:
            FileNotFoundError: 如果找不到配置文件
        """
        possible_paths = [
            os.path.join(os.path.dirname(__file__), 'config.yaml'),  # 与当前文件同目录
            os.path.join(os.getcwd(), 'config.yaml'),  # 当前工作目录
            os.path.join(os.getcwd(), 'config', 'config.yaml'),  # config 子目录
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find config.yaml in default locations") 