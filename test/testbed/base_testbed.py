#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
抽象测试床基类

定义测试生命周期的模板方法，提供统一的测试环境管理、数据库初始化和清理机制。
专门为真实数据库测试场景设计。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import logging
import tempfile
import shutil
from pathlib import Path
import sys
import os

# 添加PaperIgnition路径
from test.index.config.config_loader import load_config


class TestBed(ABC):
    """抽象测试床基类，专门用于真实数据库测试
    
    提供完整的测试生命周期管理：
    1. 环境检查
    2. 数据加载
    3. 数据库初始化
    4. 测试执行
    5. 环境清理
    """
    
    def __init__(self, config_path: str):
        """初始化测试床
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config()
        self.temp_dir = None
        self.vector_db = None
        self.metadata_db = None
        self.engine = None
        self.data = None
        self.results = {}
        
        # 设置日志
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """使用PaperIgnition的配置加载机制
        
        Returns:
            配置字典
        """
        try:
            return load_config(self.config_path)
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {str(e)}")
            raise
    
    # 生命周期钩子方法
    @abstractmethod
    def check_environment(self) -> Tuple[bool, str]:
        """检查测试环境是否满足要求
        
        Returns:
            Tuple[bool, str]: (是否就绪, 错误信息)
        """
        pass
    
    @abstractmethod
    def load_data(self) -> Any:
        """加载测试数据
        
        Returns:
            测试数据
        """
        pass
    
    @abstractmethod
    def initialize_databases(self, data: Any) -> None:
        """初始化真实数据库
        
        Args:
            data: 要索引的数据
        """
        pass
    
    @abstractmethod
    def run_tests(self) -> Dict[str, Any]:
        """运行具体测试
        
        Returns:
            测试结果字典
        """
        pass
    
    # 模板方法
    def setup(self) -> None:
        """设置测试环境"""
        self.logger.info(f"Setting up {self.__class__.__name__} test environment...")
        
        # 检查环境
        is_ready, error_msg = self.check_environment()
        if not is_ready:
            raise RuntimeError(f"Environment check failed: {error_msg}")
        
        self.logger.info("Environment check passed")
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # 加载数据
        self.logger.info("Loading test data...")
        self.data = self.load_data()
        self.logger.info(f"Loaded {len(self.data) if hasattr(self.data, '__len__') else 'test'} data items")
        
        # 初始化数据库
        self.logger.info("Initializing databases...")
        self.initialize_databases(self.data)
        self.logger.info("Database initialization completed")
    
    def teardown(self) -> None:
        """清理测试环境"""
        self.logger.info("Tearing down test environment...")
        
        # 清理数据库
        self._cleanup_databases()
        
        # 清理临时目录
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        
        self.logger.info("Test environment cleanup completed")
    
    def execute(self) -> Dict[str, Any]:
        """执行完整测试流程
        
        Returns:
            测试结果字典
        """
        self.logger.info(f"Starting {self.__class__.__name__} test execution...")
        
        try:
            self.setup()
            self.results = self.run_tests()
            self.logger.info("Test execution completed successfully")
            return self.results
        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}")
            raise
        finally:
            self.teardown()
    
    def _cleanup_databases(self) -> None:
        """清理数据库"""
        self.logger.info("Cleaning up databases...")
        
        # 清理向量数据库文件
        if self.vector_db and hasattr(self.vector_db, 'db_path'):
            try:
                vector_db_path = Path(self.vector_db.db_path)
                if vector_db_path.exists():
                    shutil.rmtree(vector_db_path.parent, ignore_errors=True)
                    self.logger.info(f"Cleaned up vector database at {vector_db_path.parent}")
            except Exception as e:
                self.logger.warning(f"Failed to clean up vector database: {str(e)}")
        
        # 清理元数据数据库表
        if self.engine:
            try:
                from AIgnite.db.metadata_db import Base
                Base.metadata.drop_all(self.engine)
                self.engine.dispose()
                self.logger.info("Cleaned up metadata database tables")
            except Exception as e:
                self.logger.warning(f"Failed to clean up metadata database: {str(e)}")
    

    
    def log_test_result(self, test_name: str, success: bool, details: str = "") -> None:
        """记录测试结果
        
        Args:
            test_name: 测试名称
            success: 是否成功
            details: 详细信息
        """
        status = "✅ PASSED" if success else "❌ FAILED"
        self.logger.info(f"{test_name}: {status}")
        if details:
            self.logger.info(f"  Details: {details}")
