"""
TestBed - 统一测试框架

提供抽象的测试床基类，用于管理测试环境、数据库初始化和测试执行。
支持真实数据库的测试场景，专门为PaperIndexer等组件设计。
"""

from .base_testbed import TestBed

__all__ = [
    'TestBed'
]

__version__ = '1.0.0'
