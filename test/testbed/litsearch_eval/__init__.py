#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LitSearch 搜索策略评估包

提供完整的评估框架，用于测试不同搜索策略在 LitSearch 数据集上的性能。
直接使用AIgnite项目的搜索模块，确保代码一致性和可维护性。
"""

# 导入AIgnite的搜索策略
from AIgnite.index.search_strategy import SearchStrategy, SearchResult, VectorSearchStrategy, TFIDFSearchStrategy, HybridSearchStrategy
from AIgnite.db.vector_db import VectorDB
from AIgnite.db.metadata_db import MetadataDB

# 导入评估框架组件
from .evaluator import SearchEvaluator
from .visualization import ResultVisualizer
from .litsearch_evaluator import LitSearchEvaluator 