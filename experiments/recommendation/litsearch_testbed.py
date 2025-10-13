#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LitSearchTestBed专用测试床

继承自TestBed，专门用于LitSearch数据集上的搜索策略测试。
使用真实数据库（VectorDB, MetadataDB）进行完整的集成测试。
"""

from experiments.recommendation.testbed.base_testbed import TestBed
from experiments.recommendation.testbed.litsearch_eval.evaluator import SearchEvaluator
from AIgnite.data.docset import DocSet, TextChunk, ChunkType
from AIgnite.db.metadata_db import MetadataDB, Base
from AIgnite.db.vector_db import VectorDB
from AIgnite.index.paper_indexer import PaperIndexer
from AIgnite.index.search_strategy import VectorSearchStrategy, TFIDFSearchStrategy, HybridSearchStrategy

#from eval.litsearch_eval_new.evaluator import SearchEvaluator
from sqlalchemy import create_engine, text, inspect
from typing import Dict, Any, Tuple, List, Optional
import os
import shutil
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import json
import logging
import sys

# 添加路径以便导入评估模块
#sys.path.append('/data3/guofang/AIgnite-Solutions/AIgnite/eval/litsearch_eval_new')

logger = logging.getLogger(__name__)

class LitSearchTestBed(TestBed):
    """LitSearch专用测试床
    
    提供完整的LitSearch数据集搜索策略测试，包括：
    - 向量搜索测试
    - TF-IDF搜索测试
    - 混合搜索测试
    - 搜索策略性能比较
    - 评估指标计算
    """
    
    def __init__(self, config_path: str):
        """初始化LitSearch测试床
        
        Args:
            config_path: 配置文件路径
        """
        super().__init__(config_path)
        
        # 加载完整配置文件以获取LITSEARCH_TESTBED配置
        self.full_config = self._load_full_config(config_path)
        self.litsearch_config = self.full_config.get('LITSEARCH_TESTBED', {})
        
        self.vector_db = None
        self.metadata_db = None
        self.engine = None
        self.corpus_data = None
        self.ground_truth = None
        self.search_strategies = {}
        self.evaluation_results = {}
        
        
        # 验证LitSearch特定配置
        self._validate_litsearch_config()
    
    def _load_full_config(self, config_path: str) -> Dict[str, Any]:
        """加载完整配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            完整的配置字典
        """
        import yaml
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            return full_config
        except Exception as e:
            self.logger.error(f"Failed to load full config from {config_path}: {str(e)}")
            raise
    
    def _validate_litsearch_config(self) -> None:
        """验证LitSearch特定配置"""
        required_sections = ['dataset', 'search_strategies', 'evaluation']
        for section in required_sections:
            if section not in self.litsearch_config:
                raise ValueError(f"Missing required section '{section}' in LITSEARCH_TESTBED configuration")
        
        # 验证数据集配置
        dataset_config = self.litsearch_config.get('dataset', {})
        if 'name' not in dataset_config:
            raise ValueError("Missing 'name' in dataset configuration")
        
        # 验证搜索策略配置
        strategies_config = self.litsearch_config.get('search_strategies', {})
        if not any(strategies_config.get(strategy, {}).get('enabled', False) 
                   for strategy in ['vector', 'tfidf', 'hybrid']):
            raise ValueError("At least one search strategy must be enabled")
        
    
    
    def check_environment(self) -> Tuple[bool, str]:
        """检查测试环境
        
        Returns:
            Tuple[bool, str]: (是否就绪, 错误信息)
        """
        try:
            # 检查配置文件
            if not os.path.exists(self.config_path):
                return False, f"Config file not found: {self.config_path}"
            
            # 检查数据库连接
            db_url = self.config['metadata_db']['db_url']
            if not db_url:
                return False, "Database URL not found in config"
            
            # 尝试连接数据库，如果不存在则创建
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                engine.dispose()
            except Exception as db_error:
                # 如果数据库不存在，尝试创建
                if "does not exist" in str(db_error):
                    self.logger.info("Database does not exist, attempting to create...")
                    try:
                        # 解析数据库URL以获取数据库名称
                        from urllib.parse import urlparse
                        parsed_url = urlparse(db_url)
                        db_name = parsed_url.path[1:]  # 移除开头的 '/'
                        
                        # 连接到默认数据库来创建新数据库
                        default_url = db_url.replace(f'/{db_name}', '/postgres')
                        # 添加isolation_level参数来避免事务块
                        default_engine = create_engine(default_url, isolation_level='AUTOCOMMIT')
                        
                        with default_engine.connect() as conn:
                            # 检查数据库是否存在
                            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'"))
                            if not result.fetchone():
                                # 创建数据库
                                conn.execute(text(f"CREATE DATABASE {db_name}"))
                                self.logger.info(f"Created database: {db_name}")
                            else:
                                self.logger.info(f"Database {db_name} already exists")
                        
                        default_engine.dispose()
                        
                        # 重新测试连接
                        engine = create_engine(db_url)
                        with engine.connect() as conn:
                            conn.execute(text("SELECT 1"))
                        engine.dispose()
                        
                    except Exception as create_error:
                        return False, f"Failed to create database: {str(create_error)}"
                else:
                    return False, f"Database connection failed: {str(db_error)}"
            
            # 检查向量数据库路径权限
            vector_db_path = self.config['vector_db']['db_path']
            if not vector_db_path:
                return False, "Vector database path not found in config"
            
            vector_db_dir = Path(vector_db_path).parent
            vector_db_dir.mkdir(parents=True, exist_ok=True)
            
            # 测试写权限
            test_file = vector_db_dir / "test_write_permission"
            test_file.write_text("test")
            test_file.unlink()
            
            return True, "Environment check passed"
            
        except Exception as e:
            return False, f"Environment check failed: {str(e)}"
    
    def load_data(self) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """加载LitSearch数据集
        
        Returns:
            Tuple[List[Dict], Dict[str, List[str]]]: (语料库数据, 查询真值)
        """
        self.logger.info("Loading LitSearch dataset...")
        
        dataset_config = self.litsearch_config.get('dataset', {})
        dataset_name = dataset_config.get('name', 'princeton-nlp/LitSearch')
        corpus_config = dataset_config.get('corpus_config', 'corpus_clean')
        query_config = dataset_config.get('query_config', 'query')
        sample_size = dataset_config.get('sample_size')
        enable_sampling = dataset_config.get('enable_sampling', True)
        
        try:
            # 加载查询数据集
            query_dataset = load_dataset(dataset_name, query_config, split="full")
            # 加载语料库数据集
            corpus_dataset = load_dataset(dataset_name, corpus_config, split="full")
            
            # 将数据集转换为字典列表
            corpus = [doc for doc in corpus_dataset]
            queries = query_dataset
            
            self.logger.info(f"Successfully loaded dataset: {len(corpus)} documents, {len(queries)} queries")
            
            # 提取查询和真值
            ground_truth = {}
            for query in queries:
                query_text = query["query"]
                relevant_docs = [str(doc_id) for doc_id in query["corpusids"]]
                ground_truth[query_text] = relevant_docs
            
            self.logger.info(f"Extracted {len(ground_truth)} queries with ground truth")
            
            # 数据采样（如果需要）
            if enable_sampling and sample_size and sample_size < len(corpus):
                corpus, ground_truth = self._sample_data(corpus, ground_truth, sample_size)
                self.logger.info(f"Sampled to {len(corpus)} documents and {len(ground_truth)} queries")
            
            return corpus, ground_truth
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def _sample_data(self, corpus: List[Dict], ground_truth: Dict[str, List[str]], sample_size: int) -> Tuple[List[Dict], Dict[str, List[str]]]:
        """采样数据
        
        Args:
            corpus: 语料库
            ground_truth: 查询真值
            sample_size: 采样大小
            
        Returns:
            Tuple[List[Dict], Dict[str, List[str]]]: 采样后的语料库和查询真值
        """
        self.logger.info(f"Sampling {sample_size} documents and related queries...")
        
        try:
            # 1. 首先随机选择一些查询
            query_texts = list(ground_truth.keys())
            query_count = min(sample_size // 10, len(query_texts))  # 选择查询数量
            query_indices = np.random.choice(len(query_texts), query_count, replace=False)
            sampled_queries = [query_texts[i] for i in query_indices]
            
            # 2. 获取这些查询相关的所有文档ID
            relevant_doc_ids = set()
            for query in sampled_queries:
                relevant_doc_ids.update(ground_truth[query])
            
            self.logger.info(f"Selected {len(sampled_queries)} queries, involving {len(relevant_doc_ids)} relevant documents")
            
            # 3. 如果相关文档数量不足，则随机添加一些文档
            corpus_ids = [str(doc["corpusid"]) for doc in corpus]
            corpus_id_to_index = {doc_id: i for i, doc_id in enumerate(corpus_ids)}
            
            # 找出已有的相关文档在语料库中的索引
            relevant_indices = [corpus_id_to_index[doc_id] for doc_id in relevant_doc_ids if doc_id in corpus_id_to_index]
            
            # 如果相关文档数量不足采样大小，则随机添加一些文档
            additional_count = max(0, sample_size - len(relevant_indices))
            if additional_count > 0:
                # 获取非相关文档的索引
                non_relevant_indices = [i for i in range(len(corpus)) if i not in relevant_indices]
                if non_relevant_indices:
                    # 随机选择一些非相关文档
                    additional_indices = np.random.choice(
                        non_relevant_indices, 
                        min(additional_count, len(non_relevant_indices)), 
                        replace=False
                    )
                    all_indices = list(relevant_indices) + list(additional_indices)
                else:
                    all_indices = relevant_indices
            else:
                # 如果相关文档数量已经超过采样大小，则随机选择一部分
                all_indices = np.random.choice(relevant_indices, sample_size, replace=False)
            
            # 4. 采样语料库
            corpus_sample = [corpus[i] for i in all_indices]
            
            # 5. 更新 ground_truth，只保留采样文档中的相关文档
            sampled_corpus_ids = set(str(doc["corpusid"]) for doc in corpus_sample)
            sampled_ground_truth = {}
            for query in sampled_queries:
                relevant_docs = [doc_id for doc_id in ground_truth[query] if doc_id in sampled_corpus_ids]
                if relevant_docs:  # 只保留有相关文档的查询
                    sampled_ground_truth[query] = relevant_docs
            
            # 如果没有查询有相关文档，记录警告
            if not sampled_ground_truth:
                self.logger.warning("After sampling, no queries have relevant documents! Using original queries and ground truth.")
                return corpus_sample, ground_truth
            
            self.logger.info(f"After sampling: {len(corpus_sample)} documents, {len(sampled_ground_truth)} queries")
            self.logger.info(f"Average relevant documents per query: {sum(len(docs) for docs in sampled_ground_truth.values()) / len(sampled_ground_truth):.2f}")
            
            return corpus_sample, sampled_ground_truth
        except Exception as e:
            self.logger.error(f"Data sampling failed: {str(e)}")
            return corpus, ground_truth
    
    def setup(self, clean_before_test: bool = True) -> None:
        """设置测试环境
        
        Args:
            clean_before_test: 是否在测试前清理环境，默认为True
        """
        self.logger.info(f"Setting up {self.__class__.__name__} test environment...")
        
        # 检查环境
        is_ready, error_msg = self.check_environment()
        if not is_ready:
            raise RuntimeError(f"Environment check failed: {error_msg}")
        
        self.logger.info("Environment check passed")
        
        # 测试前清理（如果需要）
        if clean_before_test:
            self._cleanup_before_test()
        
        # 创建临时目录
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # 加载数据
        self.logger.info("Loading test data...")
        self.data = self.load_data()
        self.logger.info(f"Loaded {len(self.data) if hasattr(self.data, '__len__') else 'test'} data items")
        
        # 从配置中读取load_previous_db参数
        load_previous_db = self.litsearch_config.get('load_previous_db', False)
        self.logger.info(f"Load previous database setting: {load_previous_db}")
        
        # 初始化数据库
        self.logger.info("Initializing databases...")
        self.initialize_databases(self.data, load_previous_db=load_previous_db)
        self.logger.info("Database initialization completed")
    
    def initialize_databases(self, data: Tuple[List[Dict], Dict[str, List[str]]], load_previous_db: bool = False) -> None:
        """初始化真实数据库
        
        Args:
            data: (语料库数据, 查询真值) 元组
            load_previous_db: 是否加载已存在的数据库，True=跳过数据索引，False=重新索引数据
        """
        self.logger.info("Initializing real databases...")
        
        corpus_data, ground_truth = data
        self.corpus_data = corpus_data
        self.ground_truth = ground_truth
        
        
        # 初始化向量数据库
        vector_db_path = self.config['vector_db']['db_path']
        model_name = self.config['vector_db'].get('model_name', 'BAAI/bge-base-en-v1.5')
        print('MODEL NAME: ', model_name)
        vector_dim = self.config['vector_db'].get('vector_dim', 768)
        print('VECTOR DIM: ', vector_dim)
        self.vector_db = VectorDB(
            db_path=vector_db_path,
            model_name=model_name,
            vector_dim=vector_dim
        )
        self.logger.info(f"Vector database initialized: {vector_db_path}")
        
        # 初始化元数据数据库
        db_url = self.config['metadata_db']['db_url']
        self.engine = create_engine(db_url)
        
        '''
        # 重新创建表
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.logger.info("Metadata database tables recreated")
        '''
        
        self.metadata_db = MetadataDB(db_path=db_url)
        self.logger.info("Metadata database initialized")
        

        # 显示存储信息
        self._display_storage_info()

        # 初始化PaperIndexer
        self.paper_indexer = PaperIndexer(self.vector_db, self.metadata_db, None)
        self.logger.info("PaperIndexer initialized with real databases")

        # 根据load_previous_db参数决定是否索引语料库数据
        if load_previous_db:
            self.logger.info("Loading previous database - skipping corpus data indexing")
        else:
            self.logger.info("Indexing corpus data...")
            self._index_corpus_data(corpus_data)
        
        # 初始化搜索策略
        self._initialize_search_strategies()
        
        self.logger.info("Database initialization completed")
        
        # 显示存储信息
        self._display_storage_info()
    
    def _index_corpus_data(self, corpus_data: List[Dict]) -> None:
        """索引语料库数据到数据库
        
        Args:
            corpus_data: 语料库数据列表
        """
        self.logger.info(f"Converting {len(corpus_data)} documents to DocSet format...")
        
        # 步骤1: 转换数据格式 Dict -> DocSet
        docsets = []
        for doc in tqdm(corpus_data, desc="Converting to DocSet format"):
            try:
                docset = DocSet(
                    doc_id=str(doc["corpusid"]),
                    title=doc.get("title", ""),
                    abstract=doc.get("abstract", ""),
                    authors=[],  # LitSearch 数据集没有作者信息
                    categories=[],  # LitSearch 数据集没有分类信息
                    published_date="",  # LitSearch 数据集没有发布日期
                    text_chunks=[],  # 空列表
                    figure_chunks=[],  # 空列表
                    table_chunks=[],  # 空列表
                    metadata={},  # 空字典
                    pdf_path="",  # 空字符串
                    HTML_path=None,
                    comments=None
                )
                docsets.append(docset)
            except Exception as e:
                self.logger.error(f"Failed to convert document {doc.get('corpusid', 'unknown')}: {str(e)}")
        
        self.logger.info(f"Successfully converted {len(docsets)} documents to DocSet format")
        
        # 步骤2: 使用 PaperIndexer 索引文档
        self.logger.info(f"Indexing {len(docsets)} documents using PaperIndexer...")
        indexing_status = self.paper_indexer.index_papers(
            papers=docsets,
            store_images=False,  # LitSearch 数据集不包含图片
            keep_temp_image=False
        )
        
        # 步骤3: 统计索引结果
        total_docs = len(indexing_status)
        metadata_success = sum(1 for status in indexing_status.values() if status.get("metadata", False))
        vector_success = sum(1 for status in indexing_status.values() if status.get("vectors", False))
        
        self.logger.info(f"Indexing completed:")
        self.logger.info(f"  Total documents: {total_docs}")
        self.logger.info(f"  Metadata indexed: {metadata_success}/{total_docs}")
        self.logger.info(f"  Vectors indexed: {vector_success}/{total_docs}")
        
        # 记录失败的文档
        failed_docs = [doc_id for doc_id, status in indexing_status.items() 
                       if not status.get("metadata", False) or not status.get("vectors", False)]
        if failed_docs:
            self.logger.warning(f"Failed to fully index {len(failed_docs)} documents: {failed_docs[:10]}...")
    
    def _initialize_search_strategies(self) -> None:
        """根据已初始化的数据库初始化搜索策略"""
        self.logger.info("Initializing search strategies...")
        
        strategies_config = self.litsearch_config.get('search_strategies', {})
        
        # 初始化向量搜索策略（如果向量数据库已初始化且启用）
        if self.vector_db is not None and strategies_config.get('vector', {}).get('enabled', False):
            vector_config = strategies_config.get('vector', {})
            # 注意：这里需要根据实际的VectorSearchStrategy构造函数调整
            # 暂时使用简化版本
            self.search_strategies['vector'] = {
                'strategy': None,  # 将在实际搜索时创建
                'config': vector_config
            }
            self.logger.info("Vector search strategy configured")
        
        # 初始化TF-IDF搜索策略（如果元数据数据库已初始化且启用）
        if self.metadata_db is not None and strategies_config.get('tfidf', {}).get('enabled', False):
            tfidf_config = strategies_config.get('tfidf', {})
            self.search_strategies['tfidf'] = {
                'strategy': None,  # 将在实际搜索时创建
                'config': tfidf_config
            }
            self.logger.info("TF-IDF search strategy configured")
        
        # 初始化混合搜索策略（如果两个数据库都初始化且启用）
        if (self.vector_db is not None and self.metadata_db is not None and 
            strategies_config.get('hybrid', {}).get('enabled', False)):
            hybrid_config = strategies_config.get('hybrid', {})
            self.search_strategies['hybrid'] = {
                'strategy': None,  # 将在实际搜索时创建
                'config': hybrid_config
            }
            self.logger.info("Hybrid search strategy configured")
        
        self.logger.info(f"Initialized {len(self.search_strategies)} search strategies")
    
    '''
    def execute(self) -> Dict[str, Any]:
        """执行完整的测试流程（集成清理控制）
        
        Returns:
            测试结果字典
        """
        # 使用基类的 execute 方法，它会自动调用 setup() -> run_tests() -> teardown()
        return super().execute()
    '''
    
    def run_tests(self) -> Dict[str, Any]:
        """运行LitSearch测试
        
        Returns:
            测试结果字典
        """
        self.logger.info("Running LitSearch tests with PaperIndexer...")
        
        results = {}
        '''
        # 测试向量搜索策略
        self.logger.info("Testing vector search strategy...")
        vector_result = self._test_paper_indexer_vector_search()
        results['vector_search'] = vector_result
        
        # 测试TF-IDF搜索策略
        self.logger.info("Testing TF-IDF search strategy...")
        tfidf_result = self._test_paper_indexer_tfidf_search()
        results['tfidf_search'] = tfidf_result
        
        
        # 测试混合搜索策略
        self.logger.info("Testing hybrid search strategy...")
        hybrid_result = self._test_paper_indexer_hybrid_search()
        results['hybrid_search'] = hybrid_result
        '''
        # 运行完整评估
        self.logger.info("Running comprehensive evaluation...")
        evaluation_result = self._run_comprehensive_evaluation()
        results['comprehensive_evaluation'] = evaluation_result
        
        # 统计测试结果
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('success', False))
        
        self.logger.info(f"Test Results: {passed_tests}/{total_tests} tests passed")
        
        return results
    
    def _test_paper_indexer_vector_search(self) -> Dict[str, Any]:
        """使用PaperIndexer测试向量搜索策略
        
        Returns:
            测试结果字典
        """
        try:
            # 选择一些测试查询
            test_queries = list(self.ground_truth.keys())[:10]  # 使用前10个查询进行测试
            
            # 执行搜索
            search_results = {}
            for query in test_queries:
                results = self.paper_indexer.find_similar_papers(
                    query=query,
                    top_k=5,
                    search_strategies=[('vector', 0.8)],
                    result_include_types=['metadata', 'search_parameters']
                )
                search_results[query] = [result.get('doc_id') for result in results if result.get('doc_id')]
            
            # 计算基本指标
            total_queries = len(test_queries)
            successful_queries = len([q for q in test_queries if search_results.get(q)])
            
            success = successful_queries > 0
            details = f"Vector search: {successful_queries}/{total_queries} queries returned results"
            
            self.log_test_result("Vector Search", success, details)
            return {
                'success': success, 
                'successful_queries': successful_queries,
                'total_queries': total_queries,
                'details': details,
                'search_results': search_results
            }
            
        except Exception as e:
            self.log_test_result("Vector Search", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_paper_indexer_tfidf_search(self) -> Dict[str, Any]:
        """使用PaperIndexer测试TF-IDF搜索策略
        
        Returns:
            测试结果字典
        """
        try:
            # 选择一些测试查询
            test_queries = list(self.ground_truth.keys())[:10]  # 使用前10个查询进行测试
            
            # 执行搜索
            search_results = {}
            for query in test_queries:
                results = self.paper_indexer.find_similar_papers(
                    query=query,
                    top_k=5,
                    search_strategies=[('tf-idf', 0.5)],
                    result_include_types=['metadata', 'search_parameters']
                )
                search_results[query] = [result.get('doc_id') for result in results if result.get('doc_id')]
            
            # 计算基本指标
            total_queries = len(test_queries)
            successful_queries = len([q for q in test_queries if search_results.get(q)])
            
            success = successful_queries > 0
            details = f"TF-IDF search: {successful_queries}/{total_queries} queries returned results"
            
            self.log_test_result("TF-IDF Search", success, details)
            return {
                'success': success, 
                'successful_queries': successful_queries,
                'total_queries': total_queries,
                'details': details,
                'search_results': search_results
            }
            
        except Exception as e:
            self.log_test_result("TF-IDF Search", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_paper_indexer_hybrid_search(self) -> Dict[str, Any]:
        """使用PaperIndexer测试混合搜索策略
        
        Returns:
            测试结果字典
        """
        try:
            # 选择一些测试查询
            test_queries = list(self.ground_truth.keys())[:10]  # 使用前10个查询进行测试
            
            # 执行搜索
            search_results = {}
            for query in test_queries:
                results = self.paper_indexer.find_similar_papers(
                    query=query,
                    top_k=5,
                    search_strategies=[('vector', 0.8), ('tf-idf', 0.5)],
                    result_include_types=['metadata', 'search_parameters']
                )
                search_results[query] = [result.get('doc_id') for result in results if result.get('doc_id')]
            
            # 计算基本指标
            total_queries = len(test_queries)
            successful_queries = len([q for q in test_queries if search_results.get(q)])
            
            success = successful_queries > 0
            details = f"Hybrid search: {successful_queries}/{total_queries} queries returned results"
            
            self.log_test_result("Hybrid Search", success, details)
            return {
                'success': success, 
                'successful_queries': successful_queries,
                'total_queries': total_queries,
                'details': details,
                'search_results': search_results
            }
            
        except Exception as e:
            self.log_test_result("Hybrid Search", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _test_search_strategy(self, strategy_name: str, strategy_info: Dict) -> Dict[str, Any]:
        """测试单个搜索策略
        
        Args:
            strategy_name: 策略名称
            strategy_info: 策略配置信息
            
        Returns:
            测试结果字典
        """
        try:
            # 选择一些测试查询
            test_queries = list(self.ground_truth.keys())[:10]  # 使用前10个查询进行测试
            
            # 执行搜索
            search_results = {}
            for query in test_queries:
                if strategy_name == 'vector':
                    results = self.vector_db.search(query, top_k=5)
                    search_results[query] = [result[0].doc_id for result in results]
                elif strategy_name == 'tfidf':
                    results = self.metadata_db.search(query, top_k=5)
                    search_results[query] = [result['doc_id'] for result in results]
                elif strategy_name == 'hybrid':
                    # 混合搜索需要结合两种策略
                    vector_results = self.vector_db.search(query, top_k=10)
                    tfidf_results = self.metadata_db.search(query, top_k=10)
                    
                    # 简单的混合策略：合并结果并去重
                    vector_docs = [result[0].doc_id for result in vector_results]
                    tfidf_docs = [result['doc_id'] for result in tfidf_results]
                    
                    # 合并并去重，保持顺序
                    combined_docs = []
                    seen = set()
                    for doc_id in vector_docs + tfidf_docs:
                        if doc_id not in seen:
                            combined_docs.append(doc_id)
                            seen.add(doc_id)
                    
                    search_results[query] = combined_docs[:5]  # 取前5个
            
            # 计算基本指标
            total_queries = len(test_queries)
            successful_queries = len([q for q in test_queries if search_results.get(q)])
            
            success = successful_queries > 0
            details = f"Strategy {strategy_name}: {successful_queries}/{total_queries} queries returned results"
            
            self.log_test_result(f"{strategy_name.title()} Search", success, details)
            return {
                'success': success, 
                'successful_queries': successful_queries,
                'total_queries': total_queries,
                'details': details,
                'search_results': search_results
            }
            
        except Exception as e:
            self.log_test_result(f"{strategy_name.title()} Search", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """运行综合评估
        
        Returns:
            评估结果字典
        """
        try:
            # 使用SearchEvaluator进行综合评估
            evaluator = SearchEvaluator()
            
            # 准备评估数据
            retrieval_results = {}
            
            # 使用PaperIndexer为每个策略准备检索结果
            strategies = {
                'vector': [('vector', self.litsearch_config.get('search_strategies', {}).get('vector', {}).get('similarity_cutoff', 0.5))],
               #'tfidf': [('tf-idf', 0.5)],
                #'hybrid': [('vector', 0.8), ('tf-idf', 0.5)]
            }
            
            for strategy_name, search_strategies in strategies.items():
                strategy_results = {}
                
                for query in list(self.ground_truth.keys()):
                    try:
                        results = self.paper_indexer.find_similar_papers(
                            query=query,
                            top_k=10,
                            search_strategies=search_strategies,
                            result_include_types=['metadata', 'search_parameters']
                        )
                        strategy_results[query] = [result.get('doc_id') for result in results if result.get('doc_id')]
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to search query '{query}' with {strategy_name} strategy: {str(e)}")
                        strategy_results[query] = []
                
                retrieval_results[strategy_name] = strategy_results
            
            # 计算评估指标
            evaluation_results = SearchEvaluator.compare_strategies(retrieval_results, self.ground_truth)
            
            # 保存评估结果
            self.evaluation_results = evaluation_results
            
            # 生成评估报告
            self._generate_evaluation_report(evaluation_results)
            
            success = len(evaluation_results) > 0
            details = f"Comprehensive evaluation completed for {len(evaluation_results)} strategies"
            
            self.log_test_result("Comprehensive Evaluation", success, details)
            return {
                'success': success,
                'evaluation_results': evaluation_results,
                'details': details
            }
            
        except Exception as e:
            self.log_test_result("Comprehensive Evaluation", False, f"Error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _generate_evaluation_report(self, evaluation_results: Dict[str, Dict[str, float]]) -> None:
        """生成评估报告
        
        Args:
            evaluation_results: 评估结果字典
        """
        try:
            # 创建结果目录
            results_path = self.litsearch_config.get('test', {}).get('results_path', 'litsearch_test_results')
            os.makedirs(results_path, exist_ok=True)
            
            # 生成Markdown报告
            report_path = os.path.join(results_path, 'evaluation_report.md')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# LitSearch 搜索策略评估报告\n\n")
                f.write("## 概述\n\n")
                
                # 找出最佳策略
                if evaluation_results:
                    best_strategy = max(evaluation_results.items(), key=lambda x: x[1].get("map", 0))[0]
                    f.write(f"**最佳策略**: {best_strategy}\n\n")
                
                f.write("## 详细结果\n\n")
                f.write("| 策略 | MAP | Precision@1 | Precision@5 | Precision@10 | Recall@10 | 查询数量 |\n")
                f.write("| ---- | --- | ----------- | ----------- | ------------ | --------- | -------- |\n")
                
                for strategy_name, metrics in evaluation_results.items():
                    f.write(f"| {strategy_name} | {metrics.get('map', 0):.4f} | "
                           f"{metrics.get('precision@1', 0):.4f} | {metrics.get('precision@5', 0):.4f} | "
                           f"{metrics.get('precision@10', 0):.4f} | {metrics.get('recall@10', 0):.4f} | "
                           f"{metrics.get('query_count', 0)} |\n")
                
                f.write("\n## 策略比较\n\n")
                
                # MAP比较
                f.write("### MAP (平均精度均值)\n\n")
                map_sorted = sorted(evaluation_results.items(), key=lambda x: x[1].get("map", 0), reverse=True)
                for i, (strategy, metrics) in enumerate(map_sorted, 1):
                    f.write(f"{i}. **{strategy}**: {metrics.get('map', 0):.4f}\n")
                
                f.write("\n### Precision@10\n\n")
                p10_sorted = sorted(evaluation_results.items(), key=lambda x: x[1].get("precision@10", 0), reverse=True)
                for i, (strategy, metrics) in enumerate(p10_sorted, 1):
                    f.write(f"{i}. **{strategy}**: {metrics.get('precision@10', 0):.4f}\n")
                
                f.write("\n### Recall@10\n\n")
                r10_sorted = sorted(evaluation_results.items(), key=lambda x: x[1].get("recall@10", 0), reverse=True)
                for i, (strategy, metrics) in enumerate(r10_sorted, 1):
                    f.write(f"{i}. **{strategy}**: {metrics.get('recall@10', 0):.4f}\n")
                
                f.write("\n## 结论\n\n")
                if evaluation_results:
                    best_strategy = max(evaluation_results.items(), key=lambda x: x[1].get("map", 0))[0]
                    best_map = evaluation_results[best_strategy].get('map', 0)
                    f.write(f"基于评估结果，**{best_strategy}** 策略在整体性能上表现最佳，其 MAP 分数为 {best_map:.4f}。\n")
            
            # 保存JSON格式的详细结果
            json_path = os.path.join(results_path, 'evaluation_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Evaluation report saved to: {report_path}")
            self.logger.info(f"Detailed results saved to: {json_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate evaluation report: {str(e)}")
    
    def _format_bytes(self, bytes_size: int) -> str:
        """Format bytes into human readable format.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted string with appropriate unit
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} PB"
    
    def _display_storage_info(self) -> None:
        """Display current storage information for metadata_db and vector_db.
        
        Shows detailed information about database storage including file sizes,
        record counts, and database connectivity status.
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("📊 CURRENT STORAGE INFORMATION")
            self.logger.info("=" * 60)
            
            # Vector DB Storage Information
            vector_db_path = self.config['vector_db']['db_path']
            self.logger.info(f"🗂️  Vector Database Path: {vector_db_path}")
            
            # Check if vector database exists and get info
            if os.path.exists(vector_db_path):
                index_file = os.path.join(vector_db_path, "index.pkl")
                faiss_file = os.path.join(vector_db_path, "index.faiss")
                
                if os.path.exists(index_file) and os.path.exists(faiss_file):
                    # Get file sizes
                    index_size = os.path.getsize(index_file)
                    faiss_size = os.path.getsize(faiss_file)
                    total_size = index_size + faiss_size
                    
                    self.logger.info(f"   ✅ Vector DB exists")
                    self.logger.info(f"   📁 Index file size: {self._format_bytes(index_size)}")
                    self.logger.info(f"   📁 FAISS file size: {self._format_bytes(faiss_size)}")
                    self.logger.info(f"   📊 Total size: {self._format_bytes(total_size)}")
                    
                    # Try to get vector count
                    try:
                        # First try to use initialized VectorDB object if available
                        if hasattr(self, 'vector_db') and self.vector_db and hasattr(self.vector_db, 'faiss_store'):
                            vector_count = len(self.vector_db.faiss_store.docstore._dict)
                            self.logger.info(f"   🔢 Vector count: {vector_count}")
                        else:
                            # Fallback: Quick check without full model loading
                            import pickle
                            with open(index_file, 'rb') as f:
                                index_data = pickle.load(f)
                                if hasattr(index_data, 'docstore') and hasattr(index_data.docstore, '_dict'):
                                    vector_count = len(index_data.docstore._dict)
                                    self.logger.info(f"   🔢 Vector count: {vector_count}")
                                else:
                                    self.logger.info(f"   🔢 Vector count: Unable to determine")
                    except Exception as e:
                        self.logger.info(f"   🔢 Vector count: Unable to determine ({str(e)})")
                else:
                    self.logger.info(f"   ❌ Vector DB files incomplete or missing")
            else:
                self.logger.info(f"   ❌ Vector DB directory does not exist")
            
            self.logger.info("")
            
            # Metadata DB Storage Information
            db_url = self.config['metadata_db']['db_url']
            self.logger.info(f"🗄️  Metadata Database URL: {db_url}")
            
            # Try to connect and get metadata info
            try:
                engine = create_engine(db_url)
                with engine.connect() as conn:
                    # Check if tables exist
                    inspector = inspect(engine)
                    tables = inspector.get_table_names()
                    
                    if 'papers' in tables:
                        # Get paper count
                        result = conn.execute(text("SELECT COUNT(*) FROM papers"))
                        paper_count = result.scalar()
                        self.logger.info(f"   ✅ Metadata DB connected")
                        self.logger.info(f"   📄 Papers table exists")
                        self.logger.info(f"   🔢 Paper count: {paper_count}")
                        
                        # Get text chunks count if table exists
                        if 'text_chunks' in tables:
                            result = conn.execute(text("SELECT COUNT(*) FROM text_chunks"))
                            chunk_count = result.scalar()
                            self.logger.info(f"   📝 Text chunks count: {chunk_count}")
                        
                        # Get database size (PostgreSQL specific)
                        if 'postgresql' in db_url:
                            try:
                                result = conn.execute(text("""
                                    SELECT pg_size_pretty(pg_database_size(current_database())) as db_size
                                """))
                                db_size = result.scalar()
                                self.logger.info(f"   💾 Database size: {db_size}")
                            except Exception:
                                self.logger.info(f"   💾 Database size: Unable to determine")
                    else:
                        self.logger.info(f"   ❌ Papers table does not exist")
                        self.logger.info(f"   📋 Available tables: {', '.join(tables) if tables else 'None'}")
                        
            except Exception as e:
                self.logger.info(f"   ❌ Cannot connect to metadata DB: {str(e)}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"Failed to display storage information: {str(e)}")


#python3 -m test.index.litsearch_testbed

if __name__ == '__main__':
    config_path = Path("/data3/guofang/AIgnite-Solutions/AIgnite/experiments/recommendation/configs/litsearch_testbed_config_gritlm.yaml")
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Please create the configuration file or specify a different path with -c")
        sys.exit(1)
    
    # 创建并运行测试床
    logger.info("Initializing LitSearchTestBed...")
    testbed = LitSearchTestBed(str(config_path))
    #testbed.setup()
    #testbed.execute()  # 这会调用 check_environment(), load_data(), initialize_databases(), run_tests() 并包含清理控制

    cleanup_before_test = testbed.litsearch_config.get('cleanup_before_test', True)
    cleanup_after_test = testbed.litsearch_config.get('cleanup_after_test', True)

    print(f"清理设置: 清理前={cleanup_before_test}, 清理后={cleanup_after_test}")
    
    # 执行测试，使用配置文件中的清理设置
    #testbed.setup(clean_before_test=cleanup_before_test)
    testbed.execute(clean_before_test=cleanup_before_test, clean_after_test=cleanup_after_test)
