#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LitSearch 评估框架

提供完整的评估框架，用于测试不同搜索策略在 LitSearch 数据集上的性能。
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from tqdm import tqdm
from datasets import load_dataset
import json

# 删除对独立搜索策略的导入，改为导入AIgnite的搜索模块
from AIgnite.index.search_strategy import SearchStrategy, VectorSearchStrategy, TFIDFSearchStrategy, HybridSearchStrategy
from AIgnite.db.vector_db import VectorDB
from AIgnite.db.metadata_db import MetadataDB
from .evaluator import SearchEvaluator
from .visualization import ResultVisualizer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LitSearchEvaluator:
    """LitSearch数据集评估框架
    
    支持灵活配置数据库初始化，可以选择性地初始化向量数据库和/或元数据数据库。
    通过构造函数参数或方法参数控制初始化行为。
    """
    
    def __init__(
        self,
        db_config: Dict[str, Any],
        init_vector_db: bool = True,
        init_metadata_db: bool = True,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        vector_weight: float = 0.7,
        vector_db_path: str = "./litsearch_vector_db"
    ):
        """初始化评估框架
        
        Args:
            db_config: 数据库配置
            embedding_model: 嵌入模型名称
            vector_weight: 混合搜索中向量搜索的权重
            init_vector_db: 是否初始化向量数据库
            init_metadata_db: 是否初始化元数据数据库
        """
        self.db_config = db_config
        self.init_vector_db = init_vector_db
        self.init_metadata_db = init_metadata_db
        if self.init_vector_db:
            self.vector_db_path = vector_db_path
            self.embedding_model = embedding_model
            self.vector_weight = vector_weight
        
        
        
        # 初始化数据库
        self.vector_db = None
        self.tfidf_db = None
        
        # 初始化搜索策略
        self.vector_strategy = None
        self.tfidf_strategy = None
        self.hybrid_strategy = None
        
        logger.info(f"初始化 LitSearch 评估框架")
        logger.info(f"数据库配置: {db_config}")
        logger.info(f"嵌入模型: {embedding_model}")
        logger.info(f"向量权重: {vector_weight}")
        logger.info(f"初始化向量数据库: {init_vector_db}")
        logger.info(f"初始化元数据数据库: {init_metadata_db}")
    
    def load_data(self) -> Tuple[Any, Dict[str, List[str]]]:
        """加载LitSearch数据集
        
        Returns:
            Tuple[Any, Dict[str, List[str]]]: 语料库和查询真值
        """
        logger.info("加载 LitSearch 数据集...")
        
        try:
            # 加载查询数据集
            query_dataset = load_dataset("princeton-nlp/LitSearch", "query", split="full")
            # 加载语料库数据集（使用corpus_clean配置，也可以根据需要选择corpus_s2orc）
            corpus_dataset = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
            
            # 将数据集转换为字典列表，以便于处理
            corpus = [doc for doc in corpus_dataset]
            queries = query_dataset
            
            logger.info(f"成功加载数据集: {len(corpus)} 篇文档, {len(queries)} 个查询")
            
            # 提取查询和真值
            ground_truth = {}
            for query in queries:
                query_text = query["query"]  
                relevant_docs = [str(doc_id) for doc_id in query["corpusids"]] 
                ground_truth[query_text] = relevant_docs
            
            logger.info(f"提取了 {len(ground_truth)} 个查询的真值数据")
            
            return corpus, ground_truth
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise
    
    def sample_data(
        self,
        corpus: Any,
        ground_truth: Dict[str, List[str]],
        sample_size: int
    ) -> Tuple[Any, Dict[str, List[str]]]:
        """采样数据
        
        Args:
            corpus: 语料库
            ground_truth: 查询真值
            sample_size: 采样大小
            
        Returns:
            Tuple[Any, Dict[str, List[str]]]: 采样后的语料库和查询真值
        """
        logger.info(f"采样 {sample_size} 篇文档和相关查询...")
        
        try:
            # 1. 首先随机选择一些查询
            query_texts = list(ground_truth.keys())
            query_count = min(sample_size, len(query_texts))
            query_indices = np.random.choice(len(query_texts), query_count, replace=False)
            sampled_queries = [query_texts[i] for i in query_indices]
            
            # 2. 获取这些查询相关的所有文档ID
            relevant_doc_ids = set()
            for query in sampled_queries:
                relevant_doc_ids.update(ground_truth[query])
            
            logger.info(f"选择了 {len(sampled_queries)} 个查询，涉及 {len(relevant_doc_ids)} 个相关文档")
            
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
                logger.warning("采样后没有查询有相关文档！将使用原始查询和真值。")
                return corpus_sample, ground_truth
            
            logger.info(f"采样后: {len(corpus_sample)} 篇文档, {len(sampled_ground_truth)} 个查询")
            logger.info(f"采样后的查询平均有 {sum(len(docs) for docs in sampled_ground_truth.values()) / len(sampled_ground_truth):.2f} 个相关文档")
            
            return corpus_sample, sampled_ground_truth
        except Exception as e:
            logger.error(f"采样数据失败: {str(e)}")
            return corpus, ground_truth
    
    def initialize_databases(
        self, 
        corpus: Any, 
        batch_size: int = 32,
        init_vector_db: bool = None,
        init_metadata_db: bool = None
    ) -> None:
        """初始化数据库并索引文档
        
        Args:
            corpus: 语料库
            batch_size: 批处理大小
            init_vector_db: 是否初始化向量数据库，None表示使用构造函数中的设置
            init_metadata_db: 是否初始化元数据数据库，None表示使用构造函数中的设置
        """
        # 使用构造函数中的默认值，如果参数为None
        if init_vector_db is None:
            init_vector_db = self.init_vector_db
        if init_metadata_db is None:
            init_metadata_db = self.init_metadata_db
            
        # 验证至少有一个数据库被初始化
        if not init_vector_db and not init_metadata_db:
            raise ValueError("至少需要初始化一个数据库")
            
        
        
        try:
            # 初始化向量数据库（如果需要）
            logger.info("初始化数据库...")
            if init_vector_db:
                logger.info(f"初始化向量数据库: {init_vector_db}")
                self.initialize_vector_database(corpus, batch_size)
            
            # 初始化元数据数据库（如果需要）
            if init_metadata_db:
                logger.info(f"初始化元数据数据库: {init_metadata_db}")
                self.initialize_metadata_database(corpus, batch_size)
            
            # 初始化搜索策略
            self._initialize_search_strategies()
            logger.info("数据库和搜索策略初始化完成")
        except Exception as e:
            logger.error(f"初始化数据库失败: {str(e)}")
            raise
    
    def initialize_vector_database(self, corpus: Any, batch_size: int = 32) -> None:
        """初始化向量数据库并索引文档
        
        Args:
            corpus: 语料库
            batch_size: 批处理大小
        """
        logger.info(f"初始化向量数据库，使用模型: {self.embedding_model}")
        
        try:
            # 初始化向量数据库
            if not os.path.exists(self.vector_db_path):
                os.makedirs(self.vector_db_path)
                
                self.vector_db = VectorDB(
                    db_path=self.vector_db_path,  # 临时路径
                    model_name=self.embedding_model
                )
                # 批量添加文档到向量数据库
                logger.info(f"索引 {len(corpus)} 篇文档到向量数据库...")
                for doc in tqdm(corpus, desc="向量数据库索引"):
                    success = self.vector_db.add_document(
                        doc_id=str(doc["corpusid"]),
                        abstract=doc.get("abstract", ""),
                        text_chunks=[],  # 空列表，简化存储
                        metadata={
                            'title': doc.get("title", ""),
                            'categories': [],  # 空列表
                            'text_chunk_ids': []  # 空列表
                        }
                    )
                    if not success:
                        logger.warning(f"文档 {doc['corpusid']} 添加到向量数据库失败")
                
                # 保存向量数据库
                    save_success = self.vector_db.save()
                    if not save_success:
                        logger.error("保存向量数据库失败")
                    
                    logger.info("向量数据库初始化完成")
            else:
                logger.info(f"向量数据库路径已存在: {self.vector_db_path}")
            
        except Exception as e:
            logger.error(f"初始化向量数据库失败: {str(e)}")
            raise
    
    def initialize_metadata_database(self, corpus: Any, batch_size: int = 32) -> None:
        """初始化元数据数据库并索引文档
        
        Args:
            corpus: 语料库
            batch_size: 批处理大小
        """
        logger.info("初始化元数据数据库")
        
        try:
            # 初始化元数据数据库
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['name']}"
            self.tfidf_db = MetadataDB(connection_string)
            
            # 批量添加文档到元数据数据库
            logger.info(f"索引 {len(corpus)} 篇文档到元数据数据库...")
            success_count = 0
            fail_count = 0
            
            for doc in tqdm(corpus, desc="元数据数据库索引"):
                try:
                    success = self.tfidf_db.save_paper(
                        doc_id=str(doc["corpusid"]),
                        pdf_path="",  # 设为空字符串，表示没有PDF文件
                        metadata={
                            'title': doc.get("title", ""),
                            'abstract': doc.get("abstract", ""),
                            'authors': [],  # 设为空列表
                            'categories': [],  # 设为空列表
                            'published_date': "",  # 设为空字符串
                            'chunk_ids': [],  # 空列表
                            'figure_ids': [],  # 空列表
                            'table_ids': [],  # 空列表
                            'metadata': {},  # 空字典
                            'HTML_path': None,  # 设为None
                            'blog': None,  # 设为None
                            'comments': ""  # 设为空字符串
                        }
                    )
                    if success:
                        success_count += 1
                        logger.debug(f"文档 {doc['corpusid']} 成功添加到元数据数据库")
                    else:
                        fail_count += 1
                        logger.warning(f"文档 {doc['corpusid']} 添加到元数据数据库失败")
                except Exception as e:
                    fail_count += 1
                    logger.error(f"文档 {doc['corpusid']} 添加异常: {str(e)}")
            
            logger.info(f"元数据数据库索引完成: 成功 {success_count} 篇，失败 {fail_count} 篇")
        except Exception as e:
            logger.error(f"初始化元数据数据库失败: {str(e)}")
            raise
    
    def _initialize_search_strategies(self) -> None:
        """根据已初始化的数据库初始化搜索策略"""
        logger.info("初始化搜索策略")
        
        # 初始化向量搜索策略（如果向量数据库已初始化）
        if self.vector_db is not None:
            self.vector_strategy = VectorSearchStrategy(self.vector_db)
            logger.info("向量搜索策略初始化完成")
        else:
            self.vector_strategy = None
            logger.info("跳过向量搜索策略初始化（向量数据库未初始化）")
        
        # 初始化TF-IDF搜索策略（如果元数据数据库已初始化）
        if self.tfidf_db is not None:
            self.tfidf_strategy = TFIDFSearchStrategy(self.tfidf_db)
            logger.info("TF-IDF搜索策略初始化完成")
        else:
            self.tfidf_strategy = None
            logger.info("跳过TF-IDF搜索策略初始化（元数据数据库未初始化）")
        
        # 初始化混合搜索策略（只有当两个数据库都初始化时才创建）
        if self.vector_db is not None and self.tfidf_db is not None:
            self.hybrid_strategy = HybridSearchStrategy(
                self.vector_strategy,
                self.tfidf_strategy,
                vector_weight=self.vector_weight
            )
            logger.info("混合搜索策略初始化完成")
        else:
            self.hybrid_strategy = None
            logger.warning("跳过混合搜索策略初始化（需要两个数据库都初始化）")
    
    def evaluate_strategy(
        self,
        strategy: SearchStrategy,
        strategy_name: str,
        queries: Dict[str, List[str]],
        top_k: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[str]]:
        """评估搜索策略
        
        Args:
            strategy: 搜索策略
            strategy_name: 策略名称
            queries: 查询真值
            top_k: 返回结果数量
            batch_size: 批处理大小
            
        Returns:
            Dict[str, List[str]]: 查询结果
        """
        logger.info(f"评估 {strategy_name} 策略...")
        
        results = {}
        query_texts = list(queries.keys())
        
        # 批量处理查询
        total_batches = (len(query_texts) + batch_size - 1) // batch_size
        for i in range(0, len(query_texts), batch_size):
            batch_queries = query_texts[i:i+batch_size]
            logger.info(f"处理查询批次 {i//batch_size + 1}/{total_batches}，共 {len(batch_queries)} 个查询")
            
            for query_text in tqdm(batch_queries, desc=f"{strategy_name} 策略检索"):
                try:
                    search_results = strategy.search(query=query_text, top_k=top_k)
                    retrieved_docs = [result.doc_id for result in search_results]
                    results[query_text] = retrieved_docs
                except Exception as e:
                    logger.error(f"查询 '{query_text}' 失败: {str(e)}")
                    results[query_text] = []
        
        logger.info(f"{strategy_name} 策略评估完成，处理了 {len(results)} 个查询")
        return results
    
    def evaluate_all_strategies(
        self,
        queries: Dict[str, List[str]],
        top_k: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Dict[str, List[str]]]:
        """评估所有搜索策略
        
        Args:
            queries: 查询真值
            top_k: 返回结果数量
            batch_size: 批处理大小
            
        Returns:
            Dict[str, Dict[str, List[str]]]: 各策略的查询结果
        """
        logger.info(f"评估所有搜索策略，共 {len(queries)} 个查询...")
        
        # 只评估已初始化的策略
        strategies = {}
        if self.vector_strategy is not None:
            strategies["vector"] = self.vector_strategy
        if self.tfidf_strategy is not None:
            strategies["tf-idf"] = self.tfidf_strategy
        if self.hybrid_strategy is not None:
            strategies["hybrid"] = self.hybrid_strategy
        
        if not strategies:
            logger.warning("没有可用的搜索策略进行评估")
            return {}
        
        logger.info(f"将评估以下策略: {list(strategies.keys())}")
        
        results = {}
        for name, strategy in strategies.items():
            strategy_results = self.evaluate_strategy(
                strategy=strategy,
                strategy_name=name,
                queries=queries,
                top_k=top_k,
                batch_size=batch_size
            )
            results[name] = strategy_results
        
        logger.info("所有策略评估完成")
        return results
    
    def run_evaluation(
        self,
        top_k: int = 100,
        sample_size: Optional[int] = None,
        batch_size: int = 32,
        init_vector_db: bool = None,
        init_metadata_db: bool = None
    ) -> Dict[str, Dict[str, float]]:
        """运行完整评估流程
        
        Args:
            top_k: 返回结果数量
            sample_size: 采样大小
            batch_size: 批处理大小
            init_vector_db: 是否初始化向量数据库，None表示使用构造函数中的设置
            init_metadata_db: 是否初始化元数据数据库，None表示使用构造函数中的设置
            
        Returns:
            Dict[str, Dict[str, float]]: 评估结果
        """
        logger.info(f"开始评估流程，top_k={top_k}, sample_size={sample_size}, batch_size={batch_size}")
        logger.info(f"初始化向量数据库: {init_vector_db if init_vector_db is not None else self.init_vector_db}")
        logger.info(f"初始化元数据数据库: {init_metadata_db if init_metadata_db is not None else self.init_metadata_db}")
        
        try:
            # 加载数据
            corpus, ground_truth = self.load_data()
            
            # 采样（如果需要）
            if sample_size:
                corpus, ground_truth = self.sample_data(corpus, ground_truth, sample_size)
            
            # 初始化数据库
            self.initialize_databases(
                corpus, 
                batch_size=batch_size,
                init_vector_db=init_vector_db,
                init_metadata_db=init_metadata_db
            )
            
            print('FINISH INITIALIZE DATABASES')
            # 评估所有策略
            results = self.evaluate_all_strategies(ground_truth, top_k=top_k, batch_size=batch_size)
            
            # 计算指标
            logger.info("计算评估指标...")
            evaluation_results = SearchEvaluator.compare_strategies(results, ground_truth)
            
            # 打印评估结果
            for name, metrics in evaluation_results.items():
                logger.info(f"\n{name} 策略的评估结果:")
                logger.info(f"  MAP: {metrics.get('map', 0):.4f}")
                logger.info(f"  Precision@10: {metrics.get('precision@10', 0):.4f}")
                logger.info(f"  Recall@10: {metrics.get('recall@10', 0):.4f}")
            
            # 找出最佳策略
            best_strategy = max(evaluation_results.items(), key=lambda x: x[1].get("map", 0))[0]
            logger.info(f"\n最佳策略: {best_strategy} (MAP: {evaluation_results[best_strategy]['map']:.4f})")
            
            return evaluation_results
        except Exception as e:
            logger.error(f"评估流程失败: {str(e)}")
            raise 