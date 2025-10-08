#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
搜索策略评估器

提供评估搜索策略性能的工具。
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEvaluator:
    """搜索策略评估器"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """计算Precision@k
        
        Args:
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 相关文档ID集合
            k: 截断位置
            
        Returns:
            float: Precision@k值
        """
        if not retrieved_docs or k <= 0:
            return 0.0
        
        # 截断到前k个结果
        retrieved_k = retrieved_docs[:k]
        
        # 计算相关文档数量
        relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        
        # 计算精确率
        return relevant_count / min(k, len(retrieved_k))
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """计算Recall@k
        
        Args:
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 相关文档ID集合
            k: 截断位置
            
        Returns:
            float: Recall@k值
        """
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0
        
        # 截断到前k个结果
        retrieved_k = retrieved_docs[:k]
        
        # 计算相关文档数量
        relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant_docs)
        
        # 计算召回率
        return relevant_count / len(relevant_docs)
    
    @staticmethod
    def average_precision(retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """计算平均精确率（Average Precision）
        
        Args:
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 相关文档ID集合
            
        Returns:
            float: 平均精确率值
        """
        if not retrieved_docs or not relevant_docs:
            return 0.0
        
        # 计算每个位置的精确率
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        
        # 计算平均精确率
        if precisions:
            return sum(precisions) / len(relevant_docs)
        else:
            return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_docs: List[str], relevant_docs: Set[str], k: int) -> float:
        """计算NDCG@k（归一化折损累积增益）
        
        Args:
            retrieved_docs: 检索到的文档ID列表
            relevant_docs: 相关文档ID集合
            k: 截断位置
            
        Returns:
            float: NDCG@k值
        """
        if not retrieved_docs or not relevant_docs or k <= 0:
            return 0.0
        
        # 截断到前k个结果
        retrieved_k = retrieved_docs[:k]
        
        # 计算DCG
        dcg = 0
        for i, doc_id in enumerate(retrieved_k):
            if doc_id in relevant_docs:
                # 使用二元相关性（0或1）
                dcg += 1.0 / np.log2(i + 2)  # i+2是因为log2(1)=0
        
        # 计算理想DCG（将所有相关文档排在前面）
        idcg = 0
        for i in range(min(len(relevant_docs), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        # 计算NDCG
        if idcg > 0:
            return dcg / idcg
        else:
            return 0.0
    
    @staticmethod
    def evaluate_strategy(
        strategy_name: str,
        retrieval_results: Dict[str, List[str]],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """评估单个搜索策略的性能
        
        Args:
            strategy_name: 策略名称
            retrieval_results: 检索结果，查询到文档ID列表的映射
            ground_truth: 真值数据，查询到相关文档ID列表的映射
            k_values: 评估的k值列表
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        # 初始化评估指标
        metrics = {
            "strategy": strategy_name,
            "query_count": 0,
            "map": 0.0
        }
        
        # 为每个k值初始化精确率和召回率指标
        for k in k_values:
            metrics[f"precision@{k}"] = 0.0
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
        
        # 计算每个查询的指标
        valid_queries = 0
        for query, retrieved_docs in retrieval_results.items():
            # 跳过不在真值数据中的查询
            if query not in ground_truth:
                continue
            
            # 获取相关文档
            relevant_docs = set(ground_truth[query])
            

           #print(f"relevant_docs: {relevant_docs}")
           # print(f"retrieved_docs: {retrieved_docs}")
            #exit()
            # 如果没有相关文档，跳过
            if not relevant_docs:
                continue
            

            # 计算平均精确率
            ap = SearchEvaluator.average_precision(retrieved_docs, relevant_docs)
            metrics["map"] += ap
            
            # 计算各个k值的精确率和召回率
            for k in k_values:
                metrics[f"precision@{k}"] += SearchEvaluator.precision_at_k(retrieved_docs, relevant_docs, k)
                metrics[f"recall@{k}"] += SearchEvaluator.recall_at_k(retrieved_docs, relevant_docs, k)
                metrics[f"ndcg@{k}"] += SearchEvaluator.ndcg_at_k(retrieved_docs, relevant_docs, k)
            
            valid_queries += 1
        
        # 计算平均值
        if valid_queries > 0:
            metrics["map"] /= valid_queries
            metrics["query_count"] = valid_queries
            
            for k in k_values:
                metrics[f"precision@{k}"] /= valid_queries
                metrics[f"recall@{k}"] /= valid_queries
                metrics[f"ndcg@{k}"] /= valid_queries
        
        return metrics
    
    @staticmethod
    def compare_strategies(
        strategies_results: Dict[str, Dict[str, List[str]]],
        ground_truth: Dict[str, List[str]],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, Dict[str, float]]:
        """比较多个搜索策略的性能
        
        Args:
            strategies_results: 各策略的检索结果，策略名称到查询结果的映射
            ground_truth: 真值数据，查询到相关文档ID列表的映射
            k_values: 评估的k值列表
            
        Returns:
            Dict[str, Dict[str, float]]: 各策略的评估指标
        """
        results = {}
        
        for strategy_name, retrieval_results in strategies_results.items():
            metrics = SearchEvaluator.evaluate_strategy(
                strategy_name,
                retrieval_results,
                ground_truth,
                k_values
            )
            results[strategy_name] = metrics
        
        return results 