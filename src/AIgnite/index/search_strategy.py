from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

def to_python_type(value):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value

@dataclass
class SearchResult:
    """Standardized search result format for all search strategies"""
    doc_id: str
    score: float
    metadata: Dict[str, Any]
    search_method: str
    matched_text: Optional[str] = None
    chunk_id: Optional[str] = None

class SearchStrategy(ABC):
    """Abstract base class for search strategies"""
    
    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """Execute search using the strategy's method
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            similarity_cutoff: Minimum similarity score to include in results
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of SearchResult objects containing search results
        """
        pass

class VectorSearchStrategy(SearchStrategy):
    """Vector-based semantic search implementation"""
    
    def __init__(self, search_strategies):
        """Initialize with vector database instance.
        
        Args:
            vector_db: Instance of VectorDB or ToyVectorDB
        """
        #self.vector_db = vector_db
        self.search_strategies = search_strategies
        assert len(self.search_strategies) == 1
        #print('VECTOR SEARCH STRATEGY INITIALIZED')
    '''
    def transfer_filters(self, filters: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Transfer filters to vector_db's search implementation.
        
        Args:
            filters: Optional filters to apply to the search
        """
        if 'doc_ids' in filters['include']:
            return {"doc_ids": filters['include']['doc_ids']}
        return None
    '''
    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            similarity_cutoff: Minimum similarity score to include in results
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of SearchResult objects containing search results
        """
        try:
            # Use vector_db's search implementation which uses FAISS
            #if filters:
            #    filters=self.transfer_filters(filters)
            vector_results = []
        
        # 1. 调用每个搜索策略
            for strategy, strategy_cutoff in self.search_strategies:
                #print('IN VECTOR SEARCH STRATEGY')
                strategy_results = strategy.search(
                    query=query, 
                    top_k=top_k,  # 获取更多候选结果 # 获取更多候选结果
                    filters=filters,  # 使用策略特定的相似度阈值
                )
                vector_results.extend(strategy_results)
            #vector_results = self.vector_db.search(query, k=top_k, filters=filters)
            
            # Process results
            #print('Within search strategy')
            results = []
            for entry, score in vector_results:
                print(entry,score)
                if score > strategy_cutoff:
                    continue
                
                
                results.append(SearchResult(
                    doc_id=entry.doc_id,
                    score=float(to_python_type(score)),
                    metadata={
                        "vector_score": float(to_python_type(score)),
                        "text": entry.text,
                        "text_type": entry.text_type,
                        "chunk_id": entry.chunk_id
                    },
                    search_method="vector",
                    matched_text=entry.text,
                    chunk_id=entry.chunk_id
                ))
            
            logger.debug(f"Vector search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise

class TFIDFSearchStrategy(SearchStrategy):
    """TF-IDF based search implementation using PostgreSQL's full-text search capabilities"""
    
    def __init__(self, search_strategies):
        self.search_strategies = search_strategies
        #self.metadata_db = metadata_db
        assert len(self.search_strategies) == 1

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.1,
        **kwargs
    ) -> List[SearchResult]:
        try:
            # Use metadata_db's search implementation
            search_results = []
        
        # 1. 调用每个搜索策略
            for strategy, strategy_cutoff in self.search_strategies:
                strategy_results = strategy.search(
                    query=query, 
                    top_k=top_k,  # 获取更多候选结果
                    filters=filters, 
                    similarity_cutoff=strategy_cutoff  # 使用策略特定的相似度阈值
                )
                search_results.extend(strategy_results)
            '''
            search_results = self.metadata_db.search_papers(
                query=query,
                top_k=top_k,
                similarity_cutoff=similarity_cutoff,
                filters=filters
            )
            '''
            #print('IN TFIDFSearchStrategy')
            #print(search_results)
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                results.append(SearchResult(
                    doc_id=result['doc_id'],
                    score=float(to_python_type(result['score'])),
                    metadata=result['metadata'],
                    search_method='tf-idf',
                    matched_text=result['matched_text']
                ))
            
            logger.debug(f"TF-IDF search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {str(e)}")
            raise
## 

class HybridSearchStrategy(SearchStrategy):
    def __init__(
        self,
        search_strategies: List[Tuple[SearchStrategy, float]]
    ):
        """
        初始化混合搜索策略
        
        Args:
            search_strategies: 搜索策略列表，每个元素为(策略实例, 相似度阈值)的元组
                            例如: [(vector_strategy, 0.5), (tfidf_strategy, 0.1)]
        """
        self.search_strategies = search_strategies
        assert len(self.search_strategies) >1

    def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, 
           similarity_cutoff: float = 0.5, **kwargs) -> List[SearchResult]:
        """
        执行混合搜索
        
        Args:
            query: 搜索查询字符串
            top_k: 返回结果数量
            filters: 可选的过滤条件
            similarity_cutoff: 相似度阈值
            **kwargs: 额外参数
            
        Returns:
            重排序后的搜索结果列表
        """
        all_results = []
        print('IN HYBRID SEARCH STRATEGY')
        # 1. 调用每个搜索策略
        for strategy, strategy_cutoff in self.search_strategies:
            strategy_results = strategy.search(
                query, 
                top_k,  # 获取更多候选结果
                filters, 
                strategy_cutoff  # 使用策略特定的相似度阈值
            )
            all_results.extend(strategy_results)
        
        # 2. 按文档ID分组结果
        doc_results = {}
        for result in all_results:
            if result.doc_id not in doc_results:
                doc_results[result.doc_id] = []
            doc_results[result.doc_id].append(result)
        
        # 3. 对每个文档的结果进行倒数排名重排序
        reranked_results = []
        for doc_id, results in doc_results.items():
            # 按搜索方法分组
            method_results = {}
            for result in results:
                if result.search_method not in method_results:
                    method_results[result.search_method] = []
                method_results[result.search_method].append(result)
            
            # 计算每个方法的倒数排名分数
            method_scores = {}
            for method, method_res in method_results.items():
                # 按分数排序
                method_res.sort(key=lambda x: x.score, reverse=True)
                # 计算倒数排名分数
                for rank, result in enumerate(method_res[:top_k], 1):
                    reciprocal_score = 1.0 / rank
                    if method not in method_scores:
                        method_scores[method] = 0
                    method_scores[method] += reciprocal_score
            
            # 4. 合并不同方法的分数
            combined_score = sum(method_scores.values()) / len(method_scores)
            
            # 5. 创建重排序后的结果
            best_result = max(results, key=lambda x: x.score)
            # Convert method_scores to Python native types
            python_method_scores = {k: float(to_python_type(v)) for k, v in method_scores.items()}
            reranked_results.append(SearchResult(
                doc_id=doc_id,
                score=float(to_python_type(combined_score)),
                metadata={
                    **best_result.metadata,
                    "reranked_score": float(to_python_type(combined_score)),
                    "method_scores": python_method_scores
                },
                search_method="hybrid_reranked",
                matched_text=best_result.matched_text,
                chunk_id=best_result.chunk_id
            ))
        
        # 6. 按重排序分数排序并返回top_k结果
        reranked_results.sort(key=lambda x: x.score, reverse=True)
        return reranked_results[:top_k]

'''
class HybridSearchStrategy(SearchStrategy):
    """Combined vector and TF-IDF search implementation"""
    
    def __init__(
        self,
        vector_strategy: VectorSearchStrategy,
        tfidf_strategy: TFIDFSearchStrategy,
        vector_weight: float = 0.7
    ):
        self.vector_strategy = vector_strategy
        self.tfidf_strategy = tfidf_strategy
        self.vector_weight = vector_weight
        self.tfidf_weight = 1 - vector_weight

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        try:
            # Get results from both strategies
            
            vector_results = self.vector_strategy.search(
                query, 
                top_k * 2,
                filters,
                similarity_cutoff
            )
            tfidf_results = self.tfidf_strategy.search(
                query,
                top_k * 2,
                filters,
                similarity_cutoff
            )
            
            # Combine results
            combined_results = {}
            
            # Process vector results
            for result in vector_results:
                combined_results[result.doc_id] = {
                    "vector_score": result.score,
                    "matched_text": result.matched_text,
                    "chunk_id": result.chunk_id,
                    "metadata": result.metadata
                }
            
            # Process TF-IDF results
            for result in tfidf_results:
                if result.doc_id in combined_results:
                    combined_results[result.doc_id]["tfidf_score"] = result.score
                else:
                    combined_results[result.doc_id] = {
                        "tfidf_score": result.score,
                        "metadata": result.metadata
                    }
            
            # Calculate combined scores
            results = []
            for doc_id, data in combined_results.items():
                vector_score = data.get("vector_score", 0)
                tfidf_score = data.get("tfidf_score", 0)
                
                # Calculate weighted score
                combined_score = (
                    vector_score * self.vector_weight +
                    tfidf_score * self.tfidf_weight
                )
                
                if combined_score >= similarity_cutoff:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        score=float(to_python_type(combined_score)),
                        metadata={
                            "vector_score": float(to_python_type(vector_score)),
                            "tfidf_score": float(to_python_type(tfidf_score)),
                            "combined_score": float(to_python_type(combined_score)),
                            **data.get("metadata", {})
                        },
                        search_method="hybrid",
                        matched_text=data.get("matched_text"),
                        chunk_id=data.get("chunk_id")
                    ))
            
            # Sort by combined score and return top_k results
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
            
            logger.debug(f"Hybrid search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise 
'''