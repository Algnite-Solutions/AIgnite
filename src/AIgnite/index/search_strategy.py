from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llama_index.core import VectorStoreIndex
import logging

# Set up logging
logger = logging.getLogger(__name__)

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
    
    def __init__(self, vector_db):
        """Initialize with vector database instance.
        
        Args:
            vector_db: Instance of VectorDB or ToyVectorDB
        """
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        try:
            # Use vector_db's search implementation which uses FAISS

            vector_results = self.vector_db.search(query, k=top_k, filters=filters)
            
            # Process results
            results = []
            for entry, score in vector_results:
                if score < similarity_cutoff:
                    continue
                    
                results.append(SearchResult(
                    doc_id=entry.doc_id,
                    score=score,
                    metadata={
                        "vector_score": score,
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
    
    def __init__(self, metadata_db):
        self.metadata_db = metadata_db

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
            search_results = self.metadata_db.search_papers(
                query=query,
                top_k=top_k,
                similarity_cutoff=similarity_cutoff,
                filters=filters
            )
            #print('IN TFIDFSearchStrategy')
            #print(search_results)
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                results.append(SearchResult(
                    doc_id=result['doc_id'],
                    score=result['score'],
                    metadata=result['metadata'],
                    search_method='tf-idf',
                    matched_text=result['matched_text']
                ))
            
            logger.debug(f"TF-IDF search found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {str(e)}")
            raise

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
                        score=combined_score,
                        metadata={
                            "vector_score": vector_score,
                            "tfidf_score": tfidf_score,
                            "combined_score": combined_score,
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