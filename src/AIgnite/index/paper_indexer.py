from typing import List, Optional, Dict, Any, Tuple
from .base_indexer import BaseIndexer
from ..data.docset import DocSet
from ..db.vector_db import VectorDB
from ..db.metadata_db import MetadataDB
from ..db.image_db import MinioImageDB
from .search_strategy import SearchStrategy, VectorSearchStrategy, TFIDFSearchStrategy, HybridSearchStrategy, SearchResult
from .filter_parser import FilterParser
from .data_retriever import DataRetriever
from .result_combiner import ResultCombiner
import logging
from tqdm import tqdm
import os

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PaperIndexer(BaseIndexer):
    def __init__(
        self,
        vector_db: Optional[VectorDB] = None,
        metadata_db: Optional[MetadataDB] = None,
        image_db: Optional[MinioImageDB] = None
    ):
        """Initialize the paper indexer with optional database instances.
        Databases can be initialized later using set_databases method.
        
        Args:
            vector_db: Optional VectorDB instance for text embeddings
            metadata_db: Optional MetadataDB instance for paper metadata
            image_db: Optional MinioImageDB instance for storing figures
        """
        logger.debug("Initializing PaperIndexer")
        self.vector_db = vector_db
        self.metadata_db = metadata_db
        self.image_db = image_db
        self.search_strategy = None
        self.filter_parser = FilterParser()
        
        # 初始化数据获取器和结果合并器
        self.data_retriever = DataRetriever(metadata_db, image_db)
        self.result_combiner = ResultCombiner()

    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'vector_db'):
            self.vector_db = None
        if hasattr(self, 'metadata_db'):
            self.metadata_db = None
        if hasattr(self, 'image_db'):
            self.image_db = None

    def set_databases(
        self,
        vector_db: Optional[VectorDB] = None,
        metadata_db: Optional[MetadataDB] = None,
        image_db: Optional[MinioImageDB] = None
    ) -> None:
        """Set or update database instances after initialization.
        
        Args:
            vector_db: Optional VectorDB instance for text embeddings
            metadata_db: Optional MetadataDB instance for paper metadata
            image_db: Optional MinioImageDB instance for storing figures
        """
        if vector_db is not None:
            self.vector_db = vector_db
            logger.debug("Vector database set")
            
        if metadata_db is not None:
            self.metadata_db = metadata_db
            logger.debug("Metadata database set")
            
        if image_db is not None:
            self.image_db = image_db
            logger.debug("Image database set")
        
        # 更新数据获取器
        self.data_retriever = DataRetriever(metadata_db, image_db)

    def set_search_strategy(self, search_strategies: List[Tuple[SearchStrategy, float]]) -> None:
        """Set the search strategy to use.
        
        Args:
            search_strategies: List of search strategies and their thresholds
            
        Raises:
            ValueError: If strategy_type is invalid or required database is not available
        """
        if len(search_strategies) == 1:
            if search_strategies[0][0] == 'vector':
                if self.vector_db is None:
                    raise ValueError("Vector database is required for vector search strategy")
                self.search_strategy = VectorSearchStrategy([(self.vector_db, search_strategies[0][1])])
            elif search_strategies[0][0] == 'tf-idf':
                if self.metadata_db is None:
                    raise ValueError("Metadata database is required for TF-IDF search strategy")
                self.search_strategy = TFIDFSearchStrategy([(self.metadata_db, search_strategies[0][1])])
            else:
                raise ValueError(f"Invalid strategy type. Must be one of: vector, tf-idf")
            logger.debug(f"Search strategy set to: {search_strategies[0]}")
        else:
            if self.vector_db is None or self.metadata_db is None:
                raise ValueError("Both vector and metadata databases are required for hybrid search strategy")
            db_search_strategies = []
            for strategy, threshold in search_strategies:
                if strategy == 'vector':
                    db_search_strategies.append((VectorSearchStrategy([(self.vector_db, threshold)]), threshold))
                elif strategy == 'tf-idf':
                    db_search_strategies.append((TFIDFSearchStrategy([(self.metadata_db, threshold)]), threshold))
            self.search_strategy = HybridSearchStrategy(db_search_strategies)
            #self.search_strategy = HybridSearchStrategy(search_strategies)
            logger.debug(f"Search strategy set to: hybrid")

        '''
        if strategy_type == 'vector':
            if self.vector_db is None:
                raise ValueError("Vector database is required for vector search strategy")
            self.search_strategy = VectorSearchStrategy(self.vector_db)
            
        elif strategy_type == 'tf-idf':
            if self.metadata_db is None:
                raise ValueError("Metadata database is required for TF-IDF search strategy")
            self.search_strategy = TFIDFSearchStrategy(self.metadata_db)
            
        elif strategy_type == 'hybrid':
            if self.vector_db is None or self.metadata_db is None:
                raise ValueError("Both vector and metadata databases are required for hybrid search strategy")
            vector_strategy = VectorSearchStrategy(self.vector_db)
            tfidf_strategy = TFIDFSearchStrategy(self.metadata_db)
            self.search_strategy = HybridSearchStrategy(vector_strategy, tfidf_strategy)
        else:
            raise ValueError(f"Invalid strategy type. Must be one of: vector, tf-idf, hybrid")
        '''
       

    def index_papers(self, papers: List[DocSet]) -> Dict[str, Dict[str, bool]]:
        """Index a list of papers into available databases.
        
        Args:
            papers: List of DocSet objects containing paper information
            
        Returns:
            Dictionary mapping doc_ids to their indexing status for each database type
        """
        indexing_status = {}
        
        try:
            # Main progress bar for papers
            for paper in tqdm(papers, desc="Indexing papers", unit="paper"):
                paper_status = {
                    "metadata": False,
                    "text_chunks": False,   # 文本块存储状态（新增）
                    "vectors": False,
                    "images": False
                }
                
                # Create metadata
                metadata = {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_date": paper.published_date,
                    "chunk_ids": [chunk.id for chunk in paper.text_chunks],
                    "figure_ids": [chunk.id for chunk in paper.figure_chunks],
                    "comments": paper.comments  # Store comments field
                }
                # Store metadata if database is available
                if self.metadata_db is not None and hasattr(paper, 'pdf_path'):
                    try:
                        success = self.metadata_db.save_paper(paper.doc_id, paper.pdf_path, metadata,paper.text_chunks)
                        paper_status["metadata"] = success
                        # 检查文本块存储状态（新增）
                        if success and paper.text_chunks:
                            paper_status["text_chunks"] = True
                        else:
                            paper_status["text_chunks"] = False
                        logger.debug(f"Stored paper metadata for {paper.doc_id}: {success}")
                    except Exception as e:
                        logger.error(f"Failed to store metadata for {paper.doc_id}: {str(e)}")
                        paper_status["metadata"] = False
                        paper_status["text_chunks"] = False
                
                # Store vectors if database is available
                if self.vector_db is not None:
                    try:
                        #text_chunks = [chunk.text for chunk in paper.text_chunks]
                        success = self.vector_db.add_document(
                            vector_db_id=paper.doc_id+'_abstract',
                            text_to_emb=paper.title+' . '+paper.abstract,
                            doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
                        )
                        paper_status["vectors"] = success
                        logger.debug(f"Added vectors for {paper.doc_id}: {success}")
                        
                        if success:
                            save_success = self.vector_db.save()
                            if not save_success:
                                logger.error(f"Failed to save vector database after adding {paper.doc_id}")
                                paper_status["vectors"] = False
                    except Exception as e:
                        logger.error(f"Failed to store vectors for {paper.doc_id}: {str(e)}")
                
                # Store images if database is available and paper has figures
                if self.image_db is not None and paper.figure_chunks:
                    try:
                        image_successes = []
                        for figure in tqdm(paper.figure_chunks, desc=f"Storing figures for {paper.doc_id}", leave=False):
                            success = self.image_db.save_image(
                                doc_id=paper.doc_id,
                                image_id=figure.id,
                                image_path=figure.image_path
                            )
                            image_successes.append(success)
                            logger.debug(f"Saved image {figure.id} for {paper.doc_id}: {success}")
                        paper_status["images"] = all(image_successes)
                    except Exception as e:
                        logger.error(f"Failed to store images for {paper.doc_id}: {str(e)}")
                
                indexing_status[paper.doc_id] = paper_status
                
        except Exception as e:
            logger.error(f"Failed to index papers: {str(e)}")
            raise RuntimeError(f"Failed to index papers: {str(e)}")
            
        return indexing_status

    def find_similar_papers(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        search_strategies: List[Tuple[SearchStrategy, float]] = None,
        result_include_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find papers similar to the query using the selected search strategy.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            search_strategies: List of search strategies and their thresholds
            result_include_types: Optional list of data types to include in results
            
        Returns:
            combine_results: The combined results according to the include_types and search results
            search_results: The original search results
            
        Raises:
            ValueError: If no search strategy is available or required database is missing
        """
        try:
            logger.debug(f"Searching for query: {query}")
            
            # 1. 执行搜索
            search_results = self._execute_search(query, top_k, filters, search_strategies)
            
            if not search_results:
                logger.debug("No search results found")
                return []
            
            # 2. 获取doc_ids
            doc_ids = [result.doc_id for result in search_results]
            logger.debug(f"Found {len(doc_ids)} documents: {doc_ids}")
            
            # 3. 获取数据
            data_dict = {}
            for data_type in result_include_types or ["metadata"]:
                if data_type != "search_parameters":
                    data_dict[data_type] = self.data_retriever.get_data_by_type(doc_ids, data_type)
            
            # 4. 合并结果
            combine_results= self.result_combiner.combine(search_results, data_dict, result_include_types or ["metadata"])
            return combine_results
            
        except Exception as e:
            logger.error(f"Failed to find similar papers: {str(e)}")
            raise RuntimeError(f"Failed to find similar papers: {str(e)}")
    
    def _execute_search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]], 
                       search_strategies: List[Tuple[SearchStrategy, float]]) -> List[SearchResult]:
        """执行搜索操作
        
        Args:
            query: 搜索查询
            top_k: 返回结果数量
            filters: 过滤条件
            similarity_cutoff: 相似度阈值
            strategy_type: 搜索策略类型
            
        Returns:
            搜索结果列表
        """
        # 确保有可用的数据库
        if search_strategies is None or len(search_strategies) == 0:
            raise ValueError("No search strategies provided")
        
        if search_strategies[0][0] == 'vector' and self.vector_db is None:
            raise ValueError("Vector database is required for vector search")
        elif search_strategies[0][0] == 'tf-idf' and self.metadata_db is None:
            raise ValueError("Metadata database is required for TF-IDF search")
        elif search_strategies[0][0] == 'hybrid' and (self.vector_db is None or self.metadata_db is None):
            raise ValueError("Both vector and metadata databases are required for hybrid search")
        
        # 设置或临时更改策略
        self.set_search_strategy(search_strategies) 
        try:
            # 解析和验证过滤器
            #parsed_filters = self.filter_parser.parse_filters(filters)
            # 执行搜索
            search_results = self.search_strategy.search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to execute search: {str(e)}")
            raise RuntimeError(f"Failed to execute search: {str(e)}")

    def get_paper_metadata(self, doc_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific paper.
        
        Args:
            doc_id: The document ID of the paper
            
        Returns:
            Dictionary containing paper metadata or None if not found or metadata db not available
        """
        if self.metadata_db is None:
            logger.warning("Metadata database not available")
            return None
            
        metadata = self.metadata_db.get_metadata(doc_id)
        logger.debug(f"Retrieved metadata for {doc_id}: {metadata is not None}")
        return metadata

    def delete_paper(self, doc_id: str) -> Dict[str, bool]:
        """Delete a paper and all its associated data from available databases.
        
        Args:
            doc_id: The document ID of the paper to delete
            
        Returns:
            Dictionary indicating deletion status for each database type
        """
        deletion_status = {
            "metadata": True,
            "text_chunks": True,   # 文本块存储状态（新增）
            "vectors": True,
            "images": True
        }
        
        try:
            logger.debug(f"Deleting paper {doc_id}")
            
            # Get metadata if available to find associated chunks and images
            metadata = None
            if self.metadata_db is not None:
                metadata = self.metadata_db.get_metadata(doc_id)
            
            # Delete from vector database if available
            if self.vector_db is not None:
                try:
                    success = self.vector_db.delete_document(doc_id)
                    deletion_status["vectors"] = success
                    logger.debug(f"Deleted from vector DB: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete from vector DB: {str(e)}")
            
            # Delete images if available and metadata exists
            if self.image_db is not None and metadata and metadata.get("figure_ids"):
                try:
                    success = self.image_db.delete_doc_images(doc_id)
                    deletion_status["images"] = success
                    logger.debug(f"Deleted images: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete images: {str(e)}")
            
            # Delete from metadata database if available
            if self.metadata_db is not None:
                try:
                    success = self.metadata_db.delete_paper(doc_id)
                    deletion_status["metadata"] = success[0]
                    deletion_status["text_chunks"] = success[1]
                    logger.debug(f"Deleted from metadata DB: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete from metadata DB: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to delete paper {doc_id}: {str(e)}")
        
        print('Deletion status:', deletion_status)
        return deletion_status

'''
if __name__ == "__main__":
    try:
        indexer = PaperIndexer()
        print("PaperIndexer initialized successfully")
    finally:
        # Explicitly clean up the embedding model
        if hasattr(indexer, 'embedding_model') and indexer.embedding_model is not None:
            try:
                indexer.embedding_model.model.stop_self_pool()
            except Exception:
                pass
            indexer.embedding_model.model = None
'''