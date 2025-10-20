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
       

    def index_papers(self, papers: List[DocSet],store_images: bool = False,keep_temp_image: bool = False) -> Dict[str, Dict[str, bool]]:
        """Index a list of papers into available databases.
        
        Args:
            papers: List of DocSet objects containing paper information
            store_images: Whether to store images to MinIO (default: False)
            
        Returns:
            Dictionary mapping doc_ids to their indexing status for each database type
        """
        indexing_status = {}
        vector_add_results = {}  # 记录向量添加结果
        
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
                    "image_storage": {f"{paper.doc_id}_{chunk.id}": False for chunk in paper.figure_chunks},
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
                            vector_db_id=paper.doc_id + '_abstract',
                            text_to_emb=paper.title + ' . ' + paper.abstract,
                            doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
                        )
                        vector_add_results[paper.doc_id] = success  # 记录添加结果
                        paper_status["vectors"] = success
                        logger.debug(f"Added vectors for {paper.doc_id}: {success}")
                        
                    except Exception as e:
                        logger.error(f"Failed to add vectors for {paper.doc_id}: {str(e)}")
                        vector_add_results[paper.doc_id] = False
                        paper_status["vectors"] = False
                
                # Store images if database is available and paper has figures
                if store_images and self.image_db is not None and paper.figure_chunks:
                    try:
                        image_successes = []
                        for figure in tqdm(paper.figure_chunks, desc=f"Storing figures for {paper.doc_id}", leave=False):
                            success = self._save_image(figure.image_path, paper.doc_id+'_'+figure.id,keep_temp_image)
                            image_successes.append(success)
                            logger.debug(f"Saved image {figure.id} for {paper.doc_id}: {success}")
                        paper_status["images"] = all(image_successes)
                    except Exception as e:
                        logger.error(f"Failed to store images for {paper.doc_id}: {str(e)}")
                        paper_status["images"] = False
                
                indexing_status[paper.doc_id] = paper_status
            
            # 所有论文处理完成后，统一保存向量数据库到磁盘
            if self.vector_db is not None and vector_add_results:
                try:
                    logger.info(f"Saving vector database with {len(vector_add_results)} papers to disk...")
                    save_success = self.vector_db.save()
                    
                    if not save_success:
                        logger.error(f"Failed to save vector database after adding {len(papers)} papers")
                        # 保存失败，更新所有论文的向量状态为 False
                        for doc_id in vector_add_results.keys():
                            if doc_id in indexing_status:
                                indexing_status[doc_id]["vectors"] = False
                    else:
                        logger.info(f"Successfully saved vector database with {len(vector_add_results)} papers")
                        
                except Exception as e:
                    logger.error(f"Failed to save vector database: {str(e)}")
                    # 保存出错，更新所有论文的向量状态为 False
                    for doc_id in vector_add_results.keys():
                        if doc_id in indexing_status:
                            indexing_status[doc_id]["vectors"] = False
                
        except Exception as e:
            logger.error(f"Failed to index papers: {str(e)}")
            raise RuntimeError(f"Failed to index papers: {str(e)}")
        
        return indexing_status

    def save_vectors(self, papers: List[DocSet], indexing_status: Dict[str, Dict[str, bool]] = None):
        """批量保存论文向量到向量数据库
        
        Args:
            papers: 论文列表
            indexing_status: 可选的索引状态字典，如果为None则自动创建
            
        Returns:
            indexing_status 字典，包含所有论文的向量保存状态
        """
        try:
            # 初始化 indexing_status（如果为 None）
            if indexing_status is None:
                indexing_status = {}
                for paper in papers:
                    indexing_status[paper.doc_id] = {
                        "metadata": False,
                        "text_chunks": False,
                        "vectors": False,
                        "images": False
                    }
            
            # 记录每篇论文的添加状态
            add_results = {}
            
            # 循环添加所有论文到内存
            for paper in papers:

                success = self.vector_db.add_document(
                    vector_db_id=paper.doc_id + '_abstract',
                    text_to_emb=paper.title + ' . ' + paper.abstract,
                    doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
                )
                if not success:
                    logger.error(f"Failed to add vectors for {paper.doc_id}")
                    add_results[paper.doc_id] = False
                    continue
                add_results[paper.doc_id] = success
                logger.debug(f"Added vectors for {paper.doc_id}: {success}")
            
            # 统一保存到磁盘（所有论文处理完后只保存一次）
            save_success = self.vector_db.save()
            
            if not save_success:
                logger.error(f"Failed to save vector database after adding {len(papers)} papers")
                # 如果保存失败，所有论文的向量状态都应该标记为失败
                for paper in papers:
                    indexing_status[paper.doc_id]["vectors"] = False
                return indexing_status
            
            # 保存成功，更新所有成功添加的论文状态
            for paper in papers:
                # 只有添加成功且保存成功的才标记为 True
                indexing_status[paper.doc_id]["vectors"] = add_results.get(paper.doc_id, False)
            
            logger.info(f"Successfully saved vectors for {len(papers)} papers to disk")
            return indexing_status
            
        except Exception as e:
            logger.error(f"Failed to save vectors: {str(e)}")
            raise RuntimeError(f"Failed to save vectors: {str(e)}")


    def store_images(self, papers: List[DocSet], indexing_status: Dict[str, Dict[str, bool]] = None, keep_temp_image: bool = False):
        """Store images from papers to MinIO storage.
        
        Args:
            papers: List of DocSet objects containing papers with figure_chunks
            indexing_status: Optional dictionary to track indexing status
            keep_temp_image: If False, delete temporary image files after successful storage
            
        Returns:
            indexing_status dictionary with updated image storage status
        """
        try:
            for paper in papers:
                for figure in paper.figure_chunks:
                    success = self._save_image(figure.image_path, paper.doc_id+'_'+figure.id,keep_temp_image)
                    logger.debug(f"Saved image {figure.id} for {paper.doc_id}: {success}")
                    
                    """
                    # Delete temporary image file if storage was successful and keep_temp_image is False
                    if success and not keep_temp_image and figure.image_path:
                        try:
                            import os
                            if os.path.exists(figure.image_path):
                                os.remove(figure.image_path)
                                logger.debug(f"Deleted temporary image file: {figure.image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary image {figure.image_path}: {str(e)}")
                    """
                if indexing_status:
                    indexing_status[paper.doc_id]["images"] = True
        except Exception as e:
            logger.error(f"Failed to store images: {str(e)}")
            raise RuntimeError(f"Failed to store images: {str(e)}")
            return False
        return indexing_status
        
    '''
    The logic of saving image is:
    1. Save image to MinIO
    2. Update storage status in metadata database
    3. Return the result. If the result is True, the image is saved successfully or image is already in MinIO, and the storage status is updated in metadata database.
    '''    
    def _save_image(self, image_path: str, object_name: str,keep_temp_image: bool = False):
        if self.image_db is not None and image_path:
            try:
                minio_success = self.image_db.save_image(
                    object_name=object_name,
                    image_path=image_path
                )
                
                # Update storage status in metadata database
                if minio_success and self.metadata_db is not None:
                    # Extract docid and figureid from object_name (format: docid_figureid)
                    if '_' in object_name and object_name.count('_') == 1:
                        doc_id, figure_id = object_name.split('_', 1)
                        metadata_success = self.metadata_db.update_image_storage_status(doc_id, figure_id, True)
                        logger.debug(f"Updated storage status for {object_name}: minio_success={minio_success}, metadata_success={metadata_success}")
                    else:
                        logger.warning(f"Invalid object_name format: {object_name}")
                    if not keep_temp_image:
                        try:
                            import os
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                logger.debug(f"Deleted temporary image file: {image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary image {image_path}: {str(e)}")
                    
                elif not minio_success :
                    # Update storage status to False if save failed
                    if '_' in object_name:
                        doc_id, figure_id = object_name.split('_', 1)
                        metadata_success = self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
                        logger.debug(f"Updated storage status for {object_name}: minio_success={minio_success}, metadata_success={metadata_success}")
                
                return minio_success and metadata_success
            except Exception as e:
                logger.error(f"Failed to save image for {object_name}: {str(e)}")
                # Update storage status to False on error
                if self.metadata_db is not None and '_' in object_name:
                    doc_id, figure_id = object_name.split('_', 1)
                    self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
                return False
        else:
            logger.error(f"Failed to save image for {object_name}: image_db is None or image_path is empty")
            return False

    def _get_image(self, object_name: str):
        if self.image_db is not None:
            try:
                return self.image_db.get_image(object_name)
            except Exception as e:
                logger.error(f"Failed to get image for {object_name}: {str(e)}")
                return None
        else:
            logger.error(f"Failed to get image for {object_name}: image_db is None")
            return None


    def _delete_image(self, image_id: str):
        if self.image_db is not None:
            try:
                [doc_id, figure_id] = image_id.split("_")
                
                # Delete from MinIO storage
                minio_success = self.image_db.delete_image(image_id)
                
                # Update metadata database
                metadata_success = False
                if self.metadata_db is not None:
                    # Also update storage status to False
                    metadata_success = self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
                    logger.debug(f"Updated storage status for {image_id}: stored=False")
                
                return minio_success and metadata_success
            except Exception as e:
                logger.error(f"Failed to delete image for {image_id}: {str(e)}")
                # Update storage status to False on error
                if self.metadata_db is not None and '_' in image_id:
                    doc_id, figure_id = image_id.split('_', 1)
                    self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
                return False
        else:
            logger.error(f"Failed to delete image for {image_id}: image_db is None")
            return False

    def _list_image_ids(self, doc_id: str):
        """
        List all image IDs for a document. 

        Args:
            doc_id: Document ID

        Returns:
            List of image IDs

        Please Note: The image IDs are in the format of {doc_id}_{figure_id}. It does not represent the image storage status.
        """
        if self.metadata_db is not None:
            try:
                return self.metadata_db.get_image_ids(doc_id)
            except Exception as e:
                logger.error(f"Failed to list images for {doc_id}: {str(e)}")
                return []
        else:
            logger.error(f"Failed to list images for {doc_id}: image_db is None or metadata_db is None")
            return []

    def get_image_storage_status_for_doc(self, doc_id: str) -> Dict[str, bool]:
        """Get storage status for all figures in a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dictionary mapping figure_id to storage status
        """
        if self.metadata_db is not None:
            try:
                return self.metadata_db.get_image_storage_status_for_doc(doc_id)
            except Exception as e:
                logger.error(f"Failed to get figure storage status for {doc_id}: {str(e)}")
                return {}
        else:
            logger.error(f"Failed to get figure storage status for {doc_id}: metadata_db is None")
            return {}

    def _delete_images_by_doc_id(self, doc_id: str):
        if self.metadata_db is not None:
            #image_ids = self._list_image_ids(doc_id)
            image_storage_status=self.get_image_storage_status_for_doc(doc_id)
            image_ids = [image_id if image_storage_status[image_id] else None for image_id in image_storage_status.keys()]
            print('IN _delete_images_by_doc_id')
            print(image_storage_status)
            print(image_ids)
            for image_id in image_ids:
                self._delete_image(image_id)
            print(f"Deleted {len(image_ids)} images for {doc_id}")
            return True
        else:
            logger.error(f"Failed to delete images for {doc_id}: metadata_db is None")
            return False

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
            logger.debug(f"Searching for query: {query}, filters: {filters is not None}")
            
            # 1. 预过滤：如果有filters，先通过metadata_db获取候选doc_ids
            if filters and self.metadata_db is None:
                raise ValueError("Metadata database is required for filtering")
            
            simplified_filters = None
            if filters:
                candidate_doc_ids = self.metadata_db.get_filtered_doc_ids(filters)
                
                if not candidate_doc_ids:
                    logger.info("No documents match the filters, returning empty results")
                    return []
                
                logger.info(f"Filter applied: {len(candidate_doc_ids)} candidate documents")
                
                # 将候选doc_ids转换为简化的filter格式
                #vector_search_candidate_doc_ids = [docid+'_abstract' for docid in candidate_doc_ids]
                simplified_filters = {"include": {"doc_ids": candidate_doc_ids}}
                
            else:
                logger.debug("No filters provided")
            
            # 2. 执行搜索（使用简化后的filters）
            search_results = self._execute_search(query, top_k, simplified_filters, search_strategies)
            
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
                    success = self._delete_images_by_doc_id(doc_id)
                    deletion_status["images"] = success
                    logger.debug(f"Deleted images: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete images: {str(e)}")
                    deletion_status["images"] = False
            
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