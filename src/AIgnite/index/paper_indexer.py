from typing import List, Optional, Dict, Any
from .base_indexer import BaseIndexer
from ..data.docset import DocSet
from ..db.vector_db import VectorDB
from ..db.metadata_db import MetadataDB
from ..db.image_db import MinioImageDB
from .search_strategy import SearchStrategy, VectorSearchStrategy, TFIDFSearchStrategy, HybridSearchStrategy, SearchResult
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
        metadata_db: MetadataDB = None,
        image_db: Optional[MinioImageDB] = None
    ):
        """Initialize the paper indexer with required metadata_db and optional database instances.
        Args:
            vector_db: Optional VectorDB instance for text embeddings
            metadata_db: Required MetadataDB instance for document metadata
            image_db: Optional MinioImageDB instance for storing figures
        Raises:
            ValueError: If metadata_db is None
        """
        #if metadata_db is None:
        #    raise ValueError("metadata_db is required for PaperIndexer initialization.")
        logger.debug("Initializing PaperIndexer")
        self.vector_db = vector_db
        self.metadata_db = metadata_db
        self.image_db = image_db
        self.search_strategy = None

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
            metadata_db: Optional MetadataDB instance for document metadata (must not be None)
            image_db: Optional MinioImageDB instance for storing figures
        Raises:
            ValueError: If metadata_db is set to None
        """
        if vector_db is not None:
            self.vector_db = vector_db
            logger.debug("Vector database set")
        if metadata_db is not None:
            self.metadata_db = metadata_db
            logger.debug("Metadata database set")
        #elif metadata_db is None:
        #    raise ValueError("metadata_db cannot be set to None for PaperIndexer.")
        if image_db is not None:
            self.image_db = image_db
            logger.debug("Image database set")

    def set_search_strategy(self, strategy_type: str) -> None:
        """Set the search strategy to use.
        
        Args:
            strategy_type: One of 'vector', 'tf-idf', or 'hybrid'
            
        Raises:
            ValueError: If strategy_type is invalid or required database is not available
        """
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
            
        logger.debug(f"Search strategy set to: {strategy_type}")

    def index_papers(self, documents: List[DocSet]) -> Dict[str, Dict[str, bool]]:
        """Index a list of documents into available databases.
        Args:
            documents: List of DocSet objects containing document information
        Returns:
            Dictionary mapping doc_ids to their indexing status for each database type
        """
        indexing_status = {}
        try:
            for document in tqdm(documents, desc="Indexing documents", unit="document"):
                doc_status = {
                    "metadata": False,
                    "vectors": False,
                    "images": False
                }
                metadata = {
                    "title": document.title,
                    "abstract": document.abstract,
                    "authors": document.authors,
                    "categories": document.categories,
                    "published_date": document.published_date,
                    "chunk_ids": [chunk.id for chunk in document.text_chunks],
                    "figure_ids": [chunk.id for chunk in document.figure_chunks]
                }
                if self.metadata_db is not None and hasattr(document, 'pdf_path'):
                    try:
                        success = self.metadata_db.add_document(document.doc_id, document.pdf_path, metadata)
                        doc_status["metadata"] = success
                        logger.debug(f"Stored document metadata for {document.doc_id}: {success}")
                    except Exception as e:
                        logger.error(f"Failed to store metadata for {document.doc_id}: {str(e)}")
                if self.vector_db is not None:
                    try:
                        text_chunks = [chunk.text for chunk in document.text_chunks]
                        success = self.vector_db.add_document(
                            doc_id=document.doc_id,
                            abstract=document.abstract,
                            text_chunks=text_chunks,
                            metadata=metadata
                        )
                        doc_status["vectors"] = success
                        logger.debug(f"Added vectors for {document.doc_id}: {success}")
                        if success:
                            save_success = self.vector_db.save()
                            if not save_success:
                                logger.error(f"Failed to save vector database after adding {document.doc_id}")
                                doc_status["vectors"] = False
                    except Exception as e:
                        logger.error(f"Failed to store vectors for {document.doc_id}: {str(e)}")
                if self.image_db is not None and document.figure_chunks:
                    try:
                        image_successes = []
                        for figure in tqdm(document.figure_chunks, desc=f"Storing figures for {document.doc_id}", leave=False):
                            success = self.image_db.save_image(
                                doc_id=document.doc_id,
                                image_id=figure.id,
                                image_path=figure.image_path
                            )
                            image_successes.append(success)
                            logger.debug(f"Saved image {figure.id} for {document.doc_id}: {success}")
                        doc_status["images"] = all(image_successes)
                    except Exception as e:
                        logger.error(f"Failed to store images for {document.doc_id}: {str(e)}")
                indexing_status[document.doc_id] = doc_status
        except Exception as e:
            logger.error(f"Failed to index documents: {str(e)}")
            raise RuntimeError(f"Failed to index documents: {str(e)}")
        return indexing_status

    def find_similar_papers(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5,
        strategy_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find documents similar to the query using the selected search strategy.
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            similarity_cutoff: Minimum similarity score to include in results
            strategy_type: Optional strategy to use for this search ('vector', 'tf-idf', 'hybrid')
        Returns:
            List of dictionaries containing document information and similarity scores
        Raises:
            ValueError: If no search strategy is available or required database is missing
        """
        try:
            logger.debug(f"Searching for query: {query}")
            if strategy_type == 'vector' and self.vector_db is None:
                raise ValueError("Vector database is required for vector search")
            elif strategy_type == 'tf-idf' and self.metadata_db is None:
                raise ValueError("Metadata database is required for TF-IDF search")
            elif strategy_type == 'hybrid' and (self.vector_db is None or self.metadata_db is None):
                raise ValueError("Both vector and metadata databases are required for hybrid search")
            if self.search_strategy is None and strategy_type is None:
                if self.vector_db is not None:
                    strategy_type = 'vector'
                elif self.metadata_db is not None:
                    strategy_type = 'tf-idf'
                else:
                    raise ValueError("No search strategy available - requires either vector or metadata database")
            original_strategy = self.search_strategy
            if strategy_type:
                self.set_search_strategy(strategy_type)
            search_results = self.search_strategy.search(
                query=query,
                top_k=top_k,
                filters=filters,
                similarity_cutoff=similarity_cutoff
            )
            processed_results = []
            for result in search_results:
                if self.metadata_db is not None:
                    metadata = self.metadata_db.get_document(result.doc_id)
                    if metadata:
                        doc_info = metadata.copy()
                        doc_info["similarity_score"] = result.score
                        doc_info["matched_text"] = result.matched_text
                        doc_info["search_method"] = result.search_method
                        if result.chunk_id:
                            doc_info["chunk_id"] = result.chunk_id
                        processed_results.append(doc_info)
                        logger.debug(f"Added result: {doc_info['title']} (score: {result.score})")
                    else:
                        logger.warning(f"No metadata found for doc_id: {result.doc_id}")
                else:
                    processed_results.append({
                        "doc_id": result.doc_id,
                        "similarity_score": result.score,
                        "matched_text": result.matched_text,
                        "search_method": result.search_method,
                        "chunk_id": result.chunk_id
                    })
            if strategy_type and original_strategy is not None:
                self.search_strategy = original_strategy
            logger.debug(f"Returning {len(processed_results)} results")
            return processed_results
        except Exception as e:
            logger.error(f"Failed to find similar documents: {str(e)}")
            raise RuntimeError(f"Failed to find similar documents: {str(e)}")

    def get_paper_metadata(self, doc_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific document.
        Args:
            doc_id: The document ID of the document
        Returns:
            Dictionary containing document metadata or None if not found or metadata db not available
        """
        if self.metadata_db is None:
            logger.warning("Metadata database not available")
            return None
        metadata = self.metadata_db.get_document(doc_id)
        logger.debug(f"Retrieved metadata for {doc_id}: {metadata is not None}")
        return metadata

    def delete_paper(self, doc_id: str) -> Dict[str, bool]:
        """Delete a document and all its associated data from available databases.
        Args:
            doc_id: The document ID of the document to delete
        Returns:
            Dictionary indicating deletion status for each database type
        """
        deletion_status = {
            "metadata": False,
            "vectors": False,
            "images": False
        }
        try:
            logger.debug(f"Deleting document {doc_id}")
            metadata = None
            if self.metadata_db is not None:
                metadata = self.metadata_db.get_document(doc_id)
            if self.vector_db is not None:
                try:
                    success = self.vector_db.delete_document(doc_id)
                    deletion_status["vectors"] = success
                    logger.debug(f"Deleted from vector DB: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete from vector DB: {str(e)}")
            if self.image_db is not None and metadata and metadata.get("figure_ids"):
                try:
                    success = self.image_db.delete_doc_images(doc_id)
                    deletion_status["images"] = success
                    logger.debug(f"Deleted images: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete images: {str(e)}")
            if self.metadata_db is not None:
                try:
                    success = self.metadata_db.delete_document(doc_id)
                    deletion_status["metadata"] = success
                    logger.debug(f"Deleted from metadata DB: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete from metadata DB: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {str(e)}")
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