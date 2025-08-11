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
                    "vectors": False,
                    "images": False
                }
                
                # Create metadata
                metadata = {
                    "doc_id": paper.doc_id,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_date": paper.published_date,
                    "chunk_ids": [chunk.id for chunk in paper.text_chunks],
                    "figure_ids": [chunk.id for chunk in paper.figure_chunks]
                }
                
                # Store metadata if database is available
                if self.metadata_db is not None and hasattr(paper, 'pdf_path'):
                    try:
                        success = self.metadata_db.save_paper(paper.doc_id, paper.pdf_path, metadata)
                        paper_status["metadata"] = success
                        logger.debug(f"Stored paper metadata for {paper.doc_id}: {success}")
                    except Exception as e:
                        logger.error(f"Failed to store metadata for {paper.doc_id}: {str(e)}")
                
                # Store vectors if database is available
                if self.vector_db is not None:
                    try:
                        text_chunks = [chunk.text for chunk in paper.text_chunks]
                        success = self.vector_db.add_document(
                            doc_id=paper.doc_id,
                            abstract=paper.abstract,
                            text_chunks=text_chunks,
                            metadata=metadata
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
        similarity_cutoff: float = 0.5,
        strategy_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find papers similar to the query using the selected search strategy.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            similarity_cutoff: Minimum similarity score to include in results
            strategy_type: Optional strategy to use for this search ('vector', 'tf-idf', 'hybrid')
            
        Returns:
            List of dictionaries containing paper information and similarity scores
            
        Raises:
            ValueError: If no search strategy is available or required database is missing
        """
        try:
            logger.debug(f"Searching for query: {query}")
            
            # Ensure we have required databases for the strategy
            if strategy_type == 'vector' and self.vector_db is None:
                raise ValueError("Vector database is required for vector search")
            elif strategy_type == 'tf-idf' and self.metadata_db is None:
                raise ValueError("Metadata database is required for TF-IDF search")
            elif strategy_type == 'hybrid' and (self.vector_db is None or self.metadata_db is None):
                raise ValueError("Both vector and metadata databases are required for hybrid search")
            
            # Default to vector search if available, otherwise TF-IDF
            if self.search_strategy is None and strategy_type is None:
                if self.vector_db is not None:
                    strategy_type = 'vector'
                elif self.metadata_db is not None:
                    strategy_type = 'tf-idf'
                else:
                    raise ValueError("No search strategy available - requires either vector or metadata database")
            
            # Set or temporarily change strategy
            original_strategy = self.search_strategy
            if strategy_type:
                self.set_search_strategy(strategy_type)
            
            # Perform search
            search_results = self.search_strategy.search(
                query=query,
                top_k=top_k,
                filters=filters,
                similarity_cutoff=similarity_cutoff
            )
            #print(111)
            #print(search_results)
            # Process results and add full metadata
            processed_results = []
            for result in search_results:
                if self.metadata_db is not None:
                    metadata = self.metadata_db.get_metadata(result.doc_id)
            #        print('METADATA')
            #        print(metadata)
            #        print('--------------------------------')
                    if metadata:
                        paper_info = metadata.copy()
                        paper_info["similarity_score"] = result.score
                        paper_info["matched_text"] = result.matched_text
                        paper_info["search_method"] = result.search_method
                        if result.chunk_id:
                            paper_info["chunk_id"] = result.chunk_id
                        
                        processed_results.append(paper_info)
                        logger.debug(f"Added result: {paper_info['title']} (score: {result.score})")
                    else:
                        logger.warning(f"No metadata found for doc_id: {result.doc_id}")
                else:
                    # If no metadata db, return basic result info
                    processed_results.append({
                        "doc_id": result.doc_id,
                        "similarity_score": result.score,
                        "matched_text": result.matched_text,
                        "search_method": result.search_method,
                        "chunk_id": result.chunk_id
                    })
            
            # Restore original strategy if temporarily changed
            if strategy_type and original_strategy is not None:
                self.search_strategy = original_strategy
                
            logger.debug(f"Returning {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to find similar papers: {str(e)}")
            raise RuntimeError(f"Failed to find similar papers: {str(e)}")

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
            "metadata": False,
            "vectors": False,
            "images": False
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
                    deletion_status["metadata"] = success
                    logger.debug(f"Deleted from metadata DB: {success}")
                except Exception as e:
                    logger.error(f"Failed to delete from metadata DB: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to delete paper {doc_id}: {str(e)}")
            
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