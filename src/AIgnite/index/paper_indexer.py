from typing import List, Optional, Dict, Any
from .base_indexer import BaseIndexer
from ..data.docset import DocSet
from ..db.vector_db import VectorDB
from ..db.metadata_db import MetadataDB
from ..db.image_db import MinioImageDB
import logging
from tqdm import tqdm
import os

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class PaperIndexer(BaseIndexer):
    def __init__(self, vector_db: VectorDB, metadata_db: MetadataDB, image_db: MinioImageDB):
        """Initialize the paper indexer with pre-initialized database instances.
        
        Args:
            vector_db: Initialized VectorDB instance for text embeddings
            metadata_db: Initialized MetadataDB instance for paper metadata
            image_db: Initialized MinioImageDB instance for storing figures
        """
        logger.debug("Initializing PaperIndexer with pre-initialized databases")
        
        self.vector_db = vector_db
        self.metadata_db = metadata_db
        self.image_db = image_db

    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'vector_db'):
            self.vector_db = None
        if hasattr(self, 'metadata_db'):
            self.metadata_db = None
        if hasattr(self, 'image_db'):
            self.image_db = None

    def index_papers(self, papers: List[DocSet]) -> None:
        """Index a list of papers into the vector store.
        
        Args:
            papers: List of DocSet objects containing paper information
        """
        try:
            # Main progress bar for papers
            for paper in tqdm(papers, desc="Indexing papers", unit="paper"):
                # Create metadata
                metadata = {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_date": paper.published_date,
                    "chunk_ids": [chunk.id for chunk in paper.text_chunks],
                    "figure_ids": [chunk.id for chunk in paper.figure_chunks]
                }
                
                # Store PDF if available
                if hasattr(paper, 'pdf_path'):
                    success = self.metadata_db.save_paper(paper.doc_id, paper.pdf_path, metadata)
                    logger.debug(f"Stored paper and metadata for {paper.doc_id}: {success}")
                else:
                    logger.warning(f"No PDF file found for {paper.doc_id}")
                    continue
                
                # Store text chunks in vector database with progress bar
                text_chunks = [chunk.text for chunk in paper.text_chunks]
                success = self.vector_db.add_document(
                    doc_id=paper.doc_id,
                    abstract=paper.abstract,
                    text_chunks=text_chunks,
                    metadata=metadata
                )
                logger.debug(f"Added vectors for {paper.doc_id}: {success}")
                
                # Save vector database after successful addition
                if success:
                    save_success = self.vector_db.save()
                    logger.debug(f"Saved vector database after adding {paper.doc_id}: {save_success}")
                    if not save_success:
                        raise RuntimeError(f"Failed to save vector database after adding {paper.doc_id}")
                
                # Store figures in image database with progress bar
                if paper.figure_chunks:
                    for figure in tqdm(paper.figure_chunks, desc=f"Storing figures for {paper.doc_id}", leave=False):
                        success = self.image_db.save_image(
                            doc_id=paper.doc_id,
                            image_id=figure.id,
                            image_path=figure.image_path
                        )
                        logger.debug(f"Saved image {figure.id} for {paper.doc_id}: {success}")
                
        except Exception as e:
            logger.error(f"Failed to index papers: {str(e)}")
            # Clean up on failure
            self.delete_paper(paper.doc_id)
            raise RuntimeError(f"Failed to index papers: {str(e)}")

    def get_paper_metadata(self, doc_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific paper.
        
        Args:
            doc_id: The document ID of the paper
            
        Returns:
            Dictionary containing paper metadata or None if not found
        """
        metadata = self.metadata_db.get_metadata(doc_id)
        logger.debug(f"Retrieved metadata for {doc_id}: {metadata is not None}")
        return metadata

    def find_similar_papers(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Find papers similar to the query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional filters to apply to the search
            similarity_cutoff: Minimum similarity score to include in results (0.0 to 1.0)
            
        Returns:
            List of dictionaries containing paper information and similarity scores
        """
        try:
            logger.debug(f"Searching for query: {query}")
            
            # Search in vector database
            results = self.vector_db.search(query, k=top_k * 2)
            logger.debug(f"Got {len(results)} initial results")
            
            # Process and filter results
            processed_results = []
            seen_doc_ids = set()
            
            for entry, similarity_score in results:
                doc_id = entry.doc_id
                if doc_id in seen_doc_ids:
                    logger.debug(f"Skipping duplicate doc_id: {doc_id}")
                    continue
                
                # Get full metadata from metadata database
                metadata = self.metadata_db.get_metadata(doc_id)
                if metadata:
                    paper_info = metadata.copy()
                    paper_info["similarity_score"] = similarity_score
                    paper_info["matched_text"] = entry.text
                    paper_info["match_type"] = entry.text_type
                    if entry.chunk_id is not None:
                        paper_info["chunk_id"] = entry.chunk_id
                    
                    processed_results.append(paper_info)
                    seen_doc_ids.add(doc_id)
                    logger.debug(f"Added result: {paper_info['title']} (score: {similarity_score})")
                    
                    if len(processed_results) >= top_k:
                        break
                else:
                    logger.warning(f"No metadata found for doc_id: {doc_id}")
            
            # Sort by similarity score
            processed_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            logger.debug(f"Returning {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to find similar papers: {str(e)}")
            raise RuntimeError(f"Failed to find similar papers: {str(e)}")

    def delete_paper(self, doc_id: str) -> bool:
        """Delete a paper and all its associated data from all databases.
        
        Args:
            doc_id: The document ID of the paper to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.debug(f"Deleting paper {doc_id}")
            
            # Get metadata to find associated chunks and images
            metadata = self.metadata_db.get_metadata(doc_id)
            if metadata:
                # Delete from vector database
                success = self.vector_db.delete_document(doc_id)
                logger.debug(f"Deleted from vector DB: {success}")
                
                # Delete images if they exist
                if metadata.get("figure_ids"):
                    success = self.image_db.delete_doc_images(doc_id)
                    logger.debug(f"Deleted images for {doc_id}: {success}")
                
                # Delete from metadata database last
                success = self.metadata_db.delete_paper(doc_id)
                logger.debug(f"Deleted from metadata DB: {success}")
                
                return True
            else:
                logger.warning(f"No metadata found for doc_id: {doc_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete paper {doc_id}: {str(e)}")
            return False

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