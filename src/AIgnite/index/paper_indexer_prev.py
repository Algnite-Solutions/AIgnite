from typing import List, Optional, Dict, Any
import faiss
from llama_index.core import VectorStoreIndex, Document
from FlagEmbedding import FlagModel
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field

from llama_index.core import StorageContext
from .base_indexer import BaseIndexer
from ..data.docset import DocSet
import logging

logger = logging.getLogger(__name__)

class BGEEmbedding(BaseEmbedding):
    """Wrapper for BGE model to implement BaseEmbedding interface."""
    
    model: Any = Field(default=None, description="The BGE model instance")
    
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', **kwargs):
        super().__init__()
        self.model = FlagModel(model_name, **kwargs)
        self._dimension = 768  # BGE base model dimension
        
    def __del__(self):
        """Cleanup method to properly close the model."""
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.stop_self_pool()
            except Exception:
                pass
            self.model = None
        
    @property
    def dimensions(self) -> int:
        return self._dimension
        
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()
        
    def _get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
        
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
        
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


class PaperIndexer(BaseIndexer):
    def __init__(self, embedding_model=None):
        # Use BGE model by default
        self.embedding_model = embedding_model or BGEEmbedding(
            'BAAI/bge-base-en-v1.5'
        )
        # Create a FAISS index
        dimension = self.embedding_model.dimensions
        print(f"Using embedding dimension: {dimension}")
        faiss_index = faiss.IndexFlatL2(dimension)
        
        # Initialize FAISS vector store with the index
        self.vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # Create storage context with the vector store
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        ) 
        # Create an initial document
        initial_doc = Document(
            text="Initial document",
            metadata={"type": "initial"}
        )
        # Create the index with the initial document
        self.index = VectorStoreIndex.from_documents(
            [initial_doc],
            storage_context=storage_context,
            embed_model=self.embedding_model
        )
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.search_strategy = None

    def index_papers(self, papers: List[DocSet]) -> None:
        """Index a list of papers into the vector store.
        
        Args:
            papers: List of DocSet objects containing paper information
        """
        try:
            for paper in papers:
                # Create document with metadata
                
                # Add text chunks as separate nodes
                for chunk in paper.text_chunks:
                    doc_chunk = Document(
                        text=chunk.text,
                        metadata={
                            "doc_id": paper.doc_id,
                            "title": paper.title,
                            "authors": paper.authors,
                            "categories": paper.categories,
                            "published_date": paper.published_date
                        }
                    )
                    self.index.insert(doc_chunk)
                # Store full metadata
                self.metadata_store[paper.doc_id] = {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_date": paper.published_date
                }
            
            
            
        except Exception as e:
            raise RuntimeError(f"Failed to index papers: {str(e)}")

    def get_paper_metadata(self, doc_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific paper.
        
        Args:
            doc_id: The document ID of the paper
            
        Returns:
            Dictionary containing paper metadata or None if not found
        """
        return self.metadata_store.get(doc_id)

    def find_similar_papers(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None,
        similarity_cutoff: float = 0.8,
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
            raise RuntimeError(f"Failed to find similar papers: {str(e)}")

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