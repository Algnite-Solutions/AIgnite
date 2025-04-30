<<<<<<< HEAD
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from typing import List, Optional, Dict, Any
import faiss
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from FlagEmbedding import FlagModel
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field

from llama_index.core import StorageContext
from AIgnite.index.base_indexer import BaseIndexer
from AIgnite.data.docset import DocSet


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
            'BAAI/bge-base-en-v1.5',
            use_fp16=True,  # Use half precision for faster inference
            device='cuda'  # Use GPU if available
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

    def index_papers(self, papers: List[DocSet]) -> None:
        """Index a list of papers into the vector store.
        
        Args:
            papers: List of DocSet objects containing paper information
        """
        try:
            for paper in papers:
                # Create document with metadata
                
                # Add text chunks as separate nodes
                for chunk in paper.chunks:
                    if chunk.type == "text":
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
        similarity_cutoff: float = 0.8
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
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            response = retriever.retrieve(query)
            # Process results
            results = []
            for node in response:
                # Skip results below similarity cutoff
                if node.score > similarity_cutoff:
                    continue
                    
                doc_id = node.metadata.get("doc_id")
                if doc_id and doc_id in self.metadata_store:
                    paper_info = self.metadata_store[doc_id].copy()
                    paper_info["similarity_score"] = node.score
                    paper_info["matched_text"] = node.text
                    results.append(paper_info)
            
            print(f"Found {len(results)} results for query: {query}")
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Failed to find similar papers: {str(e)}")


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
=======
from .base_indexer import BaseIndexer


class PaperIndexer(BaseIndexer):
    # TODO: @Fang, implement paper indexer
    pass
>>>>>>> 452a018494be4206c748e6559afa232fd2bef792
