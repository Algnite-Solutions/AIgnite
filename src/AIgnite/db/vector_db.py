from typing import List, Dict, Any, Optional, Tuple
import faiss
import numpy as np
from FlagEmbedding import FlagModel
import logging
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class VectorEntry:
    """Class for storing vector database entries."""
    doc_id: str
    text: str
    text_type: str  # 'abstract' or 'chunk' or 'combined'
    chunk_id: Optional[str] = None
    vector: Optional[np.ndarray] = None

class VectorDB:
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', vector_dim: int = 768):
        """Initialize vector database with embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            vector_dim: Dimension of the embedding vectors (default: 768 for BGE base model)
        """
        # Initialize embedding model
        self.model = FlagModel(model_name)
        self.vector_dim = vector_dim
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.vector_dim)  # Changed to Inner Product similarity
        
        # Store mapping of vector IDs to entries
        self.entries: List[VectorEntry] = []
        
    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'model'):
            try:
                self.model.stop_self_pool()
            except:
                pass
            self.model = None

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embedding vector
        """
        # Clean and normalize the text
        text = text.lower().strip()
        
        # Get embedding
        vector = self.model.encode(text)
        
        # Ensure proper shape
        if len(vector.shape) == 1:
            vector = vector.reshape(1, -1)
        
        # Convert to float32 for FAISS
        vector = vector.astype(np.float32)
        
        # Normalize the vector
        faiss.normalize_L2(vector)
        return vector

    def add_document(
        self,
        doc_id: str,
        abstract: str,
        text_chunks: List[str],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document and its chunks to the vector database.
        
        Args:
            doc_id: Document ID
            abstract: Document abstract
            text_chunks: List of text chunks
            metadata: Document metadata (not used for vectors)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create combined text for better document-level matching
            title = metadata.get('title', '')
            categories = ' '.join(metadata.get('categories', []))
            combined_text = f"{title} {categories} {abstract}"
            
            # Add combined vector
            combined_entry = VectorEntry(
                doc_id=doc_id,
                text=combined_text,
                text_type='combined'
            )
            combined_entry.vector = self._get_embedding(combined_text)
            self.entries.append(combined_entry)
            self.index.add(combined_entry.vector)
            
            # Add abstract vector
            abstract_entry = VectorEntry(
                doc_id=doc_id,
                text=abstract,
                text_type='abstract'
            )
            abstract_entry.vector = self._get_embedding(abstract)
            self.entries.append(abstract_entry)
            self.index.add(abstract_entry.vector)
            
            # Add text chunk vectors
            chunk_ids = metadata.get('text_chunk_ids', [])
            for chunk_text, chunk_id in zip(text_chunks, chunk_ids):
                chunk_entry = VectorEntry(
                    doc_id=doc_id,
                    text=chunk_text,
                    text_type='chunk',
                    chunk_id=chunk_id
                )
                chunk_entry.vector = self._get_embedding(chunk_text)
                self.entries.append(chunk_entry)
                self.index.add(chunk_entry.vector)
                
            return True
            
        except Exception as e:
            logging.error(f"Failed to add document {doc_id} to vector database: {str(e)}")
            return False

    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Tuple[VectorEntry, float]]:
        """Search for similar vectors.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of tuples containing (VectorEntry, similarity_score)
        """
        try:
            if not self.entries:
                return []
                
            # Get query embedding
            query_vector = self._get_embedding(query)
            
            # Search index with more results to account for deduplication
            k_search = min(k * 3, len(self.entries))  # Search for more results initially
            distances, indices = self.index.search(
                query_vector,
                k=k_search
            )
            
            # Process results with document-level deduplication
            seen_doc_ids = set()
            results = []
            
            for idx, score in zip(indices[0], distances[0]):
                if idx >= len(self.entries):  # Safety check
                    continue
                    
                entry = self.entries[idx]
                
                # Skip if we've already seen this document
                if entry.doc_id in seen_doc_ids:
                    continue
                    
                results.append((entry, float(score)))
                seen_doc_ids.add(entry.doc_id)
                
                # Break if we have enough unique documents
                if len(results) >= k:
                    break
            
            # Sort by similarity score (higher is better for inner product)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            logging.error(f"Failed to search vector database: {str(e)}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Delete all vectors for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find indices to remove
            indices_to_remove = []
            entries_to_keep = []
            vectors_to_keep = []
            
            for i, entry in enumerate(self.entries):
                if entry.doc_id == doc_id:
                    indices_to_remove.append(i)
                else:
                    entries_to_keep.append(entry)
                    vectors_to_keep.append(entry.vector)
            
            if not indices_to_remove:
                return False
                
            # Create new index with remaining vectors
            self.index = faiss.IndexFlatIP(self.vector_dim)
            
            if vectors_to_keep:
                vectors_array = np.vstack(vectors_to_keep)
                self.index.add(vectors_array)
            
            # Update entries
            self.entries = entries_to_keep
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete document {doc_id} from vector database: {str(e)}")
            return False 