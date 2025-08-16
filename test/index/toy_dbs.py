"""Toy database implementations for testing."""
from typing import List, Dict, Any, Optional, Tuple
import io
from dataclasses import dataclass
from AIgnite.db.vector_db import VectorDB, VectorEntry
from AIgnite.db.metadata_db import MetadataDB
from AIgnite.db.image_db import MinioImageDB
from AIgnite.index.filter_parser import FilterParser
import logging
import sys
import traceback
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.schema import TextNode, NodeWithScore
import numpy as np
import faiss
import os

logger = logging.getLogger(__name__)

@dataclass
class VectorEntry:
    """Class for storing vector database entries."""
    doc_id: str
    text: str
    text_type: str  # 'abstract' or 'chunk' or 'combined'
    chunk_id: Optional[str] = None

class ToyVectorDB(VectorDB):
    """A toy vector database for testing that stores everything in memory."""
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5', vector_dim: int = 768):
        """Initialize toy vector database with embedding model.
        
        Args:
            model_name: Name of the embedding model to use
            vector_dim: Dimension of the embedding vectors
        """
        # Pass a temporary path to parent since it requires one, though we won't use it
        super().__init__("/tmp/toy_vector_db", model_name, vector_dim)
        self.documents = {}  # doc_id -> {text_chunks, metadata}
        self.entries = []    # List[VectorEntry]
        self.index = faiss.IndexFlatIP(self.vector_dim)  # Initialize FAISS index

    def add_document(self, doc_id: str, abstract: str, text_chunks: List[str], metadata: Dict[str, Any]) -> bool:
        """Store document in memory and update the index."""
        try:
            # Store basic document info
            self.documents[doc_id] = {
                'abstract': abstract,
                'text_chunks': text_chunks,
                'metadata': metadata
            }
            
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
            chunk_ids = metadata.get('chunk_ids', [])  # Use 'chunk_ids' instead of 'text_chunk_ids'
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
            error_msg = f"Failed to add document: {str(e)}\n"
            error_msg += f"Full traceback: {''.join(traceback.format_exception(*sys.exc_info()))}"
            logger.error(error_msg)
            return False
        
    def search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[VectorEntry, float]]:
        """Search using FAISS index."""
        try:
            if not self.entries:
                return []
            
            # Get query embedding
            
            query_vector = self._get_embedding(query)
            
            logger.debug(f"Query vector shape: {query_vector.shape}")
            
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
                
                # Apply filters if provided
                if filters:
                    if "include" in filters or "exclude" in filters:
                        # New filter structure - apply memory filtering
                        filter_parser = FilterParser()
                        
                        # Get metadata for filtering (if available)
                        def get_field_value(item, field):
                            if field == "doc_ids":
                                return item.doc_id
                            # For other fields, we'd need metadata - for now, skip complex filtering
                            return None
                        
                        # Apply filters to this entry
                        if not filter_parser.apply_memory_filters([entry], filters, get_field_value):
                            continue
                    elif "doc_ids" in filters:
                        # Backward compatibility
                        allowed_doc_ids = set(filters["doc_ids"])
                        if entry.doc_id not in allowed_doc_ids:
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
            logger.error(f"Search failed: {str(e)}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from memory and index."""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                
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
            return False
        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False

class ToyMetadataDB(MetadataDB):
    """A toy metadata database for testing that stores everything in memory."""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self.metadata = {}  # doc_id -> metadata
        self.pdfs = {}      # doc_id -> pdf_content
        
    def save_paper(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any]) -> bool:
        """Store paper metadata and PDF content in memory."""
        try:
            # Read and store PDF content
            with open(pdf_path, 'rb') as f:
                self.pdfs[doc_id] = f.read()
            # Store metadata
            self.metadata[doc_id] = metadata.copy()  # Make a copy to avoid reference issues
            return True
        except Exception:
            return False
        
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a document."""
        metadata = self.metadata.get(doc_id)
        if metadata:
            # Ensure doc_id is included in the returned metadata
            metadata_with_doc_id = metadata.copy()
            metadata_with_doc_id['doc_id'] = doc_id
            return metadata_with_doc_id
        return None
        
    def get_pdf(self, doc_id: str, save_path: Optional[str] = None) -> Optional[bytes]:
        """Retrieve PDF content for a document."""
        pdf_data = self.pdfs.get(doc_id)
        if pdf_data and save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(pdf_data)
            return None
        return pdf_data
        
    def delete_paper(self, doc_id: str) -> bool:
        """Delete paper metadata and PDF content."""
        if doc_id in self.metadata:
            del self.metadata[doc_id]
            if doc_id in self.pdfs:
                del self.pdfs[doc_id]
            return True
        return False

    def search_papers(
        self,
        query: str,
        top_k: int = 10,
        similarity_cutoff: float = 0.1,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search papers using simple TF-IDF-like matching.
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            similarity_cutoff: Minimum similarity score to include in results
            filters: Optional filters to apply to search results
            
        Returns:
            List of paper metadata dictionaries with search scores
        """
        try:
            # Tokenize query
            query_terms = set(query.lower().split())
            
            # Calculate scores for each document
            results = []
            for doc_id, metadata in self.metadata.items():
                # Apply filters if provided
                if filters:
                    if "include" in filters or "exclude" in filters:
                        # New filter structure - apply memory filtering
                        
                        filter_parser = FilterParser()
                        
                        # Create a mock item for filtering
                        mock_item = {
                            'doc_id': doc_id,
                            'categories': metadata.get('categories', []),
                            'authors': metadata.get('authors', []),
                            'published_date': metadata.get('published_date', ''),
                            'title': metadata.get('title', ''),
                            'abstract': metadata.get('abstract', '')
                        }
                        
                        # Apply filters to this item
                        if not filter_parser.apply_memory_filters([mock_item], filters, lambda item, field: item.get(field)):
                            continue
                    elif "doc_ids" in filters:
                        # Backward compatibility
                        allowed_doc_ids = set(filters["doc_ids"])
                        if doc_id not in allowed_doc_ids:
                            continue
                
                # Create document text from title and abstract
                doc_text = f"{metadata['title']} {metadata['abstract']}"
                doc_terms = set(doc_text.lower().split())
                
                # Calculate simple TF-IDF-like score
                matching_terms = query_terms.intersection(doc_terms)
                if matching_terms:
                    score = len(matching_terms) / len(query_terms)  # Simple scoring
                    if score >= similarity_cutoff:
                        # Create search result
                        result = {
                            'doc_id': doc_id,
                            'score': score,
                            'metadata': metadata,
                            'matched_text': doc_text  # For simplicity, return full text
                        }
                        results.append(result)
            
            # Sort by score and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def add_blog(self, doc_id: str, blog: str) -> bool:
        """Save or update the blog text for a given document ID in memory."""
        if doc_id not in self.metadata:
            return False
        self.metadata[doc_id]['blog'] = blog
        return True

    def get_blog(self, doc_id: str) -> Optional[str]:
        """Retrieve the blog text for a given document ID from memory."""
        meta = self.metadata.get(doc_id)
        if not meta:
            return None
        return meta.get('blog')

class ToyImageDB(MinioImageDB):
    """A toy image database for testing that stores everything in memory."""
    def __init__(self, endpoint: str, access_key: str, secret_key: str, 
                 bucket_name: str, secure: bool = False):
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.images = {}  # (doc_id, image_id) -> image_content
        
    def save_image(self, doc_id: str, image_id: str, image_path: str = None, image_data: bytes = None) -> bool:
        """Store image content in memory."""
        try:
            if image_path:
                with open(image_path, 'rb') as f:
                    self.images[(doc_id, image_id)] = f.read()
            elif image_data:
                self.images[(doc_id, image_id)] = image_data
            else:
                raise ValueError("Either image_path or image_data must be provided")
            return True
        except Exception:
            return False
        
    def get_image(self, doc_id: str, image_id: str, save_path: Optional[str] = None) -> Optional[bytes]:
        """Retrieve image content from memory."""
        image_data = self.images.get((doc_id, image_id))
        if image_data and save_path:
            with open(save_path, 'wb') as f:
                f.write(image_data)
            return None
        return image_data
        
    def list_doc_images(self, doc_id: str) -> List[str]:
        """List all image IDs for a document."""
        return [img_id for (d_id, img_id) in self.images.keys() 
                if d_id == doc_id]
        
    def delete_doc_images(self, doc_id: str) -> bool:
        """Delete all images for a document from memory."""
        try:
            # Find all keys for this doc_id
            keys_to_delete = []
            for key in list(self.images.keys()):
                if key[0] == doc_id:  # key is a tuple of (doc_id, img_id)
                    keys_to_delete.append(key)
            
            # Delete all found images
            for key in keys_to_delete:
                del self.images[key]
            
            logger.debug(f"Deleted {len(keys_to_delete)} images for doc_id {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete images for {doc_id}: {str(e)}")
            return False 