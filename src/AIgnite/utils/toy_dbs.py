from typing import List, Dict, Any, Optional, Tuple
import io
from dataclasses import dataclass
from ..db.vector_db import VectorDB, VectorEntry
from ..db.metadata_db import MetadataDB
from ..db.image_db import MinioImageDB

@dataclass
class VectorEntry:
    """Class for storing vector database entries."""
    doc_id: str
    text: str
    text_type: str  # 'abstract' or 'chunk' or 'combined'
    chunk_id: Optional[str] = None

class ToyVectorDB(VectorDB):
    """A toy vector database for testing that stores everything in memory."""
    def __init__(self, model_name: str = 'BAAI/bge-base-en-v1.5'):
        self.model_name = model_name
        self.documents = {}  # doc_id -> {text_chunks, metadata}
        self.entries = []    # List[VectorEntry]
        
    def add_document(self, doc_id: str, abstract: str, text_chunks: List[str], metadata: Dict[str, Any]) -> bool:
        """Store document in memory."""
        try:
            self.documents[doc_id] = {
                'abstract': abstract,
                'text_chunks': text_chunks,
                'metadata': metadata
            }
            
            # Create entries similar to real VectorDB
            # Combined entry
            title = metadata.get('title', '')
            categories = ' '.join(metadata.get('categories', []))
            combined_text = f"{title} {categories} {abstract}"
            self.entries.append(VectorEntry(
                doc_id=doc_id,
                text=combined_text,
                text_type='combined'
            ))
            
            # Abstract entry
            self.entries.append(VectorEntry(
                doc_id=doc_id,
                text=abstract,
                text_type='abstract'
            ))
            
            # Chunk entries
            for i, chunk in enumerate(text_chunks):
                self.entries.append(VectorEntry(
                    doc_id=doc_id,
                    text=chunk,
                    text_type='chunk',
                    chunk_id=str(i)
                ))
            return True
        except Exception:
            return False
        
    def search(self, query: str, k: int = 5) -> List[Tuple[VectorEntry, float]]:
        """Simple mock search using basic text matching."""
        results = []
        query_terms = query.lower().split()
        
        for entry in self.entries:
            # Simple text matching (not real vector similarity)
            score = sum(1 for term in query_terms if term in entry.text.lower())
            if score > 0:
                results.append((entry, score / len(query_terms)))
        
        # Sort by score and deduplicate by doc_id
        results.sort(key=lambda x: x[1], reverse=True)
        seen_doc_ids = set()
        filtered_results = []
        
        for entry, score in results:
            if entry.doc_id not in seen_doc_ids:
                filtered_results.append((entry, score))
                seen_doc_ids.add(entry.doc_id)
                if len(filtered_results) >= k:
                    break
        
        return filtered_results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from memory."""
        try:
            if doc_id in self.documents:
                del self.documents[doc_id]
                self.entries = [e for e in self.entries if e.doc_id != doc_id]
                return True
            return False
        except Exception:
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
            self.metadata[doc_id] = metadata
            return True
        except Exception:
            return False
        
    def get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve metadata from memory."""
        metadata = self.metadata.get(doc_id)
        if metadata:
            return {
                'doc_id': doc_id,
                'title': metadata['title'],
                'abstract': metadata['abstract'],
                'authors': metadata['authors'],
                'categories': metadata['categories'],
                'published_date': metadata['published_date'],
                'chunk_ids': metadata.get('chunk_ids', []),
                'image_ids': metadata.get('image_ids', [])
            }
        return None
        
    def get_pdf(self, doc_id: str, save_path: Optional[str] = None) -> Optional[bytes]:
        """Retrieve PDF content and optionally save to file."""
        pdf_content = self.pdfs.get(doc_id)
        if pdf_content and save_path:
            with open(save_path, 'wb') as f:
                f.write(pdf_content)
            return None
        return pdf_content
        
    def delete_paper(self, doc_id: str) -> bool:
        """Delete paper metadata and PDF content from memory."""
        try:
            if doc_id in self.metadata:
                del self.metadata[doc_id]
                del self.pdfs[doc_id]
                return True
            return False
        except Exception:
            return False

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
            keys_to_delete = [(d_id, img_id) for (d_id, img_id) 
                            in self.images.keys() if d_id == doc_id]
            for key in keys_to_delete:
                del self.images[key]
            return True
        except Exception:
            return False 