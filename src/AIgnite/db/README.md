# Feature: Database Layer for Paper Indexing

## Description
This module provides the database abstraction and implementations for storing, searching, and managing academic paper data. It includes classes for handling vector-based semantic search, metadata storage using SQL databases, and image/figure storage using MinIO-compatible object storage. Each class exposes a clear API for integration with the rest of the system.

## Entry Point
- `AIgnite/src/AIgnite/db/vector_db.py`
  - Class: `VectorDB`
    - `__init__(self, db_path: str, model_name: str = ..., vector_dim: int = 768)`
    - `save(self) -> bool`
    - `load(self) -> bool`
    - `exists(self) -> bool`
    - `add_document(self, doc_id: str, abstract: str, text_chunks: List[str], metadata: Dict[str, Any]) -> bool`
    - `search(self, query: str, k: int = 5) -> List[Tuple[VectorEntry, float]]`
    - `delete_document(self, doc_id: str) -> bool`
- `AIgnite/src/AIgnite/db/metadata_db.py`
  - Class: `MetadataDB`
    - `__init__(self, db_path: str = None)`
    - `save_paper(self, doc_id: str, pdf_path: str, metadata: Dict[str, Any]) -> bool`
    - `get_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]`
    - `get_pdf(self, doc_id: str, save_path: str = None) -> Optional[bytes]`
    - `delete_paper(self, doc_id: str) -> bool`
    - `save_blog(self, doc_id: str, blog: str) -> bool`
    - `get_blog(self, doc_id: str) -> Optional[str]`
- `AIgnite/src/AIgnite/db/image_db.py`
  - Class: `MinioImageDB`
    - `__init__(self, endpoint: str, access_key: str, secret_key: str, bucket_name: str, secure: bool = False)`
    - `save_image(self, doc_id: str, image_id: str, image_path: str = None, image_data: bytes = None) -> bool`
    - `get_image(self, doc_id: str, image_id: str, save_path: str = None) -> Optional[bytes]`
    - `list_doc_images(self, doc_id: str) -> List[str]`
    - `delete_doc_images(self, doc_id: str) -> bool`

## Implementations

### ✅ VectorDB
- `AIgnite/src/AIgnite/db/vector_db.py`
  - Class: `VectorDB`
    - See above for public methods
- Purpose: Stores and manages text embeddings for semantic search using FAISS and a transformer-based embedding model. Supports adding, searching, and deleting document vectors.

### ✅ MetadataDB
- `AIgnite/src/AIgnite/db/metadata_db.py`
  - Class: `MetadataDB`
    - See above for public methods
- Purpose: Stores paper metadata, relationships, and PDF files using a SQL database (e.g., PostgreSQL). Supports full-text search, metadata retrieval, and blog storage.

### ✅ MinioImageDB
- `AIgnite/src/AIgnite/db/image_db.py`
  - Class: `MinioImageDB`
    - See above for public methods
- Purpose: Manages storage and retrieval of paper figures and images using a MinIO-compatible object storage backend.

## Linked Tests
- `AIgnite/test/test_db/test_vector_db.py::TestVectorDB`
- `AIgnite/test/test_db/test_metadata_db.py::TestMetadataDB`
- `AIgnite/test/test_db/test_image_db.py::TestMinioImageDB`

## Status
✅ Abstract Interface Defined  
✅ All Subclasses Implemented  
✅ Unit-tested  
⬜ Integration-tested 