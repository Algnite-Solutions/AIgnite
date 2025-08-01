# Feature: Indexing and Search Coordination

## Description
This module defines the abstract interface and concrete implementations for indexing, searching, and managing academic papers. It provides a unified API for ingesting papers, storing their metadata and embeddings, and performing semantic and keyword-based search. The module also supports pluggable search strategies and is designed for extensibility and integration with various storage and retrieval backends.

## Entry Point
- `AIgnite/src/AIgnite/index/base_indexer.py`
  - Abstract Class: `BaseIndexer`
    - `index_papers(self, papers: List[DocSet])`
    - `get_paper_metadata(self, arxiv_id: str) -> Optional[dict]`
    - `find_similar_papers(self, query_embedding: List[float], top_k: int = 5, filters: dict = None) -> List[dict]`

## Implementations

### ✅ PaperIndexer
- `AIgnite/src/AIgnite/index/paper_indexer.py`
  - Class: `PaperIndexer(BaseIndexer)`
    - `__init__(self, metadata_db, vector_db=None, image_db=None)`
    - `set_databases(self, vector_db=None, metadata_db=None, image_db=None) -> None`
    - `set_search_strategy(self, strategy_type: str) -> None`
    - `index_documents(self, documents: List[DocSet]) -> Dict[str, Dict[str, bool]]`
    - `find_similar_documents(self, query: str, top_k=5, filters=None, similarity_cutoff=0.5, strategy_type=None) -> List[Dict[str, Any]]`
    - `get_document(self, doc_id: str) -> Optional[dict]`
    - `delete_document(self, doc_id: str) -> Dict[str, bool]`
- Purpose: Coordinates the ingestion, indexing, search, and deletion of academic documents across multiple storage backends. Supports multiple search strategies (vector, TF-IDF, hybrid) and ensures atomic operations across all databases.
- **Note:** `metadata_db` is a required argument for PaperIndexer initialization and must not be None. `vector_db` and `image_db` are optional and can be omitted or set to None if not needed.

### ✅ Search Strategies
- `AIgnite/src/AIgnite/index/search_strategy.py`
  - Abstract Class: `SearchStrategy`
    - `search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, similarity_cutoff: float = 0.5, **kwargs) -> List[SearchResult]`
  - Concrete Classes:
    - `VectorSearchStrategy(SearchStrategy)`
    - `TFIDFSearchStrategy(SearchStrategy)`
    - `HybridSearchStrategy(SearchStrategy)`
- Purpose: Provides pluggable search strategies for semantic (vector-based), keyword (TF-IDF), and hybrid search, enabling flexible retrieval approaches.

## Linked Tests
- `AIgnite/test/index/test_paper_indexer.py::TestPaperIndexerWithToyDBs`

## Status
✅ Abstract Interface Defined  
✅ All Subclasses Implemented  
✅ Unit-tested  
⬜ Integration-tested 