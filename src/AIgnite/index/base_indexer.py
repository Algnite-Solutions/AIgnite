from typing import List, Optional
from abc import ABC, abstractmethod
from ..data.docset import DocSet

class BaseIndexer(ABC):
    @abstractmethod
    def index_papers(self, documents: List[DocSet]):
        """Index a list of documents."""
        pass

    @abstractmethod
    def get_paper_metadata(self, doc_id: str) -> Optional[dict]:
        """Retrieve metadata for a specific document."""
        pass

    @abstractmethod
    def find_similar_papers(self, query: str, top_k: int = 5, filters: dict = None) -> List[dict]:
        """Find documents similar to the query."""
        pass