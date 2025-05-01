from typing import List, Optional
from abc import ABC, abstractmethod
from ..data.docset import DocSet

class BaseIndexer(ABC):
    @abstractmethod
    def index_papers(self, papers: List[DocSet]):
        pass

    @abstractmethod
    def get_paper_metadata(self, arxiv_id: str) -> Optional[dict]:
        pass

    @abstractmethod
    def find_similar_papers(self, query_embedding: List[float], top_k: int = 5, filters: dict = None) -> List[dict]:
        pass