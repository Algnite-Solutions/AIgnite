from typing import List, Optional
from abc import ABC, abstractmethod
<<<<<<< HEAD
from AIgnite.data.docset import DocSet
=======
from ..docset import DocSet
>>>>>>> 452a018494be4206c748e6559afa232fd2bef792

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