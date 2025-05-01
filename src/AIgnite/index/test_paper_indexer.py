import unittest
from typing import List
from datetime import datetime
import sys
import os
os.environ['http_proxy'] = "http://127.0.0.1:7890" 
os.environ['https_proxy'] = "http://127.0.0.1:7890" 

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = ""  # Replace with your actual API key

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..data.docset import DocSet, Chunk, TextChunk, FigureChunk, TableChunk, ChunkType
from .paper_indexer import PaperIndexer


def create_sample_papers() -> List[DocSet]:
    """Create a list of sample papers for testing."""
    papers = []
    
    # Paper 1: About Machine Learning
    paper1 = DocSet(
        doc_id="2101.12345",
        title="Advances in Deep Learning for Computer Vision",
        authors=["John Smith", "Jane Doe"],
        categories=["cs.CV", "cs.LG"],
        published_date="2021-01-01",
        abstract="This paper presents recent advances in deep learning for computer vision tasks...",
        chunks=[
            TextChunk(
                id="chunk1",
                type=ChunkType.TEXT,
                text="Deep learning has revolutionized computer vision in recent years..."
            ),
            FigureChunk(
                id="fig1",
                type=ChunkType.FIGURE,
                image_path="path/to/figure1.png",
                alt_text="Architecture of the proposed model"
            ),
            TextChunk(
                id="chunk2",
                type=ChunkType.TEXT,
                text="Our experiments show significant improvements over previous methods..."
            )
        ]
    )
    
    # Paper 2: About Natural Language Processing
    paper2 = DocSet(
        doc_id="2102.23456",
        title="Transformer Models for Natural Language Understanding",
        authors=["Alice Johnson", "Bob Wilson"],
        categories=["cs.CL", "cs.LG"],
        published_date="2021-02-01",
        abstract="This paper explores the effectiveness of transformer models in NLP tasks...",
        chunks=[
            TextChunk(
                id="chunk1",
                type=ChunkType.TEXT,
                text="Transformer models have become the standard in natural language processing..."
            ),
            TableChunk(
                id="table1",
                type=ChunkType.TABLE,
                table_html="<table>...</table>"
            ),
            TextChunk(
                id="chunk2",
                type=ChunkType.TEXT,
                text="Our results demonstrate superior performance on various benchmarks..."
            )
        ]
    )
    
    # Paper 3: About Reinforcement Learning
    paper3 = DocSet(
        doc_id="2103.34567",
        title="Deep Reinforcement Learning for Game Playing",
        authors=["Charlie Brown", "David Miller"],
        categories=["cs.AI", "cs.LG"],
        published_date="2021-03-01",
        abstract="This paper investigates deep reinforcement learning approaches for game playing...",
        chunks=[
            TextChunk(
                id="chunk1",
                type=ChunkType.TEXT,
                text="Reinforcement learning has shown great promise in game playing scenarios..."
            ),
            FigureChunk(
                id="fig1",
                type=ChunkType.FIGURE,
                image_path="path/to/figure2.png",
                alt_text="Training process visualization"
            ),
            TextChunk(
                id="chunk2",
                type=ChunkType.TEXT,
                text="Our agent achieves superhuman performance in several games..."
            )
        ]
    )
    
    papers.extend([paper1, paper2, paper3])
    return papers


class TestPaperIndexer(unittest.TestCase):
    '''
    def test_init(self):
        """Test basic initialization of PaperIndexer."""
        try:
            indexer = PaperIndexer()
            self.assertIsNotNone(indexer)
            self.assertIsNotNone(indexer.embedding_model)
            self.assertIsNotNone(indexer.vector_store)
            self.assertIsNotNone(indexer.index)
            self.assertIsInstance(indexer.metadata_store, dict)
            print("PaperIndexer initialized successfully")
        except Exception as e:
            self.fail(f"Failed to initialize PaperIndexer: {str(e)}")
    '''
    
    def setUp(self):
        """Set up test fixtures."""
        self.indexer = PaperIndexer()
        self.sample_papers = create_sample_papers()
        self.indexer.index_papers(self.sample_papers)

        
    
    def test_index_papers(self):
        """Test paper indexing functionality."""
        # Test if metadata is stored correctly
        for paper in self.sample_papers:
            metadata = self.indexer.get_paper_metadata(paper.doc_id)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["title"], paper.title)
            self.assertEqual(metadata["authors"], paper.authors)
            self.assertEqual(metadata["categories"], paper.categories)
            self.assertEqual(metadata["published_date"], paper.published_date)
            self.assertEqual(metadata["abstract"], paper.abstract)

        print("Test indexing paper successfully")
    
    def test_get_paper_metadata(self):
        """Test metadata retrieval functionality."""
        # Test existing paper
        paper = self.sample_papers[0]
        metadata = self.indexer.get_paper_metadata(paper.doc_id)
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["title"], paper.title)
        
        # Test non-existent paper
        metadata = self.indexer.get_paper_metadata("non_existent_id")
        self.assertIsNone(metadata)

        print("Test getting paper metadata successfully")
    
    def test_find_similar_papers(self):
        """Test paper similarity search functionality."""
        # Test basic search
        results = self.indexer.find_similar_papers(
            query="deep learning computer vision",
            top_k=2
        )
        self.assertGreater(len(results), 0, "Should find at least one result for relevant query")
        self.assertTrue(all("similarity_score" in r for r in results))
        self.assertTrue(all("matched_text" in r for r in results))
        
        # Test search with filters
        results = self.indexer.find_similar_papers(
            query="natural language processing",
            top_k=2,
            filters={"categories": ["cs.CL"]}
        )
        self.assertGreater(len(results), 0, "Should find at least one result for filtered query")
        self.assertTrue(all("cs.CL" in r["categories"] for r in results))
        
        # Test search with very unrelated topic
        results = self.indexer.find_similar_papers(
            query="quantum physics black holes dark matter",
            top_k=2
        )
        self.assertEqual(len(results), 0, "Should find no results for completely unrelated topic")

        print("Test finding similar paper successfully")
        
if __name__ == '__main__':
    unittest.main() 