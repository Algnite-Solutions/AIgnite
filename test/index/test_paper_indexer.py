import unittest
from typing import List
from AIgnite.data.docset import DocSet, Chunk, TextChunk, FigureChunk, TableChunk, ChunkType
from AIgnite.index.paper_indexer import PaperIndexer


def create_sample_papers() -> List[DocSet]:
    """Create a list of sample papers for testing."""
    papers = []

    # Paper 1
    paper1 = DocSet(
        doc_id="2101.12345",
        title="Advances in Deep Learning for Computer Vision",
        authors=["John Smith", "Jane Doe"],
        categories=["cs.CV", "cs.LG"],
        published_date="2021-01-01",
        abstract="Recent advances in deep learning for CV...",
        pdf_path="dummy/path/to/pdf1.pdf",
        text_chunks=[
            TextChunk(id="chunk1", type=ChunkType.TEXT, text="Deep learning revolutionized CV."),
            TextChunk(id="chunk2", type=ChunkType.TEXT, text="Experiments show improvements.")
        ],
        figure_chunks=[
            FigureChunk(id="fig1", type=ChunkType.FIGURE, image_path="path/to/figure1.png", alt_text="Model architecture")
        ],
        table_chunks=[],
    )

    # Paper 2
    paper2 = DocSet(
        doc_id="2102.23456",
        title="Transformer Models for NLP",
        authors=["Alice Johnson", "Bob Wilson"],
        categories=["cs.CL", "cs.LG"],
        published_date="2021-02-01",
        abstract="Effectiveness of transformers in NLP...",
        pdf_path="dummy/path/to/pdf2.pdf",
        text_chunks=[
            TextChunk(id="chunk1", type=ChunkType.TEXT, text="Transformers became standard in NLP."),
            TextChunk(id="chunk2", type=ChunkType.TEXT, text="Superior benchmark results.")
        ],
        figure_chunks=[],
        table_chunks=[
            TableChunk(id="table1", type=ChunkType.TABLE, table_html="<table>...</table>")
        ],
    )

    # Paper 3
    paper3 = DocSet(
        doc_id="2103.34567",
        title="Deep RL for Game Playing",
        authors=["Charlie Brown", "David Miller"],
        categories=["cs.AI", "cs.LG"],
        published_date="2021-03-01",
        abstract="Investigating deep RL for games...",
        pdf_path="dummy/path/to/pdf3.pdf",
        text_chunks=[
            TextChunk(id="chunk1", type=ChunkType.TEXT, text="RL shows promise in games."),
            TextChunk(id="chunk2", type=ChunkType.TEXT, text="Achieves superhuman performance.")
        ],
        figure_chunks=[
            FigureChunk(id="fig1", type=ChunkType.FIGURE, image_path="path/to/figure2.png", alt_text="Training visualization")
        ],
        table_chunks=[],
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