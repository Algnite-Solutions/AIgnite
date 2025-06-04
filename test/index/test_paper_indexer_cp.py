import unittest
import os
import tempfile
from PIL import Image
import io
from typing import List
from AIgnite.index.paper_indexer import PaperIndexer
from AIgnite.data.docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType
from toy_dbs import ToyVectorDB, ToyMetadataDB, ToyImageDB

class TestPaperIndexerWithToyDBs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and PaperIndexer instance with toy databases."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        cls.test_images = {}
        for i in range(3):
            image_path = os.path.join(cls.temp_dir, f"test_image_{i}.png")
            img = Image.new('RGB', (100 + i*50, 100 + i*50), color=f'rgb({i*50}, {i*50}, {i*50})')
            img.save(image_path)
            cls.test_images[f"fig{i+1}"] = image_path
            
        # Create test PDF files
        cls.test_pdfs = {}
        for i in range(5):
            pdf_path = os.path.join(cls.temp_dir, f"test_paper_{i}.pdf")
            with open(pdf_path, 'wb') as f:
                f.write(f"Test PDF content for paper {i}".encode())
            cls.test_pdfs[f"pdf{i+1}"] = pdf_path
        
        # Initialize toy databases
        cls.vector_db = ToyVectorDB(model_name='BAAI/bge-base-en-v1.5')
        cls.metadata_db = ToyMetadataDB(db_path='memory://test')
        cls.image_db = ToyImageDB(
            endpoint='memory://localhost:9000',
            access_key='test_key',
            secret_key='test_secret',
            bucket_name='test-bucket'
        )
        
        # Initialize indexer with toy databases
        cls.indexer = PaperIndexer(cls.vector_db, cls.metadata_db, cls.image_db)
        
        # Create test papers
        cls.test_papers = [
            DocSet(
                doc_id="2106.14834",
                title="Machine Learning Paper",
                abstract="This is a test abstract about machine learning and AI applications.",
                authors=["Author 1", "Author 2"],
                categories=["cs.AI", "cs.LG"],
                published_date="2021-06-28",
                text_chunks=[
                    TextChunk(id="chunk1", type=ChunkType.TEXT, text="First chunk about deep learning models and their applications."),
                    TextChunk(id="chunk2", type=ChunkType.TEXT, text="Second chunk discussing neural networks and training methods."),
                    TextChunk(id="chunk3", type=ChunkType.TEXT, text="Third chunk covering evaluation metrics and results.")
                ],
                figure_chunks=[
                    FigureChunk(id="fig1", type=ChunkType.FIGURE, image_path=cls.test_images["fig1"], alt_text="Model architecture"),
                    FigureChunk(id="fig2", type=ChunkType.FIGURE, image_path=cls.test_images["fig2"], alt_text="Training curves")
                ],
                pdf_path=cls.test_pdfs["pdf1"],
                HTML_path=None
            ),
            DocSet(
                doc_id="2106.14835",
                title="NLP Advances",
                abstract="Recent advances in natural language processing and transformer models.",
                authors=["Author 3", "Author 4"],
                categories=["cs.CL"],
                published_date="2021-06-29",
                text_chunks=[
                    TextChunk(id="chunk4", type=ChunkType.TEXT, text="Overview of transformer architecture and attention mechanisms."),
                    TextChunk(id="chunk5", type=ChunkType.TEXT, text="BERT and GPT models in modern NLP applications."),
                    TextChunk(id="chunk6", type=ChunkType.TEXT, text="Future directions in language model research.")
                ],
                figure_chunks=[
                    FigureChunk(id="fig3", type=ChunkType.FIGURE, image_path=cls.test_images["fig3"], alt_text="Attention visualization")
                ],
                pdf_path=cls.test_pdfs["pdf2"],
                HTML_path=None
            )
        ]

    def test_1_index_papers(self):
        """Test paper indexing across all toy databases."""
        # Index test papers
        self.indexer.index_papers(self.test_papers)
        
        # Test metadata storage
        for paper in self.test_papers:
            metadata = self.indexer.get_paper_metadata(paper.doc_id)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["title"], paper.title)
            self.assertEqual(metadata["abstract"], paper.abstract)
            self.assertEqual(metadata["authors"], paper.authors)
            self.assertEqual(metadata["categories"], paper.categories)
            self.assertEqual(metadata["published_date"], paper.published_date)
            
            # Verify chunk IDs are stored
            self.assertEqual(len(metadata["chunk_ids"]), len(paper.text_chunks))
            self.assertEqual(len(metadata["figure_ids"]), len(paper.figure_chunks))
            
            # Verify PDF content
            pdf_path = os.path.join(self.temp_dir, "test_output.pdf")
            pdf_content = self.indexer.metadata_db.get_pdf(paper.doc_id, save_path=pdf_path)
            self.assertTrue(os.path.exists(pdf_path))
            with open(pdf_path, 'rb') as f:
                saved_content = f.read()
            with open(paper.pdf_path, 'rb') as f:
                original_content = f.read()
            self.assertEqual(saved_content, original_content)
            os.remove(pdf_path)
            
            # Verify images are stored
            for figure in paper.figure_chunks:
                image_data = self.indexer.image_db.get_image(paper.doc_id, figure.id)
                self.assertIsNotNone(image_data)
                img = Image.open(figure.image_path)
                stored_img = Image.open(io.BytesIO(image_data))
                self.assertEqual(img.size, stored_img.size)

    def test_2_find_similar_papers(self):
        """Test paper similarity search with toy vector database."""
        # Test basic search
        results = self.indexer.find_similar_papers(
            query="machine learning deep learning",
            top_k=2
        )
        self.assertGreater(len(results), 0)
        self.assertTrue(all("similarity_score" in r for r in results))
        self.assertTrue(all("matched_text" in r for r in results))
        
        # Verify first result is ML paper
        self.assertEqual(results[0]["title"], "Machine Learning Paper")
        
        # Test NLP-focused search
        results = self.indexer.find_similar_papers(
            query="transformer models and BERT",
            top_k=2
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["title"], "NLP Advances")

    def test_3_delete_paper(self):
        """Test paper deletion from all toy databases."""
        # Get a paper that hasn't been deleted yet
        paper = self.test_papers[0]  # Use ML paper
        
        # Delete paper
        success = self.indexer.delete_paper(paper.doc_id)
        self.assertTrue(success)
        
        # Verify metadata is deleted
        metadata = self.indexer.get_paper_metadata(paper.doc_id)
        self.assertIsNone(metadata)
        
        # Verify PDF is deleted
        pdf_data = self.indexer.metadata_db.get_pdf(paper.doc_id)
        self.assertIsNone(pdf_data)
        
        # Verify images are deleted
        images = self.indexer.image_db.list_doc_images(paper.doc_id)
        self.assertEqual(len(images), 0)
        
        # Verify vector search no longer returns the paper
        results = self.indexer.find_similar_papers(
            query="machine learning",
            top_k=5
        )
        self.assertTrue(all(r["title"] != paper.title for r in results))

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(cls.temp_dir)
        except:
            pass
        
        # Clean up indexer
        if hasattr(cls, 'indexer'):
            cls.indexer.vector_db = None
            cls.indexer.metadata_db = None
            cls.indexer.image_db = None

if __name__ == '__main__':
    unittest.main() 