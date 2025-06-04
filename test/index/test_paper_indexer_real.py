import unittest
import os
import tempfile
from PIL import Image
import io
import yaml
from typing import List
from AIgnite.index.paper_indexer import PaperIndexer
from AIgnite.db.db_init import init_databases
from AIgnite.data.docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType

class TestPaperIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and PaperIndexer instance."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test config file
        cls.config_path = os.path.join(cls.temp_dir, 'test_config.yaml')
        test_config = {
            'vector_db': {
                'model_name': 'BAAI/bge-base-en-v1.5'
            },
            'metadata_db': {
                'db_url': 'postgresql://postgres:11111@localhost:5432/aignite_test'
            },
            'minio_db': {
                'endpoint': 'localhost:9000',
                'access_key': 'Wc58W6KOCOfxaAc2MjzM',
                'secret_key': 'Seqa7Xs3TO6EaClfD4l6y4b4teW7t2Y1Slu92VKw',
                'bucket_name': 'aignite-test-papers',
                'secure': False
            }
        }
        with open(cls.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
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
        
        # Initialize databases using config
        vector_db, metadata_db, image_db = init_databases(cls.config_path)
        
        # Initialize indexer with pre-initialized databases
        cls.indexer = PaperIndexer(vector_db, metadata_db, image_db)
        
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
                pdf_path=cls.test_pdfs["pdf1"]
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
                pdf_path=cls.test_pdfs["pdf2"]
            ),
            DocSet(
                doc_id="2106.14836",
                title="Vision Systems",
                abstract="Computer vision techniques for object detection and recognition.",
                authors=["Author 5", "Author 6"],
                categories=["cs.CV"],
                published_date="2021-06-30",
                text_chunks=[
                    TextChunk(id="chunk7", type=ChunkType.TEXT, text="CNN architectures for image processing tasks."),
                    TextChunk(id="chunk8", type=ChunkType.TEXT, text="Object detection frameworks and their comparisons."),
                    TextChunk(id="chunk9", type=ChunkType.TEXT, text="Real-time object tracking implementations.")
                ],
                pdf_path=cls.test_pdfs["pdf3"]
            ),
            DocSet(
                doc_id="2106.14837",
                title="Robotic Learning",
                abstract="Robotic control systems using reinforcement learning.",
                authors=["Author 7", "Author 8"],
                categories=["cs.RO", "cs.AI"],
                published_date="2021-07-01",
                text_chunks=[
                    TextChunk(id="chunk10", type=ChunkType.TEXT, text="Robot manipulation and control strategies."),
                    TextChunk(id="chunk11", type=ChunkType.TEXT, text="RL algorithms for robotic task learning."),
                    TextChunk(id="chunk12", type=ChunkType.TEXT, text="Experimental results in real-world scenarios.")
                ],
                pdf_path=cls.test_pdfs["pdf4"]
            ),
            DocSet(
                doc_id="2106.14838",
                title="Quantum Computing",
                abstract="Quantum computing algorithms and their applications.",
                authors=["Author 9", "Author 10"],
                categories=["quant-ph", "cs.ET"],
                published_date="2021-07-02",
                text_chunks=[
                    TextChunk(id="chunk13", type=ChunkType.TEXT, text="Introduction to quantum computing principles."),
                    TextChunk(id="chunk14", type=ChunkType.TEXT, text="Quantum algorithms for optimization problems."),
                    TextChunk(id="chunk15", type=ChunkType.TEXT, text="Experimental results on quantum hardware.")
                ],
                pdf_path=cls.test_pdfs["pdf5"]
            )
        ]
        
        # Clean up any existing test data and index the papers
        for paper in cls.test_papers:
            try:
                cls.indexer.delete_paper(paper.doc_id)
            except:
                pass
        
        # Index all test papers
        cls.indexer.index_papers(cls.test_papers)

    def test_1_index_papers(self):
        """Test paper indexing across all databases."""
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
            self.assertEqual(len(metadata["image_ids"]), len(paper.figure_chunks))
            
            # Verify PDF content
            pdf_path = os.path.join(self.temp_dir, "test_output.pdf")
            self.indexer.metadata_db.get_pdf(paper.doc_id, save_path=pdf_path)
            self.assertTrue(os.path.exists(pdf_path))
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            with open(paper.pdf_path, 'rb') as f:
                original_content = f.read()
            self.assertEqual(pdf_content, original_content)
            os.remove(pdf_path)
            
            # Verify images are stored
            for figure in paper.figure_chunks:
                image_data = self.indexer.image_db.get_image(paper.doc_id, figure.id)
                self.assertIsNotNone(image_data)
                img = Image.open(figure.image_path)
                stored_img = Image.open(io.BytesIO(image_data))
                self.assertEqual(img.size, stored_img.size)

    def test_2_find_similar_papers(self):
        """Test paper similarity search."""
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

    def test_3_nonexistent_paper(self):
        """Test handling of non-existent papers."""
        # Try to get metadata
        metadata = self.indexer.get_paper_metadata("nonexistent")
        self.assertIsNone(metadata)
        
        # Try to get PDF
        pdf_data = self.indexer.metadata_db.get_pdf("nonexistent")
        self.assertIsNone(pdf_data)
        
        # Try to get images
        images = self.indexer.image_db.list_doc_images("nonexistent")
        self.assertEqual(len(images), 0)
        
        # Try to delete
        success = self.indexer.delete_paper("nonexistent")
        self.assertFalse(success)  # Should return False since paper doesn't exist

    def test_4_delete_paper(self):
        """Test paper deletion from all databases."""
        # Get a paper that hasn't been deleted yet
        paper = self.test_papers[1]  # Use NLP paper
        
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
            query="natural language processing",
            top_k=5
        )
        self.assertTrue(all(r["title"] != paper.title for r in results))

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Delete test papers from all databases
        for paper in cls.test_papers:
            try:
                cls.indexer.delete_paper(paper.doc_id)
            except:
                pass
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(cls.temp_dir)
        except:
            pass
        
        # Clean up indexer
        if hasattr(cls, 'indexer'):
            if hasattr(cls.indexer, 'vector_db'):
                cls.indexer.vector_db = None
            if hasattr(cls.indexer, 'metadata_db'):
                cls.indexer.metadata_db = None
            if hasattr(cls.indexer, 'image_db'):
                cls.indexer.image_db = None

if __name__ == '__main__':
    unittest.main() 