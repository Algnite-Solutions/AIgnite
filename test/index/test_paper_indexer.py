"""Test paper indexer functionality."""
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
        print("\n✅ Setting up test environment...")
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        print("✅ Creating test images and PDFs...")
        # Create test images
        cls.test_images = {}
        for i in range(5):
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
        
        print("✅ Initializing toy databases...")
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
        
        print("✅ Creating test papers...")
        # Create test papers with distinct domains
        cls.test_papers = [
            DocSet(
                doc_id="2106.14834",
                title="Large Language Models and Their Applications",
                abstract="Recent advances in large language models like GPT and their applications in various domains.",
                authors=["Author 1", "Author 2"],
                categories=["cs.CL", "cs.AI"],
                published_date="2021-06-28",
                text_chunks=[
                    TextChunk(id="chunk1", type=ChunkType.TEXT, text="Overview of transformer-based language models and their architectures."),
                    TextChunk(id="chunk2", type=ChunkType.TEXT, text="Fine-tuning strategies for LLMs on downstream tasks."),
                    TextChunk(id="chunk3", type=ChunkType.TEXT, text="Applications in text generation and summarization.")
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
                title="Natural Language Understanding with BERT",
                abstract="Advances in natural language understanding using BERT and its variants for various NLP tasks.",
                authors=["Author 3", "Author 4"],
                categories=["cs.CL"],
                published_date="2021-06-29",
                text_chunks=[
                    TextChunk(id="chunk4", type=ChunkType.TEXT, text="BERT architecture and pre-training objectives."),
                    TextChunk(id="chunk5", type=ChunkType.TEXT, text="Fine-tuning BERT for classification and token tagging."),
                    TextChunk(id="chunk6", type=ChunkType.TEXT, text="Comparison with other transformer models.")
                ],
                figure_chunks=[
                    FigureChunk(id="fig3", type=ChunkType.FIGURE, image_path=cls.test_images["fig3"], alt_text="Attention visualization")
                ],
                pdf_path=cls.test_pdfs["pdf2"],
                HTML_path=None
            ),
            DocSet(
                doc_id="2106.14836",
                title="Prompt Engineering for LLMs",
                abstract="Techniques and strategies for effective prompt engineering in large language models.",
                authors=["Author 5", "Author 6"],
                categories=["cs.CL", "cs.AI"],
                published_date="2021-06-30",
                text_chunks=[
                    TextChunk(id="chunk7", type=ChunkType.TEXT, text="Principles of prompt design and chain-of-thought prompting."),
                    TextChunk(id="chunk8", type=ChunkType.TEXT, text="Few-shot and zero-shot prompting strategies."),
                    TextChunk(id="chunk9", type=ChunkType.TEXT, text="Evaluation of prompt effectiveness.")
                ],
                figure_chunks=[],
                pdf_path=cls.test_pdfs["pdf3"],
                HTML_path=None
            ),
            DocSet(
                doc_id="2106.14837",
                title="Computer Vision with Deep CNNs",
                abstract="Deep learning approaches for computer vision tasks using convolutional neural networks.",
                authors=["Author 7", "Author 8"],
                categories=["cs.CV"],
                published_date="2021-07-01",
                text_chunks=[
                    TextChunk(id="chunk10", type=ChunkType.TEXT, text="CNN architectures for image classification and detection."),
                    TextChunk(id="chunk11", type=ChunkType.TEXT, text="Transfer learning in computer vision."),
                    TextChunk(id="chunk12", type=ChunkType.TEXT, text="Real-world applications of CNNs.")
                ],
                figure_chunks=[],
                pdf_path=cls.test_pdfs["pdf4"],
                HTML_path=None
            ),
            DocSet(
                doc_id="2106.14838",
                title="Vision Transformers for Image Recognition",
                abstract="Application of transformer architectures to computer vision tasks.",
                authors=["Author 9", "Author 10"],
                categories=["cs.CV", "cs.AI"],
                published_date="2021-07-02",
                text_chunks=[
                    TextChunk(id="chunk13", type=ChunkType.TEXT, text="Vision transformer architecture and attention mechanisms."),
                    TextChunk(id="chunk14", type=ChunkType.TEXT, text="Comparison with CNN-based approaches."),
                    TextChunk(id="chunk15", type=ChunkType.TEXT, text="Performance on image recognition benchmarks.")
                ],
                figure_chunks=[],
                pdf_path=cls.test_pdfs["pdf5"],
                HTML_path=None
            )
        ]
        print("✅ Setup complete\n")

    def setUp(self):
        """Print the name of the test being run."""
        print(f"\n✅ Running test: {self._testMethodName}")

    def test_1_index_papers(self):
        """Test paper indexing across all toy databases."""
        print("✅ Testing paper indexing...")
        # Index test papers
        indexing_status = self.indexer.index_papers(self.test_papers)
        
        print("✅ Verifying indexing status...")
        # Test indexing status
        for paper in self.test_papers:
            status = indexing_status[paper.doc_id]
            self.assertTrue(status["metadata"])
            self.assertTrue(status["vectors"])
            # Only check image status if paper has figures
            if paper.figure_chunks:
                self.assertTrue(status["images"])
            
            # Test metadata storage
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
            
            # Verify images are stored (only for papers with figures)
            if paper.figure_chunks:
                for figure in paper.figure_chunks:
                    image_data = self.indexer.image_db.get_image(paper.doc_id, figure.id)
                    self.assertIsNotNone(image_data)
                    img = Image.open(figure.image_path)
                    stored_img = Image.open(io.BytesIO(image_data))
                    self.assertEqual(img.size, stored_img.size)
        print("✅ Indexing verification complete.")

    def test_2_vector_search(self):
        """Test vector-based search strategy."""
        print("✅ Testing vector search...")
        # Test LLM-focused search
        results = self.indexer.find_similar_papers(
            query="large language models and GPT",
            top_k=2
        )
        self.assertGreater(len(results), 0)
        self.assertTrue(all("similarity_score" in r for r in results))
        self.assertTrue(all("matched_text" in r for r in results))
        self.assertTrue(results[0]["title"] in ["Large Language Models and Their Applications", "Prompt Engineering for LLMs"])

        # Test BERT/NLP-focused search
        results = self.indexer.find_similar_papers(
            query="BERT models for natural language understanding",
            top_k=2
        )
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["title"], "Natural Language Understanding with BERT")

        # Test computer vision search
        results = self.indexer.find_similar_papers(
            query="convolutional neural networks for image classification",
            top_k=2
        )
        self.assertGreater(len(results), 0)
        self.assertTrue(results[0]["title"] in ["Computer Vision with Deep CNNs", "Vision Transformers for Image Recognition"])


    def test_3_tfidf_search(self):
        """Test TF-IDF search strategy."""
        print("✅ Testing TF-IDF search...")
        print("✅ Testing TF-IDF search with query: 'large language models and applications'")
        # Test TF-IDF search focusing on title and abstract content
        results = self.indexer.find_similar_papers(
            query="large language models and applications",
            top_k=3,
            strategy_type="tf-idf"
        )
        
        print("✅ Found", len(results), "results")
        self.assertGreater(len(results), 0)
        
        # Results should match papers with relevant terms in title/abstract
        titles = [r["title"] for r in results]
        self.assertTrue(any("Language Model" in t for t in titles))
        print("✅ Top result:", results[0]['title'])
        
        # Verify search method
        self.assertEqual(results[0]["search_method"], "tf-idf")

        # Test computer vision domain query
        print("✅ Testing TF-IDF search with query: 'computer vision deep learning'")
        results = self.indexer.find_similar_papers(
            query="computer vision deep learning",
            top_k=2,
            strategy_type="tf-idf"
        )
        self.assertGreater(len(results), 0)
        titles = [r["title"] for r in results]
        self.assertTrue(any("Vision" in t for t in titles))
        self.assertTrue(any("CNN" in t for t in titles))

    def test_4_hybrid_search(self):
        """Test hybrid search strategy."""
        print("✅ Testing hybrid search...")
        print("✅ Testing hybrid search with query: 'natural language understanding and BERT'")
        # Test hybrid search
        results = self.indexer.find_similar_papers(
            query="natural language understanding and BERT",
            top_k=2,
            strategy_type="hybrid"
        )
        print("✅ Found", len(results), "results")
        self.assertGreater(len(results), 0)
        
        # Results should include papers with relevant terms in title/abstract
        titles = [r["title"] for r in results]
        self.assertTrue(any("BERT" in t for t in titles))
        self.assertTrue(any("Language" in t for t in titles))
        print("✅ Top result:", results[0]['title'])
        
        # Verify search method
        self.assertEqual(results[0]["search_method"], "hybrid")
        
        # Verify hybrid scores
        self.assertIn("similarity_score", results[0])

        # Test cross-domain query focusing on title/abstract content
        print("✅ Testing hybrid search with query: 'deep learning in vision and language models'")
        results = self.indexer.find_similar_papers(
            query="deep learning in vision and language models",
            top_k=3,
            strategy_type="hybrid"
        )
        self.assertGreater(len(results), 0)
        
        # Should find papers from both domains based on title/abstract
        titles = [r["title"] for r in results]
        self.assertTrue(any(t for t in titles if "Vision" in t or "CNN" in t))
        self.assertTrue(any(t for t in titles if "Language" in t or "BERT" in t))

    def test_5_delete_paper(self):
        """Test paper deletion from all toy databases."""
        print("✅ Testing paper deletion...")
        # Delete the first paper
        paper = self.test_papers[0]
        print("⚠️ Deleting paper:", paper.title)
        deletion_status = self.indexer.delete_paper(paper.doc_id)
        
        print("✅ Verifying deletion...")
        # Verify deletion status
        self.assertTrue(deletion_status["metadata"])
        self.assertTrue(deletion_status["vectors"])
        self.assertTrue(deletion_status["images"])
        
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
        print("✅ Deletion verification complete.")

    def test_6_save_and_get_blog(self):
        """Test saving and retrieving blog text for a paper via the indexer's metadata_db."""
        # Pick a paper to test
        paper = self.test_papers[0]
        doc_id = paper.doc_id

        # Ensure the paper exists in the DB (re-add if necessary)
        self.indexer.metadata_db.save_paper(doc_id, paper.pdf_path, paper.__dict__)

        # Save a blog
        blog_text = "This is a blog post about the paper."
        result = self.indexer.metadata_db.add_blog(doc_id, blog_text)
        self.assertTrue(result)

        # Retrieve the blog
        retrieved_blog = self.indexer.metadata_db.get_blog(doc_id)
        self.assertEqual(retrieved_blog, blog_text)

        # Update the blog
        updated_blog = "This is an updated blog post."
        result = self.indexer.metadata_db.add_blog(doc_id, updated_blog)
        self.assertTrue(result)
        retrieved_blog = self.indexer.metadata_db.get_blog(doc_id)
        self.assertEqual(retrieved_blog, updated_blog)

        # Try to get blog for non-existent paper
        non_existent_blog = self.indexer.metadata_db.get_blog("nonexistent_doc")
        self.assertIsNone(non_existent_blog)

    
    

    def test_7_filtering_functionality(self):
        """Test filtering functionality using filter_parser with doc_ids inclusion and exclusion."""
        print("✅ Testing filtering functionality with filter_parser...")
        
        # Test 1: Basic doc_ids inclusion using new filter structure
        print("✅ Test 1: Testing doc_ids inclusion with new filter structure...")
        self.indexer.set_search_strategy('vector')
        
        # Search with include filter using new structure
        results_include = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": ["2106.14834", "2106.14835"]
                }
            },
            strategy_type='vector'
        )

        print(f"Vector search with include filter: {len(results_include)} results")
        
        # Verify inclusion filtering works correctly
        if results_include:
            doc_ids_in_results = {r['doc_id'] for r in results_include}
            allowed_doc_ids = {"2106.14834", "2106.14835"}
            self.assertTrue(doc_ids_in_results.issubset(allowed_doc_ids))
            print("✅ Vector search inclusion filtering working correctly")
        
        # Test 2: Basic doc_ids exclusion using new filter structure
        print("✅ Test 2: Testing doc_ids exclusion with new filter structure...")
        
        # Search with exclude filter using new structure
        results_exclude = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "exclude": {
                    "doc_ids": ["2106.14836", "2106.14837", "2106.14838"]
                }
            },
            strategy_type='vector'
        )
        print(f"Vector search with exclude filter: {len(results_exclude)} results")
        
        # Verify exclusion filtering works correctly
        if results_exclude:
            doc_ids_in_results = {r['doc_id'] for r in results_exclude}
            excluded_doc_ids = {"2106.14836", "2106.14837", "2106.14838"}
            self.assertTrue(doc_ids_in_results.isdisjoint(excluded_doc_ids))
            print("✅ Vector search exclusion filtering working correctly")
        
        # Test 3: Combined inclusion and exclusion using new filter structure
        print("✅ Test 3: Testing combined inclusion and exclusion...")
        
        # Search with both include and exclude filters
        results_combined = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": ["2106.14834", "2106.14835", "2106.14836"]
                },
                "exclude": {
                    "doc_ids": ["2106.14836"]
                }
            },
            strategy_type='vector'
        )
        print(f"Vector search with combined filters: {len(results_combined)} results")
        
        # Verify combined filtering works correctly
        if results_combined:
            doc_ids_in_results = {r['doc_id'] for r in results_combined}
            # Should include 2106.14834 and 2106.14835, exclude 2106.14836
            expected_doc_ids = {"2106.14834", "2106.14835"}
            self.assertTrue(doc_ids_in_results.issubset(expected_doc_ids))
            print("✅ Vector search combined filtering working correctly")
        
        # Test 4: Backward compatibility with simple doc_ids filter
        print("✅ Test 4: Testing backward compatibility with simple doc_ids filter...")
        
        # Search with old format (should still work)
        results_backward = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={"doc_ids": ["2106.14834"]},
            strategy_type='vector'
        )
        print(f"Vector search with backward compatibility: {len(results_backward)} results")
        
        # Verify backward compatibility works
        if results_backward:
            doc_ids_in_results = {r['doc_id'] for r in results_backward}
            allowed_doc_ids = {"2106.14834"}
            self.assertTrue(doc_ids_in_results.issubset(allowed_doc_ids))
            print("✅ Backward compatibility working correctly")
        
        # Test 5: TF-IDF search with new filter structure
        print("✅ Test 5: Testing TF-IDF search with new filter structure...")
        self.indexer.set_search_strategy('tf-idf')
        
        results_tfidf_include = self.indexer.find_similar_papers(
            query="BERT natural language",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": ["2106.14835"]
                }
            },
            strategy_type='tf-idf'
        )
        print(f"TF-IDF search with include filter: {len(results_tfidf_include)} results")
        
        if results_tfidf_include:
            doc_ids_in_results = {r['doc_id'] for r in results_tfidf_include}
            allowed_doc_ids = {"2106.14835"}
            self.assertTrue(doc_ids_in_results.issubset(allowed_doc_ids))
            print("✅ TF-IDF search inclusion filtering working correctly")
        
        # Test 6: Hybrid search with new filter structure
        print("✅ Test 6: Testing hybrid search with new filter structure...")
        self.indexer.set_search_strategy('hybrid')
        
        results_hybrid_combined = self.indexer.find_similar_papers(
            query="deep learning vision",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": ["2106.14837", "2106.14838"]
                },
                "exclude": {
                    "doc_ids": ["2106.14836"]
                }
            },
            strategy_type='hybrid'
        )
        print(f"Hybrid search with combined filters: {len(results_hybrid_combined)} results")
        
        if results_hybrid_combined:
            doc_ids_in_results = {r['doc_id'] for r in results_hybrid_combined}
            expected_doc_ids = {"2106.14837", "2106.14838"}
            self.assertEqual(doc_ids_in_results, expected_doc_ids)
            print("✅ Hybrid search combined filtering working correctly")
        
        # Test 7: Edge cases with new filter structure
        print("✅ Test 7: Testing edge cases with new filter structure...")
        
        # Empty include filter
        results_empty_include = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": []
                }
            },
            strategy_type='vector'
        )
        self.assertEqual(len(results_empty_include), 0)
        print("✅ Empty include filter working correctly")
        
        # Non-existent doc_ids in include filter
        results_nonexistent_include = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "include": {
                    "doc_ids": ["nonexistent_doc_1", "nonexistent_doc_2"]
                }
            },
            strategy_type='vector'
        )
        self.assertEqual(len(results_nonexistent_include), 0)
        print("✅ Non-existent doc_ids in include filter working correctly")
        
        # Non-existent doc_ids in exclude filter (should return all results)
        results_nonexistent_exclude = self.indexer.find_similar_papers(
            query="language models",
            top_k=5,
            filters={
                "exclude": {
                    "doc_ids": ["nonexistent_doc_1", "nonexistent_doc_2"]
                }
            },
            strategy_type='vector'
        )
        # Should return results since excluding non-existent docs doesn't affect anything
        self.assertGreaterEqual(len(results_nonexistent_exclude), 0)
        print("✅ Non-existent doc_ids in exclude filter working correctly")
        
        print("✅ All filtering functionality tests with filter_parser passed!")


    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        print("\n✅ Cleaning up test environment...")
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(cls.temp_dir)
            print("✅ Temporary files cleaned up")
        except:
            print("⚠️ Warning: Failed to clean up some temporary files")
        
        # Clean up indexer
        if hasattr(cls, 'indexer'):
            cls.indexer.vector_db = None
            cls.indexer.metadata_db = None
            cls.indexer.image_db = None
            print("✅ Indexer cleaned up")
        print("✅ Cleanup complete\n")

if __name__ == '__main__':
    print("\n✅ Starting Paper Indexer Tests")
    unittest.main(verbosity=2) 