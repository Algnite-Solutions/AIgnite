import unittest
import numpy as np
import os
import shutil
from AIgnite.db.vector_db import VectorDB, VectorEntry

class TestVectorDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and VectorDB instance."""
        # Create a temporary directory for test database
        cls.test_db_path = "test_vector_db"
        os.makedirs(cls.test_db_path, exist_ok=True)
        
        cls.vector_db = VectorDB(
            db_path=os.path.join(cls.test_db_path, "test_db"),
            model_name='BAAI/bge-base-en-v1.5'
        )
        
        # Test documents data
        cls.test_docs = {
            "ml_paper": {
                "doc_id": "2106.14834",
                "abstract": "This is a test abstract about machine learning and AI applications.",
                "text_chunks": [
                    "First chunk about deep learning models and their applications.",
                    "Second chunk discussing neural networks and training methods.",
                    "Third chunk covering evaluation metrics and results."
                ],
                "metadata": {
                    "title": "Machine Learning Paper",
                    "authors": ["Author 1", "Author 2"],
                    "categories": ["cs.AI", "cs.LG"],
                    "published_date": "2021-06-28"
                }
            },
            "nlp_paper": {
                "doc_id": "2106.14835",
                "abstract": "Recent advances in natural language processing and transformer models.",
                "text_chunks": [
                    "Overview of transformer architecture and attention mechanisms.",
                    "BERT and GPT models in modern NLP applications.",
                    "Future directions in language model research."
                ],
                "metadata": {
                    "title": "NLP Advances",
                    "authors": ["Author 3", "Author 4"],
                    "categories": ["cs.CL"],
                    "published_date": "2021-06-29"
                }
            },
            "vision_paper": {
                "doc_id": "2106.14836",
                "abstract": "Computer vision techniques for object detection and recognition.",
                "text_chunks": [
                    "CNN architectures for image processing tasks.",
                    "Object detection frameworks and their comparisons.",
                    "Real-time object tracking implementations."
                ],
                "metadata": {
                    "title": "Vision Systems",
                    "authors": ["Author 5", "Author 6"],
                    "categories": ["cs.CV"],
                    "published_date": "2021-06-30"
                }
            },
            "robotics_paper": {
                "doc_id": "2106.14837",
                "abstract": "Robotic control systems using reinforcement learning.",
                "text_chunks": [
                    "Robot manipulation and control strategies.",
                    "RL algorithms for robotic task learning.",
                    "Experimental results in real-world scenarios."
                ],
                "metadata": {
                    "title": "Robotic Learning",
                    "authors": ["Author 7", "Author 8"],
                    "categories": ["cs.RO", "cs.AI"],
                    "published_date": "2021-07-01"
                }
            },
            "quantum_paper": {
                "doc_id": "2106.14838",
                "abstract": "Quantum computing algorithms and their applications.",
                "text_chunks": [
                    "Introduction to quantum computing principles.",
                    "Quantum algorithms for optimization problems.",
                    "Experimental results on quantum hardware."
                ],
                "metadata": {
                    "title": "Quantum Computing",
                    "authors": ["Author 9", "Author 10"],
                    "categories": ["quant-ph", "cs.ET"],
                    "published_date": "2021-07-02"
                }
            }
        }

    def setUp(self):
        """Clean up any existing test data before each test."""
        # Delete and recreate vector database for each test
        if os.path.exists(f"{self.vector_db.db_path}.index"):
            os.remove(f"{self.vector_db.db_path}.index")
        if os.path.exists(f"{self.vector_db.db_path}.entries"):
            os.remove(f"{self.vector_db.db_path}.entries")

    def test_01_add_multiple_documents(self):
        """Test 1: Adding multiple documents and verifying duplicate prevention."""
        # Add all test documents
        for doc in self.test_docs.values():
            success = self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
            self.assertTrue(success)
        
        # Try to add the same document again - should fail
        doc = next(iter(self.test_docs.values()))
        success = self.vector_db.add_document(
            doc_id=doc["doc_id"],
            abstract=doc["abstract"],
            text_chunks=doc["text_chunks"],
            metadata=doc["metadata"]
        )
        self.assertFalse(success)
        
        # Save the database
        self.assertTrue(self.vector_db.save())

    def test_02_search_abstract(self):
        """Test 2: Searching with focus on abstract."""
        # Add all test documents
        for doc in self.test_docs.values():
            self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
        
        # Test different queries
        test_queries = [
            ("machine learning AI applications", "2106.14834"),  # Should match ML document
            ("natural language processing transformers", "2106.14835"),  # Should match NLP document
        ]
        
        for query, expected_doc_id in test_queries:
            results = self.vector_db.search_documents(query, top_k=5)
            self.assertTrue(len(results) > 0)
            entry, score = results[0]
            self.assertEqual(entry.doc_id, expected_doc_id)
            self.assertTrue(0 <= score <= 1)

    def test_03_delete_document(self):
        """Test 3: Deleting a document."""
        # Create a test document with unique ID
        test_doc = {
            "doc_id": "test_delete_doc_id",  # Unique ID for this test
            "abstract": "Test abstract for deletion",
            "text_chunks": ["Test chunk 1", "Test chunk 2"],
            "metadata": {
                "title": "Test Document",
                "authors": ["Test Author"],
                "categories": ["test"],
                "published_date": "2024-01-01"
            }
        }
        
        # First add the document
        success = self.vector_db.add_document(
            doc_id=test_doc["doc_id"],
            abstract=test_doc["abstract"],
            text_chunks=test_doc["text_chunks"],
            metadata=test_doc["metadata"]
        )
        self.assertTrue(success)
        
        # Save the database
        self.assertTrue(self.vector_db.save())
        
        # Delete the document
        success = self.vector_db.delete_document(test_doc["doc_id"])
        self.assertTrue(success)
        
        # Save after deletion
        self.assertTrue(self.vector_db.save())
        
        # Try to add the same document again - should succeed now
        success = self.vector_db.add_document(
            doc_id=test_doc["doc_id"],
            abstract=test_doc["abstract"],
            text_chunks=test_doc["text_chunks"],
            metadata=test_doc["metadata"]
        )
        self.assertTrue(success)

    def test_04_nonexistent_document(self):
        """Test 4: Handling of non-existent documents."""
        # Try to delete non-existent document
        result = self.vector_db.delete_document("nonexistent")
        self.assertFalse(result)

    def test_05_load_existing_database(self):
        """Test 5: Loading from an existing database file."""
        # Create test documents with unique IDs
        test_docs = [
            {
                "doc_id": "test_load_doc_1",
                "abstract": "Test abstract for loading 1",
                "text_chunks": ["Test chunk 1", "Test chunk 2"],
                "metadata": {
                    "title": "Test Document 1",
                    "authors": ["Test Author 1"],
                    "categories": ["test"],
                    "published_date": "2024-01-01"
                }
            },
            {
                "doc_id": "test_load_doc_2",
                "abstract": "Test abstract for loading 2",
                "text_chunks": ["Test chunk 3", "Test chunk 4"],
                "metadata": {
                    "title": "Test Document 2",
                    "authors": ["Test Author 2"],
                    "categories": ["test"],
                    "published_date": "2024-01-02"
                }
            }
        ]
        
        # First add documents and save
        for doc in test_docs:
            success = self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
            self.assertTrue(success)
        
        # Save the current state
        self.assertTrue(self.vector_db.save())
        
        # Create a new VectorDB instance - should load existing data
        new_vector_db = VectorDB(
            db_path=self.vector_db.db_path,
            model_name='BAAI/bge-base-en-v1.5'
        )
        
        # Verify the data was loaded
        for doc in test_docs:
            # Try to add the same document - should fail if loaded correctly
            success = new_vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
            self.assertFalse(success, f"Document {doc['doc_id']} should already exist")

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Clean up the test directory
        if os.path.exists(cls.test_db_path):
            shutil.rmtree(cls.test_db_path)
        
        # Clean up model
        if hasattr(cls.vector_db, 'model'):
            cls.vector_db.model = None

if __name__ == '__main__':
    unittest.main() 