import unittest
import numpy as np
from AIgnite.db.vector_db import VectorDB, VectorEntry

class TestVectorDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and VectorDB instance."""
        cls.vector_db = VectorDB(
            model_name='BAAI/bge-base-en-v1.5',
            dimension=768
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
        for doc in self.test_docs.values():
            self.vector_db.delete_document(doc["doc_id"])

    def test_add_multiple_documents(self):
        """Test adding multiple documents with abstract and chunks."""
        # Add all test documents
        for doc in self.test_docs.values():
            self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
        
        # Verify all documents were added correctly
        for doc in self.test_docs.values():
            vectors = self.vector_db.get_document_vectors(doc["doc_id"])
            self.assertEqual(len(vectors), len(doc["text_chunks"]) + 1)  # chunks + abstract
            
            # Verify abstract
            abstract_entry = next(v for v in vectors if v.text_type == "abstract")
            self.assertEqual(abstract_entry.text, doc["abstract"])
            self.assertEqual(abstract_entry.metadata, doc["metadata"])
            
            # Verify chunks
            chunk_entries = [v for v in vectors if v.text_type == "chunk"]
            self.assertEqual(len(chunk_entries), len(doc["text_chunks"]))
            for i, chunk in enumerate(chunk_entries):
                self.assertEqual(chunk.text, doc["text_chunks"][i])
                self.assertEqual(chunk.chunk_id, i)

    def test_search_abstract(self):
        """Test searching with focus on abstract."""
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
            ("machine learning AI applications", "2106.14834"),  # Should match ML paper
            ("quantum computing algorithms", "2106.14838"),      # Should match quantum paper
            ("robotic control reinforcement", "2106.14837"),    # Should match robotics paper
        ]
        
        for query, expected_doc_id in test_queries:
            results = self.vector_db.search(query, k=5, filter_type="abstract")
            self.assertTrue(len(results) > 0)
            entry, score = results[0]
            self.assertEqual(entry.doc_id, expected_doc_id)
            self.assertEqual(entry.text_type, "abstract")
            self.assertTrue(0 <= score <= 1)

    def test_search_chunks(self):
        """Test searching text chunks."""
        # Add all test documents
        for doc in self.test_docs.values():
            self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
        
        # Test different chunk-specific queries
        test_queries = [
            ("BERT GPT language models", "2106.14835"),        # Should match NLP paper
            ("CNN image processing", "2106.14836"),            # Should match vision paper
            ("quantum optimization problems", "2106.14838"),    # Should match quantum paper
        ]
        
        for query, expected_doc_id in test_queries:
            results = self.vector_db.search(query, k=5, filter_type="chunk")
            self.assertTrue(len(results) > 0)
            entry, score = results[0]
            self.assertEqual(entry.doc_id, expected_doc_id)
            self.assertEqual(entry.text_type, "chunk")
            self.assertTrue(0 <= score <= 1)

    def test_delete_multiple_documents(self):
        """Test deleting multiple documents."""
        # First add all documents
        for doc in self.test_docs.values():
            self.vector_db.add_document(
                doc_id=doc["doc_id"],
                abstract=doc["abstract"],
                text_chunks=doc["text_chunks"],
                metadata=doc["metadata"]
            )
        
        # Delete each document and verify
        for doc in self.test_docs.values():
            # Verify document exists
            vectors_before = self.vector_db.get_document_vectors(doc["doc_id"])
            self.assertTrue(len(vectors_before) > 0)
            
            # Delete document
            result = self.vector_db.delete_document(doc["doc_id"])
            self.assertTrue(result)
            
            # Verify document was deleted
            vectors_after = self.vector_db.get_document_vectors(doc["doc_id"])
            self.assertEqual(len(vectors_after), 0)

    def test_vector_id_generation(self):
        """Test vector ID generation."""
        doc_id = "test123"
        
        # Test abstract ID
        abstract_id = self.vector_db._generate_vector_id(doc_id, "abstract")
        self.assertEqual(abstract_id, "test123:abstract")
        
        # Test chunk ID
        chunk_id = self.vector_db._generate_vector_id(doc_id, "chunk", 0)
        self.assertEqual(chunk_id, "test123:chunk:0")

    def test_nonexistent_document(self):
        """Test handling of non-existent documents."""
        # Try to get vectors for non-existent document
        vectors = self.vector_db.get_document_vectors("nonexistent")
        self.assertEqual(len(vectors), 0)
        
        # Try to delete non-existent document
        result = self.vector_db.delete_document("nonexistent")
        self.assertFalse(result)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        # Clean up all test documents
        for doc in cls.test_docs.values():
            try:
                cls.vector_db.delete_document(doc["doc_id"])
            except:
                pass
        
        # Clean up model
        if hasattr(cls.vector_db, 'model'):
            cls.vector_db.model = None

if __name__ == '__main__':
    unittest.main() 