import unittest
import os
import tempfile
import json
from sqlalchemy import create_engine, text
from AIgnite.db.metadata_db import MetadataDB, Base

class TestMetadataDB(unittest.TestCase):
    def setUp(self):
        """Set up test database and sample data."""
        # Use a test database
        self.db_url = "postgresql://postgres:11111@localhost:5432/aignite_test"
        
        # First drop and recreate tables
        engine = create_engine(self.db_url)
        
        # Drop existing tables
        Base.metadata.drop_all(engine)
        
        # Create new tables
        Base.metadata.create_all(engine)
        
        # Initialize MetadataDB
        self.db = MetadataDB(self.db_url)
        
        # Create a temporary PDF file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_pdf_path = os.path.join(self.temp_dir, "test.pdf")
        with open(self.test_pdf_path, "wb") as f:
            f.write(b"Test PDF content")
        
        # Sample metadata
        self.test_doc_id = "test_doc_123"
        self.test_metadata = {
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "authors": ["Author 1", "Author 2"],
            "categories": ["Category 1", "Category 2"],
            "published_date": "2024-03-20",
            "chunk_ids": ["chunk1", "chunk2", "chunk3"],
            "image_ids": ["img1", "img2"]
        }

    def tearDown(self):
        """Clean up after tests."""
        # Delete test paper if it exists
        self.db.delete_paper(self.test_doc_id)
        
        # Clean up temporary files
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_1_save_paper(self):
        """Test saving a paper with PDF and metadata."""
        # Save paper
        result = self.db.save_paper(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        self.assertTrue(result)
        
        # Verify metadata was saved
        saved_metadata = self.db.get_metadata(self.test_doc_id)
        self.assertIsNotNone(saved_metadata)
        self.assertEqual(saved_metadata["title"], self.test_metadata["title"])
        self.assertEqual(saved_metadata["abstract"], self.test_metadata["abstract"])
        self.assertEqual(saved_metadata["authors"], self.test_metadata["authors"])
        self.assertEqual(saved_metadata["categories"], self.test_metadata["categories"])
        self.assertEqual(saved_metadata["chunk_ids"], self.test_metadata["chunk_ids"])
        self.assertEqual(saved_metadata["image_ids"], self.test_metadata["image_ids"])

    def test_2_get_pdf(self):
        """Test retrieving PDF data."""
        # First save the paper
        self.db.save_paper(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        
        # Test getting PDF as binary
        pdf_data = self.db.get_pdf(self.test_doc_id)
        self.assertIsNotNone(pdf_data)
        self.assertEqual(pdf_data, b"Test PDF content")
        
        # Test saving PDF to file
        output_path = os.path.join(self.temp_dir, "output.pdf")
        self.db.get_pdf(self.test_doc_id, save_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "rb") as f:
            saved_content = f.read()
        self.assertEqual(saved_content, b"Test PDF content")
        
        # Clean up output file
        os.remove(output_path)

    def test_3_update_paper(self):
        """Test updating an existing paper."""
        # First save the paper
        self.db.save_paper(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        
        # Update metadata
        updated_metadata = self.test_metadata.copy()
        updated_metadata["title"] = "Updated Title"
        updated_metadata["chunk_ids"].append("chunk4")
        
        # Save updated version
        result = self.db.save_paper(self.test_doc_id, self.test_pdf_path, updated_metadata)
        self.assertTrue(result)
        
        # Verify updates
        saved_metadata = self.db.get_metadata(self.test_doc_id)
        self.assertEqual(saved_metadata["title"], "Updated Title")
        self.assertEqual(len(saved_metadata["chunk_ids"]), 4)
        self.assertIn("chunk4", saved_metadata["chunk_ids"])

    def test_4_get_metadata(self):
        """Test retrieving metadata."""
        # First save the paper
        self.db.save_paper(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        
        # Get metadata
        metadata = self.db.get_metadata(self.test_doc_id)
        
        # Verify all fields
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata["doc_id"], self.test_doc_id)
        self.assertEqual(metadata["title"], self.test_metadata["title"])
        self.assertEqual(metadata["abstract"], self.test_metadata["abstract"])
        self.assertEqual(metadata["authors"], self.test_metadata["authors"])
        self.assertEqual(metadata["categories"], self.test_metadata["categories"])
        self.assertEqual(metadata["published_date"], self.test_metadata["published_date"])
        self.assertEqual(metadata["chunk_ids"], self.test_metadata["chunk_ids"])
        self.assertEqual(metadata["image_ids"], self.test_metadata["image_ids"])

    def test_5_delete_paper(self):
        """Test deleting a paper."""
        # First save the paper
        self.db.save_paper(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        
        # Delete paper
        result = self.db.delete_paper(self.test_doc_id)
        self.assertTrue(result)
        
        # Verify paper is deleted
        metadata = self.db.get_metadata(self.test_doc_id)
        self.assertIsNone(metadata)
        
        pdf_data = self.db.get_pdf(self.test_doc_id)
        self.assertIsNone(pdf_data)

    def test_6_nonexistent_paper(self):
        """Test operations on non-existent papers."""
        fake_doc_id = "nonexistent_doc"
        
        # Try to get metadata
        metadata = self.db.get_metadata(fake_doc_id)
        self.assertIsNone(metadata)
        
        # Try to get PDF
        pdf_data = self.db.get_pdf(fake_doc_id)
        self.assertIsNone(pdf_data)
        
        # Try to delete
        result = self.db.delete_paper(fake_doc_id)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 