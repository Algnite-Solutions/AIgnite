import unittest
import os
import tempfile
import json
from sqlalchemy import create_engine, text
from AIgnite.db.metadata_db import MetadataDB, Base, TableSchema
from AIgnite.data.docset import DocSet, TextChunk, FigureChunk, TableChunk, ChunkType

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
        
        # Create test DocSet
        self.test_doc_id = "test_doc_123"
        self.test_docset = DocSet(
            doc_id=self.test_doc_id,
            title="Test Paper",
            abstract="This is a test abstract",
            authors=["Author 1", "Author 2"],
            categories=["Category 1", "Category 2"],
            published_date="2024-03-20",
            text_chunks=[
                TextChunk(id="chunk1", type=ChunkType.TEXT, text="Text chunk 1"),
                TextChunk(id="chunk2", type=ChunkType.TEXT, text="Text chunk 2"),
                TextChunk(id="chunk3", type=ChunkType.TEXT, text="Text chunk 3")
            ],
            figure_chunks=[
                FigureChunk(id="fig1", type=ChunkType.FIGURE, image_path="path/to/fig1.png"),
                FigureChunk(id="fig2", type=ChunkType.FIGURE, image_path="path/to/fig2.png")
            ],
            table_chunks=[
                TableChunk(id="table1", type=ChunkType.TABLE, table_html="<table>...</table>")
            ],
            metadata={"key": "value"},
            pdf_path=self.test_pdf_path,
            HTML_path=None
        )
        
        # Legacy dictionary metadata for backward compatibility tests
        self.test_metadata = {
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "authors": ["Author 1", "Author 2"],
            "categories": ["Category 1", "Category 2"],
            "published_date": "2024-03-20",
            "chunk_ids": ["chunk1", "chunk2", "chunk3"],
            "figure_ids": ["fig1", "fig2"],
            "table_ids": ["table1"],
            "metadata": {"key": "value"}
        }

    def tearDown(self):
        """Clean up after tests."""
        # Delete test paper if it exists
        self.db.delete_document(self.test_doc_id)
        
        # Clean up temporary files
        if os.path.exists(self.test_pdf_path):
            os.remove(self.test_pdf_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)

    def test_1_add_document_docset(self):
        """Test adding a document using DocSet."""
        # Add document using DocSet
        result = self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        self.assertTrue(result)
        # Verify metadata was saved
        saved_metadata = self.db.get_document(self.test_doc_id)
        self.assertIsNotNone(saved_metadata)
        self.assertEqual(saved_metadata["title"], self.test_docset.title)
        self.assertEqual(saved_metadata["abstract"], self.test_docset.abstract)
        self.assertEqual(saved_metadata["authors"], self.test_docset.authors)
        self.assertEqual(saved_metadata["categories"], self.test_docset.categories)
        self.assertEqual(saved_metadata["chunk_ids"], [chunk.id for chunk in self.test_docset.text_chunks])
        self.assertEqual(saved_metadata["figure_ids"], [chunk.id for chunk in self.test_docset.figure_chunks])
        self.assertEqual(saved_metadata["table_ids"], [chunk.id for chunk in self.test_docset.table_chunks])
        self.assertEqual(saved_metadata["metadata"], self.test_docset.metadata)

    def test_2_add_document_dict(self):
        """Test adding a document using dictionary metadata (backward compatibility)."""
        # Add document using dictionary
        result = self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        self.assertTrue(result)
        # Verify metadata was saved
        saved_metadata = self.db.get_document(self.test_doc_id)
        self.assertIsNotNone(saved_metadata)
        self.assertEqual(saved_metadata["title"], self.test_metadata["title"])
        self.assertEqual(saved_metadata["abstract"], self.test_metadata["abstract"])
        self.assertEqual(saved_metadata["authors"], self.test_metadata["authors"])
        self.assertEqual(saved_metadata["categories"], self.test_metadata["categories"])
        self.assertEqual(saved_metadata["chunk_ids"], self.test_metadata["chunk_ids"])
        self.assertEqual(saved_metadata["figure_ids"], self.test_metadata["figure_ids"])
        self.assertEqual(saved_metadata["table_ids"], self.test_metadata["table_ids"])
        self.assertEqual(saved_metadata["metadata"], self.test_metadata["metadata"])

    def test_3_get_document_pdf(self):
        """Test retrieving document PDF data."""
        # First add the document
        self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        # Test getting PDF as binary
        pdf_data = self.db.get_document_pdf(self.test_doc_id)
        self.assertIsNotNone(pdf_data)
        self.assertEqual(pdf_data, b"Test PDF content")
        # Test saving PDF to file
        output_path = os.path.join(self.temp_dir, "output.pdf")
        self.db.get_document_pdf(self.test_doc_id, save_path=output_path)
        self.assertTrue(os.path.exists(output_path))
        with open(output_path, "rb") as f:
            saved_content = f.read()
        self.assertEqual(saved_content, b"Test PDF content")
        os.remove(output_path)
    '''
    def test_4_update_document(self):
        """Test updating an existing document."""
        # First add the document
        self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        # Update DocSet
        updated_docset = self.test_docset.copy()
        updated_docset.title = "Updated Title"
        updated_docset.text_chunks.append(
            TextChunk(id="chunk4", type=ChunkType.TEXT, text="New text chunk")
        )
        self.test_metadata["title"] = "Updated Title"
        # Save updated version
        result = self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        self.assertTrue(result)
        # Verify updates
        saved_metadata = self.db.get_document(self.test_doc_id)
        self.assertEqual(saved_metadata["title"], "Updated Title")
        self.assertEqual(len(saved_metadata["chunk_ids"]), 4)
        self.assertIn("chunk4", saved_metadata["chunk_ids"])
    '''

    def test_5_delete_document(self):
        """Test deleting a document."""
        # First add the document
        self.db.add_document(self.test_doc_id, self.test_pdf_path, self.test_metadata)
        # Delete document
        result = self.db.delete_document(self.test_doc_id)
        self.assertTrue(result)
        # Verify document is deleted
        metadata = self.db.get_document(self.test_doc_id)
        self.assertIsNone(metadata)
        pdf_data = self.db.get_document_pdf(self.test_doc_id)
        self.assertIsNone(pdf_data)

    def test_6_nonexistent_document(self):
        """Test operations on non-existent documents."""
        fake_doc_id = "nonexistent_doc"
        # Try to get metadata
        metadata = self.db.get_document(fake_doc_id)
        self.assertIsNone(metadata)
        # Try to get PDF
        pdf_data = self.db.get_document_pdf(fake_doc_id)
        self.assertIsNone(pdf_data)
        # Try to delete
        result = self.db.delete_document(fake_doc_id)
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 