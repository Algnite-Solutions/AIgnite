import unittest
import os
import tempfile
import shutil
from PIL import Image
import io
import logging
from AIgnite.db.image_db import MinioImageDB

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class TestMinioImageDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test MinIO connection and test data"""
        # MinIO test server configuration
        cls.endpoint = "localhost:9000"
        cls.access_key = "Wc58W6KOCOfxaAc2MjzM"  # Default MinIO access key
        cls.secret_key = "Seqa7Xs3TO6EaClfD4l6y4b4teW7t2Y1Slu92VKw"  # Default MinIO secret key
        cls.bucket_name = "aignite-test-images"
        
        # Initialize MinIO client
        cls.minio_db = MinioImageDB(
            endpoint=cls.endpoint,
            access_key=cls.access_key,
            secret_key=cls.secret_key,
            bucket_name=cls.bucket_name,
            secure=False  # Use HTTP for local testing
        )
        
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        cls.test_doc_id = "2106.14834"
        cls.test_images = {}
        
        # Create multiple test images with different sizes and content
        for i in range(3):
            image_id = f"{cls.test_doc_id}_img_{i}"
            image_path = os.path.join(cls.temp_dir, f"test_image_{i}.png")
            
            # Create a test image using PIL
            img = Image.new('RGB', (100 + i*50, 100 + i*50), color=f'rgb({i*50}, {i*50}, {i*50})')
            img.save(image_path)
            
            cls.test_images[image_id] = image_path

    def setUp(self):
        """Clean up any existing test images before each test"""
        self.minio_db.delete_doc_images(self.test_doc_id)

    def test_save_image(self):
        """Test saving a single image"""
        image_id = f"{self.test_doc_id}_img_0"
        image_path = self.test_images[image_id]
        
        # Save image
        result = self.minio_db.save_image(
            doc_id=self.test_doc_id,
            image_id=image_id,
            image_path=image_path
        )
        self.assertTrue(result)
        
        # Verify image was saved by listing images
        saved_images = self.minio_db.list_doc_images(self.test_doc_id)
        self.assertIn(image_id, saved_images)

    def test_save_multiple_images(self):
        """Test saving multiple images for the same document"""
        # Save all test images
        for image_id, image_path in self.test_images.items():
            result = self.minio_db.save_image(
                doc_id=self.test_doc_id,
                image_id=image_id,
                image_path=image_path
            )
            self.assertTrue(result)
        
        # Verify all images were saved
        saved_images = self.minio_db.list_doc_images(self.test_doc_id)
        self.assertEqual(len(saved_images), len(self.test_images))
        for image_id in self.test_images.keys():
            self.assertIn(image_id, saved_images)

    def test_get_image_as_bytes(self):
        """Test retrieving image as bytes"""
        image_id = f"{self.test_doc_id}_img_0"
        image_path = self.test_images[image_id]
        
        # First save the image
        self.minio_db.save_image(
            doc_id=self.test_doc_id,
            image_id=image_id,
            image_path=image_path
        )
        
        # Retrieve image as bytes
        image_data = self.minio_db.get_image(
            doc_id=self.test_doc_id,
            image_id=image_id
        )
        
        self.assertIsNotNone(image_data)
        # Verify it's valid image data by trying to open it with PIL
        img = Image.open(io.BytesIO(image_data))
        self.assertEqual(img.size, (100, 100))  # First test image size

    def test_get_image_save_to_file(self):
        """Test retrieving image and saving to file"""
        image_id = f"{self.test_doc_id}_img_1"
        original_path = self.test_images[image_id]
        
        # First save the image
        self.minio_db.save_image(
            doc_id=self.test_doc_id,
            image_id=image_id,
            image_path=original_path
        )
        
        # Retrieve and save to new file
        output_path = os.path.join(self.temp_dir, "retrieved_image.png")
        self.minio_db.get_image(
            doc_id=self.test_doc_id,
            image_id=image_id,
            save_path=output_path
        )
        
        # Verify file exists and is an image
        self.assertTrue(os.path.exists(output_path))
        img = Image.open(output_path)
        self.assertEqual(img.size, (150, 150))  # Second test image size

    def test_delete_doc_images(self):
        """Test deleting all images for a document"""
        # First save all test images
        for image_id, image_path in self.test_images.items():
            self.minio_db.save_image(
                doc_id=self.test_doc_id,
                image_id=image_id,
                image_path=image_path
            )
        
        # Verify images were saved
        saved_images = self.minio_db.list_doc_images(self.test_doc_id)
        self.assertEqual(len(saved_images), len(self.test_images))
        
        # Delete all images
        result = self.minio_db.delete_doc_images(self.test_doc_id)
        self.assertTrue(result)
        
        # Verify images were deleted
        remaining_images = self.minio_db.list_doc_images(self.test_doc_id)
        self.assertEqual(len(remaining_images), 0)

    def test_list_doc_images(self):
        """Test listing all images for a document"""
        # First save all test images
        for image_id, image_path in self.test_images.items():
            self.minio_db.save_image(
                doc_id=self.test_doc_id,
                image_id=image_id,
                image_path=image_path
            )
        
        # List images
        image_list = self.minio_db.list_doc_images(self.test_doc_id)
        
        # Verify list contents
        self.assertEqual(len(image_list), len(self.test_images))
        for image_id in self.test_images.keys():
            self.assertIn(image_id, image_list)

    def test_nonexistent_image(self):
        """Test handling of non-existent images"""
        # Try to get non-existent image
        image_data = self.minio_db.get_image(
            doc_id=self.test_doc_id,
            image_id="nonexistent_image"
        )
        self.assertIsNone(image_data)

    @classmethod
    def tearDownClass(cls):
        """Clean up test data and MinIO bucket"""
        # Delete test bucket and all its contents
        try:
            cls.minio_db.delete_doc_images(cls.test_doc_id)
        except:
            pass  # Ignore errors during cleanup
        
        # Clean up temporary directory
        try:
            shutil.rmtree(cls.temp_dir)
        except:
            pass  # Ignore errors during cleanup

if __name__ == '__main__':
    unittest.main() 