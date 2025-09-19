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
        print("✅ Initializing test MinIO image database...")
        
        # MinIO test server configuration
        cls.endpoint = "localhost:9081"
        cls.access_key = "XOrv2wfoWfPypp2zGIae"  # Default MinIO access key
        cls.secret_key = "k9agaJuX2ZidOtaBxdc9Q2Hz5GnNKncNBnEZIoK3"  # Default MinIO secret key
        cls.bucket_name = "aignite-test-paper-test"
        
        # Initialize MinIO client
        print(f"✅ Connecting to MinIO server at {cls.endpoint}...")
        cls.minio_db = MinioImageDB(
            endpoint=cls.endpoint,
            access_key=cls.access_key,
            secret_key=cls.secret_key,
            bucket_name=cls.bucket_name,
            secure=False  # Use HTTP for local testing
        )
        
        # Create temporary directory for test files
        print("✅ Creating temporary test files...")
        cls.temp_dir = tempfile.mkdtemp()
        
        # Create test images
        cls.test_doc_id = "2106.14834"
        cls.test_images = {}
        
        print(f"✅ Generating {3} test images...")
        # Create multiple test images with different sizes and content
        for i in range(3):
            image_id = f"{cls.test_doc_id}_img_{i}"
            image_path = os.path.join(cls.temp_dir, f"test_image_{i}.png")
            
            # Create a test image using PIL
            img = Image.new('RGB', (100 + i*50, 100 + i*50), color=f'rgb({i*50}, {i*50}, {i*50})')
            img.save(image_path)
            
            cls.test_images[image_id] = image_path
        
        print("✅ Test setup completed successfully!")

    def setUp(self):
        """Clean up any existing test images before each test"""
        # Clean up test images by deleting them individually
        for image_id in self.test_images.keys():
            try:
                self.minio_db.delete_image(image_id)
            except:
                pass  # Ignore errors during cleanup

    def test_save_image(self):
        """Test saving a single image"""
        print("🧪 Running test: save_image")
        image_id = f"{self.test_doc_id}_img_0"
        image_path = self.test_images[image_id]
        
        # Save image
        result = self.minio_db.save_image(
            object_name=image_id,
            image_path=image_path
        )
        self.assertTrue(result)
        
        # Verify image was saved by retrieving it
        image_data = self.minio_db.get_image(image_id)
        self.assertIsNotNone(image_data)
        print("✅ test_save_image passed!")

    def test_save_multiple_images(self):
        """Test saving multiple images"""
        print("🧪 Running test: save_multiple_images")
        # Save all test images
        for image_id, image_path in self.test_images.items():
            result = self.minio_db.save_image(
                object_name=image_id,
                image_path=image_path
            )
            self.assertTrue(result)
        
        # Verify all images were saved by retrieving them
        for image_id in self.test_images.keys():
            image_data = self.minio_db.get_image(image_id)
            self.assertIsNotNone(image_data)
        print("✅ test_save_multiple_images passed!")

    def test_get_image_as_bytes(self):
        """Test retrieving image as bytes"""
        print("🧪 Running test: get_image_as_bytes")
        image_id = f"{self.test_doc_id}_img_0"
        image_path = self.test_images[image_id]
        
        # First save the image
        self.minio_db.save_image(
            object_name=image_id,
            image_path=image_path
        )
        
        # Retrieve image as bytes
        image_data = self.minio_db.get_image(image_id)
        
        self.assertIsNotNone(image_data)
        # Verify it's valid image data by trying to open it with PIL
        img = Image.open(io.BytesIO(image_data))
        self.assertEqual(img.size, (100, 100))  # First test image size
        print("✅ test_get_image_as_bytes passed!")

    def test_delete_existing_image(self):
        """Test deleting an existing image"""
        print("🧪 Running test: delete_existing_image")
        image_id = f"{self.test_doc_id}_img_1"
        image_path = self.test_images[image_id]
        
        # First save the image
        result = self.minio_db.save_image(
            object_name=image_id,
            image_path=image_path
        )
        self.assertTrue(result)
        
        # Verify image exists
        image_data = self.minio_db.get_image(image_id)
        self.assertIsNotNone(image_data)
        
        # Delete the image
        delete_result = self.minio_db.delete_image(image_id)
        self.assertTrue(delete_result)
        
        # Verify image is deleted
        deleted_image_data = self.minio_db.get_image(image_id)
        self.assertIsNone(deleted_image_data)
        print("✅ test_delete_existing_image passed!")

    '''
    def test_delete_nonexistent_image(self):
        """Test deleting a non-existent image"""
        non_existent_image_id = "non_existent_image"

        # Ensure the image doesn't exist first
        if self.minio_db.get_image(non_existent_image_id) is not None:
            self.minio_db.delete_image(non_existent_image_id)
            self.asser
        # Try to delete non-existent image
        
        self.assertFalse(delete_result)
    '''

    def test_delete_image_verification(self):
        """Test complete workflow: save, verify, delete, verify"""
        print("🧪 Running test: delete_image_verification")
        image_id = f"{self.test_doc_id}_img_2"
        image_path = self.test_images[image_id]
        
        # Step 1: Save image
        save_result = self.minio_db.save_image(
            object_name=image_id,
            image_path=image_path
        )
        self.assertTrue(save_result)
        
        # Step 2: Verify image exists
        image_data = self.minio_db.get_image(image_id)
        self.assertIsNotNone(image_data)
        
        # Step 3: Delete image
        delete_result = self.minio_db.delete_image(image_id)
        self.assertTrue(delete_result)
        
        # Step 4: Verify image is deleted
        deleted_image_data = self.minio_db.get_image(image_id)
        self.assertIsNone(deleted_image_data)
        print("✅ test_delete_image_verification passed!")


    @classmethod
    def tearDownClass(cls):
        """Clean up test data and MinIO bucket"""
        print("🧹 Cleaning up test data...")
        
        # Delete all test images
        try:
            for image_id in cls.test_images.keys():
                cls.minio_db.delete_image(image_id)
        except:
            pass  # Ignore errors during cleanup
        
        # Clean up temporary directory
        try:
            shutil.rmtree(cls.temp_dir)
        except:
            pass  # Ignore errors during cleanup
        
        print("✅ Test cleanup completed!")

if __name__ == '__main__':
    unittest.main() 