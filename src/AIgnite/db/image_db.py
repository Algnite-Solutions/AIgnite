from typing import Optional, List
import os
import logging
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io

class MinioImageDB:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        secure: bool = False
    ):
        """Initialize MinIO client for image storage.
        
        Args:
            endpoint: MinIO server endpoint
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: MinIO bucket name
            secure: Whether to use HTTPS
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        
        self.bucket_name = bucket_name
        
        # Create bucket if it doesn't exist
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logging.info(f"Created bucket {bucket_name}")
        except S3Error as e:
            logging.error(f"Error initializing MinIO bucket: {str(e)}")
            raise RuntimeError(f"Failed to initialize MinIO: {str(e)}")

    def save_image(
        self,
        doc_id: str,
        image_id: str,
        image_path: str = None,
        image_data: bytes = None
    ) -> bool:
        """Save an image to MinIO storage.
        
        Args:
            doc_id: Document ID
            image_id: Image ID
            image_path: Path to image file (mutually exclusive with image_data)
            image_data: Raw image bytes (mutually exclusive with image_path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not image_path and not image_data:
                raise ValueError("Either image_path or image_data must be provided")
            
            if image_path and image_data:
                raise ValueError("Only one of image_path or image_data should be provided")
            
            # Generate MinIO object name
            object_name = f"{doc_id}/{image_id}"
            
            # Handle image path
            if image_path:
                # Verify file exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                
                # Upload file
                self.client.fput_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    file_path=image_path
                )
            
            # Handle image data
            else:
                # Verify it's valid image data
                try:
                    Image.open(io.BytesIO(image_data))
                except:
                    raise ValueError("Invalid image data")
                
                # Upload bytes
                self.client.put_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name,
                    data=io.BytesIO(image_data),
                    length=len(image_data)
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save image {image_id} for doc {doc_id}: {str(e)}")
            return False

    def get_image(
        self,
        doc_id: str,
        image_id: str,
        save_path: str = None
    ) -> Optional[bytes]:
        """Retrieve an image from MinIO storage.
        
        Args:
            doc_id: Document ID
            image_id: Image ID
            save_path: Optional path to save the image file
            
        Returns:
            Image bytes if save_path is None, otherwise None
        """
        try:
            object_name = f"{doc_id}/{image_id}"
            
            # Get object data
            try:
                response = self.client.get_object(
                    bucket_name=self.bucket_name,
                    object_name=object_name
                )
                image_data = response.read()
                response.close()
                response.release_conn()
            except S3Error as e:
                if e.code == 'NoSuchKey':
                    return None
                raise
            
            # Save to file if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'wb') as f:
                    f.write(image_data)
                return None
            
            return image_data
            
        except Exception as e:
            logging.error(f"Failed to get image {image_id} for doc {doc_id}: {str(e)}")
            return None

    def list_doc_images(self, doc_id: str) -> List[str]:
        """List all images for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of image IDs
        """
        try:
            prefix = f"{doc_id}/"
            images = []
            
            # List all objects with prefix
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix
            )
            
            # Extract image IDs from object names
            for obj in objects:
                image_id = obj.object_name.split('/')[-1]
                images.append(image_id)
            
            return images
            
        except Exception as e:
            logging.error(f"Failed to list images for doc {doc_id}: {str(e)}")
            return []

    def delete_doc_images(self, doc_id: str) -> bool:
        """Delete all images for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            prefix = f"{doc_id}/"
            
            # List all objects to delete
            objects = self.client.list_objects(
                bucket_name=self.bucket_name,
                prefix=prefix
            )
            
            # Delete each object
            for obj in objects:
                self.client.remove_object(
                    bucket_name=self.bucket_name,
                    object_name=obj.object_name
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete images for doc {doc_id}: {str(e)}")
            return False 