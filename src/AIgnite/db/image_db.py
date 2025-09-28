from typing import Optional, List
import os
import logging
from minio import Minio
from minio.error import S3Error
from PIL import Image
import io

'''
docker run -p 9081:9081 -p 9091:9091 \
  -e MINIO_ROOT_USER=XOrv2wfoWfPypp2zGIae \
  -e MINIO_ROOT_PASSWORD=k9agaJuX2ZidOtaBxdc9Q2Hz5GnNKncNBnEZIoK3 \
  -v /home/guofang/AIgnite-Solutions/AIgnite/src/AIgnite/db/minio/data:/data \
  -v /home/guofang/AIgnite-Solutions/AIgnite/src/AIgnite/db/minio/config:/root/.minio \
  quay.io/minio/minio server /data \
  --address ":9081" --console-address ":9091"
'''

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
        object_name: str,
        image_path: str = None,
        image_data: bytes = None
    ) -> bool:
        """Save an image to MinIO storage.
        
        Args:
            object_name: Object name to use for storage in MinIO
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
            
            # Handle image path
            if self.get_image(object_name):
                logging.warning(f"Image with object_name {object_name} already exists")
                return True
            
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
            logging.error(f"Failed to save image with object_name {object_name}: {str(e)}")
            return False

    def get_image(
        self,
        object_name: str,
    ) -> Optional[bytes]:
        """Retrieve an image from MinIO storage.
        
        Args:
            object_name: Object name in MinIO storage
            save_path: Optional path to save the image file
            
        Returns:
            Image bytes if save_path is None, otherwise None
        """
        try:
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
            return image_data
            
        except Exception as e:
            logging.error(f"Failed to get image with object_name {object_name}: {str(e)}")
            return None

    def delete_image(self, image_id: str) -> bool:
        """Delete an image from MinIO storage by image_id.
        
        Args:
            image_id: Image ID (used as object_name in MinIO)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove object from MinIO
            self.client.remove_object(
                bucket_name=self.bucket_name,
                object_name=image_id
            )
            logging.info(f"Successfully deleted image with image_id: {image_id}")
            return True
            
        except S3Error as e:
            if e.code == 'NoSuchKey':
                logging.warning(f"Image not found with image_id: {image_id}")
            else:
                logging.error(f"Failed to delete image with image_id {image_id}: {str(e)}")
            return False
            
        except Exception as e:
            logging.error(f"Failed to delete image with image_id {image_id}: {str(e)}")
            return False 