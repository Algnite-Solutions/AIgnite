from typing import List, Dict, Any, Optional
from ..db.metadata_db import MetadataDB
from ..db.image_db import MinioImageDB
import logging

logger = logging.getLogger(__name__)

class DataRetriever:
    """简化的数据获取器，支持批量获取不同类型的数据"""
    
    def __init__(self, metadata_db: Optional[MetadataDB] = None, image_db: Optional[MinioImageDB] = None):
        """初始化数据获取器
        
        Args:
            metadata_db: MetadataDB实例，用于获取元数据和文本数据
            image_db: MinioImageDB实例，用于获取图像数据
        """
        self.metadata_db = metadata_db
        self.image_db = image_db
        
        # 支持的数据类型
        self.supported_data_types = {
            "metadata": "论文元数据",
            "text_chunks": "文本块内容", 
            "full_text": "完整文本",
            "images": "图像数据"
        }
    
    def get_data_by_type(self, doc_ids: List[str], data_type: str) -> Dict[str, Any]:
        """根据数据类型批量获取数据
        
        Args:
            doc_ids: 文档ID列表
            data_type: 数据类型，支持 "metadata", "text_chunks", "full_text", "images"
            
        Returns:
            以doc_id为key的数据字典
        """
        if not doc_ids:
            return {}
        
        if data_type not in self.supported_data_types:
            logger.warning(f"Unsupported data type: {data_type}")
            return {}
        
        try:
            if data_type == "metadata":
                return self._get_metadata(doc_ids)
            elif data_type == "text_chunks":
                return self._get_text_chunks(doc_ids)
            elif data_type == "full_text":
                return self._get_full_text(doc_ids)
            elif data_type == "images":
                return self._get_images(doc_ids)
            else:
                return {}
        except Exception as e:
            logger.error(f"Failed to get {data_type} data for doc_ids {doc_ids}: {str(e)}")
            return {}
    
    def _get_metadata(self, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量获取元数据
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            以doc_id为key的元数据字典
        """
        if not self.metadata_db:
            logger.warning("MetadataDB not available")
            return {}
        
        result = {}
        for doc_id in doc_ids:
            try:
                metadata = self.metadata_db.get_metadata(doc_id)
                if metadata:
                    result[doc_id] = metadata
                else:
                    logger.debug(f"No metadata found for doc_id: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to get metadata for doc_id {doc_id}: {str(e)}")
                continue
        
        logger.debug(f"Retrieved metadata for {len(result)} out of {len(doc_ids)} documents")
        return result
    
    def _get_text_chunks(self, doc_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """批量获取文本块
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            以doc_id为key的文本块列表字典
        """
        if not self.metadata_db:
            logger.warning("MetadataDB not available")
            return {}
        
        result = {}
        for doc_id in doc_ids:
            try:
                text_chunks = self.metadata_db.get_text_chunks(doc_id)
                if text_chunks:
                    result[doc_id] = text_chunks
                else:
                    logger.debug(f"No text chunks found for doc_id: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to get text chunks for doc_id {doc_id}: {str(e)}")
                continue
        
        logger.debug(f"Retrieved text chunks for {len(result)} out of {len(doc_ids)} documents")
        return result
    
    def _get_full_text(self, doc_ids: List[str]) -> Dict[str, str]:
        """批量获取完整文本
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            以doc_id为key的完整文本字典
        """
        if not self.metadata_db:
            logger.warning("MetadataDB not available")
            return {}
        
        result = {}
        for doc_id in doc_ids:
            try:
                full_text = self.metadata_db.get_full_text(doc_id)
                if full_text:
                    result[doc_id] = full_text
                else:
                    logger.debug(f"No full text found for doc_id: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to get full text for doc_id {doc_id}: {str(e)}")
                continue
        
        logger.debug(f"Retrieved full text for {len(result)} out of {len(doc_ids)} documents")
        return result
    
    def _get_images(self, doc_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """批量获取图像信息
        
        Args:
            doc_ids: 文档ID列表
            
        Returns:
            以doc_id为key的图像信息列表字典
        """
        if not self.image_db:
            logger.warning("ImageDB not available")
            return {}
        
        result = {}
        for doc_id in doc_ids:
            try:
                # 获取文档的所有图像ID
                image_ids = self.image_db.list_doc_images(doc_id)
                if image_ids:
                    # 构建图像信息列表
                    images_info = []
                    for image_id in image_ids:
                        images_info.append({
                            'image_id': image_id,
                            'doc_id': doc_id,
                            'object_name': f"{doc_id}/{image_id}"
                        })
                    result[doc_id] = images_info
                else:
                    logger.debug(f"No images found for doc_id: {doc_id}")
            except Exception as e:
                logger.error(f"Failed to get images for doc_id {doc_id}: {str(e)}")
                continue
        
        logger.debug(f"Retrieved image info for {len(result)} out of {len(doc_ids)} documents")
        return result
    
    def get_supported_data_types(self) -> Dict[str, str]:
        """获取支持的数据类型
        
        Returns:
            支持的数据类型字典
        """
        return self.supported_data_types.copy()
    
    def is_data_type_supported(self, data_type: str) -> bool:
        """检查数据类型是否支持
        
        Args:
            data_type: 数据类型
            
        Returns:
            是否支持该数据类型
        """
        return data_type in self.supported_data_types
