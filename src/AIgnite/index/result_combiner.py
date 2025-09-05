from typing import List, Dict, Any
from .search_strategy import SearchResult
import logging

logger = logging.getLogger(__name__)

class ResultCombiner:
    """简化的结果合并器，将搜索结果与获取的数据进行合并"""
    
    def __init__(self):
        """初始化结果合并器"""
        # 支持的数据类型
        self.supported_include_types = {
            "metadata": "论文元数据",
            "search_parameters": "搜索参数",
            "text_chunks": "文本块内容",
            "full_text": "完整文本",
            "images": "图像数据"
        }
        
        # 默认包含的数据类型
        self.default_include_types = ["metadata", "search_parameters"]
    
    def combine(self, search_results: List[SearchResult], 
                data_dict: Dict[str, Dict[str, Any]], 
                include_types: List[str]) -> List[Dict[str, Any]]:
        """合并搜索结果和数据
        
        Args:
            search_results: 搜索结果列表
            data_dict: 以数据类型为key的数据字典，每个数据类型包含以doc_id为key的数据
            include_types: 需要包含的数据类型列表
            
        Returns:
            合并后的结果列表
        """
        if not search_results:
            return []
        
        # 验证和清理include_types
        validated_types = self._validate_include_types(include_types)
        
        combined_results = []
        for result in search_results:
            try:
                combined = {"doc_id": result.doc_id}
                
                # 添加搜索参数
                if "search_parameters" in validated_types:
                    self._add_search_parameters(combined, result)
                
                # 添加其他数据类型
                for data_type in validated_types:
                    if data_type != "search_parameters":
                        self._add_data_type(combined, result.doc_id, data_type, data_dict)
                
                combined_results.append(combined)
                
            except Exception as e:
                logger.error(f"Failed to combine result for doc_id {result.doc_id}: {str(e)}")
                # 添加基本结果，即使合并失败
                basic_result = {
                    "doc_id": result.doc_id,
                    "search_parameters": {
                        "similarity_score": result.score,
                        "search_method": result.search_method,
                    },
                    "error": f"Failed to combine data: {str(e)}"
                }
                combined_results.append(basic_result)
        
        logger.debug(f"Successfully combined {len(combined_results)} results")
        return combined_results
    
    def _validate_include_types(self, include_types: List[str]) -> List[str]:
        """验证包含的数据类型
        
        Args:
            include_types: 数据类型列表
            
        Returns:
            验证后的数据类型列表
        """
        if not include_types:
            return self.default_include_types
        
        valid_types = []
        for type_name in include_types:
            if type_name in self.supported_include_types:
                valid_types.append(type_name)
            else:
                logger.warning(f"Unsupported include type: {type_name}")
        
        # 如果没有有效类型，使用默认类型
        if not valid_types:
            logger.warning("No valid include types found, using defaults")
            return self.default_include_types
        
        return valid_types
    
    def _add_search_parameters(self, combined: Dict[str, Any], search_result: SearchResult):
        """添加搜索参数到合并结果
        
        Args:
            combined: 合并结果字典
            search_result: 搜索结果对象
        """
        combined['search_parameters'] = {"similarity_score": search_result.score,
            "search_method": search_result.search_method,
            "matched_text": search_result.matched_text,
            "text_type": search_result.metadata.get("text_type", None),
            "chunk_id": search_result.chunk_id
        }
        
        # 添加元数据中的搜索相关信息
        # if hasattr(search_result, 'metadata') and search_result.metadata:
            # 添加向量搜索相关信息
            # if 'vector_score' in search_result.metadata:
                # combined['vector_score'] = search_result.metadata['vector_score']
            # if 'text_type' in search_result.metadata:
                # combined['text_type'] = search_result.metadata['text_type']
    
    def _add_data_type(self, combined: Dict[str, Any], doc_id: str, 
                       data_type: str, data_dict: Dict[str, Dict[str, Any]]):
        """添加指定类型的数据到合并结果
        
        Args:
            combined: 合并结果字典
            doc_id: 文档ID
            data_type: 数据类型
            data_dict: 数据字典
        """
        if data_type in data_dict and doc_id in data_dict[data_type]:
            combined[data_type] = data_dict[data_type][doc_id]
        else:
            # 如果数据不存在，添加空值或默认值
            if data_type == "metadata":
                combined[data_type] = None
            elif data_type == "text_chunks":
                combined[data_type] = []
            elif data_type == "full_text":
                combined[data_type] = ""
            elif data_type == "images":
                combined[data_type] = []
            else:
                combined[data_type] = None
            
            logger.debug(f"No {data_type} data found for doc_id: {doc_id}")
    
    def get_supported_include_types(self) -> Dict[str, str]:
        """获取支持的包含数据类型
        
        Returns:
            支持的包含数据类型字典
        """
        return self.supported_include_types.copy()
    
    def get_default_include_types(self) -> List[str]:
        """获取默认的包含数据类型
        
        Returns:
            默认的包含数据类型列表
        """
        return self.default_include_types.copy()
    
    def is_include_type_supported(self, include_type: str) -> bool:
        """检查包含数据类型是否支持
        
        Args:
            include_type: 包含数据类型
            
        Returns:
            是否支持该包含数据类型
        """
        return include_type in self.supported_include_types
