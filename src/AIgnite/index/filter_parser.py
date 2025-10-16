from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class FilterParser:
    """Parser for handling include/exclude filters in search operations."""
    
    def __init__(self):
        """Initialize the filter parser."""
        self.supported_fields = {
            'categories', 'authors', 'published_date', 'doc_ids',
            'title_keywords', 'abstract_keywords', 'text_type'  # 新增text_type支持
        }
    
    def parse_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Parse and validate filter conditions.
        
        Args:
            filters: Raw filter dictionary
            
        Returns:
            Parsed and validated filter structure, or None if no filters provided
            
        Raises:
            ValueError: If filter structure is invalid
        """
        # Return None for no filters (None or empty dict)
        if not filters:
            return None
        
        # Validate filter structure
        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary")
            
        # Initialize parsed filters structure
        parsed_filters = {
            "include": {},
            "exclude": {}
        }
        
        # Parse include filters
        if "include" in filters:
            include_filters = filters["include"]
            if not isinstance(include_filters, dict):
                raise ValueError("Include filters must be a dictionary")
            parsed_filters["include"] = self._validate_field_filters(include_filters, "include")
            
        # Parse exclude filters
        if "exclude" in filters:
            exclude_filters = filters["exclude"]
            if not isinstance(exclude_filters, dict):
                raise ValueError("Exclude filters must be a dictionary")
            parsed_filters["exclude"] = self._validate_field_filters(exclude_filters, "exclude")
        
        # Check if the parsed filters actually contain any meaningful content
        if not parsed_filters["include"] and not parsed_filters["exclude"]:
            return None
            
        return parsed_filters
    
    def _validate_field_filters(self, field_filters: Dict[str, Any], filter_type: str) -> Dict[str, Any]:
        """Validate individual field filters.
        
        Args:
            field_filters: Dictionary of field-specific filters
            filter_type: Either 'include' or 'exclude'
            
        Returns:
            Validated field filters
            
        Raises:
            ValueError: If field filters are invalid
        """
        validated_filters = {}
        
        for field, value in field_filters.items():
            if field not in self.supported_fields:
                logger.warning(f"Unsupported filter field: {field}")
                continue
                
            if field == "published_date":
                validated_value = self._validate_date_filter(value, filter_type)
            elif field in ["categories", "authors", "doc_ids", "title_keywords", "abstract_keywords", "text_type"]:
                validated_value = self._validate_list_filter(value, field)
            else:
                logger.warning(f"Unknown field type for {field}")
                raise ValueError(f"Unknown field type for {field}")
                continue
                
            if validated_value is not None:
                validated_filters[field] = validated_value
            else:
                logger.warning(f"Unknown value type for {value}")
                continue
                
        return validated_filters
    
    def _validate_date_filter(self, value: Any, filter_type: str) -> Optional[Dict[str, str]]:
        """Validate date filter values.
        
        Args:
            value: Date filter value (can be string or list)
            filter_type: Either 'include' or 'exclude'
            
        Returns:
            Validated date filter or None if invalid
        """
        if isinstance(value, str):
            # Single date - treat as exact match
            try:
                datetime.fromisoformat(value)
                return {"exact": value}
            except ValueError:
                logger.error(f"Invalid date format: {value}")
                return None
        elif isinstance(value, list) and len(value) == 2:
            # Date range
            #print('VALIDATE DATE RANGE: ', value)
            try:
                start_date = datetime.fromisoformat(value[0])
                end_date = datetime.fromisoformat(value[1])
                end_date = end_date + timedelta(days=1)
                #print('END DATE: ', end_date.isoformat())
                if start_date > end_date:
                    logger.error(f"Start date {value[0]} is after end date {value[1]}")
                    return None
                return {"range": [value[0], end_date.isoformat()]}
            except (ValueError, IndexError):
                logger.error(f"Invalid date range format: {value}")
                return None
        else:
            logger.error(f"Invalid date filter format: {value}")
            return None
    
    def _validate_list_filter(self, value: Any, field: str) -> Optional[List[str]]:
        """Validate list-based filter values.
        
        Args:
            value: List filter value
            field: Field name for logging
            
        Returns:
            Validated list or None if invalid
        """
        if field == "text_type":
            # 验证text_type值是否有效
            valid_types = {'abstract', 'chunk', 'combined'}
            if isinstance(value, str):
                if value in valid_types:
                    return [value]
                else:
                    raise ValueError(f"Invalid text_type value: {value}")
            elif isinstance(value, list):
                if all(t in valid_types for t in value):
                    return value
                else:
                    raise ValueError(f"Invalid text_type values: {value}")
            #    else:
            #        logger.error(f"Invalid text_type values: {value}")
            #        raise ValueError(f"Invalid text_type values: {value}")
            #        
            #        raise ValueError(f"Invalid text_type values: {value}")
            #        return None
            else:
                #logger.error(f"Invalid text_type filter format: {value}")
                raise ValueError(f"Invalid text_type filter format: {value}")
                #return None
        
        # 其他字段的原有逻辑
        if isinstance(value, str):
            # Single value - convert to list
            return [value]
        elif isinstance(value, list):
            # Ensure all items are strings
            if all(isinstance(item, str) for item in value):
                return value
            else:
                raise ValueError(f"Invalid {field} filter format: {value}")
                #logger.error(f"All values in {field} filter must be strings")
                #return None
        else:
            logger.error(f"Invalid {field} filter format: {value}")
            return None
    
    def get_sql_conditions(self, filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate SQL WHERE conditions for metadata database.
        
        Args:
            filters: Parsed filter structure
            
        Returns:
            Tuple of (WHERE clause, parameter dictionary)
        """
        conditions = []
        params = {}
        param_counter = 0
        
        # Handle include filters
        for field, value in filters.get("include", {}).items():
            if field == "doc_ids":
                if value:
                    placeholders = ','.join([f"'{doc_id}'" for doc_id in value])
                    conditions.append(f"doc_id IN ({placeholders})")
            elif field == "categories":
                if value:
                    placeholders = ','.join([f"'{cat}'" for cat in value])
                    conditions.append(f"categories ?| ARRAY[{placeholders}]")
            elif field == "authors":
                if value:
                    author_conditions = []
                    for author in value:
                        param_name = f"author_{param_counter}"
                        author_conditions.append(f"authors::text ILIKE '%{author}%'")
                        param_counter += 1
                    conditions.append(f"({' OR '.join(author_conditions)})")
            elif field == "published_date":
                if "range" in value:
                    start_date, end_date = value["range"]
                    # Use explicit >= and <= for closed interval (inclusive of both boundaries)
                    conditions.append("published_date >= :filter_start_date AND published_date <= :filter_end_date")
                    params["filter_start_date"] = start_date
                    params["filter_end_date"] = end_date
                elif "exact" in value:
                    conditions.append("published_date = :filter_exact_date")
                    params["filter_exact_date"] = value["exact"]
            elif field in ["title_keywords", "abstract_keywords"]:
                if value:
                    keyword_conditions = []
                    for keyword in value:
                        param_name = f"keyword_{param_counter}"
                        if field == "title_keywords":
                            keyword_conditions.append(f"title ILIKE '%{keyword}%'")
                        else:
                            keyword_conditions.append(f"abstract ILIKE '%{keyword}%'")
                        param_counter += 1
                    conditions.append(f"({' OR '.join(keyword_conditions)})")
        
        # Handle exclude filters
        for field, value in filters.get("exclude", {}).items():
            if field == "doc_ids":
                if value:
                    placeholders = ','.join([f"'{doc_id}'" for doc_id in value])
                    conditions.append(f"doc_id NOT IN ({placeholders})")
            elif field == "categories":
                if value:
                    placeholders = ','.join([f"'{cat}'" for cat in value])
                    conditions.append(f"NOT (categories ?| ARRAY[{placeholders}])")
            elif field == "authors":
                if value:
                    author_conditions = []
                    for author in value:
                        author_conditions.append(f"authors::text NOT ILIKE '%{author}%'")
                    conditions.append(f"({' AND '.join(author_conditions)})")
            elif field == "published_date":
                if "range" in value:
                    start_date, end_date = value["range"]
                    # Exclude closed interval: NOT (date >= start AND date <= end)
                    conditions.append("NOT (published_date >= :filter_exclude_start_date AND published_date <= :filter_exclude_end_date)")
                    params["filter_exclude_start_date"] = start_date
                    params["filter_exclude_end_date"] = end_date
                elif "exact" in value:
                    conditions.append("published_date != :filter_exclude_exact_date")
                    params["filter_exclude_exact_date"] = value["exact"]
            elif field in ["title_keywords", "abstract_keywords"]:
                if value:
                    keyword_conditions = []
                    for keyword in value:
                        if field == "title_keywords":
                            keyword_conditions.append(f"title NOT ILIKE '%{keyword}%'")
                        else:
                            keyword_conditions.append(f"abstract NOT ILIKE '%{keyword}%'")
                    conditions.append(f"({' AND '.join(keyword_conditions)})")
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params
    
    def apply_memory_filters(self, items: List[Any], filters: Dict[str, Any], 
                           get_field_value: callable) -> List[Any]:
        """Apply filters to items in memory (for vector database).
        
        Args:
            items: List of items to filter
            filters: Parsed filter structure
            get_field_value: Function to extract field values from items
            
        Returns:
            Filtered list of items
        """
        if not filters:
            return items
            
        filtered_items = []
        
        for item in items:
            include_item = True
            
            # Check include filters
            for field, value in filters.get("include", {}).items():
                if not self._check_include_filter(item, field, value, get_field_value):
                    include_item = False
                    break
            
            if not include_item:
                continue
                
            # Check exclude filters
            for field, value in filters.get("exclude", {}).items():
                if self._check_exclude_filter(item, field, value, get_field_value):
                    include_item = False
                    break
            
            if include_item:
                filtered_items.append(item)
                
        return filtered_items
    
    def _check_include_filter(self, item: Any, field: str, value: Any, 
                            get_field_value: callable) -> bool:
        """Check if item passes include filter.
        
        Args:
            item: Item to check
            field: Field name
            value: Expected value(s)
            get_field_value: Function to extract field value
            
        Returns:
            True if item passes filter, False otherwise
        """
        item_value = get_field_value(item, field)
        if item_value is None:
            return False
            
        if field == "text_type":
            # text_type过滤：检查item的text_type是否在允许的类型列表中
            return item_value in value
        elif field == "categories":
            return any(cat in item_value for cat in value)
        elif field == "authors":
            return any(author.lower() in str(item_value).lower() for author in value)
        elif field == "doc_ids":
            return item_value in value
        elif field in ["title_keywords", "abstract_keywords"]:
            text_value = str(item_value).lower()
            return any(keyword.lower() in text_value for keyword in value)
        elif field == "published_date":
            if "range" in value:
                start_date, end_date = value["range"]
                return start_date <= item_value <= end_date
            elif "exact" in value:
                return item_value == value["exact"]
        
        return False
    
    def _check_exclude_filter(self, item: Any, field: str, value: Any, 
                            get_field_value: callable) -> bool:
        """Check if item passes exclude filter.
        
        Args:
            item: Item to check
            field: Field name
            value: Value(s) to exclude
            get_field_value: Function to extract field value
            
        Returns:
            True if item should be excluded, False otherwise
        """
        item_value = get_field_value(item, field)
        if item_value is None:
            return False
            
        if field == "text_type":
            # text_type过滤：检查item的text_type是否在排除的类型列表中
            return item_value in value
        elif field == "categories":
            return any(cat in item_value for cat in value)
        elif field == "authors":
            return any(author.lower() in str(item_value).lower() for author in value)
        elif field == "doc_ids":
            return item_value in value
        elif field in ["title_keywords", "abstract_keywords"]:
            text_value = str(item_value).lower()
            return any(keyword.lower() in text_value for keyword in value)
        elif field == "published_date":
            if "range" in value:
                start_date, end_date = value["range"]
                return start_date <= item_value <= end_date
            elif "exact" in value:
                return item_value == value["exact"]
        
        return False