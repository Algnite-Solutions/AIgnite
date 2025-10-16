### 过滤器实现机制 - 两阶段过滤架构

## 架构概述

系统采用**两阶段过滤架构**，将复杂的过滤逻辑与搜索逻辑解耦，提升性能和代码可维护性。

### 两阶段过滤流程

**第一阶段：预过滤（PaperIndexer + MetadataDB）**
```python
# 在 PaperIndexer.find_similar_papers 中
if filters:
    # 使用 MetadataDB 处理所有复杂过滤条件
    candidate_doc_ids = self.metadata_db.get_filtered_doc_ids(filters)
    # 转换为简化的 doc_ids 过滤
    simplified_filters = {"include": {"doc_ids": candidate_doc_ids}}
```

**第二阶段：简化过滤（SearchStrategy + VectorDB/MetadataDB）**
```python
# SearchStrategy 只接收简化的 doc_ids 过滤
search_results = self.search_strategy.search(
    query=query,
    top_k=top_k,
    filters=simplified_filters  # 仅包含 doc_ids
)
```

### 架构优势

1. **职责分离**：
   - MetadataDB：负责复杂过滤（categories、authors、published_date等）
   - SearchStrategy：专注于搜索（向量搜索或全文搜索）

2. **性能优化**：
   - 利用PostgreSQL索引进行高效过滤
   - 减少搜索空间，提升向量搜索效率
   - 避免在搜索层重复过滤逻辑

3. **代码简化**：
   - 搜索策略不需要理解复杂的过滤语法
   - 过滤逻辑集中在一处，易于维护和扩展

4. **灵活性**：
   - 可以轻松添加新的过滤字段，无需修改搜索策略
   - 支持复杂的过滤组合（include/exclude）

## 各层过滤器实现详解

### 0. PaperIndexer 层 - 过滤协调

**职责**：协调两阶段过滤流程

```python
def find_similar_papers(self, query, top_k, filters, search_strategies, result_include_types):
    # 第一阶段：预过滤
    simplified_filters = None
    if filters:
        # 使用 MetadataDB 处理复杂过滤条件
        candidate_doc_ids = self.metadata_db.get_filtered_doc_ids(filters)
        
        if not candidate_doc_ids:
            return []  # 没有候选文档，直接返回
        
        # 转换为简化格式
        simplified_filters = {"include": {"doc_ids": candidate_doc_ids}}
    
    # 第二阶段：执行搜索（使用简化过滤）
    search_results = self._execute_search(query, top_k, simplified_filters, search_strategies)
    
    # 后续处理：数据获取和结果合并
    # ...
```

**用户输入的完整过滤格式**：
```python
filters = {
    "include": {
        "categories": ["cs.AI", "cs.LG"],
        "authors": ["John Smith"],
        "published_date": ["2023-01-01", "2024-01-01"],
        "title_keywords": ["machine learning"],
        "abstract_keywords": ["deep learning"]
    },
    "exclude": {
        "categories": ["cs.CR"],
        "authors": ["Bob Wilson"]
    }
}
```

**传递给搜索策略的简化格式**：
```python
simplified_filters = {
    "include": {
        "doc_ids": ["doc1", "doc2", "doc3", ...]  # 预过滤后的候选ID列表
    }
}
```

#### 1. FilterParser核心功能

**字段验证**
```python
class FilterParser:
    def __init__(self):
        self.supported_fields = {
            'categories', 'authors', 'published_date', 'doc_ids',
            'title_keywords', 'abstract_keywords', 'text_type'
        }
    
    def _validate_list_filter(self, value: Any, field: str) -> Optional[List[str]]:
        if field == "text_type":
            # 验证text_type值是否有效
            valid_types = {'abstract', 'chunk', 'combined'}
            if isinstance(value, str):
                return [value] if value in valid_types else None
            elif isinstance(value, list):
                if all(t in valid_types for t in value):
                    return value
                else:
                    logger.error(f"Invalid text_type values: {value}")
                    return None
        # ... 其他字段处理逻辑
```

**SQL条件生成**
```python
def get_sql_conditions(self, filters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """为MetadataDB生成SQL WHERE条件"""
    # 支持PostgreSQL特定的操作符
    # categories: JSON数组包含查询 (?|)
    # authors: ILIKE模糊匹配
    # published_date: BETWEEN范围查询
    # title_keywords/abstract_keywords: ILIKE关键词匹配
```

**内存过滤**
```python
def apply_memory_filters(self, items: List[Any], filters: Dict[str, Any], 
                       get_field_value: callable) -> List[Any]:
    """为VectorDB应用内存过滤"""
    # 支持text_type、doc_ids等字段的内存过滤
    # 适用于向量数据库的快速过滤
```

#### 2. MetadataDB 层 - 预过滤执行

**职责**：在第一阶段处理所有复杂过滤条件，返回候选文档ID列表

**核心方法：get_filtered_doc_ids**
```python
def get_filtered_doc_ids(self, filters: Optional[Dict[str, Any]] = None) -> List[str]:
    """Get document IDs that match the filter criteria.
    
    Args:
        filters: 完整的过滤条件（支持复杂字段）
        
    Returns:
        List of document IDs that match the filters
        
    支持的过滤字段：
        - categories: 论文分类
        - authors: 作者
        - published_date: 发布日期
        - title_keywords: 标题关键词
        - abstract_keywords: 摘要关键词
        - doc_ids: 文档ID（用于白名单/黑名单）
    """
    # 使用 FilterParser 生成 SQL 条件
    sql_conditions, params = self.filter_parser.get_sql_conditions(filters)
    
    # 执行 SQL 查询
    query = f"SELECT doc_id FROM papers WHERE {sql_conditions}"
    result = session.execute(text(query), params)
    
    return [row[0] for row in result]
```

**搜索方法（第二阶段）**
```python
def search_papers(self, query: str, filters: Optional[Dict[str, Any]] = None, 
                 top_k: int = 10, similarity_cutoff: float = 0.1):
    """
    在两阶段架构中，MetadataDB的搜索方法只接收简化的doc_ids过滤
    
    Args:
        filters: 简化的过滤条件，格式为 {"include": {"doc_ids": [...]}}
    """
    # 提取候选文档ID列表
    candidate_doc_ids = filters.get("include", {}).get("doc_ids", []) if filters else []
    
    if not candidate_doc_ids:
        # 如果没有过滤条件，搜索所有文档
        search_query = """
            SELECT * FROM papers 
            WHERE to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', :query)
        """
    else:
        # 在候选文档中进行全文搜索
        search_query = """
            SELECT * FROM papers 
            WHERE doc_id = ANY(:candidate_doc_ids)
            AND to_tsvector('english', title || ' ' || abstract) @@ to_tsquery('english', :query)
        """
    
    search_results = session.execute(text(search_query), 
                                    {'candidate_doc_ids': candidate_doc_ids, 'query': query})
    return search_results
```

**SQL过滤条件示例**
```sql
-- 按分类过滤
WHERE categories ?| ARRAY['cs.AI', 'cs.CL']

-- 按作者过滤
WHERE authors::text ILIKE '%John Smith%'

-- 按日期范围过滤
WHERE published_date BETWEEN '2023-01-01' AND '2024-01-01'

-- 按标题关键词过滤
WHERE title ILIKE '%machine learning%'
```

#### 3. VectorDB 层 - 简化过滤应用

**职责**：在第二阶段接收简化的doc_ids过滤，执行向量搜索

**内存过滤机制（简化版）**
```python
def _apply_filters_first(self, filters: Optional[Dict[str, Any]]) -> List[VectorEntry]:
    """
    在两阶段架构中，只接收简化的doc_ids过滤
    
    Args:
        filters: 简化的过滤条件，格式为 {"include": {"doc_ids": [...]}}
    
    Returns:
        过滤后的向量条目列表
    """
    if not filters:
        return self.entries
    
    # 提取候选文档ID列表
    candidate_doc_ids = filters.get("include", {}).get("doc_ids", [])
    if not candidate_doc_ids:
        return self.entries
    
    # 只需要进行doc_id过滤
    def get_field_value(item, field):
        if field == "doc_ids":
            return item.doc_id
        return None
    
    # 使用FilterParser进行内存过滤（只处理doc_ids）
    return filter_parser.apply_memory_filters(self.entries, filters, get_field_value)
```

**向量搜索方法（第二阶段）**
```python
def search(self, query: str, filters: Optional[Dict[str, Any]] = None, k: int = 5):
    """
    在两阶段架构中，只接收简化的doc_ids过滤
    
    Args:
        filters: 简化的过滤条件，格式为 {"include": {"doc_ids": [...]}}
    """
    # 第一步：应用简化过滤器获取候选条目
    candidate_entries = self._apply_filters_first(filters)
    
    if not candidate_entries:
        return []
    
    # 第二步：在候选条目中进行向量搜索
    temp_vectors = np.vstack([entry.vector for entry in candidate_entries])
    temp_index = faiss.IndexFlatIP(self.vector_dim)
    temp_index.add(temp_vectors)
    
    # 搜索临时索引，避免对无关向量进行计算
    distances, indices = temp_index.search(query_vector, k_search)
    
    # 返回搜索结果
    return [candidate_entries[idx] for idx in indices[0]]
```

**架构说明**：
- VectorDB 不再需要理解复杂的过滤条件（categories、authors等）
- 只需处理预过滤后的doc_ids白名单
- 利用内存过滤快速筛选候选向量
- 在减小的候选集上执行向量搜索，提升效率