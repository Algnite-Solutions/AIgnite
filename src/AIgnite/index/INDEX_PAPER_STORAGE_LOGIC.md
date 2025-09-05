# PaperIndexer 数据存储逻辑详解

## 概述

本文档详细描述了AIgnite Index System中index paper过程中，paper信息是如何被分解、处理和存储到三个不同数据库中的完整流程。该系统采用**并行多数据库存储**架构，每个数据库负责不同类型的数据，通过doc_id进行关联，形成完整的论文索引和检索系统。

## 系统架构

```
数据储存架构
├── 数据准备阶段
│   └── 构建metadata对象
├── 并行存储阶段
│   ├── MetadataDB (PostgreSQL) - 元数据存储 + 全文内容存储
│   ├── VectorDB (FAISS) - 向量存储
│   └── MinioImageDB (MinIO) - 图像存储
└── 状态跟踪
    └── 返回索引状态报告

数据检索架构（重构后）
├── 搜索策略模块 (SearchStrategy)
│   ├── VectorSearchStrategy - 向量搜索（策略组合模式）
│   ├── TFIDFSearchStrategy - TF-IDF搜索（策略组合模式）
│   └── HybridSearchStrategy - 混合搜索（倒数排名重排序）
├── 过滤器模块 (FilterParser)
│   ├── 字段验证和解析
│   ├── SQL条件生成
│   └── 内存过滤应用
└── 结果合并模块 (CombineStrategy) - 新增
    ├── DataRetriever - 数据获取接口
    └── ResultCombiner - 结果合并器
```

## 数据储存架构详解

### 1. 数据准备阶段

#### 输入数据结构
```python
class DocSet:
    doc_id: str                    # 论文唯一标识符
    title: str                     # 论文标题
    abstract: str                  # 论文摘要
    authors: List[str]             # 作者列表
    categories: List[str]          # 分类列表
    published_date: str            # 发布日期
    text_chunks: List[Chunk]      # 文本块列表
    figure_chunks: List[Chunk]    # 图像块列表
    table_chunks: List[Chunk]     # 表格块列表
    pdf_path: str                  # PDF文件路径
    HTML_path: str                 # HTML文件路径
    comments: str | None           # 论文评论/备注
```

#### Metadata构建
```python
metadata = {
    "title": paper.title,                    # 论文标题
    "abstract": paper.abstract,              # 论文摘要
    "authors": paper.authors,                # 作者列表
    "categories": paper.categories,          # 分类列表
    "published_date": paper.published_date,  # 发布日期
    "chunk_ids": [chunk.id for chunk in paper.text_chunks],     # 文本块ID列表
    "figure_ids": [chunk.id for chunk in paper.figure_chunks],  # 图像块ID列表
    "table_ids": [chunk.id for chunk in paper.table_chunks],    # 表格块ID列表
    "comments": paper.comments               # 论文评论/备注
}
```

### 2. 并行存储阶段

#### MetadataDB (PostgreSQL) - 元数据存储

**存储条件**
```python
if self.metadata_db is not None and hasattr(paper, 'pdf_path'):
    success = self.metadata_db.save_paper(paper.doc_id, paper.pdf_path, metadata, paper.text_chunks)
```

**存储内容**
- 结构化数据：标题、摘要、作者、分类、发布日期
- PDF二进制数据：完整的PDF文件内容
- 文本块内容：完整的文本块内容，使用doc_id+chunk_id作为唯一标识
- 引用关系：文本块ID、图像块ID、表格块ID
- 文件路径：PDF路径、HTML路径

**数据库表结构**
```sql
-- 论文元数据表
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    doc_id VARCHAR UNIQUE NOT NULL,
    title VARCHAR NOT NULL,
    abstract TEXT,
    authors JSON,
    categories JSON,
    published_date VARCHAR,
    pdf_data BYTEA,
    chunk_ids JSON,
    figure_ids JSON,
    table_ids JSON,
    extra_metadata JSON,
    pdf_path VARCHAR,
    HTML_path VARCHAR,
    comments TEXT
);

-- 文本块内容表
CREATE TABLE text_chunks (
    id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    chunk_id VARCHAR NOT NULL,
    text_content TEXT NOT NULL,
    chunk_order INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES papers(doc_id) ON DELETE CASCADE
);
```

#### VectorDB (FAISS) - 向量存储

VectorDB 使用简化的存储接口，只需要三个参数：`vector_db_id`、`text_to_emb`、`doc_metadata`。`vector_db_id` 是向量数据库中的唯一标识符，用于索引内容（`text_to_emb`）；`doc_metadata` 是用于区分内容的附加信息。例如，文本块可能来自同一文档，它们共享一个 `doc_id` 但各自拥有独立的 `vector_db_id`。

**存储条件**
```python
if self.vector_db is not None:
    success = self.vector_db.add_document(
        vector_db_id=unique_vector_id,  # 向量数据库中的唯一ID
        text_to_emb=text_content,       # 要嵌入的文本内容
        doc_metadata=metadata           # 文档元数据
    )
```

**存储内容**
- 单一向量存储：根据传入的 `text_to_emb` 参数创建单个向量嵌入
- 灵活存储策略：支持存储摘要、文本块或任何其他文本内容的向量表示
- 元数据关联：通过 `doc_metadata` 中的 `doc_id` 和 `text_type` 等信息进行内容分类和检索

**使用示例**
```python
# 存储论文摘要
vector_db.add_document(
    vector_db_id=f"{paper.doc_id}_abstract",
    text_to_emb=paper.abstract,
    doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
)

# 存储文本块
for chunk in paper.text_chunks:
    vector_db.add_document(
        vector_db_id=f"{paper.doc_id}_chunk_{chunk.id}",
        text_to_emb=chunk.text,
        doc_metadata={"doc_id": paper.doc_id, "text_type": "chunk", "chunk_id": chunk.id}
    )

# 存储组合内容（标题+分类+摘要）
combined_text = f"{paper.title} {' '.join(paper.categories)} {paper.abstract}"
vector_db.add_document(
    vector_db_id=f"{paper.doc_id}_combined",
    text_to_emb=combined_text,
    doc_metadata={"doc_id": paper.doc_id, "text_type": "combined"}
)
```

**向量条目结构**
```python
@dataclass
class VectorEntry:
    doc_id: str                    # 文档标识符
    text: str                      # 原始文本
    text_type: str                 # 文本类型：'abstract', 'chunk', 'combined'
    chunk_id: Optional[str] = None # 文本块ID（仅用于chunk类型）
    vector: Optional[np.ndarray] = None  # 向量表示
```

#### MinioImageDB (MinIO) - 图像存储

**存储条件**
```python
if self.image_db is not None and paper.figure_chunks:
    for figure in paper.figure_chunks:
        success = self.image_db.save_image(
            doc_id=paper.doc_id,
            image_id=figure.id,
            image_path=figure.image_path
        )
```

**存储内容**
- 图像文件：论文中的图表、图像等视觉内容
- 对象命名：`{doc_id}/{image_id}`的层次结构
- 元数据：图像ID和文档ID的关联关系

### 3. 状态跟踪

#### 状态结构
```python
paper_status = {
    "metadata": False,      # 元数据存储状态
    "text_chunks": False,   # 文本块存储状态
    "vectors": False,       # 向量存储状态
    "images": False         # 图像存储状态
}
```

## 数据检索架构详解

数据检索发生在PaperIndexer的find_similar_papers方法中，其基本输入为查询字符串query、结果数量top_k、过滤条件filters、搜索策略列表search_strategies和返回数据类型result_include_types。该方法通过四步流程实现：1)执行搜索获取候选文档，2)提取文档ID，3)根据指定类型获取详细数据，4)合并搜索结果并返回结构化数据。

### PaperIndexer 搜索策略设置

#### set_search_strategy 方法
```python
def set_search_strategy(self, search_strategies: List[Tuple[SearchStrategy, float]]) -> None:
    """Set the search strategy to use.
    
    Args:
        search_strategies: List of search strategies and their thresholds
        
    Raises:
        ValueError: If strategy_type is invalid or required database is not available
    """
```

**支持的策略类型**：
- **单一策略**：`[('vector', 0.5)]` 或 `[('tf-idf', 0.1)]`
- **混合策略**：`[('vector', 0.5), ('tf-idf', 0.1)]`

**策略设置流程**：
1. 检查策略数量（单一 vs 混合）
2. 验证所需数据库的可用性
3. 创建相应的SearchStrategy实例
4. 设置内部搜索策略

### 1. 搜索策略模块 (SearchStrategy)

#### 策略组合模式设计

所有搜索策略现在采用统一的策略组合模式：
- **初始化参数**：`search_strategies: List[Tuple[SearchStrategy, float]]`
- **策略元组**：每个元素为`(策略实例, 相似度阈值)`的元组
- **搜索流程**：调用内部策略列表，获取更多候选结果，应用策略特定阈值过滤

#### 设计优势
- **统一接口**：所有策略使用相同的初始化模式
- **灵活组合**：支持任意策略的组合和嵌套
- **阈值控制**：每个策略可以设置独立的相似度阈值
- **扩展性**：易于添加新的搜索策略类型

#### VectorSearchStrategy - 向量搜索

在向量搜索的SearchStrategy中，可以传入filters，目前支持的key为docids，格式为 "filters={"doc_ids": ["001", "004"]}"
```python
class VectorSearchStrategy(SearchStrategy):
    def __init__(self, search_strategies):
        """初始化向量搜索策略
        
        Args:
            search_strategies: 搜索策略列表，每个元素为(策略实例, 相似度阈值)的元组
        """
        self.search_strategies = search_strategies
        assert len(self.search_strategies) == 1

    def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, 
               similarity_cutoff: float = 0.5, **kwargs):
        # 调用内部策略列表进行搜索
        for strategy, strategy_cutoff in self.search_strategies:
            strategy_results = strategy.search(query, top_k * 2, filters, strategy_cutoff)
        # 过滤结果并转换为标准格式
```


在基于关键词的SearchStrategy中，可以传入filters，目前支持的key为：`categories`、`authors`、`published_date`、`doc_ids`、`title_keywords`、`abstract_keywords`、`text_type`，格式为：
```python
{
    "include": {
        "categories": ["cs.AI", "cs.LG"],           # 分类过滤，支持多个分类
        "doc_ids": ["doc1", "doc2"],                # 特定文档ID过滤
        "text_type": ["abstract", "chunk"]          # 文本类型过滤：abstract/chunk/combined
    },
    "exclude": {
        "categories": ["cs.CR"],                    # 排除特定分类
        "authors": ["Bob Wilson"],                  # 排除特定作者
        "text_type": ["chunk"]                      # 排除特定文本类型
    }
}
```
#### TFIDFSearchStrategy - TF-IDF搜索
```python
class TFIDFSearchStrategy(SearchStrategy):
    def __init__(self, search_strategies):
        """初始化TF-IDF搜索策略
        
        Args:
            search_strategies: 搜索策略列表，每个元素为(策略实例, 相似度阈值)的元组
        """
        self.search_strategies = search_strategies
        assert len(self.search_strategies) == 1

    def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, 
               similarity_cutoff: float = 0.1, **kwargs):
        # 调用内部策略列表进行搜索
        for strategy, strategy_cutoff in self.search_strategies:
            strategy_results = strategy.search(query, top_k * 2, filters, strategy_cutoff)
```

在基于混合检索的SearchStrategy中，可以传入filters，filters会被传入向量检索和关键词检索中

#### HybridSearchStrategy - 混合搜索

混合搜索策略支持多个搜索策略的组合，通过倒数排名重排序机制优化搜索结果。

**初始化设计**
```python
class HybridSearchStrategy(SearchStrategy):
    def __init__(
        self,
        search_strategies: List[Tuple[SearchStrategy, float]]
    ):
        """
        初始化混合搜索策略
        
        Args:
            search_strategies: 搜索策略列表，每个元素为(策略实例, 相似度阈值)的元组
                            例如: [(vector_strategy, 0.5), (tfidf_strategy, 0.1)]
        """
        self.search_strategies = search_strategies
        assert len(self.search_strategies) > 1
```

**搜索方法设计**
```python
def search(self, query: str, top_k: int, filters: Optional[Dict[str, Any]] = None, 
           similarity_cutoff: float = 0.5, **kwargs) -> List[SearchResult]:
    """
    执行混合搜索
    
    Args:
        query: 搜索查询字符串
        top_k: 返回结果数量
        filters: 可选的过滤条件
        similarity_cutoff: 相似度阈值
        **kwargs: 额外参数
        
    Returns:
        重排序后的搜索结果列表
    """
```

**倒数排名重排序机制**
1. **多策略搜索**：调用所有子策略获取候选结果
2. **结果分组**：按文档ID分组，按搜索方法分类
3. **倒数排名计算**：为每个搜索方法计算倒数排名分数
4. **分数合并**：将不同方法的分数进行平均合并
5. **结果重排序**：按合并后的分数排序并返回top_k结果


### 2. 过滤器模块 (FilterParser)

#### 支持的过滤字段
- **基础字段**：doc_ids、categories、authors、published_date
- **文本内容**：title_keywords、abstract_keywords、text_type

#### 过滤器语法结构
```python
filters = {
    "include": {
        "categories": ["cs.AI", "cs.CL"],
        "text_type": ["abstract", "combined"],
        "authors": ["John Smith"],
        "published_date": ["2023-01-01", "2024-01-01"]
    },
    "exclude": {
        "text_type": ["chunk"],
        "doc_ids": ["paper_001", "paper_002"]
    }
}
```

#### 先过滤后搜索架构
- **MetadataDB**：应用SQL过滤器 → 全文搜索
- **VectorDB**：内存过滤候选向量 → 向量搜索

### 3. 结果合并模块 (CombineStrategy)

#### DataRetriever - 数据获取接口
```python
class DataRetriever:
    def __init__(self, metadata_db: Optional[MetadataDB] = None, 
                 image_db: Optional[MinioImageDB] = None):
        self.metadata_db = metadata_db
        self.image_db = image_db
    
    def get_data_by_type(self, doc_ids: List[str], data_type: str) -> Dict[str, Any]:
        if data_type == "metadata":
            return self._get_metadata(doc_ids)
        elif data_type == "text_chunks":
            return self._get_text_chunks(doc_ids)
        elif data_type == "images":
            return self._get_images(doc_ids)
        elif data_type == "full_text":
            return self._get_full_text(doc_ids)
```

#### ResultCombiner - 结果合并器
```python
class ResultCombiner:
    def combine(self, search_results: List[SearchResult], 
                data_dict: Dict[str, Dict[str, Any]], 
                include_types: List[str]) -> List[Dict[str, Any]]:
        results = []
        for result in search_results:
            combined = {"doc_id": result.doc_id}
            # 添加搜索参数和其他数据类型
            # ...
        return results
```

#### 数据获取流程
1. **搜索执行**：通过SearchStrategy获取候选文档列表
2. **文档ID提取**：从搜索结果中提取doc_ids
3. **数据获取**：根据result_include_types调用DataRetriever获取相应数据
4. **结果合并**：使用ResultCombiner将搜索结果与获取的数据合并

#### 支持的数据类型
```python
SUPPORTED_INCLUDE_TYPES = {
    "metadata",           # 论文元数据
    "search_parameters",  # 搜索参数
    "text_chunks",        # 文本块内容
    "full_text",          # 完整文本
    "images"              # 图像数据
}
```

## 数据关联关系

### 主键关联
- **doc_id**：所有三个数据库的主关联键
- **doc_id + chunk_id**：文本块在全文存储表中的唯一标识
- **vector_db_id**：向量在FAISS数据库中的唯一标识（替代原来的chunk_id）
- **image_id**：图像在MinIO中的唯一标识

### 引用关系
```python
metadata = {
    "chunk_ids": ["chunk_001", "chunk_002", "chunk_003"],
    "figure_ids": ["fig_001", "fig_002"],
    "table_ids": ["table_001"]
}
```