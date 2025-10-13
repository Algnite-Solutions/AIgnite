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
│   ├── VectorDB (FAISS) - 向量存储（支持 BGE/GritLM 等多种嵌入模型）
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
    metadata: dict                 # 额外元数据
    pdf_path: str                  # PDF文件路径
    HTML_path: str                 # HTML文件路径
    comments: str | None           # 论文评论/备注
    blog: str | None               # 博客内容
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
    "image_storage": {f"{paper.doc_id}_{chunk.id}": False for chunk in paper.figure_chunks},  # 图片存储状态初始化
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
- 引用关系：文本块ID、图像块ID
- 图片存储状态：image_storage字段记录每个图片的实际存储状态（初始化为False）
- 文件路径：PDF路径、HTML路径
- 论文评论：comments字段存储论文评论/备注

**图像ID查询功能**
```python
def get_image_ids(self, doc_id: str) -> List[str]:
    """Get all image IDs for a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of image IDs
    """
```

**图片存储状态管理功能**
```python
def update_image_storage_status(self, doc_id: str, figure_id: str, stored: bool) -> bool:
    """Update storage status for a specific figure.
    
    Args:
        doc_id: Document ID
        figure_id: Figure ID to update
        stored: Storage status (True if stored, False if not)
        
    Returns:
        True if successful, False otherwise
    """

def get_image_storage_status_for_doc(self, doc_id: str) -> Dict[str, bool]:
    """Get storage status for all figures in a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Dictionary mapping figure_id to storage status
    """
```

**查询实现**
- 从papers表的image_storage字段获取图片存储状态
- 支持None值处理，返回空列表/空字典
- 完整的异常处理和数据库会话管理
- 与现有数据库查询模式保持一致

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
    image_storage JSON,  -- 图片存储状态：{"docid_figureid1": true, "docid_figureid2": false}
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

**删除功能**
```python
def delete_paper(self, doc_id: str) -> Tuple[bool, bool]:
    """Delete paper and its metadata, including all text chunks.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Tuple of (paper_metadata_deleted, text_chunks_deleted)
    """
```

**删除功能特点**
- 级联删除：先删除文本块，再删除论文元数据
- 外键约束处理：遵循数据库外键约束顺序
- 事务安全：使用数据库事务确保数据一致性
- 返回状态：分别返回元数据和文本块删除状态
- 完整清理：删除所有关联数据，包括PDF、文本块、图片引用等

**获取所有文档ID功能**
```python
def get_all_doc_ids(self) -> List[str]:
    """Get all document IDs from the metadata database.
    
    Returns:
        List of all document IDs in the database
    """
```

**功能说明**
- 查询papers表获取所有doc_id
- 返回排序后的文档ID列表
- 用于数据库一致性检查和差异识别

#### VectorDB (FAISS) - 向量存储

VectorDB 使用简化的存储接口，只需要三个参数：`vector_db_id`、`text_to_emb`、`doc_metadata`。`vector_db_id` 是向量数据库中的唯一标识符，用于索引内容（`text_to_emb`）；`doc_metadata` 是用于区分内容的附加信息。例如，文本块可能来自同一文档，它们共享一个 `doc_id` 但各自拥有独立的 `vector_db_id`。

**嵌入模型支持**

VectorDB 支持多种嵌入模型：
- **BGE 模型**（如 `BAAI/bge-base-en-v1.5`）：通用文本嵌入模型
- **GritLM 模型**（如 `GritLM/GritLM-7B`）：指令式嵌入模型，支持查询与文档的差异化处理

**模型初始化示例**
```python
# 使用 BGE 模型
vector_db = VectorDB(
    db_path="./vector_index",
    model_name='BAAI/bge-base-en-v1.5',
    vector_dim=768
)

# 使用 GritLM 模型
vector_db = VectorDB(
    db_path="./vector_index",
    model_name='GritLM/GritLM-7B',
    vector_dim=4096  # GritLM-7B 的向量维度
)
```

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
    text_to_emb=paper.title+' . '+paper.abstract
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

**向量存储场景设计**

PaperIndexer 支持两种向量存储场景：

1. **集成存储场景**：在 `index_papers` 方法中与元数据同时存储
2. **独立存储场景**：通过 `save_vectors` 方法在元数据存储后单独存储向量

**集成存储场景（index_papers）**
```python
# 在 index_papers 方法中
if self.vector_db is not None:
    success = self.vector_db.add_document(
        vector_db_id=f"{paper.doc_id}_abstract",
        text_to_emb=f"{paper.title} . {paper.abstract}",
        doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
    )
```

**独立存储场景（save_vectors）**
```python
def save_vectors(self, papers: List[DocSet], indexing_status: Dict[str, Dict[str, bool]] = None):
    """Store vectors from papers to FAISS storage.
    
    Args:
        papers: List of DocSet objects containing papers
        indexing_status: Optional dictionary to track indexing status
        
    Returns:
        indexing_status dictionary with updated vector storage status
    """
    for paper in papers:
        success = self.vector_db.add_document(
            vector_db_id=paper.doc_id+'_abstract',
            text_to_emb=paper.title+' . '+paper.abstract,
            doc_metadata={"doc_id": paper.doc_id, "text_type": "abstract"}
        )
```

**向量存储逻辑设计**

1. **双重存储场景支持**：
   - `index_papers()`：集成存储，与元数据同时处理
   - `save_vectors(papers, indexing_status)`：独立存储，元数据存储后单独处理向量

2. **存储内容**：
   - vector_db_id格式：`{doc_id}_abstract`
   - 文本内容：`{title} . {abstract}`（标题+摘要组合）
   - 元数据：`{"doc_id": doc_id, "text_type": "abstract"}`

3. **存储状态管理**：
   - 支持通过 `indexing_status` 参数跟踪向量存储状态
   - 存储成功后更新 `indexing_status[doc_id]["vectors"] = True`

4. **错误处理**：
   - 完整的异常捕获和日志记录
   - 抛出 `RuntimeError` 以便上层处理

**使用场景示例**

```python
# 场景1：集成存储（推荐用于新论文索引）
indexing_results = indexer.index_papers(papers)

# 场景2：独立存储（用于元数据已存储的论文）
# 先存储元数据
indexing_results = indexer.index_papers(papers)
# 后单独存储向量
indexer.save_vectors(papers, indexing_results)
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

**获取所有文档ID功能**
```python
def get_all_doc_ids(self) -> List[str]:
    """Get all unique document IDs from the vector database.
    
    Returns:
        List of unique document IDs stored in the vector database
    """
```

**功能说明**
- 从 FAISS docstore 提取所有唯一的 doc_id
- 自动去重并返回排序后的文档ID列表
- 用于向量数据库一致性检查和与MetadataDB的差异比对

#### MinioImageDB (MinIO) - 图像存储

**存储场景设计**

PaperIndexer 支持两种图像存储场景：

1. **集成存储场景**：在 `index_papers` 方法中与元数据同时存储
2. **独立存储场景**：通过 `store_images` 方法在元数据存储后单独存储图像

**集成存储场景（index_papers）**
```python
# 在 index_papers 方法中
if store_images and self.image_db is not None and paper.figure_chunks:
    for figure in paper.figure_chunks:
        success = self._save_image(figure.image_path, paper.doc_id+'_'+figure.id)
```

**独立存储场景（store_images）**
```python
def store_images(self, papers: List[DocSet], indexing_status: Dict[str, Dict[str, bool]] = None, keep_temp_image: bool = False):
    """Store images from papers to MinIO storage.
    
    Args:
        papers: List of DocSet objects containing papers with figure_chunks
        indexing_status: Optional dictionary to track indexing status
        keep_temp_image: If False, delete temporary image files after successful storage
        
    Returns:
        indexing_status dictionary with updated image storage status
    """
    for paper in papers:
        for figure in paper.figure_chunks:
            success = self._save_image(figure.image_path, paper.doc_id+'_'+figure.id, keep_temp_image)
```

**核心存储实现（_save_image）**
```python
def _save_image(self, image_path: str, object_name: str, keep_temp_image: bool = False):
    """Core image storage method used by both index_papers and store_images.
    
    Args:
        image_path: Path to the image file
        object_name: Object name for MinIO storage (format: doc_id_figure_id)
        keep_temp_image: If False, delete temporary image files after successful storage
        
    Returns:
        True if successful, False otherwise
    """
    if self.image_db is not None and image_path:
        try:
            # 1. Save image to MinIO
            minio_success = self.image_db.save_image(
                object_name=object_name,
                image_path=image_path
            )
            
            # 2. Update storage status in metadata database
            if minio_success and self.metadata_db is not None:
                if '_' in object_name:
                    doc_id, figure_id = object_name.split('_', 1)
                    metadata_success = self.metadata_db.update_image_storage_status(doc_id, figure_id, True)
                    
                    # 3. Handle temporary file cleanup
                    if not keep_temp_image:
                        try:
                            import os
                            if os.path.exists(image_path):
                                os.remove(image_path)
                                logger.debug(f"Deleted temporary image file: {image_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete temporary image {image_path}: {str(e)}")
                            
            elif not minio_success and self.metadata_db is not None:
                # Update storage status to False if save failed
                if '_' in object_name:
                    doc_id, figure_id = object_name.split('_', 1)
                    metadata_success = self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
            
            return minio_success and metadata_success
        except Exception as e:
            # Update storage status to False on error
            if self.metadata_db is not None and '_' in object_name:
                doc_id, figure_id = object_name.split('_', 1)
                self.metadata_db.update_image_storage_status(doc_id, figure_id, False)
            return False
```

**存储内容**
- 图像文件：论文中的图表、图像等视觉内容
- 对象命名：`{doc_id}_{figure_id}`的扁平结构（如：`2106.14834_fig1`）
- 存储方式：直接使用 `object_name` 作为 MinIO 存储的对象名
- 元数据：通过对象名中的 `doc_id` 和 `figure_id` 组合来维护关联关系

**图像存储逻辑设计**

1. **双重存储场景支持**：
   - `index_papers(store_images=True)`：集成存储，与元数据同时处理
   - `store_images(papers, keep_temp_image=False)`：独立存储，元数据存储后单独处理图像

2. **核心存储方法（_save_image）**：
   - 被 `index_papers` 和 `store_images` 共同使用
   - 负责 MinIO 存储、元数据状态更新、临时文件清理
   - 支持 `keep_temp_image` 参数控制临时文件处理

3. **存储状态管理**：
   - 成功存储：更新 `image_storage` 状态为 `True`
   - 存储失败：更新 `image_storage` 状态为 `False`
   - 状态同步：确保 MinIO 存储状态与数据库记录一致

4. **临时文件处理**：
   - `keep_temp_image=False`：存储成功后删除临时文件
   - `keep_temp_image=True`：保留临时文件（用于调试或后续处理）

**使用场景示例**

```python
# 场景1：集成存储（推荐用于新论文索引）
indexing_results = indexer.index_papers(papers, store_images=True)

# 场景2：独立存储（用于元数据已存储的论文）
# 先存储元数据
indexing_results = indexer.index_papers(papers, store_images=False)
# 后单独存储图像
indexer.store_images(papers, indexing_results, keep_temp_image=False)

# 场景3：保留临时文件的独立存储（用于调试）
indexer.store_images(papers, keep_temp_image=True)
```

**图像删除功能**
```python
def delete_image(self, image_id: str) -> bool:
    """Delete an image from MinIO storage by image_id.
    
    Args:
        image_id: Image ID (used as object_name in MinIO)
        
    Returns:
        True if successful, False otherwise
    """
```

**删除功能特点**
- 直接通过image_id删除MinIO中的图像对象
- 支持NoSuchKey错误处理（图像不存在）
- 返回布尔值表示操作成功与否
- 完整的异常处理和日志记录

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
    "images"              # 图像数据（图像ID列表）
}
```

#### 图像管理功能

**PaperIndexer图像管理方法**
```python
def _delete_image(self, image_id: str):
    """Delete an image by image_id.
    
    Args:
        image_id: Image ID to delete
        
    Returns:
        True if successful, False otherwise
    """

def _list_image_ids(self, doc_id: str):
    """List all image IDs for a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        List of image IDs
    """

def get_image_storage_status_for_doc(self, doc_id: str) -> Dict[str, bool]:
    """Get storage status for all figures in a document.
    
    Args:
        doc_id: Document ID
        
    Returns:
        Dictionary mapping figure_id to storage status
    """

def _delete_images_by_doc_id(self, doc_id: str):
    """Delete all images for a document by doc_id.
    
    Args:
        doc_id: Document ID
        
    Returns:
        True if successful, False otherwise
    """
```

**批量删除实现逻辑**
```python
def _delete_images_by_doc_id(self, doc_id: str):
    if self.metadata_db is not None:
        image_storage_status = self.get_image_storage_status_for_doc(doc_id)
        image_ids = [image_id for image_id in image_storage_status.keys() if image_storage_status[image_id]]
        for image_id in image_ids:
            self._delete_image(image_id)
        print(f"Deleted {len(image_ids)} images for {doc_id}")
        return True
```

**图像管理特点**
- 提供统一的图像管理接口
- 桥接MinIO图像存储和MetadataDB元数据查询
- 支持图片存储状态的精确跟踪和管理
- 完整的错误处理和日志记录
- 支持图像生命周期的完整管理（存储、查询、删除、状态跟踪）
- 支持批量删除功能，通过doc_id删除文档的所有图像
- 确保MinIO存储状态与数据库状态的一致性

## 数据关联关系

### 主键关联
- **doc_id**：所有三个数据库的主关联键
- **doc_id + chunk_id**：文本块在全文存储表中的唯一标识
- **vector_db_id**：向量在FAISS数据库中的唯一标识（替代原来的chunk_id）
- **object_name**：图像在MinIO中的唯一标识（格式：`{doc_id}_{figure_id}`）

### 引用关系
```python
metadata = {
    "chunk_ids": ["chunk_001", "chunk_002", "chunk_003"],
    "figure_ids": ["fig_001", "fig_002"],
    "image_storage": {"docid_fig_001": True, "docid_fig_002": False}  # 图片存储状态（初始化为False，后续动态更新）
}
```

### 图片存储状态管理
- **figure_ids**：记录文档中应该存在的图片ID列表
- **image_storage**：记录每个图片的实际存储状态（True=已存储，False=未存储）
  - 该字段在metadata构建时初始化为False状态
  - 在图片存储/删除操作时动态更新状态
- **状态同步**：图片存储/删除操作会自动更新image_storage状态
- **数据一致性**：确保MinIO存储状态与数据库记录的一致性