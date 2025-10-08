# LitSearch 搜索策略评估

这个项目提供了一个评估框架，用于在 LitSearch 数据集上评估不同搜索策略的性能。**现在直接使用AIgnite项目的搜索模块，确保代码一致性和可维护性。**

## 项目架构

**重要更新**：本评估框架现在直接导入和使用AIgnite项目的搜索模块，不再维护独立的搜索策略实现。

```
litsearch_eval/
├── litsearch_evaluator.py # 评估框架（使用AIgnite搜索模块）
├── evaluator.py          # 评估指标计算
├── visualization.py      # 结果可视化
├── run_evaluation.py     # 主程序
└── README.md             # 本文档
```

## 搜索策略（来自AIgnite项目）

1. **向量搜索（Vector Search）**：
   - 使用 BAAI/bge-base-en-v1.5 模型将文本转换为向量
   - 使用 FAISS 进行高效的向量相似度搜索
   - 适合捕捉语义相似性

2. **TF-IDF搜索（TF-IDF Search）**：
   - 使用 PostgreSQL 的全文搜索功能
   - 实现了自定义排序函数，结合标题和摘要的权重
   - 适合精确的关键词匹配

3. **混合搜索（Hybrid Search）**：
   - 结合向量搜索和TF-IDF搜索的优势
   - 通过可调整的权重融合两种策略的结果
   - 默认权重：向量搜索 0.7，TF-IDF搜索 0.3

## 依赖关系

本评估框架依赖AIgnite项目的以下模块：
- `AIgnite.index.search_strategy` - 搜索策略接口和实现
- `AIgnite.db.vector_db` - 向量数据库
- `AIgnite.db.metadata_db` - 元数据数据库

## 安装依赖

```bash
# 基础依赖
pip install datasets numpy pandas matplotlib seaborn tqdm

# 数据库依赖
pip install sqlalchemy psycopg2-binary

# AIgnite项目依赖（需要先安装AIgnite项目）
# 确保AIgnite项目在Python路径中
```

## 使用方法

### 快速测试

使用少量数据进行快速测试：

```bash
python run_evaluation.py --quick_test --gpu_devices "6" --output_dir "./quick_results"
```

### 自定义采样测试

使用指定数量的数据进行测试：

```bash
python run_evaluation.py --sample_size 1000 --gpu_devices "6,7" --output_dir "./sample_results"
```

### 完整数据集测试

使用完整数据集进行测试（需要较长时间）：

```bash
python run_evaluation.py --gpu_devices "6,7" --output_dir "./full_results"
```

## 命令行参数

### 基本参数

- `--top_k`: 检索结果数量（默认：100）
- `--output_dir`: 输出目录（默认：./results）
- `--sample_size`: 数据集采样大小（默认：使用全部数据）
- `--quick_test`: 快速测试模式，使用10篇文档（默认：False）
- `--batch_size`: 批处理大小（默认：32）
- `--log_level`: 日志级别（默认：INFO）

### 数据库参数

- `--db_host`: PostgreSQL 数据库主机（默认：localhost）
- `--db_port`: PostgreSQL 数据库端口（默认：5432）
- `--db_name`: PostgreSQL 数据库名称（默认：LitSearch）
- `--db_user`: PostgreSQL 数据库用户名（默认：postgres）
- `--db_password`: PostgreSQL 数据库密码（默认：11111）

### 模型参数

- `--embedding_model`: 向量嵌入模型名称（默认：BAAI/bge-base-en-v1.5）
- `--vector_weight`: 混合搜索中向量搜索的权重（默认：0.7）

### GPU参数

- `--gpu_devices`: 可用的GPU设备ID，用逗号分隔（默认：""，使用CPU）

## 数据存储说明

为了与AIgnite项目保持兼容，本评估框架在存储LitSearch数据时：
- 只存储必要的字段：doc_id, title, abstract
- 其他字段设为None或空值
- 使用AIgnite的标准数据库接口

## 优势

1. **代码一致性**：直接使用AIgnite的搜索模块，确保接口一致
2. **维护性**：无需维护重复的搜索策略实现
3. **可靠性**：使用经过测试的AIgnite核心模块
4. **扩展性**：自动获得AIgnite项目的功能更新

## 注意事项

- 确保AIgnite项目已正确安装并在Python路径中
- 数据库连接需要PostgreSQL支持
- 向量数据库文件会保存在`./litsearch_vector_db`目录中 