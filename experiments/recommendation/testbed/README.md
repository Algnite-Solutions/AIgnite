# TestBed - 统一测试框架

TestBed是一个统一的测试框架，专门为AIgnite项目设计，用于管理测试环境、数据库初始化和测试执行。

## 架构特点

- **统一生命周期**: 所有测试都遵循相同的生命周期（环境检查 → 数据加载 → 数据库初始化 → 测试执行 → 环境清理）
- **真实数据库支持**: 专门为真实数据库（VectorDB, MetadataDB）设计
- **配置驱动**: 使用YAML配置文件管理测试参数
- **灵活扩展**: 易于添加新的测试类型

## 目录结构

```
AIgnite/test/testbed/
├── __init__.py                    # 包初始化
├── base_testbed.py               # 抽象基类
└── README.md                     # 本文档

AIgnite/test/index/
├── paper_indexer_testbed.py      # PaperIndexer专用测试床
├── litsearch_testbed.py          # LitSearch专用测试床
└── ...

AIgnite/test/configs/
├── paper_indexer_testbed_config.yaml  # PaperIndexer配置文件
└── litsearch_testbed_config.yaml     # LitSearch配置文件
```

## 使用方法

### 1. 直接运行测试床

```bash
# 运行PaperIndexer测试床
cd AIgnite/test/index
python paper_indexer_testbed.py

# 运行LitSearch测试床
cd AIgnite/test/index
python litsearch_testbed.py
```

### 2. 在unittest中使用

```python
import unittest
from test.index.paper_indexer_testbed import PaperIndexerTestBed

class TestPaperIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = "/data3/guofang/AIgnite-Solutions/AIgnite/test/configs/paper_indexer_testbed_config.yaml"
        cls.testbed = PaperIndexerTestBed(config_path)
        cls.results = cls.testbed.execute()
    
    def test_vector_search(self):
        result = self.results.get('vector_search', {})
        self.assertTrue(result.get('success', False))
```

### 3. 直接使用TestBed

```python
from test.index.paper_indexer_testbed import PaperIndexerTestBed

# 创建测试床实例
config_path = "/data3/guofang/AIgnite-Solutions/AIgnite/test/configs/paper_indexer_testbed_config.yaml"
testbed = PaperIndexerTestBed(config_path)

# 执行测试
results = testbed.execute()

# 检查结果
for test_name, result in results.items():
    print(f"{test_name}: {'PASSED' if result['success'] else 'FAILED'}")
```

## 配置文件

配置文件使用YAML格式，包含以下主要部分：

- `INDEX_SERVICE`: 服务配置
  - `vector_db`: 向量数据库配置
  - `metadata_db`: 元数据数据库配置
  - `minio_db`: 图像数据库配置
- `test`: 测试参数（批处理大小、结果数量等）
- `logging`: 日志配置
- `search`: 搜索参数
- `test_data`: 测试数据配置
- `environment`: 环境配置
- `performance`: 性能监控配置

## 测试结果

TestBed返回结构化的测试结果，每个测试包含：

- `success`: 布尔值，表示测试是否成功
- `details`: 测试详细信息
- `results_count`: 结果数量（如果适用）
- `error`: 错误信息（如果测试失败）

## 扩展TestBed

要创建新的测试床，继承`TestBed`基类并实现抽象方法：

```python
from test.testbed.base_testbed import TestBed

class MyCustomTestBed(TestBed):
    def check_environment(self) -> Tuple[bool, str]:
        # 实现环境检查逻辑
        pass
    
    def load_data(self) -> Any:
        # 实现数据加载逻辑
        pass
    
    def initialize_databases(self, data: Any) -> None:
        # 实现数据库初始化逻辑
        pass
    
    def run_tests(self) -> Dict[str, Any]:
        # 实现测试执行逻辑
        pass
```

## 可用的测试床

### PaperIndexerTestBed
- **位置**: `AIgnite/test/index/paper_indexer_testbed.py`
- **配置**: `AIgnite/test/configs/paper_indexer_testbed_config.yaml`
- **功能**: 测试PaperIndexer的完整功能，包括向量搜索、TF-IDF搜索、混合搜索、文档删除、图像存储等

### LitSearchTestBed
- **位置**: `AIgnite/test/index/litsearch_testbed.py`
- **配置**: `AIgnite/test/configs/litsearch_testbed_config.yaml`
- **功能**: 测试LitSearch数据集上的搜索性能，包括评估指标计算

## 注意事项

1. **环境要求**: 确保PostgreSQL数据库可用，向量数据库路径有写权限
2. **配置路径**: 配置文件路径相对于运行目录
3. **清理机制**: TestBed会自动清理临时文件和数据库
4. **错误处理**: 所有错误都会被捕获并记录到日志中

## 故障排除

### 常见问题

1. **配置文件未找到**: 检查配置文件路径是否正确
2. **数据库连接失败**: 检查PostgreSQL服务是否运行，连接参数是否正确
3. **权限错误**: 检查向量数据库路径是否有写权限
4. **导入错误**: 确保项目路径设置正确

### 调试模式

使用DEBUG日志级别获取详细信息：

```bash
# 运行PaperIndexer测试床并查看详细日志
cd AIgnite/test/index
python paper_indexer_testbed.py

# 或者修改配置文件中的日志级别为DEBUG
# 在 paper_indexer_testbed_config.yaml 中设置:
# logging:
#   level: "DEBUG"
```

这将显示详细的执行过程，帮助定位问题。
