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
├── paper_indexer_testbed.py      # PaperIndexer专用测试床
├── run_paper_indexer_testbed.py  # 测试运行器
├── configs/                      # 配置文件目录
│   └── paper_indexer_testbed_config.yaml
└── README.md                     # 本文档
```

## 使用方法

### 1. 使用TestBed运行器

```bash
# 使用默认配置运行测试
cd AIgnite/test/testbed
python run_paper_indexer_testbed.py

# 使用自定义配置文件
python run_paper_indexer_testbed.py -c configs/custom_config.yaml

# 设置日志级别并保存结果
python run_paper_indexer_testbed.py -l DEBUG -o results.json

# 静默运行（只显示最终结果）
python run_paper_indexer_testbed.py -q
```

### 2. 在unittest中使用

```python
import unittest
from AIgnite.test.testbed.paper_indexer_testbed import PaperIndexerTestBed

class TestPaperIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = "configs/paper_indexer_testbed_config.yaml"
        cls.testbed = PaperIndexerTestBed(config_path)
        cls.results = cls.testbed.execute()
    
    def test_vector_search(self):
        result = self.results.get('vector_search', {})
        self.assertTrue(result.get('success', False))
```

### 3. 直接使用TestBed

```python
from AIgnite.test.testbed.paper_indexer_testbed import PaperIndexerTestBed

# 创建测试床实例
testbed = PaperIndexerTestBed("configs/paper_indexer_testbed_config.yaml")

# 执行测试
results = testbed.execute()

# 检查结果
for test_name, result in results.items():
    print(f"{test_name}: {'PASSED' if result['success'] else 'FAILED'}")
```

## 配置文件

配置文件使用YAML格式，包含以下主要部分：

- `database`: 数据库配置（向量数据库、元数据数据库）
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
from AIgnite.test.testbed.base_testbed import TestBed

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
python run_paper_indexer_testbed.py -l DEBUG
```

这将显示详细的执行过程，帮助定位问题。
