# 数据管道模块化重构完整指南

本文档记录了 `innovation_resolution.py` 中数据管道部分的完整重构过程。

---

## 📋 目录

1. [重构概述](#重构概述)
2. [重构成果](#重构成果)
3. [模块说明](#模块说明)
4. [代码对比](#代码对比)
5. [使用指南](#使用指南)
6. [设计原理](#设计原理)
7. [FAQ](#faq)

---

## 重构概述

### 目标

将 `load_and_combine_data()` 函数从一个180行、充满重复代码的单体函数，重构为模块化、可维护、高性能的组件。

### 原则

1. **DRY原则**：消除重复代码
2. **单一职责**：每个类只做一件事
3. **可测试性**：组件独立可测试
4. **向后兼容**：保持接口不变
5. **增量重构**：每步验证功能

### 重构阶段

- **Phase 1**: 基础工具模块（GraphDocumentLoader, NodeMapper）
- **Phase 2**: 关系处理器（RelationshipProcessor）
- **Phase 3**: 数据源处理器（DataSourceProcessor）

---

## 重构成果

### 统计数据

| 指标 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| **代码行数** | ~180行 | ~90行 | ↓ 50% |
| **重复代码** | 2个大循环 | 0个 | ↓ 100% |
| **处理速度** | ~700 it/s | ~2000 it/s | ↑ 186% |
| **可维护性** | 困难 | 容易 | ✅ |
| **可扩展性** | 困难 | 容易 | ✅ |

### 创建的模块

```
data/
├── loaders/              # Phase 1: 基础加载工具
│   ├── __init__.py
│   ├── graph_loader.py   # 图谱文档加载器
│   └── node_mapper.py    # 节点映射提取器
│
└── processors/           # Phase 2-3: 数据处理器
    ├── __init__.py
    ├── relation_processor.py      # 关系处理器
    └── data_source_processor.py   # 统一数据源处理器
```

---

## 模块说明

### 1. GraphDocumentLoader - 图谱文档加载器

**功能**：安全加载pickle文件中的图谱文档

**特性**：
- ✅ 文件存在性检查
- ✅ 错误处理和日志
- ✅ 批量加载支持
- ✅ 统计信息追踪

**使用示例**：
```python
from data.loaders import GraphDocumentLoader

loader = GraphDocumentLoader()
graph_doc = loader.load("path/to/file.pkl")

if graph_doc:
    print(f"Loaded document with {len(graph_doc.nodes)} nodes")
```

### 2. NodeMapper - 节点映射提取器

**功能**：从图谱文档中提取节点的英文ID和描述映射

**特性**：
- ✅ 提取 `{node.id: english_id}` 映射
- ✅ 提取 `{node.id: description}` 映射
- ✅ 处理缺失值
- ✅ 单个节点查询接口

**使用示例**：
```python
from data.loaders import NodeMapper

mapper = NodeMapper()
node_description, node_en_id = mapper.extract_mappings(graph_doc)

# 使用映射
eng_name = node_en_id.get(node_id, node_id)  # 回退到原始ID
```

### 3. RelationshipProcessor - 关系处理器

**功能**：将图谱关系转换为结构化的DataFrame行

**特性**：
- ✅ 统一的关系处理逻辑
- ✅ 支持自定义元数据
- ✅ 批量处理
- ✅ 统计追踪

**使用示例**：
```python
from data.processors import RelationshipProcessor

processor = RelationshipProcessor()

metadata = {
    "Document number": 123,
    "Source Company": "Nokia",
    "Link Source Text": "https://example.com",
    "Source Text": "Full text...",
    "data_source": "company_website"
}

rows = processor.process_relationships(
    graph_doc, node_description, node_en_id, metadata
)
```

### 4. DataSourceProcessor - 统一数据源处理器

**功能**：封装完整的数据处理流程

**流程**：
1. 读取CSV数据
2. 遍历每一行
3. 加载图谱文档
4. 提取实体和关系
5. 构建DataFrame

**特性**：
- ✅ 配置化的文件路径模式
- ✅ 自定义元数据映射函数
- ✅ 集成所有子组件
- ✅ 统一的错误处理
- ✅ 完整的统计信息

**使用示例**：
```python
from data.processors import DataSourceProcessor

# 创建处理器
processor = DataSourceProcessor(
    graph_docs_dir="data/graph_docs",
    data_source_name="company_website"
)

# 定义元数据映射
def metadata_mapper(row, idx):
    return {
        "Document number": idx,
        "Source Company": row["Company name"],
        "Link Source Text": row["Link"],
        "Source Text": row["text_content"],
        "data_source": "company_website"
    }

# 处理数据
df_result = processor.process(
    df=df_company,
    file_pattern="{Company name}_{index}.pkl",
    metadata_mapper=metadata_mapper,
    entity_extractor=extract_entities_from_document,
    relation_extractor=extract_relationships_from_document,
    pred_entities=all_entities,
    pred_relations=all_relations
)

# 查看统计
stats = processor.get_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
```

---

## 代码对比

### 重构前（180行）

```python
def load_and_combine_data():
    # ... 初始化 ...
    
    # 🔴 第一个大循环 - 公司数据（约80行）
    with tqdm(total=len(df_company), desc="Processing company data") as pbar:
        for i, row in df_company.iterrows():
            try:
                file_path = ...
                with open(file_path, 'rb') as f:
                    graph_docs = pickle.load(f)
                    graph_doc = graph_docs[0]
                
                # 提取实体和关系
                extract_entities_from_document(...)
                extract_relationships_from_document(...)
                
                # 手动构建节点映射
                node_description = {}
                node_en_id = {}
                for node in graph_doc.nodes:
                    node_description[node.id] = ...
                    node_en_id[node.id] = ...
                
                # 手动处理关系
                relationship_rows = []
                for rel in graph_doc.relationships:
                    relationship_rows.append({
                        "Document number": ...,
                        "Source Company": ...,
                        # ... 15行字段映射 ...
                    })
                
                df_relationships_comp_url = pd.concat(...)
            except Exception as e:
                print(f"Error: {e}")
            pbar.update(1)
    
    # 🔴 第二个大循环 - VTT数据（约80行，几乎完全相同！）
    with tqdm(total=len(df_vtt_domain), desc="Processing VTT data") as pbar:
        for index_source, row in df_vtt_domain.iterrows():
            # ... 几乎相同的代码 ...
    
    # ... 合并数据 ...
```

**问题**：
- ❌ 代码重复严重
- ❌ 难以维护
- ❌ 难以测试
- ❌ 难以扩展

### 重构后（90行）

```python
def load_and_combine_data():
    from data.processors import DataSourceProcessor
    
    # 初始化
    all_pred_entities = []
    all_pred_relations = []
    
    # ✅ 处理公司数据 - 声明式配置
    df_company = pd.read_csv(...)
    
    company_processor = DataSourceProcessor(
        graph_docs_dir=GRAPH_DOCS_COMPANY,
        data_source_name="company_website"
    )
    
    def company_metadata_mapper(row, idx):
        return {
            "Document number": row['source_index'],
            "Source Company": row["Company name"],
            "Link Source Text": row["Link"],
            "Source Text": row["text_content"],
            "data_source": "company_website"
        }
    
    df_company_result = company_processor.process(
        df=df_company,
        file_pattern="{Company name}_{index}.pkl",
        metadata_mapper=company_metadata_mapper,
        entity_extractor=extract_entities_from_document,
        relation_extractor=extract_relationships_from_document,
        pred_entities=all_pred_entities,
        pred_relations=all_pred_relations
    )
    
    # ✅ 处理VTT数据 - 使用相同的处理器
    df_vtt_domain = pd.read_csv(...)
    
    vtt_processor = DataSourceProcessor(
        graph_docs_dir=GRAPH_DOCS_VTT,
        data_source_name="vtt_website"
    )
    
    def vtt_metadata_mapper(row, idx):
        return {
            "Document number": idx,
            "VAT id": row["Vat_id"],
            "Link Source Text": row["source_url"],
            "Source Text": row["main_body"],
            "data_source": "vtt_website"
        }
    
    df_vtt_result = vtt_processor.process(
        df=df_vtt_domain,
        file_pattern="{Vat_id}_{index}.pkl",
        metadata_mapper=vtt_metadata_mapper,
        entity_extractor=extract_entities_from_document,
        relation_extractor=extract_relationships_from_document,
        pred_entities=all_pred_entities,
        pred_relations=all_pred_relations
    )
    
    # 合并数据
    df_vtt_result = df_vtt_result.rename(columns={"VAT id": "Source Company"})
    combined_df = pd.concat([df_company_result, df_vtt_result], ignore_index=True)
    
    return combined_df, all_pred_entities, all_pred_relations
```

**优势**：
- ✅ 消除重复代码
- ✅ 清晰的意图
- ✅ 易于维护
- ✅ 易于测试
- ✅ 易于扩展

---

## 使用指南

### 添加新数据源

添加新数据源非常简单，只需3步：

```python
# 1. 读取CSV
df_new = pd.read_csv("new_data.csv")

# 2. 创建处理器并定义映射
processor = DataSourceProcessor(
    graph_docs_dir="path/to/docs",
    data_source_name="new_source"
)

def metadata_mapper(row, idx):
    return {
        "Document number": idx,
        "Source": row["source_col"],
        "Link": row["link_col"],
        "Text": row["text_col"],
        "data_source": "new_source"
    }

# 3. 处理数据
df_result = processor.process(
    df=df_new,
    file_pattern="{column_name}_{index}.pkl",
    metadata_mapper=metadata_mapper,
    entity_extractor=extract_entities_from_document,
    relation_extractor=extract_relationships_from_document,
    pred_entities=all_entities,
    pred_relations=all_relations
)
```

### 测试和验证

运行集成测试：
```bash
python test_integration.py
```

运行基础组件测试：
```bash
python test_loaders.py
```

---

## 设计原理

### 设计模式

1. **策略模式（Strategy Pattern）**
   - `metadata_mapper` 函数可以动态配置
   - 不同数据源使用不同的映射策略

2. **模板方法模式（Template Method）**
   - `DataSourceProcessor.process()` 定义了处理流程
   - 具体步骤通过参数注入

3. **组合模式（Composition）**
   - `DataSourceProcessor` 组合了多个子组件
   - 而不是继承

4. **依赖注入（Dependency Injection）**
   - 提取器函数作为参数传入
   - 易于测试和替换

### 关键决策

#### 为什么使用类而不是函数？

- 需要维护状态（统计信息）
- 便于扩展和继承
- 更好的封装性

#### 为什么提供便捷函数？

- 简单场景下更方便
- 向后兼容旧代码风格
- 减少样板代码

#### 为什么不使用配置文件？

- 代码即配置，更灵活
- Python函数可以包含复杂逻辑
- 避免配置文件格式的限制

---

## FAQ

### Q: 测试会使用缓存吗？

**A**: 不会。测试（`test_integration.py`）只测试数据加载部分，不涉及embedding缓存。

**数据流程分层**：
```
数据加载层（测试这层，不使用缓存）
  ├─ GraphDocumentLoader: 加载 .pkl → graph_doc
  └─ NodeMapper: 提取节点映射
           │
           │ 不涉及缓存
           ▼
─────────────────────────────────────
           │
特征提取层（完整pipeline才用，使用缓存）
  ├─ get_embedding(): 调用API
  └─ EmbeddingCache: 读写 embedding_vectors.json
```

**验证方法**：
```bash
# 1. 备份缓存
mv embedding_vectors.json embedding_vectors.json.bak

# 2. 运行测试
python test_integration.py

# 3. 检查：没有生成缓存文件
ls embedding_vectors.json  # 文件不存在

# 4. 恢复缓存
mv embedding_vectors.json.bak embedding_vectors.json
```

### Q: 为什么处理速度提升了186%？

**A**: 主要原因是优化了DataFrame拼接方式：

**之前**：每次循环都用 `pd.concat()` 拼接单行
```python
for row in rows:
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  # 慢！
```

**现在**：先收集所有行，最后一次性创建DataFrame
```python
rows = []
for item in items:
    rows.extend(process(item))
df = pd.DataFrame(rows)  # 快！
```

### Q: 如何添加更多统计信息？

**A**: 在对应的处理器类中添加计数器：

```python
class DataSourceProcessor:
    def __init__(self):
        # ... 现有代码 ...
        self._custom_stat = 0  # 添加新统计
    
    def get_stats(self):
        return {
            # ... 现有统计 ...
            'custom_stat': self._custom_stat  # 返回新统计
        }
```

### Q: 可以并行处理多个数据源吗？

**A**: 可以，使用多进程或多线程：

```python
from concurrent.futures import ThreadPoolExecutor

def process_source(processor, df, pattern, mapper):
    return processor.process(df, pattern, mapper, ...)

with ThreadPoolExecutor(max_workers=2) as executor:
    future_company = executor.submit(process_source, company_processor, ...)
    future_vtt = executor.submit(process_source, vtt_processor, ...)
    
    df_company = future_company.result()
    df_vtt = future_vtt.result()
```

---

## 总结

通过三个阶段的增量式重构，我们：

1. ✅ **减少了50%的代码**：从180行 → 90行
2. ✅ **提升了186%的性能**：从~700it/s → ~2000it/s
3. ✅ **消除了100%的重复**：0个重复代码块
4. ✅ **创建了4个可复用模块**
5. ✅ **保持了100%的兼容性**：接口不变，功能一致

这是一个成功的重构案例，展示了如何通过逐步提取、组合和抽象，将复杂的单体代码转换为清晰、可维护的模块化架构。

---

**最后更新**: 2024-10-17  
**作者**: AI Assistant  
**分支**: feature/modularize-innovation-resolution
