# 数据结构文档

## 📊 核心数据结构说明

本文档详细说明项目中主要的数据结构，帮助理解数据在各个模块之间的流转。

---

## 1. `analysis_results` (分析结果)

### 来源
由 `analyze_innovation_network()` 函数生成（定义在 `data/processors/network_analyzer.py`）

### 完整结构

```python
analysis_results = {
    # 1. NetworkX 图对象
    'graph': nx.Graph,
    
    # 2. 基础统计指标
    'stats': {
        'total': int,                    # 创新总数
        'avg_sources': float,            # 平均数据源数
        'avg_developers': float,         # 平均开发者数
        'multi_source_count': int,       # 多源验证创新数量
        'multi_developer_count': int     # 多开发者创新数量
    },
    
    # 3. 多源验证的创新
    'multi_source': {
        'innovation_id': {
            'names': set[str],            # 创新名称集合
            'descriptions': set[str],     # 描述集合
            'developed_by': set[str],     # 开发组织ID集合
            'sources': set[str],          # 数据源URL集合
            'source_ids': set[str],       # 源文档ID集合
            'data_sources': set[str]      # 数据源类型集合
        },
        # ... 更多创新
    },
    
    # 4. Top N 最活跃组织（按创新数量排序）
    'top_orgs': [
        ('org_id', innovation_count),    # (组织ID, 创新数量)
        # ... 更多组织
    ],
    
    # 5. Top N 关键组织（按介数中心性排序）
    'key_orgs': [
        ('org_id', betweenness_centrality),  # (组织ID, 介数中心性分数)
        # ... 更多组织
    ],
    
    # 6. Top N 关键创新（按特征向量中心性排序）
    'key_innovations': [
        ('innovation_id', eigenvector_centrality),  # (创新ID, 特征向量中心性分数)
        # ... 更多创新
    ]
}
```

### 详细说明

#### 📈 `graph` (NetworkX 图对象)

```python
# 类型: networkx.Graph
# 节点类型:
G.nodes[node_id] = {
    'type': 'Innovation' | 'Organization',
    # 如果是创新:
    'names': str,              # 逗号分隔的名称
    'sources': int,            # 数据源数量
    'developed_by': int,       # 开发者数量
    # 如果是组织:
    'name': str               # 组织名称
}

# 边:
G.edges[source, target] = {
    'type': 'DEVELOPED_BY' | 'COLLABORATION'
}

# 可用的图操作:
analysis_results['graph'].nodes[org_id].get('name', org_id)  # 获取节点属性
analysis_results['graph'].number_of_nodes()                   # 节点总数
analysis_results['graph'].number_of_edges()                   # 边总数
```

#### 📊 `stats` (统计指标)

```python
{
    'total': 450,                  # 例子: 450 个创新
    'avg_sources': 2.3,            # 例子: 平均每个创新有 2.3 个数据源
    'avg_developers': 1.8,         # 例子: 平均每个创新有 1.8 个开发者
    'multi_source_count': 120,     # 例子: 120 个创新被多个数据源验证
    'multi_developer_count': 80    # 例子: 80 个创新由多个组织协作开发
}
```

#### 🔍 `multi_source` (多源验证的创新)

```python
{
    'canonical_innovation_123': {
        'names': {
            'AI Assistant',
            'Smart AI Helper',
            'Intelligent Assistant'
        },
        'descriptions': {
            'An AI-powered assistant',
            'Intelligent virtual helper'
        },
        'developed_by': {
            'FI01234567',    # 组织 VAT ID
            'FI89012345'
        },
        'sources': {
            'https://company-a.com/innovations',
            'https://vtt.fi/research/projects/ai'
        },
        'source_ids': {
            'doc_123',
            'doc_456'
        },
        'data_sources': {
            'company_website',
            'vtt_website'
        }
    }
}

# 注意: 所有字段都是 set 类型
# 导出时会被转换为 list
```

#### 🏢 `top_orgs` (最活跃组织)

```python
[
    ('FI01234567', 25),    # 组织 A: 25 个创新
    ('FI89012345', 18),    # 组织 B: 18 个创新
    ('FI45678901', 15),    # 组织 C: 15 个创新
    # ...
]

# 类型: List[Tuple[str, int]]
# 按创新数量降序排列
```

#### ⭐ `key_orgs` (关键组织 - 介数中心性)

```python
[
    ('FI01234567', 0.1234),    # 组织 A: 介数中心性 0.1234
    ('FI89012345', 0.0987),    # 组织 B: 介数中心性 0.0987
    ('FI45678901', 0.0765),    # 组织 C: 介数中心性 0.0765
    # ...
]

# 类型: List[Tuple[str, float]]
# 按介数中心性降序排列
# 介数中心性: 衡量节点在网络中的"桥梁"作用
# 取值范围: [0, 1]
```

#### 🚀 `key_innovations` (关键创新 - 特征向量中心性)

```python
[
    ('canonical_innovation_123', 0.2345),  # 创新 A: 特征向量中心性 0.2345
    ('canonical_innovation_456', 0.1987),  # 创新 B: 特征向量中心性 0.1987
    ('canonical_innovation_789', 0.1654),  # 创新 C: 特征向量中心性 0.1654
    # ...
]

# 类型: List[Tuple[str, float]]
# 按特征向量中心性降序排列
# 特征向量中心性: 衡量节点的"影响力"（连接到重要节点的程度）
# 取值范围: [0, 1]
```

---

## 2. `consolidated_graph` (合并知识图谱)

### 来源
由 `create_innovation_knowledge_graph()` 函数生成（定义在 `data/processors/knowledge_graph_builder.py`）

### 完整结构

```python
consolidated_graph = {
    # 1. 创新节点
    'innovations': {
        'canonical_innovation_id': {
            'id': str,                    # 创新ID
            'names': set[str],            # 所有名称变体
            'descriptions': set[str],     # 所有描述
            'developed_by': set[str],     # 开发组织ID
            'sources': set[str],          # 数据源URL
            'source_ids': set[str],       # 源文档ID
            'data_sources': set[str]      # 数据源类型
        },
        # ... 更多创新
    },
    
    # 2. 组织节点
    'organizations': {
        'org_id': {
            'id': str,                    # 组织ID (通常是 VAT ID)
            'name': str,                  # 组织名称
            'description': str            # 组织描述
        },
        # ... 更多组织
    },
    
    # 3. 关系边
    'relationships': [
        {
            'source': str,                # 源节点ID
            'target': str,                # 目标节点ID
            'type': str                   # 关系类型: 'DEVELOPED_BY' 或 'COLLABORATION'
        },
        # ... 更多关系
    ]
}
```

### 示例数据

```python
{
    'innovations': {
        'canonical_innovation_1': {
            'id': 'canonical_innovation_1',
            'names': {'AI Assistant', 'Smart AI'},
            'descriptions': {'An intelligent assistant'},
            'developed_by': {'FI01234567', 'FI89012345'},
            'sources': {
                'https://company-a.com/innovations',
                'https://vtt.fi/research'
            },
            'source_ids': {'doc_123', 'doc_456'},
            'data_sources': {'company_website', 'vtt_website'}
        }
    },
    'organizations': {
        'FI01234567': {
            'id': 'FI01234567',
            'name': 'Tech Company A',
            'description': 'Leading AI technology company'
        }
    },
    'relationships': [
        {
            'source': 'canonical_innovation_1',
            'target': 'FI01234567',
            'type': 'DEVELOPED_BY'
        }
    ]
}
```

---

## 3. `canonical_mapping` (规范映射)

### 来源
由 `resolve_innovation_duplicates()` 函数生成（定义在 `innovation_resolution.py`）

### 结构

```python
canonical_mapping = {
    'innovation_id_1': 'canonical_id_1',    # 创新1 映射到 规范创新1
    'innovation_id_2': 'canonical_id_1',    # 创新2 映射到 规范创新1（重复）
    'innovation_id_3': 'canonical_id_2',    # 创新3 映射到 规范创新2
    # ... 更多映射
}
```

### 说明

- **键 (key)**: 原始创新ID
- **值 (value)**: 规范创新ID（聚类后的代表ID）
- **用途**: 识别重复的创新并将它们映射到同一个规范ID

### 示例

```python
{
    'innovation_123': 'canonical_innovation_1',
    'innovation_124': 'canonical_innovation_1',  # 与 123 重复
    'innovation_125': 'canonical_innovation_1',  # 与 123 重复
    'innovation_126': 'canonical_innovation_2',
    'innovation_127': 'canonical_innovation_3'
}

# 在这个例子中:
# - innovation_123, 124, 125 被识别为重复，合并为 canonical_innovation_1
# - innovation_126 是独立的创新
# - innovation_127 是另一个独立的创新
```

---

## 4. 数据流转图

```
┌─────────────────────────────────────────────────────────────┐
│ 1. 数据加载 (load_and_combine_data)                         │
│    df_relationships                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 重复识别 (resolve_innovation_duplicates)                 │
│    canonical_mapping                                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 图谱构建 (create_innovation_knowledge_graph)             │
│    consolidated_graph                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 网络分析 (analyze_innovation_network)                    │
│    analysis_results                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 结果导出 (export_analysis_results)                       │
│    JSON 文件                                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 导出文件格式

### canonical_mapping.json
```json
{
  "innovation_1": "canonical_1",
  "innovation_2": "canonical_1",
  "innovation_3": "canonical_2"
}
```

### consolidated_graph.json
```json
{
  "innovations": {
    "canonical_1": {
      "id": "canonical_1",
      "names": ["Name A", "Name B"],          // set → list
      "descriptions": ["Description"],
      "developed_by": ["org_1"],
      "sources": ["source_1"],
      "source_ids": ["doc_1"],
      "data_sources": ["company_website"]
    }
  },
  "organizations": { /* ... */ },
  "relationships": [ /* ... */ ]
}
```

### innovation_stats.json
```json
{
  "total": 450,
  "avg_sources": 2.3,
  "avg_developers": 1.8,
  "multi_source_count": 120,
  "multi_developer_count": 80
}
```

### multi_source_innovations.json
```json
{
  "canonical_1": {
    "names": ["Name A", "Name B"],            // set → list
    "descriptions": ["Description"],
    "developed_by": ["org_1", "org_2"],
    "sources": ["source_1", "source_2"],
    "source_ids": ["doc_1", "doc_2"],
    "data_sources": ["company_website", "vtt_website"]
  }
}
```

### key_nodes.json
```json
{
  "key_organizations": [
    {
      "id": "org_1",
      "centrality": 0.1234,
      "name": "Organization Name"
    }
  ],
  "key_innovations": [
    {
      "id": "canonical_1",
      "centrality": 0.2345,
      "names": ["Innovation Name"]
    }
  ]
}
```

---

## 6. 类型注解参考

```python
from typing import Dict, List, Tuple, Set
import networkx as nx

# 类型别名
InnovationID = str
OrganizationID = str
CanonicalID = str

# 核心类型
AnalysisResults = Dict[str, Any]  # 包含 graph, stats, multi_source, etc.
ConsolidatedGraph = Dict[str, Any]  # 包含 innovations, organizations, relationships
CanonicalMapping = Dict[InnovationID, CanonicalID]

# 详细类型
Stats = Dict[str, Union[int, float]]
MultiSource = Dict[CanonicalID, Dict[str, Set[str]]]
TopOrgs = List[Tuple[OrganizationID, int]]
KeyOrgs = List[Tuple[OrganizationID, float]]
KeyInnovations = List[Tuple[CanonicalID, float]]
```

---

## 7. 常见操作示例

### 访问统计数据
```python
total_innovations = analysis_results['stats']['total']
avg_sources = analysis_results['stats']['avg_sources']
```

### 遍历多源创新
```python
for inno_id, inno_data in analysis_results['multi_source'].items():
    names = list(inno_data['names'])  # 转换 set 为 list
    print(f"{inno_id}: {names}")
```

### 获取前 5 个最活跃组织
```python
top_5_orgs = analysis_results['top_orgs'][:5]
for org_id, count in top_5_orgs:
    print(f"{org_id}: {count} innovations")
```

### 访问图节点属性
```python
graph = analysis_results['graph']
for node_id in graph.nodes():
    node_type = graph.nodes[node_id].get('type')
    if node_type == 'Innovation':
        names = graph.nodes[node_id].get('names')
        print(f"Innovation: {names}")
```

### 检查创新是否在图谱中
```python
if innovation_id in consolidated_graph['innovations']:
    innovation = consolidated_graph['innovations'][innovation_id]
    print(f"Names: {innovation['names']}")
```

---

## 8. 注意事项

### Set vs List
⚠️ **重要**: 在内存中，许多字段使用 `set` 类型以确保唯一性：
- `consolidated_graph['innovations'][id]['names']` → `set`
- `analysis_results['multi_source'][id]['sources']` → `set`

✅ **导出时**: `ResultExporter` 自动将所有 `set` 转换为 `list` 以支持 JSON 序列化

### 空值处理
```python
# 安全地访问可能为空的字段
org_name = graph.nodes[org_id].get('name', org_id)  # 使用 org_id 作为默认值
```

### 类型检查
```python
# 检查字段类型
if isinstance(innovation['names'], set):
    names_list = list(innovation['names'])
```

---

## 📚 相关文档

- [结果导出器文档](./RESULT_EXPORTER_REFACTORING.md)
- [网络分析器文档](./NETWORK_ANALYZER_REFACTORING.md)
- [快速参考](./RESULT_EXPORTER_QUICK_REFERENCE.md)
