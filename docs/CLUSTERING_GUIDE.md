# 聚类方法指南

本文档详细介绍了创新去重系统中使用的聚类方法。

## 🎯 概述

系统使用聚类算法来识别和合并重复的创新。默认使用 **HDBSCAN**（平面聚类）方法。

## 📊 支持的聚类方法

### 平面聚类（推荐）

#### 1. HDBSCAN（默认，推荐）⭐⭐⭐⭐⭐

**特点：**
- 自动发现簇的数量
- 基于密度的层次聚类
- 能识别噪音点（独特的创新）
- 适合大规模数据

**适用场景：**
- ✅ 不知道有多少重复创新
- ✅ 需要自动发现模式
- ✅ 数据规模较大（几百到几千个）

**参数：**
```python
method="hdbscan"
min_cluster_size=2      # 最小簇大小
metric="cosine"         # 距离度量
cluster_selection_method="eom"  # 簇选择方法
```

#### 2. K-means ⭐⭐⭐⭐

**特点：**
- 需要预先指定簇的数量
- 快速、简单
- 适合球形簇

**适用场景：**
- ✅ 已知大约有多少个独特创新
- ✅ 需要快速聚类

**参数：**
```python
method="kmeans"
n_clusters=450          # 簇的数量
random_state=42         # 随机种子
```

#### 3. Agglomerative（层次聚类）⭐⭐⭐

**特点：**
- 自底向上的层次聚类
- 可以使用余弦相似度

**参数：**
```python
method="agglomerative"
n_clusters=450
affinity="cosine"       # 相似度度量
linkage="average"       # 链接方式
```

#### 4. Spectral（谱聚类）⭐⭐

**特点：**
- 基于图谱理论
- 适合复杂形状的簇

**参数：**
```python
method="spectral"
n_clusters=450
affinity="nearest_neighbors"
n_neighbors=10
```

### 图聚类（高级用法）

⚠️ **注意：** 图聚类不是使用知识图谱数据，而是临时创建相似度图。

#### 1. Graph Threshold ⭐⭐

**特点：**
- 基于相似度阈值建图
- 使用连通分量算法
- 需要手动设置阈值

**适用场景：**
- 有明确的相似度阈值（如 85%）
- 需要可解释性

**参数：**
```python
method="graph_threshold"
similarity_threshold=0.85   # 相似度阈值
use_cosine=True
```

#### 2. Graph K-core ⭐

**特点：**
- 基于 k-core 分解
- 找到密集连接的子图

**参数：**
```python
method="graph_kcore"
similarity_threshold=0.85
k_core=15               # k-core 参数
use_cosine=True
```

## 🔄 平面聚类 vs 图聚类

### 核心区别

| 特性 | 平面聚类 | 图聚类 |
|------|---------|--------|
| **输入** | 嵌入向量矩阵 | 嵌入向量矩阵（相同） |
| **处理方式** | 直接在向量空间计算 | 先建立临时相似度图 |
| **是否创建图** | ❌ 否 | ✅ 是（临时） |
| **算法复杂度** | O(n log n) ~ O(n²) | O(n²) |
| **参数调整** | 较少 | 较多（阈值敏感） |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### 图聚类的"图"

❌ **常见误解：** 图聚类使用原始知识图谱（Innovation → Organization）

✅ **正确理解：** 图聚类临时创建相似度图（Innovation ←→ Innovation）

```
原始知识图谱（已转换为 DataFrame）:
Innovation "AI Platform" ─DEVELOPED_BY→ Organization "Fortum"

图聚类的临时图（在聚类时创建）:
Innovation "AI Platform" ←──0.98──→ Innovation "AI System"
                                      (边权重 = 相似度)
```

## 💻 使用示例

### 默认配置（推荐）

```python
from innovation_resolution import resolve_innovation_duplicates

canonical_mapping = resolve_innovation_duplicates(
    df_relationships=df_relationships,
    model=embed_model,
    cache_config=cache_config,
    method="hdbscan",           # 默认方法
    min_cluster_size=2,
    metric="cosine",
    cluster_selection_method="eom"
)
```

### 使用 K-means

```python
canonical_mapping = resolve_innovation_duplicates(
    df_relationships=df_relationships,
    model=embed_model,
    cache_config=cache_config,
    method="kmeans",
    n_clusters=450,
    random_state=42
)
```

### 使用图聚类（不推荐）

```python
canonical_mapping = resolve_innovation_duplicates(
    df_relationships=df_relationships,
    model=embed_model,
    cache_config=cache_config,
    method="graph_threshold",
    similarity_threshold=0.85
)
```

## 📈 性能对比

假设数据规模：500 个创新，1536 维嵌入向量

| 方法 | 时间复杂度 | 估计耗时 | 内存占用 | 推荐度 |
|------|-----------|---------|---------|--------|
| **HDBSCAN** | O(n log n) | 1-2秒 | 3 MB | ⭐⭐⭐⭐⭐ |
| K-means | O(nki) | 0.5秒 | 3 MB | ⭐⭐⭐⭐ |
| Agglomerative | O(n²) | 3-5秒 | 4 MB | ⭐⭐⭐ |
| Spectral | O(n³) | 10-20秒 | 5 MB | ⭐⭐ |
| Graph Threshold | O(n²) | 5-10秒 | 6 MB | ⭐⭐ |
| Graph K-core | O(n²) | 8-15秒 | 6 MB | ⭐ |

## 🎯 选择建议

### 推荐使用 HDBSCAN 如果：
- ✅ 不确定有多少重复
- ✅ 希望自动发现模式
- ✅ 需要识别噪音点
- ✅ 数据规模较大

### 考虑 K-means 如果：
- ✅ 大致知道簇的数量
- ✅ 需要快速处理
- ✅ 数据分布较均匀

### 避免图聚类 如果：
- ❌ 数据规模大（性能问题）
- ❌ 不确定合适的阈值
- ❌ 需要自动化处理

## 🔧 实现细节

聚类策略使用策略模式实现，位于 `data/processors/clustering_strategy.py`：

```python
# 策略模式架构
ClusteringStrategy (抽象基类)
├── FlatClusteringStrategy (平面聚类)
│   ├── hdbscan
│   ├── kmeans
│   ├── agglomerative
│   └── spectral
└── GraphClusteringStrategy (图聚类)
    ├── graph_threshold
    └── graph_kcore

# 工厂类
ClusteringStrategyFactory.create_strategy(method)
```

## 📚 相关文档

- [技术细节](TECHNICAL_DETAILS.md) - 深入了解数据处理流程
- [开发指南](DEVELOPMENT.md) - 开发和调试指南
- [数据管道重构指南](DATA_PIPELINE_REFACTORING_GUIDE.md) - 架构设计
