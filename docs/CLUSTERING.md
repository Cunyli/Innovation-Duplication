# 聚类算法模块文档

## 📋 概述

本项目使用多种聚类算法来识别和合并重复的创新记录。聚类模块已经过重构，提供统一的接口和完善的测试支持。

## 🎯 重构亮点

本次重构将原有的 50+ 行 if-elif 分支简化为统一的接口调用，提高了代码的可维护性和可读性。

### 代码质量改进

| 指标 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| 代码行数 | ~50 行 | ~15 行 | ⬇️ 70% |
| 圈复杂度 | 高（4+ 分支） | 低（1 个调用） | ⬇️ 显著降低 |
| 重复代码 | 高 | 无 | ✅ 消除 |
| 可读性 | 中等 | 高 | ⬆️ 提升 |
| 可维护性 | 低 | 高 | ⬆️ 提升 |

## 🔧 核心 API

### `cluster_with_stats()`

统一的聚类接口，支持所有聚类算法：

```python
from utils.cluster.cluster_algorithms import cluster_with_stats

# 基本用法
labels, stats = cluster_with_stats(
    embedding_matrix=embeddings,
    method="hdbscan",  # 或 "kmeans", "agglomerative", "spectral"
    min_cluster_size=2,
    metric="cosine"
)

# 自动打印统计信息
print(f"发现 {stats['n_clusters']} 个簇")
print(f"噪声点: {stats['n_noise']}")
```

**参数说明:**
- `embedding_matrix`: (N, D) 嵌入向量矩阵
- `method`: 聚类方法名称
  - `"hdbscan"`: 层次密度聚类（自动确定簇数）
  - `"kmeans"`: K均值聚类（需要指定 k）
  - `"agglomerative"`: 层次聚类（需要指定簇数）
  - `"spectral"`: 谱聚类（需要指定簇数）
- `**kwargs`: 各方法特定参数

**返回值:**
- `labels`: 聚类标签数组，-1 表示噪声点
- `stats`: 统计信息字典
  - `n_clusters`: 簇数量（不含噪声）
  - `n_noise`: 噪声点数量
  - `cluster_sizes`: 各簇大小列表
  - `largest_cluster`: 最大簇大小
  - `smallest_cluster`: 最小簇大小（不含噪声）
  - `avg_cluster_size`: 平均簇大小

### `get_cluster_info()`

从聚类标签中提取详细统计信息：

```python
from utils.cluster.cluster_algorithms import get_cluster_info

stats = get_cluster_info(labels)
```

## 📚 支持的聚类算法

### 1. HDBSCAN (推荐)

**优势:**
- 自动确定簇数量
- 处理不规则形状的簇
- 支持可变密度的簇
- 自动识别噪声点

**参数:**
```python
labels, stats = cluster_with_stats(
    embedding_matrix,
    method="hdbscan",
    min_cluster_size=2,           # 最小簇大小
    metric="cosine",              # 距离度量
    cluster_selection_method="eom" # 'eom' 或 'leaf'
)
```

**EOM vs Leaf:**
- `eom` (Excess of Mass): 更保守，适合大数据集
- `leaf`: 更细粒度，适合小簇检测

**注意事项:**
- EOM 在小数据集（<100 样本）上可能表现不佳
- 需要足够的样本来建立稳定的密度分布

### 2. K-Means

**优势:**
- 快速高效
- 适合球形簇
- 结果稳定

**参数:**
```python
labels, stats = cluster_with_stats(
    embedding_matrix,
    method="kmeans",
    n_clusters=450,      # 簇数量
    random_state=42      # 随机种子
)
```

### 3. Agglomerative Clustering

**优势:**
- 层次结构清晰
- 支持余弦距离
- 适合文本嵌入

**参数:**
```python
labels, stats = cluster_with_stats(
    embedding_matrix,
    method="agglomerative",
    n_clusters=450,         # 簇数量
    affinity="cosine",      # 距离度量
    linkage="average"       # 链接方式
)
```

**参数说明:**
- `affinity`: 距离度量方式
  - `"cosine"`: 余弦距离（推荐用于文本）
  - `"euclidean"`: 欧氏距离
  - `"manhattan"`: 曼哈顿距离
- `linkage`: 簇间距离计算方式
  - `"average"`: 平均链接（推荐与 cosine 搭配）
  - `"complete"`: 完全链接
  - `"single"`: 单链接

**最佳实践:**
对于文本嵌入，推荐使用 `affinity="cosine"` 和 `linkage="average"`

### 4. Spectral Clustering

**优势:**
- 处理非凸簇
- 基于图理论
- 适合复杂拓扑结构

**参数:**
```python
labels, stats = cluster_with_stats(
    embedding_matrix,
    method="spectral",
    n_clusters=450,       # 簇数量
    n_neighbors=15,       # KNN 邻居数
    random_state=42
)
```

**注意事项:**
- `n_neighbors` 必须 <= 样本数
- 自动调整机制会在样本数不足时给出警告

## 🔬 图聚类算法

除了向量聚类，还支持基于图的聚类方法（位于 `utils/cluster/graph_clustering.py`）：

### 阈值聚类（Threshold Clustering）

```python
from utils.cluster.graph_clustering import graph_threshold_clustering

clusters = graph_threshold_clustering(
    embedding_matrix=embeddings,
    ids=sample_ids,
    similarity_threshold=0.85,
    use_cosine=True
)
```

**返回格式:** `{canonical_id: [member_ids]}`

### K-Core 聚类

```python
from utils.cluster.graph_clustering import graph_kcore_clustering

clusters = graph_kcore_clustering(
    embedding_matrix=embeddings,
    ids=sample_ids,
    similarity_threshold=0.85,
    k_core=15,
    use_cosine=True
)
```

**工作原理:**
1. 构建相似度图（边 = 相似度 >= 阈值）
2. 提取 k-core 子图（节点度 >= k_core）
3. 在 k-core 中找连通分量作为簇

## 📊 使用示例

### 基本工作流

```python
from utils.cluster.cluster_algorithms import cluster_with_stats
import numpy as np

# 准备数据
embeddings = np.random.randn(2000, 1536)  # 2000 个创新，1536 维嵌入

# 运行聚类
labels, stats = cluster_with_stats(
    embedding_matrix=embeddings,
    method="hdbscan",
    min_cluster_size=2,
    metric="cosine"
)

# 查看结果
print(f"总样本数: {len(labels)}")
print(f"发现簇数: {stats['n_clusters']}")
print(f"噪声点数: {stats['n_noise']}")
print(f"最大簇大小: {stats['largest_cluster']}")
print(f"平均簇大小: {stats['avg_cluster_size']:.2f}")
```

### 在 innovation_resolution.py 中的应用

```python
# 旧代码：50+ 行 if-elif
if method_lower == "hdbscan":
    min_cluster_size = method_kwargs.get("min_cluster_size", 2)
    # ... 更多参数 ...
    labels = cluster_hdbscan(...)
elif method_lower == "kmeans":
    # ... 重复的代码 ...

# 新代码：15 行统一接口
labels, stats = cluster_with_stats(
    embedding_matrix=embedding_matrix,
    method=method_lower,
    **method_kwargs
)

print(f"\n聚类统计:")
print(f"  簇数量: {stats['n_clusters']}")
print(f"  噪声点: {stats['n_noise']}")
print(f"  最大簇: {stats['largest_cluster']}")
print(f"  平均簇大小: {stats['avg_cluster_size']:.2f}")
```

## 🧪 测试

运行聚类测试套件：

```bash
python tests/test_cluster.py
```

**测试覆盖:**
- ✅ 4 种平面聚类算法（HDBSCAN, K-Means, Agglomerative, Spectral）
- ✅ 2 种图聚类算法（Threshold, K-Core）
- ✅ 统计函数在各种场景下的表现
- ✅ 边界情况处理（小数据集、n_neighbors 调整等）

## 📈 性能基准

在 ~2000 个创新样本上的表现：

| 方法 | 创新数量 | 组织数量 | 关系数量 | 特点 |
|------|---------|---------|---------|------|
| 阈值聚类 | 1911 | 2490 | 12502 | 基准方法 |
| HDBSCAN | 1735 | 2490 | 12341 | 最保守 |
| K-Means | 1911 | 2490 | 12544 | 稳定 |
| Agglomerative | 1911 | 2490 | 12544 | 层次清晰 |
| Spectral | 1911 | 2490 | 12612 | 适合复杂结构 |

## 🔧 数学原理

### 余弦距离的归一化技巧

在聚类中使用余弦距离时，有一个巧妙的数学等价：

```python
# L2 归一化 + 欧氏距离 ≈ 余弦距离
normalized_X = normalize(X, norm='l2')
euclidean_distance(normalized_X) ∝ cosine_distance(X)
```

**数学推导:**

$$
d_{euclidean}(\mathbf{u}, \mathbf{v}) = \sqrt{2 \times (1 - \cos(\mathbf{u}, \mathbf{v}))}
$$

其中 $\mathbf{u}, \mathbf{v}$ 是 L2 归一化后的向量。

**实际应用:**
```python
# Agglomerative clustering 中的实现
if affinity == "cosine":
    X_normalized = normalize(X, norm='l2')
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',  # 在归一化数据上使用欧氏距离
        linkage=linkage
    )
    labels = model.fit_predict(X_normalized)
```

这个技巧使得可以在只支持欧氏距离的算法中使用余弦相似度。

## 🚀 扩展指南

### 添加新的聚类方法

1. 在 `utils/cluster/cluster_algorithms.py` 中实现新方法：

```python
def cluster_new_method(
    embedding_matrix: np.ndarray,
    param1: int = 10,
    param2: str = "default"
) -> np.ndarray:
    """
    新的聚类方法实现
    
    Args:
        embedding_matrix: 嵌入向量矩阵
        param1: 参数1说明
        param2: 参数2说明
        
    Returns:
        labels: 聚类标签数组
    """
    # 实现你的聚类逻辑
    labels = ...
    return labels
```

2. 在 `cluster_with_stats()` 中添加新方法的分支：

```python
def cluster_with_stats(...):
    # ...
    elif method == "new_method":
        param1 = kwargs.get("param1", 10)
        param2 = kwargs.get("param2", "default")
        labels = cluster_new_method(
            embedding_matrix=embedding_matrix,
            param1=param1,
            param2=param2
        )
```

3. 在 `tests/test_cluster.py` 中添加测试

## 📝 最佳实践

1. **选择合适的聚类方法:**
   - 不知道簇数 → HDBSCAN
   - 知道簇数 + 球形簇 → K-Means
   - 文本嵌入 + 层次结构 → Agglomerative
   - 复杂拓扑 → Spectral

2. **参数调优:**
   - HDBSCAN: 从 `min_cluster_size=2` 开始，根据噪声比例调整
   - K-Means: 使用肘部法则确定 k
   - Agglomerative: 文本用 `affinity="cosine"`
   - Spectral: 确保 `n_neighbors < n_samples`

3. **数据预处理:**
   - 对于文本嵌入，考虑 L2 归一化
   - 检查嵌入质量（避免全零向量）
   - 大数据集考虑降维（PCA, UMAP）

4. **结果验证:**
   - 检查 `stats` 中的噪声比例
   - 可视化簇的分布
   - 人工抽样验证合并质量

## 📚 相关文档

- [开发指南](DEVELOPMENT.md)
- [快速开始](GETTING_STARTED.md)
- [测试说明](../tests/README.md)

## 🤝 贡献

欢迎贡献新的聚类算法或改进现有实现！

---

**最后更新:** 2025年10月17日
