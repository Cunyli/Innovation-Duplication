# 技术细节文档

> 高层结构和包概览请先阅读 **DEVELOPMENT.md**。本文件保留深入实现细节和公式。

本文档详细说明创新去重系统的核心技术实现。

## 🎯 核心功能：`resolve_innovation_duplicates`

这是系统的核心方法，用于识别和合并重复的创新。

### 默认执行流程

```python
# 在 main() 函数中调用
canonical_mapping = resolve_innovation_duplicates(
    df_relationships=df_relationships,  # 输入：关系数据
    model=embed_model,                  # Azure OpenAI 嵌入模型
    cache_config=cache_config,          # 缓存配置
    method="hdbscan",                   # 默认聚类方法
    min_cluster_size=2,
    metric="cosine",
    cluster_selection_method="eom"
)
```

## 📊 数据流程

### 完整数据流图

```
原始文件 (CSV + PKL)
    ↓
load_and_combine_data()
    ↓
df_relationships (完整关系表，~1000行)
┌──────────────────────────────────────────────────┐
│ source_id │ source_type │ relationship_type │... │
│ I001      │ Innovation  │ DEVELOPED_BY      │... │
│ I001      │ Innovation  │ USES              │... │
│ I001      │ Innovation  │ ENABLES           │... │
│ I002      │ Innovation  │ DEVELOPED_BY      │... │
└──────────────────────────────────────────────────┘
    ↓
resolve_innovation_duplicates()
    │
    ├─ Step 1: extract_unique_innovations()
    │  └─ 提取唯一ID (去重)
    │     输入: 1000行 → 输出: 200个唯一创新ID
    │
    ├─ Step 2: build_all_features()
    │  └─ 构建创新特征（回查完整 DataFrame）
    │     为每个创新构建文本：
    │     "Innovation name: ... Description: ...
    │      Developed by: ... Relationships: ..."
    │
    ├─ Step 3: get_embeddings()
    │  └─ 生成或加载嵌入向量
    │     - 检查缓存
    │     - 调用 Azure OpenAI API
    │     - 保存到缓存
    │     输出: (200, 1536) 嵌入矩阵
    │
    └─ Step 4: cluster()
       └─ 执行聚类算法
          输入: 嵌入矩阵
          输出: canonical_mapping
          {
            'I001': 'I001',
            'I023': 'I001',  # 重复，合并到 I001
            'I045': 'I001',  # 重复，合并到 I001
            'I002': 'I002',
            ...
          }
```

## 🔍 详细步骤解析

### Step 1: 提取唯一创新

```python
# 在 data/processors/innovation_feature_builder.py
def extract_unique_innovations(df_relationships: pd.DataFrame):
    # 1. 筛选创新类型
    innovations = df_relationships[
        df_relationships["source_type"] == "Innovation"
    ]
    
    # 2. 按 source_id 去重
    unique_innovations = innovations.drop_duplicates(subset=["source_id"])
    
    return unique_innovations  # 200行
```

**重要理解：**
- 虽然去重后只有 200 行
- 但原始的 `df_relationships` 仍然保留完整的 1000 行
- 后续步骤会回查这 1000 行来构建特征

### Step 2: 构建创新特征

```python
# 在 data/processors/innovation_feature_builder.py
def build_all_features(unique_innovations, df_relationships):
    innovation_features = {}
    
    for _, row in unique_innovations.iterrows():
        innovation_id = row["source_id"]  # 例如 "I001"
        
        # 🔑 关键：调用 build_context，传入完整的 df_relationships
        context = build_context(row, df_relationships)
        innovation_features[innovation_id] = context
    
    return innovation_features

def build_context(innovation_row, df_relationships):
    innovation_id = innovation_row["source_id"]
    
    # 1. 基本信息
    context = f"Innovation name: {innovation_row['source_english_id']}. "
    context += f"Description: {innovation_row['source_description']}. "
    
    # 2. 开发者信息（查询完整 DataFrame）
    developers = df_relationships[
        (df_relationships["source_id"] == innovation_id) &
        (df_relationships["relationship_type"] == "DEVELOPED_BY")
    ]["target_english_id"].unique()
    
    context += f"Developed by: {', '.join(developers)}. "
    
    # 3. 关系上下文（查询完整 DataFrame）
    relations = df_relationships[
        df_relationships["source_id"] == innovation_id
    ]
    
    for _, rel_row in relations.iterrows():
        context += f"{rel_row['relationship_type']} {rel_row['target_english_id']}. "
    
    return context
```

**完整上下文示例：**

```python
innovation_features = {
    'I001': """Innovation name: AI Platform. 
               Description: Advanced AI system. 
               Developed by: Fortum, Nokia. 
               USES Cloud Service. 
               ENABLES Analytics Tool.""",
    'I002': """Innovation name: ML Engine. 
               Description: Machine learning tool. 
               Developed by: Nokia. 
               USES GPU Cluster.""",
    # ... 200 个创新
}
```

### Step 3: 生成嵌入向量

```python
# 在 data/processors/embedding_manager.py
def get_embeddings(self, innovation_features, model):
    # 1. 从缓存加载
    embeddings = self.cache.load()  # 从 embedding_vectors.json
    
    # 2. 找出缺失的 ID
    all_ids = list(innovation_features.keys())
    missing_ids = self.cache.get_missing_keys(all_ids)
    
    # 3. 为缺失的创新生成嵌入
    if missing_ids:
        print(f"Generating {len(missing_ids)} new embeddings...")
        
        new_embeddings = {}
        for innovation_id in tqdm(missing_ids):
            text = innovation_features[innovation_id]
            
            # 调用 Azure OpenAI API
            embedding = get_embedding(text, model)
            # 返回: [0.15, 0.22, 0.31, ..., 0.87]  # 1536维
            
            new_embeddings[innovation_id] = embedding
        
        # 4. 更新缓存
        self.cache.update(new_embeddings)
        embeddings.update(new_embeddings)
    
    # 5. 构建嵌入矩阵
    innovation_ids = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[id] for id in innovation_ids])
    # shape: (200, 1536)
    
    return innovation_ids, embedding_matrix
```

**嵌入向量的含义：**
- 每个创新的文本 → 1536 维数字向量
- 相似的创新在向量空间中距离更近
- 使用 Azure OpenAI 的 text-embedding-ada-002 模型

### Step 4: 执行聚类

```python
# 在 data/processors/clustering_strategy.py
class FlatClusteringStrategy:
    def cluster(self, embedding_matrix, innovation_ids, **kwargs):
        # 1. 调用 HDBSCAN
        labels = cluster_hdbscan(
            embedding_matrix=embedding_matrix,  # (200, 1536)
            min_cluster_size=2,
            metric="cosine"
        )
        
        # 2. HDBSCAN 返回标签
        # labels = [0, 0, 0, 1, 1, 2, -1, 3, 3, 3, ...]
        #          ^^^^^^     ^^^ ^^^  ^  ^^^^^^^^^
        #          簇0        簇1 簇2 噪音  簇3
        
        # 3. 构建簇字典
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:
                # 噪音点单独成簇
                clusters[f"noise_{innovation_ids[idx]}"] = [innovation_ids[idx]]
            else:
                clusters.setdefault(int(label), []).append(innovation_ids[idx])
        
        # 4. 转换为标准映射
        canonical_mapping = {}
        for label_key, members in clusters.items():
            canonical_id = members[0]  # 第一个作为代表
            for member_id in members:
                canonical_mapping[member_id] = canonical_id
        
        return canonical_mapping
```

## 🔄 信息保留机制

### 常见疑问：去重会丢失信息吗？

**答案：不会！** 虽然 `drop_duplicates` 看起来丢失了行，但实际上：

```python
# 看起来丢失了信息
unique_innovations = df.drop_duplicates(subset=["source_id"])
# 1000 行 → 200 行

# 但特征构建时会回查完整 DataFrame
def build_context(innovation_row, df_relationships):  # ← 传入完整的 1000 行
    innovation_id = innovation_row["source_id"]
    
    # 查询所有与该创新相关的行
    relations = df_relationships[
        df_relationships["source_id"] == innovation_id
    ]  # ← 可能查到多行
    
    # 所有关系信息都会被使用
    for _, rel_row in relations.iterrows():
        context += f"{rel_row['relationship_type']} {rel_row['target_name']}. "
```

**验证示例：**

```python
# 假设 I001 有 3 条关系记录
df_relationships:
  source_id  relationship_type  target_name
  I001       DEVELOPED_BY       Fortum
  I001       USES               Cloud
  I001       ENABLES            Analytics

# Step 1: 去重后
unique_innovations:
  source_id  ...
  I001       ...  # 只保留 1 行

# Step 2: 构建特征时
context = build_context(I001, df_relationships)
# 回查完整 DataFrame，找到 I001 的 3 条记录
# 结果: "... Developed by Fortum. Uses Cloud. Enables Analytics."

# ✅ 所有 3 条关系都被保留在文本中！
```

## 📁 涉及的文件清单

### 核心代码文件

```
innovation_resolution.py                # 主流程
├── load_and_combine_data()
├── resolve_innovation_duplicates()    # 核心方法
├── create_innovation_knowledge_graph()
└── analyze_innovation_network()

data/processors/
├── innovation_feature_builder.py      # 特征构建
│   ├── InnovationExtractor
│   └── InnovationFeatureBuilder
├── embedding_manager.py               # 嵌入管理
│   └── EmbeddingManager
├── clustering_strategy.py             # 聚类策略
│   ├── FlatClusteringStrategy
│   ├── GraphClusteringStrategy
│   └── ClusteringStrategyFactory
├── embedding_strategy.py              # 嵌入生成
│   └── get_embedding()
└── model_initializer.py              # 模型初始化
    └── initialize_openai_client()

utils/cluster/
├── cluster_algorithms.py             # 平面聚类算法
└── graph_clustering.py               # 图聚类算法

core/
└── cache.py                          # 缓存系统
    ├── CacheBackend
    ├── JsonFileCache
    ├── MemoryCache
    └── EmbeddingCache
```

### 数据文件

```
data/
├── dataframes/
│   ├── vtt_mentions_comp_domain.csv     # 公司网站数据
│   └── comp_mentions_vtt_domain.csv     # VTT 网站数据
│
└── graph_docs_names_resolved/           # 图文档（PKL 文件）
    ├── Fortum Oyj_0.pkl
    ├── Nokia Oyj_0.pkl
    └── ...

embedding_vectors.json                   # 嵌入向量缓存
```

## 🎯 关键理解点

### 1. 去重不会丢失信息

- `drop_duplicates` 只是获取唯一 ID 列表
- 特征构建时会回查完整 DataFrame
- 所有关系都会被使用

### 2. 默认使用平面聚类

- 方法：HDBSCAN
- 原因：自动、准确、高效
- 不使用图聚类（除非显式指定）

### 3. 嵌入向量有缓存

- 第一次运行：调用 API，保存到 `embedding_vectors.json`
- 后续运行：从缓存加载，只为新创新生成嵌入
- 节省时间和 API 调用费用

### 4. 输出是映射字典

```python
canonical_mapping = {
    'I001': 'I001',  # 标准ID
    'I023': 'I001',  # 重复，合并到 I001
    'I045': 'I001',  # 重复，合并到 I001
    'I002': 'I002',  # 标准ID
    'I015': 'I002',  # 重复，合并到 I002
}

# 200 个原始创新 → 150 个唯一创新
```

## 🔧 配置选项

### 缓存配置

```python
cache_config = {
    "type": "embedding",
    "backend": "json",              # 'json' 或 'memory'
    "path": "./embedding_vectors.json",
    "use_cache": True               # 是否启用缓存
}
```

### 聚类方法选择

```python
# HDBSCAN（推荐）
method="hdbscan"
min_cluster_size=2
metric="cosine"

# K-means
method="kmeans"
n_clusters=450

# 图聚类（不推荐）
method="graph_threshold"
similarity_threshold=0.85
```

## 📚 相关文档

- [聚类方法指南](CLUSTERING_GUIDE.md) - 详细的聚类方法说明
- [开发指南](DEVELOPMENT.md) - 开发和调试指南
- [数据管道重构指南](DATA_PIPELINE_REFACTORING_GUIDE.md) - 架构设计
