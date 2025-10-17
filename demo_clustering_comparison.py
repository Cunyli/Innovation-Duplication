#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图聚类 vs 平面聚类 对比演示

这个脚本用具体的例子展示两种聚类方法的区别
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from utils.cluster.cluster_algorithms import cluster_hdbscan
from utils.cluster.graph_clustering import graph_threshold_clustering

print("=" * 80)
print("图聚类 vs 平面聚类：完整对比演示")
print("=" * 80)
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 步骤 1: 准备相同的输入数据")
print("━" * 80)

# 5个创新的嵌入向量（简化为4维，实际是1536维）
innovation_ids = ['I001', 'I002', 'I003', 'I004', 'I005']

# 模拟嵌入向量：
# - I001 和 I002 是 AI 相关的创新（向量相似）
# - I003 和 I004 是 IoT 相关的创新（向量相似）
# - I005 是混合型创新（与其他都不太相似）
embedding_matrix = np.array([
    [0.1, 0.2, 0.3, 0.4],      # I001: "AI Platform"
    [0.12, 0.21, 0.31, 0.39],  # I002: "AI System" (与I001相似)
    [0.8, 0.1, 0.05, 0.02],    # I003: "IoT Sensor"
    [0.82, 0.09, 0.06, 0.01],  # I004: "IoT Device" (与I003相似)
    [0.5, 0.5, 0.5, 0.5]       # I005: "Hybrid Innovation"
])

print(f"创新ID列表: {innovation_ids}")
print(f"嵌入矩阵形状: {embedding_matrix.shape}")
print(f"\n嵌入矩阵内容:")
for i, innovation_id in enumerate(innovation_ids):
    print(f"  {innovation_id}: {embedding_matrix[i]}")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("🔵 步骤 2: 平面聚类（HDBSCAN）")
print("━" * 80)

print("执行过程:")
print("  1️⃣  归一化嵌入向量")
X_normalized = normalize(embedding_matrix, norm='l2')
print(f"     归一化后矩阵形状: {X_normalized.shape}")

print("  2️⃣  直接在向量空间中计算距离和密度")
print("     ⚠️  没有创建任何图结构！")

print("  3️⃣  使用 HDBSCAN 算法找簇")
labels = cluster_hdbscan(
    embedding_matrix=embedding_matrix,
    min_cluster_size=2,
    metric='cosine'
)

print(f"\n结果:")
print(f"  标签数组: {labels}")
print(f"  解释:")
for i, (innovation_id, label) in enumerate(zip(innovation_ids, labels)):
    if label == -1:
        print(f"    {innovation_id} → 簇 {label} (噪音点)")
    else:
        print(f"    {innovation_id} → 簇 {label}")

# 统计
unique_labels = set(labels)
n_clusters = len([l for l in unique_labels if l != -1])
n_noise = list(labels).count(-1)
print(f"\n  统计: {n_clusters} 个簇, {n_noise} 个噪音点")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("🟢 步骤 3: 图聚类（graph_threshold）")
print("━" * 80)

print("执行过程:")
print("  1️⃣  归一化嵌入向量并计算相似度矩阵")
X_normalized = normalize(embedding_matrix, norm='l2')
sim_matrix = cosine_similarity(X_normalized)

print(f"     相似度矩阵 (对称矩阵):")
print("          ", "  ".join([f"{id:6}" for id in innovation_ids]))
for i, row_id in enumerate(innovation_ids):
    row_str = f"     {row_id}  "
    for j in range(len(innovation_ids)):
        row_str += f"{sim_matrix[i, j]:6.3f} "
    print(row_str)

print("\n  2️⃣  创建 NetworkX 图对象")
print("     🆕 这是图聚类的关键步骤！")
G = nx.Graph()
for idx in innovation_ids:
    G.add_node(idx)
print(f"     图的节点: {list(G.nodes)}")

print("\n  3️⃣  根据相似度阈值添加边 (threshold=0.85)")
threshold = 0.85
edges_added = []
for i in range(len(innovation_ids)):
    for j in range(i+1, len(innovation_ids)):
        if sim_matrix[i, j] >= threshold:
            G.add_edge(innovation_ids[i], innovation_ids[j])
            edges_added.append((innovation_ids[i], innovation_ids[j], sim_matrix[i, j]))
            print(f"     ✅ 添加边: {innovation_ids[i]} ←──{sim_matrix[i, j]:.3f}──→ {innovation_ids[j]}")
        else:
            print(f"     ❌ 跳过: {innovation_ids[i]} ←──{sim_matrix[i, j]:.3f}──→ {innovation_ids[j]} (< 0.85)")

print(f"\n     图的边: {list(G.edges)}")

print("\n  4️⃣  使用 NetworkX 的连通分量算法找簇")
components = list(nx.connected_components(G))
print(f"     连通分量: {components}")

print("\n  5️⃣  转换为簇字典")
clusters = graph_threshold_clustering(
    embedding_matrix=embedding_matrix,
    ids=innovation_ids,
    similarity_threshold=threshold,
    use_cosine=True
)

print(f"\n结果:")
for canonical_id, members in clusters.items():
    print(f"  簇 {canonical_id}: {members}")

print(f"\n  统计: {len(clusters)} 个簇")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("📊 步骤 4: 对比总结")
print("━" * 80)

print("输入数据:")
print("  ✅ 平面聚类: embedding_matrix + innovation_ids")
print("  ✅ 图聚类:   embedding_matrix + innovation_ids")
print("  👉 输入完全相同！")
print()

print("中间过程:")
print("  平面聚类:")
print("    - 归一化向量")
print("    - 直接在向量空间中计算距离")
print("    - 基于密度构建层次树")
print("    - ❌ 不创建图结构")
print()
print("  图聚类:")
print("    - 归一化向量")
print("    - 计算相似度矩阵")
print("    - 🆕 创建 NetworkX 图对象")
print("    - 🆕 根据阈值添加边")
print("    - 🆕 使用图算法（连通分量）")
print()

print("输出格式:")
print("  平面聚类: labels = [0, 0, 1, 1, -1]")
print("  图聚类:   clusters = {'I001': ['I001', 'I002'], ...}")
print("  👉 格式不同，但都可以转换为相同的 canonical_mapping")
print()

print("关键区别:")
print("  🔵 平面聚类: 直接在向量空间工作，不需要图")
print("  🟢 图聚类:   先建立相似度图，再用图算法")
print()

print("图聚类的'图':")
print("  ❌ 不是原始的知识图谱 (Innovation → Organization)")
print("  ✅ 是临时创建的相似度图 (Innovation ←→ Innovation)")
print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("🎨 步骤 5: 可视化图聚类的图结构")
print("━" * 80)

print("原始知识图谱（已转换为 DataFrame，不在这里使用）:")
print("""
  Innovation "AI Platform" ─DEVELOPED_BY→ Organization "Fortum"
  Innovation "AI Platform" ─USES→ Service "Cloud"
""")

print("图聚类临时创建的相似度图（实际使用的）:")
print("""
    I001 ━━━0.98━━━ I002      (AI 相关创新)
    
    I003 ━━━0.99━━━ I004      (IoT 相关创新)
    
    I005                     (孤立节点)
""")

print("这两个'图'完全不同！")
print("  - 原始知识图谱: 节点是多种类型(Innovation/Organization/Service)")
print("  - 相似度图: 节点只有 Innovation，边代表相似度")
print()

print("=" * 80)
print("演示完成！")
print("=" * 80)
print()
print("💡 关键理解:")
print("   1. 输入数据相同（都是嵌入向量矩阵）")
print("   2. 处理方式不同（平面直接计算，图需要建图）")
print("   3. 图聚类的图是临时创建的，不是原始知识图谱")
print("   4. 最终输出可以转换为相同格式")
print()
