# tests/test_cluster.py
# 全面测试所有聚类算法

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from innovation_platform.utils.cluster.cluster_algorithms import cluster_with_stats, get_cluster_info
from innovation_platform.utils.cluster.graph_clustering import graph_threshold_clustering, graph_kcore_clustering


def create_test_data():
    """创建测试数据：4 个样本，2 个明显的簇"""
    # 簇 1: A 和 B 很相似
    # 簇 2: C 和 D 很相似
    X = np.array([
        [0.1, 0.1, 0.2],      # A
        [0.15, 0.05, 0.25],   # B (与 A 相似)
        [0.9, 0.8, 0.75],     # C
        [0.85, 0.78, 0.7]     # D (与 C 相似)
    ])
    ids = ["A", "B", "C", "D"]
    return X, ids


def test_flat_clustering_methods():
    """测试所有平面聚类算法"""
    print("=" * 70)
    print("🧪 测试平面聚类算法（HDBSCAN, K-Means, Agglomerative, Spectral）")
    print("=" * 70)
    
    X, ids = create_test_data()
    
    # 测试配置
    methods = [
        {'name': 'HDBSCAN', 'method': 'hdbscan', 'kwargs': {'min_cluster_size': 2, 'metric': 'cosine'}},
        {'name': 'K-Means', 'method': 'kmeans', 'kwargs': {'n_clusters': 2, 'random_state': 42}},
        {'name': 'Agglomerative', 'method': 'agglomerative', 'kwargs': {'n_clusters': 2, 'affinity': 'cosine'}},
        {'name': 'Spectral', 'method': 'spectral', 'kwargs': {'n_clusters': 2, 'affinity': 'nearest_neighbors', 'n_neighbors': 2}},
    ]
    
    results = []
    
    for config in methods:
        print(f"\n📌 测试 {config['name']}")
        print("-" * 70)
        
        try:
            # 使用统一接口
            labels, stats = cluster_with_stats(X, method=config['method'], **config['kwargs'])
            
            # 打印统计信息
            print(f"✅ 聚类成功！")
            print(f"   方法: {stats['method'].upper()}")
            print(f"   簇数量: {stats['n_clusters']}")
            print(f"   噪音点: {stats['n_noise']}")
            print(f"   最大簇大小: {stats['largest_cluster']}")
            print(f"   最小簇大小: {stats['smallest_cluster']}")
            print(f"   总样本数: {stats['total_samples']}")
            
            # 显示每个样本的标签
            print(f"   标签分配: ", end="")
            for i, (id_name, label) in enumerate(zip(ids, labels)):
                print(f"{id_name}→{label}", end="  ")
            print()
            
            # 保存结果
            results.append({
                'method': config['name'],
                'success': True,
                'stats': stats,
                'labels': labels
            })
            
        except Exception as e:
            print(f"❌ 失败: {str(e)}")
            results.append({
                'method': config['name'],
                'success': False,
                'error': str(e)
            })
    
    return results


def test_graph_clustering_methods():
    """测试图聚类算法"""
    print("\n\n" + "=" * 70)
    print("🧪 测试图聚类算法（Threshold-based, K-Core）")
    print("=" * 70)
    
    X, ids = create_test_data()
    
    # 测试 1: Threshold-based clustering
    print(f"\n📌 测试 Threshold-based Clustering")
    print("-" * 70)
    try:
        clusters_dict = graph_threshold_clustering(
            embedding_matrix=X,
            ids=ids,
            similarity_threshold=0.9,
            use_cosine=True
        )
        print(f"✅ 成功！找到 {len(clusters_dict)} 个簇")
        for canonical, members in clusters_dict.items():
            print(f"   簇 {canonical}: {members}")
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
    
    # 测试 2: K-Core clustering
    print(f"\n📌 测试 K-Core Clustering")
    print("-" * 70)
    try:
        clusters_dict = graph_kcore_clustering(
            embedding_matrix=X,
            ids=ids,
            similarity_threshold=0.85,
            k_core=1,  # 低阈值以确保有结果
            use_cosine=True
        )
        print(f"✅ 成功！找到 {len(clusters_dict)} 个簇")
        for canonical, members in clusters_dict.items():
            print(f"   簇 {canonical}: {members}")
    except Exception as e:
        print(f"❌ 失败: {str(e)}")


def test_get_cluster_info():
    """测试 get_cluster_info 函数"""
    print("\n\n" + "=" * 70)
    print("🧪 测试 get_cluster_info() 函数")
    print("=" * 70)
    
    # 模拟不同的标签场景
    test_cases = [
        {
            'name': '正常聚类（无噪音）',
            'labels': np.array([0, 0, 1, 1, 2, 2, 2])
        },
        {
            'name': 'HDBSCAN 聚类（有噪音点）',
            'labels': np.array([0, 0, 1, -1, 2, 2, -1])
        },
        {
            'name': '极端情况（全部噪音）',
            'labels': np.array([-1, -1, -1, -1])
        },
        {
            'name': '极端情况（全部同一簇）',
            'labels': np.array([0, 0, 0, 0, 0])
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📌 {test_case['name']}")
        print(f"   标签: {test_case['labels']}")
        
        stats = get_cluster_info(test_case['labels'])
        
        print(f"   📊 统计结果:")
        print(f"      簇数量: {stats['n_clusters']}")
        print(f"      噪音点: {stats['n_noise']}")
        print(f"      最大簇: {stats['largest_cluster']}")
        print(f"      最小簇: {stats['smallest_cluster']}")
        print(f"      总样本: {stats['total_samples']}")
        if stats['cluster_sizes']:
            print(f"      簇大小分布: {stats['cluster_sizes']}")


def run_comparison_test():
    """比较所有方法的性能"""
    print("\n\n" + "=" * 70)
    print("📊 聚类方法对比总结")
    print("=" * 70)
    
    X, ids = create_test_data()
    
    methods = ['hdbscan', 'kmeans', 'agglomerative', 'spectral']
    
    print(f"\n{'方法':<15} {'簇数':<8} {'噪音点':<10} {'最大簇':<10} {'最小簇':<10}")
    print("-" * 70)
    
    for method in methods:
        try:
            if method == 'hdbscan':
                kwargs = {'min_cluster_size': 2, 'metric': 'cosine'}
            elif method == 'spectral':
                # Spectral 需要 n_neighbors < 样本数 (这里是 4)
                kwargs = {'n_clusters': 2, 'n_neighbors': 2}
            else:
                kwargs = {'n_clusters': 2}
            
            labels, stats = cluster_with_stats(X, method=method, **kwargs)
            
            print(f"{method.upper():<15} {stats['n_clusters']:<8} {stats['n_noise']:<10} "
                  f"{stats['largest_cluster']:<10} {stats['smallest_cluster']:<10}")
        except Exception as e:
            print(f"{method.upper():<15} {'ERROR':<8} {str(e)[:40]}")


def main():
    """运行所有测试"""
    print("\n")
    print("🚀 开始聚类算法测试套件")
    print("=" * 70)
    
    # 测试 1: 平面聚类
    test_flat_clustering_methods()
    
    # 测试 2: 图聚类
    test_graph_clustering_methods()
    
    # 测试 3: 统计函数
    test_get_cluster_info()
    
    # 测试 4: 对比分析
    run_comparison_test()
    
    print("\n\n" + "=" * 70)
    print("✅ 所有测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
