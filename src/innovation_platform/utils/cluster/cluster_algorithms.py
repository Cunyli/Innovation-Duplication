# utils/cluster/cluster_algorithms.py

import numpy as np
from typing import Dict, Tuple
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import hdbscan


def cluster_hdbscan(
    embedding_matrix: np.ndarray,
    min_cluster_size: int = 2,
    metric: str = 'cosine',
    cluster_selection_method: str = 'eom'
) -> np.ndarray:
    """
    使用 HDBSCAN 对嵌入向量进行聚类。
    """
    if metric == 'cosine':
        embedding_matrix = normalize(embedding_matrix, norm='l2')
        metric_used = 'euclidean'
    else:
        metric_used = metric

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric_used,
        cluster_selection_method=cluster_selection_method
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels


def cluster_kmeans(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    random_state: int = 42
) -> np.ndarray:
    """
    使用 K-Means 对嵌入向量进行聚类。
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embedding_matrix)
    return labels


def cluster_agglomerative(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    affinity: str = 'cosine',
    linkage: str = 'average'
) -> np.ndarray:
    """
    使用层次聚类（Agglomerative Clustering）进行聚类。
    
    这是一种自底向上的层次聚类方法：
    1. 开始时每个样本都是一个簇
    2. 不断合并最相似的簇对
    3. 直到达到指定的簇数量
    
    Args:
        embedding_matrix: 嵌入向量矩阵 (N, D)
        n_clusters: 最终簇的数量
        affinity: 距离度量方式
            - 'cosine': 余弦距离（推荐用于文本嵌入）
            - 'euclidean': 欧几里得距离
            - 'manhattan': 曼哈顿距离
            - 'l1', 'l2': L1/L2 范数
        linkage: 簇间距离计算方式（如何决定合并哪两个簇）
            - 'average': 平均链接（推荐，平衡且稳定）
            - 'single': 单链接（最近点距离）
            - 'complete': 全链接（最远点距离）
            - 'ward': Ward 方差最小化（仅支持欧几里得距离）
    
    Returns:
        labels: 聚类标签数组 (N,)
    
    注意：
        - 当 affinity='cosine' 时，会自动转换为归一化 + 欧几里得距离
        - 计算复杂度: O(n²log(n))，对大数据集较慢
        - 结果是确定性的（不像 K-Means 有随机性）
    """
    if affinity == 'cosine':
        embedding_matrix = normalize(embedding_matrix, norm='l2')
        affinity_used = 'euclidean'
    else:
        affinity_used = affinity

    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=affinity_used,
        linkage=linkage
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels


def cluster_spectral(
    embedding_matrix: np.ndarray,
    n_clusters: int = 450,
    affinity: str = 'nearest_neighbors',
    n_neighbors: int = 10
) -> np.ndarray:
    """
    使用谱聚类（Spectral Clustering）对嵌入向量聚类。
    
    注意：当 affinity='nearest_neighbors' 时，n_neighbors 必须小于样本数。
    函数会自动调整 n_neighbors 以避免错误。
    """
    n_samples = len(embedding_matrix)
    
    # 自动调整 n_neighbors 以避免错误
    if affinity == 'nearest_neighbors':
        original_n_neighbors = n_neighbors
        n_neighbors = min(n_neighbors, n_samples - 1)
        
        if n_neighbors < 1:
            raise ValueError(
                f"样本数 ({n_samples}) 太少，无法使用谱聚类。"
                f"至少需要 2 个样本。"
            )
        
        # 如果调整了参数，给出警告
        if n_neighbors != original_n_neighbors:
            import warnings
            warnings.warn(
                f"n_neighbors 从 {original_n_neighbors} 自动调整为 {n_neighbors} "
                f"（样本数: {n_samples}）",
                UserWarning
            )
    
    clusterer = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=42
    )
    labels = clusterer.fit_predict(embedding_matrix)
    return labels


def get_cluster_info(labels: np.ndarray) -> Dict:
    """
    获取聚类结果的统计信息。
    
    Args:
        labels: 聚类标签数组
        
    Returns:
        Dict: 包含聚类统计信息的字典
            - n_clusters: 簇的数量（不包括噪音点）
            - n_noise: 噪音点数量（标签为 -1 的点）
            - cluster_sizes: 每个簇的大小
            - largest_cluster: 最大簇的大小
            - smallest_cluster: 最小簇的大小（不包括噪音）
    """
    unique_labels = np.unique(labels)
    n_noise = np.sum(labels == -1)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # 计算每个簇的大小（排除噪音点）
    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[int(label)] = np.sum(labels == label)
    
    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_sizes': cluster_sizes,
        'largest_cluster': max(cluster_sizes.values()) if cluster_sizes else 0,
        'smallest_cluster': min(cluster_sizes.values()) if cluster_sizes else 0,
        'total_samples': len(labels)
    }


def cluster_with_stats(
    embedding_matrix: np.ndarray,
    method: str = 'hdbscan',
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    使用指定方法进行聚类，并返回标签和统计信息。
    
    Args:
        embedding_matrix: 嵌入向量矩阵
        method: 聚类方法，可选 'hdbscan', 'kmeans', 'agglomerative', 'spectral'
        **kwargs: 传递给具体聚类方法的参数
        
    Returns:
        Tuple[np.ndarray, Dict]: (聚类标签, 统计信息字典)
        
    Example:
        >>> labels, stats = cluster_with_stats(embeddings, method='hdbscan', min_cluster_size=3)
        >>> print(f"找到 {stats['n_clusters']} 个簇")
        >>> print(f"噪音点: {stats['n_noise']} 个")
    """
    # 根据方法选择聚类算法
    if method == 'hdbscan':
        labels = cluster_hdbscan(embedding_matrix, **kwargs)
    elif method == 'kmeans':
        labels = cluster_kmeans(embedding_matrix, **kwargs)
    elif method == 'agglomerative':
        labels = cluster_agglomerative(embedding_matrix, **kwargs)
    elif method == 'spectral':
        labels = cluster_spectral(embedding_matrix, **kwargs)
    else:
        raise ValueError(f"未知的聚类方法: {method}. 可选: 'hdbscan', 'kmeans', 'agglomerative', 'spectral'")
    
    # 获取统计信息
    stats = get_cluster_info(labels)
    stats['method'] = method
    
    return labels, stats