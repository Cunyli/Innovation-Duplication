#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Clustering Strategy Module

Provides different clustering strategies for innovation deduplication.
"""

import numpy as np
from typing import Dict, List, Protocol, Optional
from abc import ABC, abstractmethod

from utils.cluster.cluster_algorithms import cluster_with_stats
from utils.cluster.graph_clustering import (
    graph_threshold_clustering,
    graph_kcore_clustering
)


class ClusteringStrategy(ABC):
    """聚类策略抽象基类"""
    
    @abstractmethod
    def cluster(
        self,
        embedding_matrix: np.ndarray,
        innovation_ids: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        执行聚类并返回标准映射
        
        Args:
            embedding_matrix: 嵌入矩阵 (N, D)
            innovation_ids: 创新 ID 列表
            **kwargs: 算法特定参数
        
        Returns:
            Dict[str, str]: {innovation_id: canonical_id}
        """
        pass


class FlatClusteringStrategy(ClusteringStrategy):
    """平面聚类策略 - 支持 HDBSCAN, K-means, Agglomerative, Spectral"""
    
    SUPPORTED_METHODS = {"hdbscan", "kmeans", "agglomerative", "spectral"}
    
    def __init__(self, method: str):
        """
        初始化平面聚类策略
        
        Args:
            method: 聚类方法名称
        """
        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method.lower()
    
    def cluster(
        self,
        embedding_matrix: np.ndarray,
        innovation_ids: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        使用平面聚类算法进行聚类
        
        Args:
            embedding_matrix: 嵌入矩阵
            innovation_ids: 创新 ID 列表
            **kwargs: 传递给聚类算法的参数
        
        Returns:
            Dict[str, str]: 标准映射
        """
        print(f"Clustering similar innovations with method='{self.method}'...")
        
        # 调用统一的聚类接口
        labels, stats = cluster_with_stats(
            embedding_matrix=embedding_matrix,
            method=self.method,
            **kwargs
        )
        
        # 打印统计信息
        self._print_stats(stats)
        
        # 构建簇字典
        clusters = self._build_clusters(labels, innovation_ids)
        
        # 转换为标准映射
        return self._clusters_to_mapping(clusters)
    
    @staticmethod
    def _print_stats(stats: Dict):
        """打印聚类统计信息"""
        print(f"✅ 聚类完成 [{stats['method'].upper()}]:")
        print(f"   📊 簇数量: {stats['n_clusters']}")
        print(f"   ⚠️  噪音点: {stats['n_noise']} ({stats['n_noise']/stats['total_samples']*100:.1f}%)")
        print(f"   📈 最大簇: {stats['largest_cluster']} 样本")
        print(f"   📉 最小簇: {stats['smallest_cluster']} 样本")
        print(f"   🔢 总样本: {stats['total_samples']}")
    
    @staticmethod
    def _build_clusters(labels: np.ndarray, innovation_ids: List[str]) -> Dict:
        """
        从标签构建簇字典
        
        Args:
            labels: 聚类标签数组
            innovation_ids: 创新 ID 列表
        
        Returns:
            Dict: {label: [innovation_ids]}
        """
        clusters = {}
        
        for idx, label in enumerate(labels):
            if label == -1:
                # HDBSCAN 的噪声点单独成簇
                key = f"noise_{innovation_ids[idx]}"
                clusters.setdefault(key, []).append(innovation_ids[idx])
            else:
                clusters.setdefault(int(label), []).append(innovation_ids[idx])
        
        return clusters
    
    @staticmethod
    def _clusters_to_mapping(clusters: Dict) -> Dict[str, str]:
        """
        将簇字典转换为标准映射
        
        Args:
            clusters: 簇字典
        
        Returns:
            Dict[str, str]: {innovation_id: canonical_id}
        """
        canonical_mapping = {}
        
        for label_key, members in clusters.items():
            # 使用簇中第一个成员作为标准 ID
            canonical_id = members[0]
            for member_id in members:
                canonical_mapping[member_id] = canonical_id
        
        return canonical_mapping


class GraphClusteringStrategy(ClusteringStrategy):
    """图聚类策略 - 支持 Graph Threshold 和 K-core"""
    
    SUPPORTED_METHODS = {"graph_threshold", "graph_kcore"}
    
    def __init__(self, method: str):
        """
        初始化图聚类策略
        
        Args:
            method: 聚类方法名称
        """
        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Choose from: {self.SUPPORTED_METHODS}"
            )
        self.method = method.lower()
    
    def cluster(
        self,
        embedding_matrix: np.ndarray,
        innovation_ids: List[str],
        **kwargs
    ) -> Dict[str, str]:
        """
        使用图聚类算法进行聚类
        
        Args:
            embedding_matrix: 嵌入矩阵
            innovation_ids: 创新 ID 列表
            **kwargs: 传递给聚类算法的参数
        
        Returns:
            Dict[str, str]: 标准映射
        """
        print(f"Clustering similar innovations with method='{self.method}'...")
        
        # 提取参数
        similarity_threshold = kwargs.get("similarity_threshold", 0.85)
        use_cosine = kwargs.get("use_cosine", True)
        
        # 调用对应的图聚类算法
        if self.method == "graph_threshold":
            clusters_dict = graph_threshold_clustering(
                embedding_matrix=embedding_matrix,
                ids=innovation_ids,
                similarity_threshold=similarity_threshold,
                use_cosine=use_cosine
            )
        else:  # graph_kcore
            k_core = kwargs.get("k_core", 15)
            clusters_dict = graph_kcore_clustering(
                embedding_matrix=embedding_matrix,
                ids=innovation_ids,
                similarity_threshold=similarity_threshold,
                k_core=k_core,
                use_cosine=use_cosine
            )
        
        # 转换为标准映射
        return self._clusters_dict_to_mapping(clusters_dict)
    
    @staticmethod
    def _clusters_dict_to_mapping(clusters_dict: Dict[str, List[str]]) -> Dict[str, str]:
        """
        将图聚类结果转换为标准映射
        
        Args:
            clusters_dict: {canonical_id: [member_ids]}
        
        Returns:
            Dict[str, str]: {innovation_id: canonical_id}
        """
        canonical_mapping = {}
        
        for canonical_id, members in clusters_dict.items():
            for member_id in members:
                canonical_mapping[member_id] = canonical_id
        
        return canonical_mapping


class ClusteringStrategyFactory:
    """聚类策略工厂"""
    
    METHOD_TO_STRATEGY = {
        "hdbscan": FlatClusteringStrategy,
        "kmeans": FlatClusteringStrategy,
        "agglomerative": FlatClusteringStrategy,
        "spectral": FlatClusteringStrategy,
        "graph_threshold": GraphClusteringStrategy,
        "graph_kcore": GraphClusteringStrategy,
    }
    
    @classmethod
    def create_strategy(cls, method: str) -> ClusteringStrategy:
        """
        根据方法名创建聚类策略
        
        Args:
            method: 聚类方法名称
        
        Returns:
            ClusteringStrategy: 聚类策略实例
        
        Raises:
            ValueError: 如果方法不支持
        """
        method_lower = method.lower()
        
        if method_lower not in cls.METHOD_TO_STRATEGY:
            raise ValueError(
                f"Unknown clustering method '{method}'.\n"
                f"请选择：{list(cls.METHOD_TO_STRATEGY.keys())}"
            )
        
        strategy_class = cls.METHOD_TO_STRATEGY[method_lower]
        return strategy_class(method_lower)
