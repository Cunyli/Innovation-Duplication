#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embedding Manager Module

Manages embedding generation, caching, and retrieval for innovations.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from tqdm import tqdm
from ...core.cache import CacheFactory


class EmbeddingManager:
    """嵌入向量管理器 - 处理嵌入生成和缓存"""
    
    DEFAULT_CACHE_CONFIG = {
        "type": "embedding",
        "backend": "json",
        "path": "./embedding_vectors.json",
        "use_cache": True
    }
    
    def __init__(
        self,
        cache_config: Optional[Dict] = None,
        embedding_function: Optional[Callable] = None
    ):
        """
        初始化嵌入管理器
        
        Args:
            cache_config: 缓存配置字典
            embedding_function: 嵌入生成函数 (text, model) -> np.ndarray
        """
        # 合并配置
        config = {**self.DEFAULT_CACHE_CONFIG}
        if cache_config:
            config.update(cache_config)
        
        self.config = config
        self.embedding_function = embedding_function
        
        # 初始化缓存
        self.cache = CacheFactory.create_cache(
            cache_type=config["type"],
            backend_type=config["backend"],
            cache_path=config["path"],
            use_cache=config["use_cache"]
        )
    
    def _generate_missing_embeddings(
        self,
        missing_ids: List[str],
        features: Dict[str, str],
        model
    ) -> Dict[str, np.ndarray]:
        """
        为缺失的 ID 生成嵌入向量
        
        Args:
            missing_ids: 缺失的创新 ID 列表
            features: 创新特征字典 {id: text}
            model: 嵌入模型
        
        Returns:
            Dict[str, np.ndarray]: 新生成的嵌入
        """
        if not missing_ids:
            return {}
        
        print(f"Generating {len(missing_ids)} new embeddings...")
        new_embeddings = {}
        
        for innovation_id in tqdm(missing_ids, desc="Embedding innovations"):
            text = features[innovation_id]
            new_embeddings[innovation_id] = self.embedding_function(text, model)
        
        return new_embeddings
    
    def get_embeddings(
        self,
        innovation_features: Dict[str, str],
        model=None
    ) -> Tuple[List[str], np.ndarray]:
        """
        获取所有创新的嵌入向量（使用缓存）
        
        Args:
            innovation_features: 创新特征字典 {id: text}
            model: 嵌入模型
        
        Returns:
            Tuple[List[str], np.ndarray]: (创新ID列表, 嵌入矩阵)
        """
        print("Generating features for similarity comparison...")
        
        # 从缓存加载已有嵌入
        embeddings: Dict[str, np.ndarray] = self.cache.load()
        
        # 找出缺失的 ID
        all_ids = list(innovation_features.keys())
        missing_ids = self.cache.get_missing_keys(all_ids)
        
        # 生成缺失的嵌入
        if missing_ids:
            new_embeddings = self._generate_missing_embeddings(
                missing_ids, innovation_features, model
            )
            
            # 更新缓存
            self.cache.update(new_embeddings)
            embeddings.update(new_embeddings)
        
        # 构建有序的嵌入矩阵
        embedding_items = [(iid, embeddings[iid]) for iid in all_ids]
        innovation_ids = [item[0] for item in embedding_items]
        embedding_matrix = np.vstack([item[1] for item in embedding_items])
        
        return innovation_ids, embedding_matrix
    
    @staticmethod
    def create_from_config(
        cache_config: Optional[Dict],
        embedding_function: Callable
    ) -> 'EmbeddingManager':
        """
        工厂方法：从配置创建嵌入管理器
        
        Args:
            cache_config: 缓存配置
            embedding_function: 嵌入函数
        
        Returns:
            EmbeddingManager: 嵌入管理器实例
        """
        return EmbeddingManager(
            cache_config=cache_config,
            embedding_function=embedding_function
        )
