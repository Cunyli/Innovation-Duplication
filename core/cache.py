#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
缓存模块 - 提供可插拔的缓存后端实现

此模块实现了一个灵活的缓存系统，支持多种后端存储方式：
- JSON 文件缓存
- 内存缓存
- 可扩展的自定义后端

主要用于缓存嵌入向量等计算密集型结果。
"""

import os
import json
from typing import Dict, Any, Optional, List, Protocol, Union


class CacheBackend(Protocol):
    """缓存后端接口协议
    
    定义了所有缓存后端必须实现的方法。
    使用 Protocol 允许鸭子类型和更灵活的实现。
    """
    
    def load(self) -> Dict:
        """加载缓存数据
        
        Returns:
            Dict: 缓存的数据字典
        """
        ...
    
    def save(self, data: Dict) -> bool:
        """保存数据到缓存
        
        Args:
            data: 要保存的数据字典
            
        Returns:
            bool: 保存是否成功
        """
        ...
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，不存在则返回 None
        """
        ...
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        ...
    
    def update(self, data: Dict) -> None:
        """批量更新缓存
        
        Args:
            data: 要更新的数据字典
        """
        ...
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """获取缓存中缺失的键
        
        Args:
            required_keys: 需要的键列表
            
        Returns:
            List[str]: 缓存中不存在的键列表
        """
        ...
    
    def contains(self, key: str) -> bool:
        """检查缓存是否包含指定键
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 是否存在
        """
        ...


class JsonFileCache:
    """基于 JSON 文件的缓存实现
    
    将缓存数据持久化到 JSON 文件中。适合需要跨会话保留数据的场景。
    
    特点：
    - 持久化存储
    - 人类可读的格式
    - 支持惰性加载
    
    Attributes:
        cache_path: JSON 文件路径
        cache_data: 内存中的缓存数据
        loaded: 是否已加载缓存
    """
    
    def __init__(self, cache_path: str):
        """初始化 JSON 文件缓存

        Args:
            cache_path: JSON 缓存文件的路径。可以传入文件名（例如 'emb.json'），
                        也可以传入相对或绝对路径。
                        如果传入的是文件名或相对路径且没有目录部分，
                        则会默认放到项目的 `data/cache/` 目录下。
        """
        # Normalize and ensure default cache directory under project data
        if not os.path.isabs(cache_path) and os.path.dirname(cache_path) == '':
            default_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'cache')
            # If package is not under repo root (rare), fallback to repo root 'data/cache'
            if not os.path.exists(default_cache_dir):
                try:
                    os.makedirs(default_cache_dir, exist_ok=True)
                except Exception:
                    # best-effort: fallback to current working directory
                    default_cache_dir = os.path.join(os.getcwd(), 'data', 'cache')
                    os.makedirs(default_cache_dir, exist_ok=True)
            self.cache_path = os.path.join(default_cache_dir, cache_path)
        else:
            self.cache_path = cache_path
        self.cache_data = {}
        self.loaded = False
    
    def load(self) -> Dict:
        """从 JSON 文件加载缓存数据
        
        采用惰性加载策略，只在第一次调用时加载文件。
        
        Returns:
            Dict: 加载的缓存数据
        """
        if self.loaded:
            return self.cache_data
            
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache_data = json.load(f)
                print(f"✅ 从缓存加载了 {len(self.cache_data)} 项: {self.cache_path}")
                self.loaded = True
                return self.cache_data
            except Exception as e:
                print(f"❌ 加载缓存时出错: {e}")
                return {}
        else:
            print(f"ℹ️  缓存文件不存在: {self.cache_path}")
            return {}
    
    def save(self, data: Dict) -> bool:
        """保存数据到 JSON 文件
        
        自动创建必要的目录结构。
        
        Args:
            data: 要保存的数据字典
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 确保目录存在
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ 保存了 {len(data)} 项到缓存: {self.cache_path}")
            self.cache_data = data
            self.loaded = True
            return True
        except Exception as e:
            print(f"❌ 保存缓存时出错: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        如果未加载，会先加载缓存。
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，不存在则返回 None
        """
        if not self.loaded:
            self.load()
        return self.cache_data.get(key, None)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项
        
        仅更新内存，不自动保存到文件。
        需要调用 save() 或 update() 来持久化。
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        if not self.loaded:
            self.load()
        self.cache_data[key] = value
    
    def update(self, data: Dict) -> None:
        """批量更新缓存并保存到文件
        
        Args:
            data: 要更新的数据字典
        """
        if not self.loaded:
            self.load()
        self.cache_data.update(data)
        self.save(self.cache_data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """获取缓存中缺失的键
        
        Args:
            required_keys: 需要的键列表
            
        Returns:
            List[str]: 缓存中不存在的键列表
        """
        if not self.loaded:
            self.load()
        return [k for k in required_keys if k not in self.cache_data]
    
    def contains(self, key: str) -> bool:
        """检查缓存是否包含指定键
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 是否存在
        """
        if not self.loaded:
            self.load()
        return key in self.cache_data


class MemoryCache:
    """基于内存的缓存实现
    
    将数据存储在内存中，不持久化。适合临时数据或测试场景。
    
    特点：
    - 速度快
    - 不持久化
    - 进程结束后数据丢失
    
    Attributes:
        cache_data: 内存中的缓存数据
    """
    
    def __init__(self):
        """初始化内存缓存"""
        self.cache_data = {}
    
    def load(self) -> Dict:
        """返回内存中的缓存数据
        
        Returns:
            Dict: 缓存数据
        """
        return self.cache_data
    
    def save(self, data: Dict) -> bool:
        """保存数据到内存
        
        Args:
            data: 要保存的数据字典
            
        Returns:
            bool: 始终返回 True
        """
        self.cache_data = data
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，不存在则返回 None
        """
        return self.cache_data.get(key, None)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        self.cache_data[key] = value
    
    def update(self, data: Dict) -> None:
        """批量更新缓存
        
        Args:
            data: 要更新的数据字典
        """
        self.cache_data.update(data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """获取缓存中缺失的键
        
        Args:
            required_keys: 需要的键列表
            
        Returns:
            List[str]: 缓存中不存在的键列表
        """
        return [k for k in required_keys if k not in self.cache_data]
    
    def contains(self, key: str) -> bool:
        """检查缓存是否包含指定键
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 是否存在
        """
        return key in self.cache_data


class EmbeddingCache:
    """嵌入向量缓存系统
    
    可插拔的嵌入向量缓存系统，支持不同的存储后端。
    主要用于缓存文本嵌入向量，避免重复计算。
    
    当前支持:
    - 文件缓存 (JSON)
    - 内存缓存
    
    可扩展支持:
    - 数据库缓存 (Redis, MongoDB, etc.)
    - 分布式缓存
    
    Attributes:
        use_cache: 是否启用缓存
        backend: 缓存后端实现
    
    Examples:
        >>> # 使用 JSON 文件缓存
        >>> cache = EmbeddingCache(backend_type="json", cache_path="./embeddings.json")
        >>> cache.set("text1", [0.1, 0.2, 0.3])
        >>> cache.get("text1")
        [0.1, 0.2, 0.3]
        
        >>> # 使用内存缓存
        >>> cache = EmbeddingCache(backend_type="memory")
        
        >>> # 禁用缓存
        >>> cache = EmbeddingCache(use_cache=False)
    """
    
    def __init__(
        self, 
        backend: Optional[CacheBackend] = None, 
        cache_path: str = "./embedding_vectors.json", 
        backend_type: str = "json", 
        use_cache: bool = True
    ):
        """初始化嵌入缓存系统
        
        Args:
            backend: 自定义缓存后端实现（可选）
            cache_path: 缓存文件路径（仅用于文件缓存）
            backend_type: 后端类型，'json' 或 'memory'
            use_cache: 是否启用缓存
            
        Raises:
            ValueError: 如果指定了不支持的 backend_type
        """
        self.use_cache = use_cache
        
        if not use_cache:
            self.backend = None
            return
            
        if backend is not None:
            self.backend = backend
        elif backend_type == "json":
            self.backend = JsonFileCache(cache_path)
        elif backend_type == "memory":
            self.backend = MemoryCache()
        else:
            raise ValueError(f"不支持的后端类型: {backend_type}")
    
    def load(self) -> Dict:
        """加载缓存数据
        
        Returns:
            Dict: 加载的缓存数据，如果缓存被禁用则返回空字典
        """
        if not self.use_cache or self.backend is None:
            return {}
        return self.backend.load()
    
    def save(self, data: Dict) -> bool:
        """保存数据到缓存
        
        Args:
            data: 要保存的数据字典
            
        Returns:
            bool: 保存是否成功，缓存被禁用时返回 False
        """
        if not self.use_cache or self.backend is None:
            return False
        return self.backend.save(data)
    
    def get(self, key: str) -> Optional[Any]:
        """从缓存获取值
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[Any]: 缓存的值，不存在或缓存被禁用时返回 None
        """
        if not self.use_cache or self.backend is None:
            return None
        return self.backend.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值
        
        Args:
            key: 缓存键
            value: 要缓存的值
        """
        if not self.use_cache or self.backend is None:
            return
        self.backend.set(key, value)
    
    def update(self, data: Dict) -> None:
        """批量更新缓存
        
        Args:
            data: 要更新的数据字典
        """
        if not self.use_cache or self.backend is None:
            return
        self.backend.update(data)
    
    def get_missing_keys(self, required_keys: List[str]) -> List[str]:
        """获取缓存中缺失的键
        
        Args:
            required_keys: 需要的键列表
            
        Returns:
            List[str]: 缓存中不存在的键列表，缓存被禁用时返回所有键
        """
        if not self.use_cache or self.backend is None:
            return required_keys
        return self.backend.get_missing_keys(required_keys)
    
    def contains(self, key: str) -> bool:
        """检查缓存是否包含指定键
        
        Args:
            key: 要检查的键
            
        Returns:
            bool: 是否存在，缓存被禁用时返回 False
        """
        if not self.use_cache or self.backend is None:
            return False
        return self.backend.contains(key)


class CacheFactory:
    """缓存工厂类
    
    用于创建不同类型的缓存实例。
    提供统一的接口来创建和配置缓存对象。
    
    Examples:
        >>> # 创建 JSON 文件缓存
        >>> cache = CacheFactory.create_cache(
        ...     cache_type="embedding",
        ...     backend_type="json",
        ...     cache_path="./cache/embeddings.json"
        ... )
        
        >>> # 创建内存缓存
        >>> cache = CacheFactory.create_cache(
        ...     cache_type="embedding",
        ...     backend_type="memory"
        ... )
        
        >>> # 禁用缓存
        >>> cache = CacheFactory.create_cache(
        ...     cache_type="embedding",
        ...     use_cache=False
        ... )
    """
    
    @staticmethod
    def create_cache(
        cache_type: str = "embedding", 
        backend_type: str = "json",
        cache_path: str = "embedding_vectors.json",
        use_cache: bool = True
    ) -> Union[EmbeddingCache, Any]:
        """创建缓存实例
        
        Args:
            cache_type: 缓存类型，当前仅支持 'embedding'
            backend_type: 后端类型，'json' 或 'memory'
            cache_path: 缓存文件路径（仅用于文件缓存）
            use_cache: 是否启用缓存
            
        Returns:
            Union[EmbeddingCache, Any]: 缓存实例
            
        Raises:
            ValueError: 如果指定了不支持的 cache_type
        """
        if cache_type == "embedding":
            return EmbeddingCache(
                backend_type=backend_type,
                cache_path=cache_path,
                use_cache=use_cache
            )
        else:
            raise ValueError(f"不支持的缓存类型: {cache_type}")


# 导出公共接口
__all__ = [
    'CacheBackend',
    'JsonFileCache',
    'MemoryCache',
    'EmbeddingCache',
    'CacheFactory'
]
