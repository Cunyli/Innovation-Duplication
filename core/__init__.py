"""
Core modules for Innovation Resolution system
"""

from .cache import (
    CacheBackend,
    JsonFileCache,
    MemoryCache,
    EmbeddingCache,
    CacheFactory
)

__all__ = [
    'CacheBackend',
    'JsonFileCache',
    'MemoryCache',
    'EmbeddingCache',
    'CacheFactory'
]
