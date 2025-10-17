#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Graph Document Loader

Provides utilities for loading graph documents from pickle files.
"""

import os
import pickle
from typing import Optional, Any, List
import logging

logger = logging.getLogger(__name__)


class GraphDocumentLoader:
    """
    图谱文档加载器
    
    负责从pickle文件中加载图谱文档，提供错误处理和日志记录。
    
    Examples:
        >>> loader = GraphDocumentLoader()
        >>> graph_doc = loader.load("path/to/file.pkl")
        >>> if graph_doc:
        >>>     print(f"Loaded document with {len(graph_doc.nodes)} nodes")
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化加载器
        
        Args:
            verbose: 是否打印详细的错误信息
        """
        self.verbose = verbose
        self._load_count = 0
        self._error_count = 0
    
    def load(self, file_path: str) -> Optional[Any]:
        """
        加载单个图谱文档
        
        Args:
            file_path: pickle文件的完整路径
            
        Returns:
            Optional[Any]: 加载成功返回图谱文档对象，失败返回None
            
        Examples:
            >>> loader = GraphDocumentLoader()
            >>> doc = loader.load("/path/to/graph_doc.pkl")
        """
        if not os.path.exists(file_path):
            if self.verbose:
                logger.debug(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, 'rb') as f:
                graph_docs = pickle.load(f)
                
            # 通常pickle文件包含一个列表，取第一个元素
            if isinstance(graph_docs, list) and len(graph_docs) > 0:
                self._load_count += 1
                return graph_docs[0]
            elif graph_docs is not None:
                self._load_count += 1
                return graph_docs
            else:
                if self.verbose:
                    logger.warning(f"Empty document in file: {file_path}")
                self._error_count += 1
                return None
                
        except Exception as e:
            if self.verbose:
                logger.error(f"Error loading {file_path}: {e}")
            self._error_count += 1
            return None
    
    def load_batch(self, file_paths: List[str]) -> List[Any]:
        """
        批量加载多个图谱文档
        
        Args:
            file_paths: pickle文件路径列表
            
        Returns:
            List[Any]: 成功加载的图谱文档列表（跳过加载失败的文件）
            
        Examples:
            >>> loader = GraphDocumentLoader()
            >>> docs = loader.load_batch([
            >>>     "/path/to/doc1.pkl",
            >>>     "/path/to/doc2.pkl"
            >>> ])
            >>> print(f"Loaded {len(docs)} documents")
        """
        documents = []
        for file_path in file_paths:
            doc = self.load(file_path)
            if doc is not None:
                documents.append(doc)
        return documents
    
    def get_stats(self) -> dict:
        """
        获取加载统计信息
        
        Returns:
            dict: 包含成功加载数和错误数的字典
            
        Examples:
            >>> loader = GraphDocumentLoader()
            >>> loader.load("file1.pkl")
            >>> loader.load("file2.pkl")
            >>> stats = loader.get_stats()
            >>> print(f"Loaded: {stats['loaded']}, Errors: {stats['errors']}")
        """
        return {
            'loaded': self._load_count,
            'errors': self._error_count,
            'success_rate': self._load_count / max(1, self._load_count + self._error_count)
        }
    
    def reset_stats(self):
        """重置统计计数器"""
        self._load_count = 0
        self._error_count = 0


def load_graph_document(file_path: str, verbose: bool = False) -> Optional[Any]:
    """
    便捷函数：加载单个图谱文档
    
    这是 GraphDocumentLoader.load() 的便捷包装函数。
    
    Args:
        file_path: pickle文件的完整路径
        verbose: 是否打印详细的错误信息
        
    Returns:
        Optional[Any]: 加载成功返回图谱文档对象，失败返回None
        
    Examples:
        >>> doc = load_graph_document("/path/to/file.pkl")
        >>> if doc:
        >>>     print("Document loaded successfully")
    """
    loader = GraphDocumentLoader(verbose=verbose)
    return loader.load(file_path)
