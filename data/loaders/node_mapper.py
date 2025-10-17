#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node Mapper

Provides utilities for extracting and mapping node information from graph documents.
"""

from typing import Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class NodeMapper:
    """
    节点映射提取器
    
    负责从图谱文档中提取节点的ID映射关系，包括：
    - 英文ID（english_id）映射
    - 描述（description）映射
    
    Examples:
        >>> mapper = NodeMapper()
        >>> node_desc, node_en_id = mapper.extract_mappings(graph_doc)
        >>> print(f"Mapped {len(node_en_id)} nodes")
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化映射器
        
        Args:
            verbose: 是否打印详细的调试信息
        """
        self.verbose = verbose
    
    def extract_mappings(self, graph_doc: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        从图谱文档中提取节点映射
        
        Args:
            graph_doc: 图谱文档对象，需要有 nodes 属性
            
        Returns:
            Tuple[Dict[str, str], Dict[str, str]]: 
                - 第一个字典: {node.id: description}
                - 第二个字典: {node.id: english_id}
                
        Examples:
            >>> mapper = NodeMapper()
            >>> descriptions, english_ids = mapper.extract_mappings(graph_doc)
            >>> 
            >>> # 使用映射
            >>> for node_id in graph_doc.nodes:
            >>>     eng_name = english_ids.get(node_id, node_id)
            >>>     desc = descriptions.get(node_id, "")
            >>>     print(f"{eng_name}: {desc}")
        """
        node_description = {}
        node_en_id = {}
        
        # 检查文档是否有nodes属性
        if not hasattr(graph_doc, 'nodes'):
            if self.verbose:
                logger.warning("Graph document has no 'nodes' attribute")
            return node_description, node_en_id
        
        # 提取每个节点的描述和英文ID
        for node in graph_doc.nodes:
            node_id = node.id
            
            # 提取描述
            if hasattr(node, 'properties') and isinstance(node.properties, dict):
                node_description[node_id] = node.properties.get('description', '')
                # 优先使用english_id，如果没有则使用原始ID
                node_en_id[node_id] = node.properties.get('english_id', node_id)
            else:
                node_description[node_id] = ''
                node_en_id[node_id] = node_id
        
        if self.verbose:
            logger.info(f"Extracted mappings for {len(node_en_id)} nodes")
        
        return node_description, node_en_id
    
    def get_english_id(self, graph_doc: Any, node_id: str) -> str:
        """
        获取单个节点的英文ID
        
        Args:
            graph_doc: 图谱文档对象
            node_id: 节点的原始ID
            
        Returns:
            str: 英文ID，如果不存在则返回原始ID
            
        Examples:
            >>> mapper = NodeMapper()
            >>> eng_id = mapper.get_english_id(graph_doc, "temp_org_123")
            >>> print(eng_id)  # 可能输出: "Nokia Corporation"
        """
        _, node_en_id = self.extract_mappings(graph_doc)
        return node_en_id.get(node_id, node_id)
    
    def get_description(self, graph_doc: Any, node_id: str) -> str:
        """
        获取单个节点的描述
        
        Args:
            graph_doc: 图谱文档对象
            node_id: 节点的原始ID
            
        Returns:
            str: 节点描述，如果不存在则返回空字符串
            
        Examples:
            >>> mapper = NodeMapper()
            >>> desc = mapper.get_description(graph_doc, "temp_org_123")
            >>> print(desc)
        """
        node_description, _ = self.extract_mappings(graph_doc)
        return node_description.get(node_id, '')


def extract_node_mappings(graph_doc: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    便捷函数：提取节点映射
    
    这是 NodeMapper.extract_mappings() 的便捷包装函数。
    
    Args:
        graph_doc: 图谱文档对象
        
    Returns:
        Tuple[Dict[str, str], Dict[str, str]]: (descriptions, english_ids)
        
    Examples:
        >>> descriptions, english_ids = extract_node_mappings(graph_doc)
        >>> print(f"Mapped {len(english_ids)} nodes")
    """
    mapper = NodeMapper()
    return mapper.extract_mappings(graph_doc)
