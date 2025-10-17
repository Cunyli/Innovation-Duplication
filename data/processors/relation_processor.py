#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Relationship Processor

Provides utilities for processing relationships from graph documents and
converting them into structured DataFrame rows.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RelationshipProcessor:
    """
    关系处理器
    
    负责从图谱文档中提取关系信息，并转换为结构化的DataFrame行。
    支持不同数据源的元数据映射。
    
    Examples:
        >>> processor = RelationshipProcessor()
        >>> 
        >>> # 定义元数据映射（数据源特定字段）
        >>> metadata = {
        >>>     "Document number": 123,
        >>>     "Source Company": "Nokia",
        >>>     "Link Source Text": "https://example.com",
        >>>     "Source Text": "Full text...",
        >>>     "data_source": "company_website"
        >>> }
        >>> 
        >>> # 处理关系
        >>> rows = processor.process_relationships(
        >>>     graph_doc=graph_doc,
        >>>     node_description=node_desc,
        >>>     node_en_id=node_en_id,
        >>>     metadata=metadata
        >>> )
    """
    
    def __init__(self, verbose: bool = False):
        """
        初始化关系处理器
        
        Args:
            verbose: 是否打印详细的调试信息
        """
        self.verbose = verbose
        self._processed_count = 0
    
    def process_relationships(
        self,
        graph_doc: Any,
        node_description: Dict[str, str],
        node_en_id: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        从图谱文档中提取关系并转换为DataFrame行
        
        Args:
            graph_doc: 图谱文档对象，需要有 relationships 属性
            node_description: 节点ID到描述的映射
            node_en_id: 节点ID到英文ID的映射
            metadata: 数据源特定的元数据字典，包含：
                - Document number: 文档编号
                - Source Company or VAT id: 来源公司
                - Link Source Text: 源链接
                - Source Text: 源文本
                - data_source: 数据源标识
        
        Returns:
            List[Dict[str, Any]]: 关系行列表，每行包含完整的关系信息
            
        Examples:
            >>> processor = RelationshipProcessor()
            >>> metadata = {
            >>>     "Document number": 1,
            >>>     "Source Company": "VTT",
            >>>     "Link Source Text": "https://vtt.fi",
            >>>     "Source Text": "VTT develops...",
            >>>     "data_source": "company_website"
            >>> }
            >>> rows = processor.process_relationships(
            >>>     graph_doc, node_desc, node_en_id, metadata
            >>> )
            >>> print(f"Extracted {len(rows)} relationships")
        """
        relationship_rows = []
        
        # 检查文档是否有relationships属性
        if not hasattr(graph_doc, 'relationships'):
            if self.verbose:
                logger.warning("Graph document has no 'relationships' attribute")
            return relationship_rows
        
        # 提取数据源特定字段（处理不同的列名）
        doc_number = metadata.get("Document number")
        source_company = metadata.get("Source Company") or metadata.get("VAT id")
        link_source = metadata.get("Link Source Text")
        source_text = metadata.get("Source Text")
        data_source = metadata.get("data_source", "unknown")
        
        # 遍历所有关系
        for rel in graph_doc.relationships:
            try:
                # 构建关系行
                rel_row = {
                    "Document number": doc_number,
                    "Source Company": source_company,
                    "relationship description": rel.properties.get('description', '') if hasattr(rel, 'properties') else '',
                    "source_id": rel.source,
                    "source_type": rel.source_type,
                    "source_english_id": node_en_id.get(rel.source, None),
                    "source_description": node_description.get(rel.source, None),
                    "relationship_type": rel.type,
                    "target_id": rel.target,
                    "target_type": rel.target_type,
                    "target_english_id": node_en_id.get(rel.target, None),
                    "target_description": node_description.get(rel.target, None),
                    "Link Source Text": link_source,
                    "Source Text": source_text,
                    "data_source": data_source
                }
                
                relationship_rows.append(rel_row)
                self._processed_count += 1
                
            except Exception as e:
                if self.verbose:
                    logger.error(f"Error processing relationship {rel.source} -> {rel.target}: {e}")
                continue
        
        if self.verbose:
            logger.info(f"Processed {len(relationship_rows)} relationships from document")
        
        return relationship_rows
    
    def process_relationships_batch(
        self,
        graph_docs: List[Any],
        node_descriptions: List[Dict[str, str]],
        node_en_ids: List[Dict[str, str]],
        metadatas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个图谱文档的关系
        
        Args:
            graph_docs: 图谱文档列表
            node_descriptions: 节点描述映射列表
            node_en_ids: 节点英文ID映射列表
            metadatas: 元数据列表
        
        Returns:
            List[Dict[str, Any]]: 所有关系行的列表
            
        Examples:
            >>> processor = RelationshipProcessor()
            >>> all_rows = processor.process_relationships_batch(
            >>>     graph_docs=[doc1, doc2, doc3],
            >>>     node_descriptions=[desc1, desc2, desc3],
            >>>     node_en_ids=[id1, id2, id3],
            >>>     metadatas=[meta1, meta2, meta3]
            >>> )
        """
        all_rows = []
        
        for graph_doc, node_desc, node_en_id, metadata in zip(
            graph_docs, node_descriptions, node_en_ids, metadatas
        ):
            rows = self.process_relationships(
                graph_doc, node_desc, node_en_id, metadata
            )
            all_rows.extend(rows)
        
        return all_rows
    
    def get_stats(self) -> dict:
        """
        获取处理统计信息
        
        Returns:
            dict: 包含处理数量的字典
            
        Examples:
            >>> processor = RelationshipProcessor()
            >>> processor.process_relationships(...)
            >>> stats = processor.get_stats()
            >>> print(f"Processed: {stats['processed']}")
        """
        return {
            'processed': self._processed_count
        }
    
    def reset_stats(self):
        """重置统计计数器"""
        self._processed_count = 0


def process_graph_relationships(
    graph_doc: Any,
    node_description: Dict[str, str],
    node_en_id: Dict[str, str],
    metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    便捷函数：处理图谱文档的关系
    
    这是 RelationshipProcessor.process_relationships() 的便捷包装函数。
    
    Args:
        graph_doc: 图谱文档对象
        node_description: 节点描述映射
        node_en_id: 节点英文ID映射
        metadata: 元数据字典
    
    Returns:
        List[Dict[str, Any]]: 关系行列表
        
    Examples:
        >>> rows = process_graph_relationships(
        >>>     graph_doc, node_desc, node_en_id, metadata
        >>> )
    """
    processor = RelationshipProcessor()
    return processor.process_relationships(
        graph_doc, node_description, node_en_id, metadata
    )
