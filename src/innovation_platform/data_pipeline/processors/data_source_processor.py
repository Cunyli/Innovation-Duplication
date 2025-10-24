#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Source Processor

Provides a unified processor for handling different data sources (company websites, VTT domain, etc.)
with a common pipeline: load CSV → process documents → extract entities/relationships → build DataFrame.
"""

import os
import pandas as pd
from typing import List, Dict, Tuple, Callable, Any, Optional
from tqdm import tqdm
import logging

from ..loaders import GraphDocumentLoader, NodeMapper
from .relation_processor import RelationshipProcessor

logger = logging.getLogger(__name__)


class DataSourceProcessor:
    """
    统一的数据源处理器
    
    封装了完整的数据处理流程：
    1. 读取CSV数据
    2. 遍历每一行
    3. 加载图谱文档
    4. 提取实体和关系
    5. 构建DataFrame
    
    支持不同数据源的配置化处理。
    
    Examples:
        >>> processor = DataSourceProcessor(
        >>>     graph_docs_dir="data/graph_docs_names_resolved",
        >>>     data_source_name="company_website"
        >>> )
        >>> 
        >>> df_result = processor.process(
        >>>     df=df_company,
        >>>     file_pattern="{Company name}_{index}.pkl",
        >>>     metadata_mapper=lambda row, idx: {...},
        >>>     entity_extractor=extract_entities_from_document,
        >>>     relation_extractor=extract_relationships_from_document,
        >>>     pred_entities=all_entities,
        >>>     pred_relations=all_relations
        >>> )
    """
    
    def __init__(
        self,
        graph_docs_dir: str,
        data_source_name: str,
        verbose: bool = False
    ):
        """
        初始化数据源处理器
        
        Args:
            graph_docs_dir: 图谱文档所在目录
            data_source_name: 数据源名称（用于日志和标识）
            verbose: 是否打印详细信息
        """
        self.graph_docs_dir = graph_docs_dir
        self.data_source_name = data_source_name
        self.verbose = verbose
        
        # 初始化工具
        self.loader = GraphDocumentLoader(verbose=False)
        self.node_mapper = NodeMapper(verbose=False)
        self.rel_processor = RelationshipProcessor(verbose=False)
        
        # 统计信息
        self._processed_docs = 0
        self._failed_docs = 0
        self._total_relationships = 0
    
    def process(
        self,
        df: pd.DataFrame,
        file_pattern: str,
        metadata_mapper: Callable[[pd.Series, int], Dict[str, Any]],
        entity_extractor: Callable,
        relation_extractor: Callable,
        pred_entities: Optional[List[Dict]] = None,
        pred_relations: Optional[List[Dict]] = None,
        index_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        处理数据源
        
        Args:
            df: 输入的DataFrame（包含要处理的行）
            file_pattern: 文件名模式，支持占位符如 "{Company name}_{index}.pkl"
            metadata_mapper: 元数据映射函数，接收 (row, index) 返回元数据字典
            entity_extractor: 实体提取函数
            relation_extractor: 关系提取函数
            pred_entities: 用于累积预测实体的列表（可选）
            pred_relations: 用于累积预测关系的列表（可选）
            index_column: 用作索引的列名（可选，默认使用DataFrame索引）
        
        Returns:
            pd.DataFrame: 处理后的关系DataFrame
            
        Examples:
            >>> # 定义元数据映射函数
            >>> def map_metadata(row, idx):
            >>>     return {
            >>>         "Document number": idx,
            >>>         "Source Company": row["Company name"],
            >>>         "Link Source Text": row["Link"],
            >>>         "Source Text": row["text_content"],
            >>>         "data_source": "company_website"
            >>>     }
            >>> 
            >>> # 处理数据
            >>> df_result = processor.process(
            >>>     df=df_company,
            >>>     file_pattern="{Company name}_{index}.pkl",
            >>>     metadata_mapper=map_metadata,
            >>>     entity_extractor=extract_entities,
            >>>     relation_extractor=extract_relations
            >>> )
        """
        relationship_rows = []
        
        desc = f"Processing {self.data_source_name} data"
        with tqdm(total=len(df), desc=desc) as pbar:
            for idx, row in df.iterrows():
                try:
                    # 1. 构建文件路径
                    file_path = self._build_file_path(row, idx, file_pattern)
                    
                    # 2. 加载图谱文档
                    graph_doc = self.loader.load(file_path)
                    
                    if graph_doc is None:
                        self._failed_docs += 1
                        pbar.update(1)
                        continue
                    
                    # 3. 提取实体和关系（用于评估）
                    if entity_extractor:
                        entity_extractor(graph_doc, pred_entities)
                    
                    if relation_extractor:
                        relation_extractor(graph_doc, pred_relations)
                    
                    # 4. 提取节点映射
                    node_description, node_en_id = self.node_mapper.extract_mappings(graph_doc)
                    
                    # 5. 处理关系
                    # 使用提供的索引列或DataFrame索引
                    actual_idx = row[index_column] if index_column and index_column in row else idx
                    metadata = metadata_mapper(row, actual_idx)
                    
                    rows = self.rel_processor.process_relationships(
                        graph_doc, node_description, node_en_id, metadata
                    )
                    
                    relationship_rows.extend(rows)
                    self._processed_docs += 1
                    self._total_relationships += len(rows)
                    
                except Exception as e:
                    if self.verbose:
                        logger.error(f"Error processing {idx}: {e}")
                    else:
                        print(f"Error processing {idx}: {e}")
                    self._failed_docs += 1
                
                pbar.update(1)
        
        # 构建结果DataFrame
        df_result = pd.DataFrame(relationship_rows)
        
        # 打印统计信息
        print(f"Processed {len(df_result)} relationships from {self.data_source_name}")
        
        return df_result
    
    def _build_file_path(self, row: pd.Series, idx: int, pattern: str) -> str:
        """
        根据模式构建文件路径
        
        Args:
            row: DataFrame行
            idx: 行索引
            pattern: 文件名模式，如 "{Company name}_{index}.pkl"
        
        Returns:
            str: 完整的文件路径
        """
        # 替换索引占位符
        file_name = pattern.replace("{index}", str(idx))
        
        # 替换列占位符
        for col in row.index:
            placeholder = f"{{{col}}}"
            if placeholder in file_name:
                # 处理列值中的空格
                value = str(row[col]).replace(' ', '_')
                file_name = file_name.replace(placeholder, value)
        
        return os.path.join(self.graph_docs_dir, file_name)
    
    def get_stats(self) -> Dict[str, int]:
        """
        获取处理统计信息
        
        Returns:
            dict: 包含处理统计的字典
        """
        total = self._processed_docs + self._failed_docs
        success_rate = self._processed_docs / total if total > 0 else 0
        
        return {
            'processed_docs': self._processed_docs,
            'failed_docs': self._failed_docs,
            'total_docs': total,
            'success_rate': success_rate,
            'total_relationships': self._total_relationships,
            'avg_relationships_per_doc': self._total_relationships / max(1, self._processed_docs)
        }
    
    def reset_stats(self):
        """重置统计计数器"""
        self._processed_docs = 0
        self._failed_docs = 0
        self._total_relationships = 0


def create_company_processor(graph_docs_dir: str) -> DataSourceProcessor:
    """
    创建公司网站数据源处理器
    
    Args:
        graph_docs_dir: 图谱文档目录
    
    Returns:
        DataSourceProcessor: 配置好的处理器
    """
    return DataSourceProcessor(
        graph_docs_dir=graph_docs_dir,
        data_source_name="company_website"
    )


def create_vtt_processor(graph_docs_dir: str) -> DataSourceProcessor:
    """
    创建VTT域名数据源处理器
    
    Args:
        graph_docs_dir: 图谱文档目录
    
    Returns:
        DataSourceProcessor: 配置好的处理器
    """
    return DataSourceProcessor(
        graph_docs_dir=graph_docs_dir,
        data_source_name="vtt_website"
    )
