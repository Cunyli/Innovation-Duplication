#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Knowledge Graph Builder Module

Provides structured classes for building consolidated innovation knowledge graphs.
"""

import pandas as pd
from typing import Dict, Set, List
from tqdm import tqdm


class ConsolidatedInnovation:
    """表示一个合并后的创新实体"""
    
    def __init__(self, canonical_id: str):
        """
        初始化合并创新
        
        Args:
            canonical_id: 标准创新ID
        """
        self.id = canonical_id
        self.names: Set[str] = set()
        self.descriptions: Set[str] = set()
        self.developed_by: Set[str] = set()
        self.sources: Set[str] = set()
        self.source_ids: Set[str] = set()
        self.data_sources: Set[str] = set()
    
    def add_innovation_record(self, row: pd.Series, innovation_id: str):
        """
        添加创新记录信息
        
        Args:
            row: DataFrame行
            innovation_id: 原始创新ID
        """
        self.source_ids.add(innovation_id)
        self.names.add(str(row['source_english_id']))
        self.descriptions.add(str(row['source_description']))
        self.sources.add(str(row['Link Source Text']))
        self.data_sources.add(str(row['data_source']))
        
        # 添加开发关系
        if row['relationship_type'] == 'DEVELOPED_BY':
            self.developed_by.add(row['target_id'])
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'id': self.id,
            'names': self.names,
            'descriptions': self.descriptions,
            'developed_by': self.developed_by,
            'sources': self.sources,
            'source_ids': self.source_ids,
            'data_sources': self.data_sources
        }


class InnovationConsolidator:
    """创新合并器 - 负责合并重复的创新"""
    
    def __init__(self, canonical_mapping: Dict[str, str]):
        """
        初始化合并器
        
        Args:
            canonical_mapping: 创新ID到标准ID的映射
        """
        self.canonical_mapping = canonical_mapping
        self.consolidated_innovations: Dict[str, ConsolidatedInnovation] = {}
    
    def consolidate(self, df_relationships: pd.DataFrame) -> Dict[str, ConsolidatedInnovation]:
        """
        合并创新记录
        
        Args:
            df_relationships: 关系数据框
        
        Returns:
            Dict[str, ConsolidatedInnovation]: 合并后的创新字典
        """
        innovation_rows = df_relationships[df_relationships['source_type'] == 'Innovation']
        
        for _, row in tqdm(innovation_rows.iterrows(), 
                          total=len(innovation_rows),
                          desc="Consolidating innovations"):
            self._process_innovation_row(row)
        
        return self.consolidated_innovations
    
    def _process_innovation_row(self, row: pd.Series):
        """
        处理单条创新记录
        
        Args:
            row: DataFrame行
        """
        innovation_id = row['source_id']
        canonical_id = self.canonical_mapping.get(innovation_id, innovation_id)
        
        # 获取或创建合并创新对象
        if canonical_id not in self.consolidated_innovations:
            self.consolidated_innovations[canonical_id] = ConsolidatedInnovation(canonical_id)
        
        # 添加记录信息
        self.consolidated_innovations[canonical_id].add_innovation_record(row, innovation_id)


class OrganizationExtractor:
    """组织提取器 - 负责提取唯一的组织"""
    
    @staticmethod
    def extract(df_relationships: pd.DataFrame) -> Dict[str, Dict]:
        """
        提取所有唯一的组织
        
        Args:
            df_relationships: 关系数据框
        
        Returns:
            Dict[str, Dict]: 组织字典
        """
        organizations = {}
        
        org_rows = df_relationships[df_relationships['target_type'] == 'Organization']
        org_rows = org_rows.drop_duplicates(subset=['target_id'])
        
        for _, row in tqdm(org_rows.iterrows(), 
                          total=len(org_rows),
                          desc="Adding organizations"):
            org_id = row['target_id']
            if org_id not in organizations:
                organizations[org_id] = {
                    'id': org_id,
                    'name': row['target_english_id'],
                    'description': row['target_description']
                }
        
        return organizations


class RelationshipBuilder:
    """关系构建器 - 负责构建知识图谱的关系"""
    
    @staticmethod
    def build_development_relationships(
        consolidated_innovations: Dict[str, ConsolidatedInnovation]
    ) -> List[Dict]:
        """
        构建创新-组织开发关系
        
        Args:
            consolidated_innovations: 合并后的创新字典
        
        Returns:
            List[Dict]: 关系列表
        """
        relationships = []
        
        for canonical_id, innovation in tqdm(consolidated_innovations.items(), 
                                            desc="Adding development relationships"):
            for org_id in innovation.developed_by:
                relationships.append({
                    'source': canonical_id,
                    'target': org_id,
                    'type': 'DEVELOPED_BY'
                })
        
        return relationships
    
    @staticmethod
    def build_collaboration_relationships(df_relationships: pd.DataFrame) -> List[Dict]:
        """
        构建组织间协作关系
        
        Args:
            df_relationships: 关系数据框
        
        Returns:
            List[Dict]: 关系列表
        """
        relationships = []
        
        collab_rows = df_relationships[
            (df_relationships['source_type'] == 'Organization') & 
            (df_relationships['relationship_type'] == 'COLLABORATION')
        ]
        
        for _, row in tqdm(collab_rows.iterrows(), 
                          total=len(collab_rows),
                          desc="Adding collaborations"):
            relationships.append({
                'source': row['source_id'],
                'target': row['target_id'],
                'type': 'COLLABORATION'
            })
        
        return relationships


class KnowledgeGraphBuilder:
    """知识图谱构建器 - 协调整个构建过程"""
    
    def __init__(self, df_relationships: pd.DataFrame, canonical_mapping: Dict[str, str]):
        """
        初始化构建器
        
        Args:
            df_relationships: 关系数据框
            canonical_mapping: 创新ID到标准ID的映射
        """
        self.df_relationships = df_relationships
        self.canonical_mapping = canonical_mapping
    
    def build(self) -> Dict:
        """
        构建完整的知识图谱
        
        Returns:
            Dict: 合并后的知识图谱
        """
        print("Creating innovation knowledge graph...")
        
        # Step 1: 合并创新
        consolidator = InnovationConsolidator(self.canonical_mapping)
        consolidated_innovations = consolidator.consolidate(self.df_relationships)
        
        # Step 2: 提取组织
        organizations = OrganizationExtractor.extract(self.df_relationships)
        
        # Step 3: 构建关系
        development_rels = RelationshipBuilder.build_development_relationships(
            consolidated_innovations
        )
        collaboration_rels = RelationshipBuilder.build_collaboration_relationships(
            self.df_relationships
        )
        
        # Step 4: 组装知识图谱
        consolidated_graph = {
            'innovations': {
                k: v.to_dict() for k, v in consolidated_innovations.items()
            },
            'organizations': organizations,
            'relationships': development_rels + collaboration_rels
        }
        
        # 打印统计信息
        self._print_statistics(consolidated_graph)
        
        return consolidated_graph
    
    @staticmethod
    def _print_statistics(consolidated_graph: Dict):
        """
        打印知识图谱统计信息
        
        Args:
            consolidated_graph: 知识图谱
        """
        print(f"Created knowledge graph with "
              f"{len(consolidated_graph['innovations'])} innovations, " 
              f"{len(consolidated_graph['organizations'])} organizations, and "
              f"{len(consolidated_graph['relationships'])} relationships")


def create_consolidated_knowledge_graph(
    df_relationships: pd.DataFrame, 
    canonical_mapping: Dict[str, str]
) -> Dict:
    """
    创建合并后的知识图谱（便捷函数）
    
    Args:
        df_relationships: 关系数据框
        canonical_mapping: 创新ID到标准ID的映射
    
    Returns:
        Dict: 合并后的知识图谱
    """
    builder = KnowledgeGraphBuilder(df_relationships, canonical_mapping)
    return builder.build()


__all__ = [
    'ConsolidatedInnovation',
    'InnovationConsolidator',
    'OrganizationExtractor',
    'RelationshipBuilder',
    'KnowledgeGraphBuilder',
    'create_consolidated_knowledge_graph'
]
