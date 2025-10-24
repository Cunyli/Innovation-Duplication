#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果导出器模块

提供结构化的方式来导出创新分析结果到各种格式。
采用策略模式和单一职责原则，使代码易于测试和扩展。
"""

import os
import json
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class DataSerializer:
    """
    数据序列化器
    
    负责将内存中的Python对象转换为可序列化的格式。
    主要处理 set → list 的转换以支持 JSON 序列化。
    """
    
    @staticmethod
    def _convert_sets_to_lists(data: Any) -> Any:
        """
        递归地将数据结构中的所有 set 转换为 list
        
        Args:
            data: 任意数据结构
            
        Returns:
            转换后的数据（set → list）
        """
        if isinstance(data, set):
            return list(data)
        elif isinstance(data, dict):
            return {k: DataSerializer._convert_sets_to_lists(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [DataSerializer._convert_sets_to_lists(item) for item in data]
        else:
            return data
    
    @staticmethod
    def serialize_canonical_mapping(canonical_mapping: Dict[str, str]) -> Dict[str, str]:
        """
        序列化规范映射
        
        Args:
            canonical_mapping: 创新ID到规范ID的映射
            
        Returns:
            可序列化的映射字典
        """
        return {k: v for k, v in canonical_mapping.items()}
    
    @staticmethod
    def serialize_consolidated_graph(consolidated_graph: Dict) -> Dict:
        """
        序列化合并后的知识图谱
        
        将图谱中的 set 对象转换为 list 以支持 JSON 序列化
        
        Args:
            consolidated_graph: 合并后的知识图谱
            
        Returns:
            可序列化的图谱字典
        """
        graph_for_json = {
            'innovations': {},
            'organizations': consolidated_graph['organizations'],
            'relationships': consolidated_graph['relationships']
        }
        
        # 转换创新数据中的 set 为 list
        for inno_id, inno_data in consolidated_graph['innovations'].items():
            graph_for_json['innovations'][inno_id] = {
                'id': inno_data['id'],
                'names': list(inno_data['names']) if isinstance(inno_data['names'], set) else inno_data['names'],
                'descriptions': list(inno_data['descriptions']) if isinstance(inno_data['descriptions'], set) else inno_data['descriptions'],
                'developed_by': list(inno_data['developed_by']) if isinstance(inno_data['developed_by'], set) else inno_data['developed_by'],
                'sources': list(inno_data['sources']) if isinstance(inno_data['sources'], set) else inno_data['sources'],
                'source_ids': list(inno_data['source_ids']) if isinstance(inno_data['source_ids'], set) else inno_data['source_ids'],
                'data_sources': list(inno_data['data_sources']) if isinstance(inno_data['data_sources'], set) else inno_data['data_sources']
            }
        
        return graph_for_json
    
    @staticmethod
    def serialize_innovation_stats(analysis_results: Dict) -> Dict:
        """
        序列化创新统计数据
        
        Args:
            analysis_results: 网络分析结果
            
        Returns:
            可序列化的统计数据
        """
        return analysis_results['stats']
    
    @staticmethod
    def serialize_multi_source_innovations(analysis_results: Dict) -> Dict:
        """
        序列化多源验证的创新数据
        
        Args:
            analysis_results: 网络分析结果
            
        Returns:
            可序列化的多源创新数据
        """
        multi_source_for_json = {}
        for inno_id, inno_data in analysis_results['multi_source'].items():
            multi_source_for_json[inno_id] = {
                'names': list(inno_data['names']) if isinstance(inno_data['names'], set) else inno_data['names'],
                'descriptions': list(inno_data['descriptions']) if isinstance(inno_data['descriptions'], set) else inno_data['descriptions'],
                'developed_by': list(inno_data['developed_by']) if isinstance(inno_data['developed_by'], set) else inno_data['developed_by'],
                'sources': list(inno_data['sources']) if isinstance(inno_data['sources'], set) else inno_data['sources'],
                'source_ids': list(inno_data['source_ids']) if isinstance(inno_data['source_ids'], set) else inno_data['source_ids'],
                'data_sources': list(inno_data['data_sources']) if isinstance(inno_data['data_sources'], set) else inno_data['data_sources']
            }
        return multi_source_for_json
    
    @staticmethod
    def serialize_key_nodes(analysis_results: Dict, consolidated_graph: Dict) -> Dict:
        """
        序列化关键节点数据（组织和创新）
        
        Args:
            analysis_results: 网络分析结果
            consolidated_graph: 合并后的知识图谱
            
        Returns:
            可序列化的关键节点数据
        """
        key_nodes = {
            'key_organizations': [
                {
                    'id': org_id,
                    'centrality': centrality,
                    'name': analysis_results['graph'].nodes[org_id].get('name', org_id)
                }
                for org_id, centrality in analysis_results['key_orgs']
            ],
            'key_innovations': [
                {
                    'id': inno_id,
                    'centrality': centrality,
                    'names': list(consolidated_graph['innovations'][inno_id]['names'])
                }
                for inno_id, centrality in analysis_results['key_innovations']
                if inno_id in consolidated_graph['innovations']
            ]
        }
        return key_nodes


class FileWriter:
    """
    文件写入器
    
    负责将序列化后的数据写入文件。
    支持多种文件格式的写入策略。
    """
    
    @staticmethod
    def write_json(
        data: Dict,
        filepath: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """
        将数据写入 JSON 文件
        
        Args:
            data: 要写入的数据
            filepath: 文件路径
            indent: JSON 缩进空格数
            ensure_ascii: 是否确保 ASCII 编码
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


class ResultExporter:
    """
    结果导出器
    
    协调数据序列化和文件写入操作，提供统一的接口来导出分析结果。
    """
    
    def __init__(
        self,
        output_dir: str,
        serializer: Optional[DataSerializer] = None,
        writer: Optional[FileWriter] = None
    ):
        """
        初始化结果导出器
        
        Args:
            output_dir: 输出目录路径
            serializer: 数据序列化器实例（可选）
            writer: 文件写入器实例（可选）
        """
        self.output_dir = output_dir
        self.serializer = serializer or DataSerializer()
        self.writer = writer or FileWriter()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_canonical_mapping(self, canonical_mapping: Dict[str, str]) -> None:
        """
        导出规范映射
        
        Args:
            canonical_mapping: 创新ID到规范ID的映射
        """
        data = self.serializer.serialize_canonical_mapping(canonical_mapping)
        filepath = os.path.join(self.output_dir, 'canonical_mapping.json')
        self.writer.write_json(data, filepath)
    
    def export_consolidated_graph(self, consolidated_graph: Dict) -> None:
        """
        导出合并后的知识图谱
        
        Args:
            consolidated_graph: 合并后的知识图谱
        """
        data = self.serializer.serialize_consolidated_graph(consolidated_graph)
        filepath = os.path.join(self.output_dir, 'consolidated_graph.json')
        self.writer.write_json(data, filepath)
    
    def export_innovation_stats(self, analysis_results: Dict) -> None:
        """
        导出创新统计数据
        
        Args:
            analysis_results: 网络分析结果
        """
        data = self.serializer.serialize_innovation_stats(analysis_results)
        filepath = os.path.join(self.output_dir, 'innovation_stats.json')
        self.writer.write_json(data, filepath)
    
    def export_multi_source_innovations(self, analysis_results: Dict) -> None:
        """
        导出多源验证的创新数据
        
        Args:
            analysis_results: 网络分析结果
        """
        data = self.serializer.serialize_multi_source_innovations(analysis_results)
        filepath = os.path.join(self.output_dir, 'multi_source_innovations.json')
        self.writer.write_json(data, filepath)
    
    def export_key_nodes(
        self,
        analysis_results: Dict,
        consolidated_graph: Dict
    ) -> None:
        """
        导出关键节点数据
        
        Args:
            analysis_results: 网络分析结果
            consolidated_graph: 合并后的知识图谱
        """
        data = self.serializer.serialize_key_nodes(analysis_results, consolidated_graph)
        filepath = os.path.join(self.output_dir, 'key_nodes.json')
        self.writer.write_json(data, filepath)
    
    def export_all(
        self,
        analysis_results: Dict,
        consolidated_graph: Dict,
        canonical_mapping: Dict[str, str]
    ) -> None:
        """
        导出所有分析结果
        
        这是一个便捷方法，一次性导出所有结果文件。
        
        Args:
            analysis_results: 网络分析结果
            consolidated_graph: 合并后的知识图谱
            canonical_mapping: 创新ID到规范ID的映射
        """
        print("Exporting results...")
        
        # 导出各个文件
        self.export_canonical_mapping(canonical_mapping)
        self.export_consolidated_graph(consolidated_graph)
        self.export_innovation_stats(analysis_results)
        self.export_multi_source_innovations(analysis_results)
        self.export_key_nodes(analysis_results, consolidated_graph)
        
        print(f"Results exported to {self.output_dir}")


def export_analysis_results(
    analysis_results: Dict,
    consolidated_graph: Dict,
    canonical_mapping: Dict[str, str],
    output_dir: str
) -> None:
    """
    导出分析结果的便捷函数
    
    这是一个顶层函数，提供简单的接口来导出所有结果。
    内部使用 ResultExporter 类来完成实际工作。
    
    Args:
        analysis_results: 网络分析结果
        consolidated_graph: 合并后的知识图谱
        canonical_mapping: 创新ID到规范ID的映射
        output_dir: 输出目录路径
    """
    exporter = ResultExporter(output_dir)
    exporter.export_all(analysis_results, consolidated_graph, canonical_mapping)
