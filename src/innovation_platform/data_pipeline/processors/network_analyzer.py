#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Analysis Module

This module provides structured components for analyzing innovation networks,
including statistics calculation, centrality analysis, and key node identification.
"""

from typing import Dict, List, Tuple, Optional
import networkx as nx


class NetworkGraphBuilder:
    """构建 NetworkX 图对象"""
    
    @staticmethod
    def build(consolidated_graph: Dict) -> nx.Graph:
        """
        从合并的知识图谱构建 NetworkX 图
        
        Args:
            consolidated_graph: 合并后的知识图谱
            
        Returns:
            nx.Graph: NetworkX 图对象
        """
        G = nx.Graph()
        
        # 添加创新节点
        for innovation_id, innovation in consolidated_graph['innovations'].items():
            G.add_node(
                innovation_id,
                type='Innovation',
                names=', '.join(innovation['names']),
                sources=len(innovation['sources']),
                developed_by=len(innovation['developed_by'])
            )
        
        # 添加组织节点
        for org_id, org in consolidated_graph['organizations'].items():
            G.add_node(
                org_id,
                type='Organization',
                name=org['name']
            )
        
        # 添加边
        for rel in consolidated_graph['relationships']:
            G.add_edge(rel['source'], rel['target'], type=rel['type'])
        
        return G


class InnovationStatisticsCalculator:
    """计算创新网络的基础统计指标"""
    
    @staticmethod
    def calculate(consolidated_graph: Dict) -> Dict:
        """
        计算创新网络的基础统计指标
        
        Args:
            consolidated_graph: 合并后的知识图谱
            
        Returns:
            Dict: 统计指标字典
        """
        innovations = consolidated_graph['innovations']
        
        if not innovations:
            return {
                'total': 0,
                'avg_sources': 0,
                'avg_developers': 0,
                'multi_source_count': 0,
                'multi_developer_count': 0
            }
        
        total_innovations = len(innovations)
        
        # 计算平均数据源数
        total_sources = sum(len(i['sources']) for i in innovations.values())
        avg_sources = total_sources / total_innovations
        
        # 计算平均开发者数
        total_developers = sum(len(i['developed_by']) for i in innovations.values())
        avg_developers = total_developers / total_innovations
        
        # 计算多源创新数量
        multi_source_count = sum(
            1 for i in innovations.values() if len(i['sources']) > 1
        )
        
        # 计算多开发者创新数量
        multi_developer_count = sum(
            1 for i in innovations.values() if len(i['developed_by']) > 1
        )
        
        return {
            'total': total_innovations,
            'avg_sources': avg_sources,
            'avg_developers': avg_developers,
            'multi_source_count': multi_source_count,
            'multi_developer_count': multi_developer_count
        }


class MultiSourceInnovationExtractor:
    """提取多数据源验证的创新"""
    
    @staticmethod
    def extract(consolidated_graph: Dict) -> Dict:
        """
        提取在多个数据源中都出现的创新
        
        Args:
            consolidated_graph: 合并后的知识图谱
            
        Returns:
            Dict: 多源创新字典
        """
        return {
            innovation_id: innovation_data
            for innovation_id, innovation_data in consolidated_graph['innovations'].items()
            if len(innovation_data['sources']) > 1
        }


class OrganizationRanker:
    """组织排序器 - 按创新数量排序"""
    
    @staticmethod
    def rank_by_innovation_count(
        consolidated_graph: Dict,
        top_n: int = 10
    ) -> List[Tuple[str, int]]:
        """
        按创新数量对组织排序
        
        Args:
            consolidated_graph: 合并后的知识图谱
            top_n: 返回前 N 个组织
            
        Returns:
            List[Tuple[str, int]]: (组织ID, 创新数量) 列表
        """
        org_innovation_counts = {}
        
        # 统计每个组织的创新数量
        for rel in consolidated_graph['relationships']:
            if (rel['type'] == 'DEVELOPED_BY' and 
                rel['target'] in consolidated_graph['organizations']):
                org_id = rel['target']
                org_innovation_counts[org_id] = org_innovation_counts.get(org_id, 0) + 1
        
        # 排序并返回 Top N
        sorted_orgs = sorted(
            org_innovation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_orgs[:top_n]


class CentralityAnalyzer:
    """中心性分析器 - 分析网络中的关键节点"""
    
    def __init__(self, graph: nx.Graph):
        """
        初始化中心性分析器
        
        Args:
            graph: NetworkX 图对象
        """
        self.graph = graph
        self.betweenness_centrality = None
        self.eigenvector_centrality = None
    
    def compute_centralities(self, max_iter: int = 1000) -> bool:
        """
        计算中心性指标
        
        Args:
            max_iter: 特征向量中心性的最大迭代次数
            
        Returns:
            bool: 是否计算成功
        """
        try:
            # 计算介数中心性（衡量节点的中介作用）
            self.betweenness_centrality = nx.betweenness_centrality(self.graph)
            
            # 计算特征向量中心性（衡量节点连接的质量）
            self.eigenvector_centrality = nx.eigenvector_centrality(
                self.graph,
                max_iter=max_iter
            )
            
            return True
        except Exception as e:
            print(f"Warning: Centrality computation failed: {e}")
            return False
    
    def get_key_organizations(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取关键组织（基于介数中心性）
        
        Args:
            top_n: 返回前 N 个组织
            
        Returns:
            List[Tuple[str, float]]: (组织ID, 中心性得分) 列表
        """
        if self.betweenness_centrality is None:
            return []
        
        # 筛选组织节点
        org_centralities = [
            (node, self.betweenness_centrality[node])
            for node in self.graph.nodes
            if self.graph.nodes[node].get('type') == 'Organization'
        ]
        
        # 排序并返回 Top N
        sorted_orgs = sorted(
            org_centralities,
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_orgs[:top_n]
    
    def get_key_innovations(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        获取关键创新（基于特征向量中心性）
        
        Args:
            top_n: 返回前 N 个创新
            
        Returns:
            List[Tuple[str, float]]: (创新ID, 中心性得分) 列表
        """
        if self.eigenvector_centrality is None:
            return []
        
        # 筛选创新节点
        innovation_centralities = [
            (node, self.eigenvector_centrality[node])
            for node in self.graph.nodes
            if self.graph.nodes[node].get('type') == 'Innovation'
        ]
        
        # 排序并返回 Top N
        sorted_innovations = sorted(
            innovation_centralities,
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_innovations[:top_n]


class InnovationNetworkAnalyzer:
    """创新网络分析器 - 协调所有分析组件"""
    
    def __init__(self, consolidated_graph: Dict):
        """
        初始化网络分析器
        
        Args:
            consolidated_graph: 合并后的知识图谱
        """
        self.consolidated_graph = consolidated_graph
        self.graph = None
        self.centrality_analyzer = None
    
    def analyze(
        self,
        top_n: int = 10,
        max_iter: int = 1000
    ) -> Dict:
        """
        执行完整的网络分析
        
        Args:
            top_n: 返回前 N 个关键节点
            max_iter: 中心性计算的最大迭代次数
            
        Returns:
            Dict: 分析结果字典
        """
        print("Analyzing innovation network...")
        
        # Step 1: 构建 NetworkX 图
        self.graph = NetworkGraphBuilder.build(self.consolidated_graph)
        
        # Step 2: 计算基础统计指标
        stats = InnovationStatisticsCalculator.calculate(self.consolidated_graph)
        
        # Step 3: 提取多源创新
        multi_source = MultiSourceInnovationExtractor.extract(self.consolidated_graph)
        
        # Step 4: 按创新数量排序组织
        top_orgs = OrganizationRanker.rank_by_innovation_count(
            self.consolidated_graph,
            top_n=top_n
        )
        
        # Step 5: 中心性分析
        self.centrality_analyzer = CentralityAnalyzer(self.graph)
        centrality_success = self.centrality_analyzer.compute_centralities(max_iter=max_iter)
        
        if centrality_success:
            key_orgs = self.centrality_analyzer.get_key_organizations(top_n=top_n)
            key_innovations = self.centrality_analyzer.get_key_innovations(top_n=top_n)
        else:
            key_orgs = []
            key_innovations = []
        
        # 返回分析结果
        return {
            'graph': self.graph,
            'stats': stats,
            'multi_source': multi_source,
            'top_orgs': top_orgs,
            'key_orgs': key_orgs,
            'key_innovations': key_innovations
        }
    
    def print_summary(self, analysis_results: Dict):
        """
        打印分析结果摘要
        
        Args:
            analysis_results: 分析结果字典
        """
        stats = analysis_results['stats']
        
        print("\n" + "="*60)
        print("创新网络分析摘要")
        print("="*60)
        
        print(f"\n📊 基础统计:")
        print(f"  - 创新总数: {stats['total']}")
        print(f"  - 平均数据源数: {stats['avg_sources']:.2f}")
        print(f"  - 平均开发者数: {stats['avg_developers']:.2f}")
        print(f"  - 多源验证创新: {stats['multi_source_count']}")
        print(f"  - 协作创新: {stats['multi_developer_count']}")
        
        print(f"\n🏢 Top 5 最活跃组织:")
        for i, (org_id, count) in enumerate(analysis_results['top_orgs'][:5], 1):
            org_name = self.consolidated_graph['organizations'].get(org_id, {}).get('name', org_id)
            print(f"  {i}. {org_name}: {count} 个创新")
        
        if analysis_results['key_orgs']:
            print(f"\n⭐ Top 5 关键组织 (介数中心性):")
            for i, (org_id, centrality) in enumerate(analysis_results['key_orgs'][:5], 1):
                org_name = self.consolidated_graph['organizations'].get(org_id, {}).get('name', org_id)
                print(f"  {i}. {org_name}: {centrality:.4f}")
        
        if analysis_results['key_innovations']:
            print(f"\n🚀 Top 5 关键创新 (特征向量中心性):")
            for i, (inno_id, centrality) in enumerate(analysis_results['key_innovations'][:5], 1):
                inno_names = list(self.consolidated_graph['innovations'][inno_id]['names'])
                inno_name = inno_names[0] if inno_names else inno_id
                print(f"  {i}. {inno_name}: {centrality:.4f}")
        
        print("\n" + "="*60)


def analyze_innovation_network(
    consolidated_graph: Dict,
    top_n: int = 10,
    max_iter: int = 1000,
    print_summary: bool = True
) -> Dict:
    """
    分析创新网络的便捷函数
    
    Args:
        consolidated_graph: 合并后的知识图谱
        top_n: 返回前 N 个关键节点
        max_iter: 中心性计算的最大迭代次数
        print_summary: 是否打印分析摘要
        
    Returns:
        Dict: 分析结果字典，包含以下字段:
            - graph: NetworkX 图对象
            - stats: 基础统计指标
            - multi_source: 多源验证的创新
            - top_orgs: Top N 最活跃组织
            - key_orgs: Top N 关键组织（介数中心性）
            - key_innovations: Top N 关键创新（特征向量中心性）
    """
    analyzer = InnovationNetworkAnalyzer(consolidated_graph)
    results = analyzer.analyze(top_n=top_n, max_iter=max_iter)
    
    if print_summary:
        analyzer.print_summary(results)
    
    return results
