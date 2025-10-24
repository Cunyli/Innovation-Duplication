#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据结构检查工具

这个工具可以帮助你检查和理解 analysis_results 的结构。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import Dict, Any
import json


def print_analysis_results_structure(analysis_results: Dict) -> None:
    """
    打印 analysis_results 的详细结构
    
    Args:
        analysis_results: 网络分析结果
    """
    print("=" * 70)
    print("📊 ANALYSIS_RESULTS 结构检查")
    print("=" * 70)
    print()
    
    # 1. 顶层键
    print("🔑 顶层键:")
    for key in analysis_results.keys():
        print(f"  ✓ {key}")
    print()
    
    # 2. Graph 信息
    if 'graph' in analysis_results:
        graph = analysis_results['graph']
        print("📈 graph (NetworkX 图对象):")
        print(f"  - 类型: {type(graph).__name__}")
        print(f"  - 节点数: {graph.number_of_nodes()}")
        print(f"  - 边数: {graph.number_of_edges()}")
        
        # 节点类型统计
        innovation_count = sum(1 for n in graph.nodes() if graph.nodes[n].get('type') == 'Innovation')
        org_count = sum(1 for n in graph.nodes() if graph.nodes[n].get('type') == 'Organization')
        print(f"  - 创新节点: {innovation_count}")
        print(f"  - 组织节点: {org_count}")
        
        # 示例节点
        if graph.number_of_nodes() > 0:
            sample_node = list(graph.nodes())[0]
            print(f"  - 示例节点属性: {dict(graph.nodes[sample_node])}")
        print()
    
    # 3. Stats 信息
    if 'stats' in analysis_results:
        stats = analysis_results['stats']
        print("📊 stats (统计指标):")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.2f}")
            else:
                print(f"  - {key}: {value}")
        print()
    
    # 4. Multi-source 信息
    if 'multi_source' in analysis_results:
        multi_source = analysis_results['multi_source']
        print("🔍 multi_source (多源验证创新):")
        print(f"  - 数量: {len(multi_source)}")
        
        if multi_source:
            # 示例创新
            sample_id = list(multi_source.keys())[0]
            sample_data = multi_source[sample_id]
            print(f"  - 示例 ID: {sample_id}")
            print(f"  - 示例数据结构:")
            for key, value in sample_data.items():
                value_type = type(value).__name__
                value_len = len(value) if hasattr(value, '__len__') else 'N/A'
                print(f"    • {key}: {value_type} (长度: {value_len})")
                if key == 'names' and value:
                    print(f"      例: {list(value)[:2]}")
        print()
    
    # 5. Top orgs 信息
    if 'top_orgs' in analysis_results:
        top_orgs = analysis_results['top_orgs']
        print("🏢 top_orgs (最活跃组织):")
        print(f"  - 数量: {len(top_orgs)}")
        if top_orgs:
            print(f"  - 类型: List[Tuple[str, int]]")
            print(f"  - Top 3:")
            for i, (org_id, count) in enumerate(top_orgs[:3], 1):
                print(f"    {i}. {org_id}: {count} 个创新")
        print()
    
    # 6. Key orgs 信息
    if 'key_orgs' in analysis_results:
        key_orgs = analysis_results['key_orgs']
        print("⭐ key_orgs (关键组织 - 介数中心性):")
        print(f"  - 数量: {len(key_orgs)}")
        if key_orgs:
            print(f"  - 类型: List[Tuple[str, float]]")
            print(f"  - Top 3:")
            for i, (org_id, centrality) in enumerate(key_orgs[:3], 1):
                print(f"    {i}. {org_id}: {centrality:.4f}")
        else:
            print(f"  ⚠️  为空（可能中心性计算失败）")
        print()
    
    # 7. Key innovations 信息
    if 'key_innovations' in analysis_results:
        key_innovations = analysis_results['key_innovations']
        print("🚀 key_innovations (关键创新 - 特征向量中心性):")
        print(f"  - 数量: {len(key_innovations)}")
        if key_innovations:
            print(f"  - 类型: List[Tuple[str, float]]")
            print(f"  - Top 3:")
            for i, (inno_id, centrality) in enumerate(key_innovations[:3], 1):
                print(f"    {i}. {inno_id}: {centrality:.4f}")
        else:
            print(f"  ⚠️  为空（可能中心性计算失败）")
        print()
    
    print("=" * 70)
    print("✅ 结构检查完成")
    print("=" * 70)


def create_sample_analysis_results() -> Dict:
    """
    创建一个示例 analysis_results 用于演示
    
    Returns:
        Dict: 示例 analysis_results
    """
    import networkx as nx
    
    # 创建示例图
    G = nx.Graph()
    
    # 添加节点
    G.add_node('innovation_1', type='Innovation', names='AI Assistant', sources=2, developed_by=1)
    G.add_node('innovation_2', type='Innovation', names='Smart Robot', sources=1, developed_by=2)
    G.add_node('org_1', type='Organization', name='Tech Company A')
    G.add_node('org_2', type='Organization', name='Research Lab B')
    
    # 添加边
    G.add_edge('innovation_1', 'org_1', type='DEVELOPED_BY')
    G.add_edge('innovation_2', 'org_1', type='DEVELOPED_BY')
    G.add_edge('innovation_2', 'org_2', type='DEVELOPED_BY')
    
    # 创建 analysis_results
    analysis_results = {
        'graph': G,
        'stats': {
            'total': 2,
            'avg_sources': 1.5,
            'avg_developers': 1.5,
            'multi_source_count': 1,
            'multi_developer_count': 1
        },
        'multi_source': {
            'innovation_1': {
                'names': {'AI Assistant', 'Smart AI'},
                'descriptions': {'An intelligent assistant'},
                'developed_by': {'org_1'},
                'sources': {'https://example.com/1', 'https://example.com/2'},
                'source_ids': {'doc_1', 'doc_2'},
                'data_sources': {'company_website', 'vtt_website'}
            }
        },
        'top_orgs': [
            ('org_1', 2),
            ('org_2', 1)
        ],
        'key_orgs': [
            ('org_1', 0.3333),
            ('org_2', 0.1667)
        ],
        'key_innovations': [
            ('innovation_1', 0.7071),
            ('innovation_2', 0.7071)
        ]
    }
    
    return analysis_results


def check_serialization_issues(analysis_results: Dict) -> None:
    """
    检查序列化可能的问题
    
    Args:
        analysis_results: 网络分析结果
    """
    print("\n🔍 序列化检查:")
    print("-" * 70)
    
    issues = []
    
    # 检查 multi_source 中的 set
    if 'multi_source' in analysis_results:
        for inno_id, data in analysis_results['multi_source'].items():
            for key, value in data.items():
                if isinstance(value, set):
                    issues.append(f"multi_source[{inno_id}][{key}] 是 set 类型")
    
    if issues:
        print("⚠️  发现需要转换的字段:")
        for issue in issues[:5]:  # 只显示前5个
            print(f"  - {issue}")
        if len(issues) > 5:
            print(f"  ... 还有 {len(issues) - 5} 个")
        print("\n💡 建议: 使用 ResultExporter 导出，它会自动处理 set → list 转换")
    else:
        print("✅ 没有发现序列化问题")
    
    print("-" * 70)


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("数据结构检查工具")
    print("=" * 70)
    print()
    
    # 创建示例数据
    print("📝 创建示例 analysis_results...")
    analysis_results = create_sample_analysis_results()
    print("✓ 示例数据创建完成\n")
    
    # 检查结构
    print_analysis_results_structure(analysis_results)
    
    # 检查序列化问题
    check_serialization_issues(analysis_results)
    
    print("\n💡 提示:")
    print("  - 这是一个示例数据，真实数据会更复杂")
    print("  - 使用 export_analysis_results() 导出时会自动处理序列化")
    print("  - 查看 docs/DATA_STRUCTURES.md 了解完整文档")
    print()


if __name__ == '__main__':
    main()
