#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果导出器使用示例

演示如何使用重构后的 ResultExporter 来导出分析结果。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from innovation_platform.data_pipeline.processors import ResultExporter, DataSerializer, export_analysis_results


def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")
    
    # 准备示例数据
    canonical_mapping = {
        'innovation_1': 'canonical_innovation_1',
        'innovation_2': 'canonical_innovation_1',
        'innovation_3': 'canonical_innovation_2'
    }
    
    consolidated_graph = {
        'innovations': {
            'canonical_innovation_1': {
                'id': 'canonical_innovation_1',
                'names': {'AI Assistant', 'Smart AI'},
                'descriptions': {'An intelligent assistant'},
                'developed_by': {'company_a', 'company_b'},
                'sources': {'website_1', 'website_2'},
                'source_ids': {'doc_1', 'doc_2'},
                'data_sources': {'company_website', 'vtt_website'}
            }
        },
        'organizations': {
            'company_a': {
                'id': 'company_a',
                'name': 'Company A',
                'description': 'Tech company'
            }
        },
        'relationships': [
            {'source': 'canonical_innovation_1', 'target': 'company_a', 'type': 'DEVELOPED_BY'}
        ]
    }
    
    analysis_results = {
        'stats': {
            'total': 2,
            'avg_sources': 1.5,
            'avg_developers': 2.0
        },
        'multi_source': {},
        'key_orgs': [('company_a', 0.85)],
        'key_innovations': [('canonical_innovation_1', 0.92)],
        'graph': type('MockGraph', (), {
            'nodes': {'company_a': {'name': 'Company A'}}
        })()
    }
    
    # 方式 1: 使用便捷函数（推荐）
    print("方式 1: 使用便捷函数")
    export_analysis_results(
        analysis_results,
        consolidated_graph,
        canonical_mapping,
        './example_results_1'
    )
    print("✓ 结果已导出到 ./example_results_1\n")
    
    # 方式 2: 使用 ResultExporter 类（更灵活）
    print("方式 2: 使用 ResultExporter 类")
    exporter = ResultExporter('./example_results_2')
    exporter.export_all(analysis_results, consolidated_graph, canonical_mapping)
    print("✓ 结果已导出到 ./example_results_2\n")


def example_custom_serialization():
    """自定义序列化示例"""
    print("=== 自定义序列化示例 ===\n")
    
    # 准备数据
    consolidated_graph = {
        'innovations': {
            'inno_1': {
                'id': 'inno_1',
                'names': {'Innovation X', 'Product X'},
                'descriptions': {'Revolutionary tech'},
                'developed_by': {'org_1'},
                'sources': {'source_1'},
                'source_ids': {'id_1'},
                'data_sources': {'company_website'}
            }
        },
        'organizations': {},
        'relationships': []
    }
    
    # 使用序列化器预处理数据
    serializer = DataSerializer()
    serialized_graph = serializer.serialize_consolidated_graph(consolidated_graph)
    
    print("原始数据 (包含 set):")
    print(f"  names 类型: {type(consolidated_graph['innovations']['inno_1']['names'])}")
    print(f"  names 值: {consolidated_graph['innovations']['inno_1']['names']}\n")
    
    print("序列化后的数据 (转换为 list):")
    print(f"  names 类型: {type(serialized_graph['innovations']['inno_1']['names'])}")
    print(f"  names 值: {serialized_graph['innovations']['inno_1']['names']}\n")


def example_selective_export():
    """选择性导出示例"""
    print("=== 选择性导出示例 ===\n")
    
    # 准备数据
    canonical_mapping = {'inno_1': 'canonical_1'}
    analysis_results = {
        'stats': {'total': 1},
        'multi_source': {},
        'key_orgs': [],
        'key_innovations': [],
        'graph': type('MockGraph', (), {'nodes': {}})()
    }
    
    # 创建导出器
    exporter = ResultExporter('./example_results_3')
    
    # 只导出特定的结果
    print("只导出规范映射和统计数据...")
    exporter.export_canonical_mapping(canonical_mapping)
    exporter.export_innovation_stats(analysis_results)
    
    print("✓ 选择性导出完成\n")


def main():
    """运行所有示例"""
    print("=" * 60)
    print("结果导出器使用示例")
    print("=" * 60)
    print()
    
    try:
        example_basic_usage()
        example_custom_serialization()
        example_selective_export()
        
        print("=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
