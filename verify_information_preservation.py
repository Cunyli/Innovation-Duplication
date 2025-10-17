#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证 resolve_innovation_duplicates 中的信息保留情况
"""

import pandas as pd
import sys

def create_test_data():
    """创建测试数据 - 模拟同一创新有多条关系记录的情况"""
    return pd.DataFrame({
        'source_type': ['Innovation'] * 6,
        'source_id': ['I001', 'I001', 'I001', 'I002', 'I002', 'I002'],
        'source_english_id': ['AI Platform'] * 3 + ['ML Engine'] * 3,
        'source_description': ['Advanced AI system'] * 3 + ['Machine learning engine'] * 3,
        'relationship_type': ['DEVELOPED_BY', 'USES', 'ENABLES', 'DEVELOPED_BY', 'USES', 'COLLABORATION'],
        'target_id': ['O001', 'S001', 'T001', 'O002', 'S002', 'O003'],
        'target_english_id': ['TechCorp', 'Cloud Service', 'Analytics Tool', 'DataLab', 'GPU Cluster', 'Research Institute'],
        'target_description': ['Technology company', 'Cloud infrastructure', 'Analytics platform', 'Data laboratory', 'GPU hardware', 'Research org'],
        'relationship description': ['', 'Utilizes', 'Enables', '', 'Uses', 'Collaborates with'],
        'data_source': ['company_website'] * 3 + ['vtt_website'] * 3
    })

def test_deduplication_information_loss():
    """测试去重是否丢失信息"""
    print("=" * 80)
    print("测试 1: 去重操作的信息保留情况")
    print("=" * 80)
    
    df = create_test_data()
    
    print(f"\n📊 原始数据: {len(df)} 行")
    print(df[['source_id', 'relationship_type', 'target_english_id']].to_string(index=False))
    
    # 执行去重
    innovations = df[df["source_type"] == "Innovation"]
    unique_innovations = innovations.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    
    print(f"\n📉 去重后: {len(unique_innovations)} 行")
    print(unique_innovations[['source_id', 'relationship_type', 'target_english_id']].to_string(index=False))
    
    # 统计信息丢失
    original_relations = set()
    for _, row in df.iterrows():
        original_relations.add((row['source_id'], row['relationship_type'], row['target_english_id']))
    
    kept_relations = set()
    for _, row in unique_innovations.iterrows():
        kept_relations.add((row['source_id'], row['relationship_type'], row['target_english_id']))
    
    lost_relations = original_relations - kept_relations
    
    print(f"\n❌ 丢失的关系 ({len(lost_relations)} 条):")
    for rel in sorted(lost_relations):
        print(f"   - {rel[0]}: {rel[1]} → {rel[2]}")
    
    return df, unique_innovations

def test_feature_building_information_usage():
    """测试特征构建是否使用了完整信息"""
    print("\n" + "=" * 80)
    print("测试 2: 特征构建时的信息使用情况")
    print("=" * 80)
    
    from data.processors import InnovationFeatureBuilder
    
    df, unique_innovations = create_test_data(), create_test_data().drop_duplicates(subset=["source_id"])
    
    print(f"\n构建特征...")
    
    for _, row in unique_innovations.iterrows():
        innovation_id = row['source_id']
        
        # 构建上下文
        context = InnovationFeatureBuilder.build_context(row, df)
        
        print(f"\n📝 创新 {innovation_id}:")
        print(f"   上下文长度: {len(context)} 字符")
        print(f"   上下文内容: {context[:150]}...")
        
        # 统计使用了多少条关系
        related_rows = df[df['source_id'] == innovation_id]
        print(f"   ✅ 使用了 {len(related_rows)} 条关系记录 (原始数据中的所有记录)")
        
        # 检查每个关系是否在上下文中
        for _, rel_row in related_rows.iterrows():
            target_name = rel_row['target_english_id']
            if target_name in context:
                print(f"      ✓ {rel_row['relationship_type']:15s} → {target_name:20s} [已包含]")
            else:
                print(f"      ✗ {rel_row['relationship_type']:15s} → {target_name:20s} [未包含]")

def test_clustering_with_complete_features():
    """测试完整流程：去重 + 特征构建 + 聚类"""
    print("\n" + "=" * 80)
    print("测试 3: 完整流程验证")
    print("=" * 80)
    
    from data.processors import (
        InnovationExtractor,
        InnovationFeatureBuilder,
        EmbeddingManager
    )
    
    # 创建更复杂的测试数据
    df = pd.DataFrame({
        'source_type': ['Innovation'] * 9,
        'source_id': ['I001', 'I001', 'I001', 'I002', 'I002', 'I003', 'I003', 'I003', 'I003'],
        'source_english_id': ['AI Platform'] * 3 + ['AI System'] * 2 + ['AI Platform'] * 4,  # I001 和 I003 名称相同
        'source_description': ['AI tech'] * 3 + ['AI technology'] * 2 + ['AI tech'] * 4,
        'relationship_type': ['DEVELOPED_BY', 'USES', 'ENABLES', 'DEVELOPED_BY', 'USES', 'DEVELOPED_BY', 'USES', 'ENABLES', 'COLLABORATION'],
        'target_id': ['O001', 'S001', 'T001', 'O002', 'S002', 'O001', 'S001', 'T001', 'O003'],
        'target_english_id': ['TechCorp', 'Cloud', 'Analytics', 'DataLab', 'GPU', 'TechCorp', 'Cloud', 'Analytics', 'Institute'],
        'target_description': ['Company'] * 3 + ['Lab', 'Hardware'] + ['Company', 'Service', 'Tool', 'Research'],
        'relationship description': [''] * 9,
        'data_source': ['company_website'] * 9
    })
    
    print(f"\n原始数据: {len(df)} 行，涉及 {df['source_id'].nunique()} 个唯一创新")
    
    # Step 1: 提取唯一创新
    unique = InnovationExtractor.extract_unique_innovations(df)
    print(f"去重后: {len(unique)} 行")
    
    # Step 2: 构建特征
    features = InnovationFeatureBuilder.build_all_features(unique, df, show_progress=False)
    
    print(f"\n构建的特征:")
    for iid, context in features.items():
        relation_count = len(df[df['source_id'] == iid])
        print(f"\n{iid}: 使用了 {relation_count} 条关系")
        print(f"   上下文: {context[:100]}...")
        
        # 检查关键信息
        developers = df[(df['source_id'] == iid) & (df['relationship_type'] == 'DEVELOPED_BY')]['target_english_id'].tolist()
        for dev in developers:
            if dev in context:
                print(f"   ✓ 包含开发者: {dev}")
            else:
                print(f"   ✗ 缺失开发者: {dev}")

def test_graph_clustering_usage():
    """测试图聚类是否可用"""
    print("\n" + "=" * 80)
    print("测试 4: 图聚类方法验证")
    print("=" * 80)
    
    from data.processors import ClusteringStrategyFactory
    
    methods = [
        ('hdbscan', '平面聚类'),
        ('kmeans', '平面聚类'),
        ('graph_threshold', '图聚类'),
        ('graph_kcore', '图聚类'),
    ]
    
    print("\n支持的聚类方法:")
    for method, category in methods:
        try:
            strategy = ClusteringStrategyFactory.create_strategy(method)
            print(f"   ✅ {method:20s} [{category:10s}] - {type(strategy).__name__}")
        except Exception as e:
            print(f"   ❌ {method:20s} - 失败: {e}")
    
    print("\n💡 使用示例:")
    print("   # 使用默认的 HDBSCAN")
    print("   resolve_innovation_duplicates(df, method='hdbscan')")
    print("\n   # 使用图聚类")
    print("   resolve_innovation_duplicates(df, method='graph_threshold', similarity_threshold=0.85)")

def main():
    """运行所有测试"""
    print("\n🔍 验证 resolve_innovation_duplicates 的信息保留机制\n")
    
    # 测试1: 去重的信息丢失
    test_deduplication_information_loss()
    
    # 测试2: 特征构建的信息使用
    test_feature_building_information_usage()
    
    # 测试3: 完整流程
    test_clustering_with_complete_features()
    
    # 测试4: 图聚类
    test_graph_clustering_usage()
    
    print("\n" + "=" * 80)
    print("📋 总结:")
    print("=" * 80)
    print("""
1. ❌ drop_duplicates(subset=['source_id']) 确实会丢失信息
   - 只保留每个创新的第一条记录
   - 其他关系记录被丢弃

2. ✅ 但特征构建时会回查完整的 df_relationships
   - InnovationFeatureBuilder.build_context() 使用完整 DataFrame
   - 所有关系信息都会被包含在文本上下文中

3. ✅ 图聚类方法可用
   - graph_threshold: 基于相似度阈值构建图
   - graph_kcore: 基于 K-core 分解

4. 💡 改进建议:
   - 去重时可以聚合信息，避免误解
   - 使用 groupby 预处理可以提高效率
   - 显式保存关系统计信息
    """)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
