#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试集成后的 load_and_combine_data 函数

验证新的 GraphDocumentLoader 和 NodeMapper 集成是否工作正常
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from innovation_resolution import load_and_combine_data


def test_load_and_combine_data():
    """测试数据加载和合并功能"""
    print("="*60)
    print("测试 load_and_combine_data() 集成")
    print("="*60)
    
    try:
        print("\n🚀 开始加载数据...")
        print("⚠️  注意: 这可能需要几分钟时间\n")
        
        # 调用函数
        df_relationships, all_pred_entities, all_pred_relations = load_and_combine_data()
        
        print("\n" + "="*60)
        print("✅ 数据加载成功!")
        print("="*60)
        
        # 显示统计信息
        print(f"\n📊 数据统计:")
        print(f"  - 关系记录数: {len(df_relationships):,}")
        print(f"  - 预测实体数: {len(all_pred_entities):,}")
        print(f"  - 预测关系数: {len(all_pred_relations):,}")
        
        # 显示DataFrame信息
        print(f"\n📋 关系DataFrame信息:")
        print(f"  - 列数: {len(df_relationships.columns)}")
        print(f"  - 数据源分布:")
        if 'data_source' in df_relationships.columns:
            print(df_relationships['data_source'].value_counts().to_string(header=False))
        
        # 显示实体类型分布
        if all_pred_entities:
            print(f"\n🏷️  实体类型分布:")
            entity_types = {}
            for entity in all_pred_entities:
                entity_type = entity.get('type', 'Unknown')
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {entity_type}: {count:,}")
        
        # 显示关系类型分布
        if all_pred_relations:
            print(f"\n🔗 关系类型分布:")
            relation_types = {}
            for relation in all_pred_relations:
                relation_type = relation.get('relation', 'Unknown')
                relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
            for relation_type, count in sorted(relation_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {relation_type}: {count:,}")
        
        # 显示样例数据
        print(f"\n📝 关系记录样例（前3条）:")
        sample_cols = ['source_english_id', 'relationship_type', 'target_english_id', 'data_source']
        available_cols = [col for col in sample_cols if col in df_relationships.columns]
        if available_cols:
            print(df_relationships[available_cols].head(3).to_string(index=False))
        
        print("\n" + "="*60)
        print("✅ 所有测试通过!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🧪 Phase 1.2 集成测试".center(60, "="))
    print()
    
    success = test_load_and_combine_data()
    
    if success:
        print("\n🎉 集成成功! 新模块工作正常。\n")
        return 0
    else:
        print("\n💥 集成失败，请检查错误信息。\n")
        return 1


if __name__ == "__main__":
    exit(main())
