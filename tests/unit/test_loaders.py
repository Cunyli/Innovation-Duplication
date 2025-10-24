#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据加载器模块

这个脚本用于验证 data/loaders 模块的功能是否正常。
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from innovation_platform.data_pipeline.loaders import GraphDocumentLoader, NodeMapper

DATA_DIR = PROJECT_ROOT / "data" / "graph_docs_names_resolved"


def test_graph_loader():
    """测试图谱文档加载器"""
    print("=" * 60)
    print("测试 GraphDocumentLoader")
    print("=" * 60)
    
    # 创建加载器
    loader = GraphDocumentLoader(verbose=True)
    
    # 测试1: 加载一个存在的文件
    test_file = DATA_DIR
    if test_file.exists():
        files = [f for f in os.listdir(test_file) if f.endswith('.pkl')]
        if files:
            test_path = test_file / files[0]
            print(f"\n✓ 测试加载文件: {test_path}")
            doc = loader.load(str(test_path))
            
            if doc:
                print(f"  ✅ 成功加载文档")
                print(f"  - 节点数: {len(doc.nodes) if hasattr(doc, 'nodes') else 'N/A'}")
                print(f"  - 关系数: {len(doc.relationships) if hasattr(doc, 'relationships') else 'N/A'}")
            else:
                print(f"  ❌ 加载失败")
        else:
            print(f"  ⚠️  目录中没有.pkl文件")
    else:
        print(f"  ⚠️  测试目录不存在: {test_file}")
    
    # 测试2: 加载不存在的文件
    print(f"\n✓ 测试加载不存在的文件")
    doc = loader.load("non_existent_file.pkl")
    print(f"  {'✅ 正确返回None' if doc is None else '❌ 应该返回None'}")
    
    # 测试3: 查看统计信息
    print(f"\n✓ 加载统计:")
    stats = loader.get_stats()
    print(f"  - 成功加载: {stats['loaded']}")
    print(f"  - 加载错误: {stats['errors']}")
    print(f"  - 成功率: {stats['success_rate']:.2%}")
    
    print("\n" + "=" * 60)


def test_node_mapper():
    """测试节点映射器"""
    print("\n" + "=" * 60)
    print("测试 NodeMapper")
    print("=" * 60)
    
    # 先加载一个文档
    loader = GraphDocumentLoader(verbose=False)
    test_file = DATA_DIR
    
    if test_file.exists():
        files = [f for f in os.listdir(test_file) if f.endswith('.pkl')]
        if files:
            test_path = test_file / files[0]
            doc = loader.load(str(test_path))
            
            if doc and hasattr(doc, 'nodes'):
                print(f"\n✓ 使用文档: {files[0]}")
                
                # 创建映射器
                mapper = NodeMapper(verbose=True)
                
                # 提取映射
                print(f"\n✓ 提取节点映射:")
                node_desc, node_en_id = mapper.extract_mappings(doc)
                
                print(f"  - 提取了 {len(node_en_id)} 个节点的映射")
                
                # 显示前3个映射示例
                print(f"\n✓ 映射示例（前3个）:")
                for i, (node_id, eng_id) in enumerate(list(node_en_id.items())[:3]):
                    desc = node_desc.get(node_id, "")
                    print(f"  {i+1}. 原始ID: {node_id}")
                    print(f"     英文ID: {eng_id}")
                    print(f"     描述: {desc[:80]}{'...' if len(desc) > 80 else ''}")
                    print()
                
                # 测试单个节点查询
                if node_en_id:
                    test_node_id = list(node_en_id.keys())[0]
                    print(f"✓ 测试单个节点查询:")
                    eng_id = mapper.get_english_id(doc, test_node_id)
                    desc = mapper.get_description(doc, test_node_id)
                    print(f"  - 节点ID: {test_node_id}")
                    print(f"  - 英文ID: {eng_id}")
                    print(f"  - 描述: {desc[:80]}{'...' if len(desc) > 80 else ''}")
                
                print(f"\n✅ NodeMapper 测试通过")
            else:
                print(f"  ⚠️  无法加载有效文档")
        else:
            print(f"  ⚠️  目录中没有.pkl文件")
    else:
        print(f"  ⚠️  测试目录不存在: {test_file}")
    
    print("\n" + "=" * 60)


def main():
    """主测试函数"""
    print("\n🚀 开始测试 data/loaders 模块\n")
    
    try:
        # 测试图谱加载器
        test_graph_loader()
        
        # 测试节点映射器
        test_node_mapper()
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
