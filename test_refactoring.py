#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
验证重构是否成功的测试脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("🔍 开始验证重构...")
print("=" * 60)

# 测试1: 验证导入
print("\n[测试1] 验证模块导入...")
try:
    from data.processors import (
        is_valid_entity_name,
        is_valid_relationship,
        extract_entities_from_document,
        extract_relationships_from_document
    )
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 验证 validators 功能
print("\n[测试2] 验证 validators 功能...")
test_cases_validator = [
    ("Innovation Platform", True, "有效的实体名称"),
    ("ab", False, "太短的名称"),
    ("123", False, "只包含数字"),
    ("null", False, "占位符名称"),
    ("AI Tech Solution", True, "有效的复合名称"),
    ("", False, "空字符串"),
    (None, False, "None值"),
]

validator_passed = 0
validator_failed = 0

for name, expected, desc in test_cases_validator:
    result = is_valid_entity_name(name)
    if result == expected:
        print(f"  ✓ {desc}: '{name}' -> {result}")
        validator_passed += 1
    else:
        print(f"  ✗ {desc}: '{name}' -> {result} (期望: {expected})")
        validator_failed += 1

print(f"\n  验证器测试: {validator_passed} 通过, {validator_failed} 失败")

# 测试3: 验证关系验证器
print("\n[测试3] 验证关系验证器...")
test_cases_relationship = [
    ("AI Platform", "Tech Corp", "DEVELOPED_BY", True, "有效关系"),
    ("AI", "ab", "DEVELOPED_BY", False, "名称过短"),
    ("AI Platform", "Tech Corp", "INVALID", False, "无效关系类型"),
    ("Tech Corp", "Tech Corp", "DEVELOPED_BY", False, "相同名称"),
]

rel_passed = 0
rel_failed = 0

for innovation, org, rel_type, expected, desc in test_cases_relationship:
    result = is_valid_relationship(innovation, org, rel_type)
    if result == expected:
        print(f"  ✓ {desc}: {result}")
        rel_passed += 1
    else:
        print(f"  ✗ {desc}: {result} (期望: {expected})")
        rel_failed += 1

print(f"\n  关系验证器测试: {rel_passed} 通过, {rel_failed} 失败")

# 测试4: 验证提取器功能
print("\n[测试4] 验证提取器功能...")

class MockNode:
    def __init__(self, id, type, properties):
        self.id = id
        self.type = type
        self.properties = properties

class MockRel:
    def __init__(self, source, target, type, source_type, target_type):
        self.source = source
        self.target = target
        self.type = type
        self.source_type = source_type
        self.target_type = target_type
        self.properties = {}

class MockDoc:
    def __init__(self, nodes, relationships=None):
        self.nodes = nodes
        self.relationships = relationships or []

# 测试实体提取
doc = MockDoc([
    MockNode("i1", "Innovation", {"english_id": "Innovation Platform", "description": "Test"}),
    MockNode("o1", "Organization", {"english_id": "Tech Corporation"}),
    MockNode("bad", "Innovation", {"english_id": "ab"})  # 应该被过滤
])

entities = extract_entities_from_document(doc)
expected_entity_count = 2  # 只有两个有效实体

if len(entities) == expected_entity_count:
    print(f"  ✓ 实体提取成功: 提取了 {len(entities)} 个有效实体")
    for entity in entities:
        print(f"    - {entity['name']} ({entity['type']})")
else:
    print(f"  ✗ 实体提取失败: 提取了 {len(entities)} 个实体 (期望: {expected_entity_count})")

# 测试关系提取
doc_with_rels = MockDoc(
    nodes=[
        MockNode("i1", "Innovation", {"english_id": "Innovation Platform"}),
        MockNode("o1", "Organization", {"english_id": "Tech Corporation"})
    ],
    relationships=[
        MockRel("i1", "o1", "DEVELOPED_BY", "Innovation", "Organization")
    ]
)

relationships = extract_relationships_from_document(doc_with_rels)
expected_rel_count = 1

if len(relationships) == expected_rel_count:
    print(f"  ✓ 关系提取成功: 提取了 {len(relationships)} 个关系")
    for rel in relationships:
        print(f"    - {rel['innovation']} -> {rel['organization']} ({rel['relation']})")
else:
    print(f"  ✗ 关系提取失败: 提取了 {len(relationships)} 个关系 (期望: {expected_rel_count})")

# 测试5: 验证主文件导入
print("\n[测试5] 验证主文件可以正常导入...")
try:
    # 只验证导入，不执行 main
    import innovation_resolution
    print("✅ 主文件导入成功")
    
    # 验证函数是否可访问
    assert hasattr(innovation_resolution, 'load_and_combine_data')
    assert hasattr(innovation_resolution, 'resolve_innovation_duplicates')
    assert hasattr(innovation_resolution, 'create_innovation_knowledge_graph')
    print("✅ 主要函数都可访问")
    
except Exception as e:
    print(f"❌ 主文件导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 总结
print("\n" + "=" * 60)
print("📊 测试总结")
print("=" * 60)

total_tests = 5
passed_tests = 0

if validator_failed == 0:
    passed_tests += 1
if rel_failed == 0:
    passed_tests += 1
if len(entities) == expected_entity_count:
    passed_tests += 1
if len(relationships) == expected_rel_count:
    passed_tests += 1
# 主文件导入已经在上面检查过，如果到这里说明通过了
passed_tests += 1

print(f"\n✅ 通过: {passed_tests}/{total_tests} 个测试")

if passed_tests == total_tests:
    print("\n🎉 所有测试通过！重构成功！")
    sys.exit(0)
else:
    print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败")
    sys.exit(1)
