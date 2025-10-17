#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Validation Module

This module provides validation functions for entities and relationships
extracted from graph documents.
"""

from typing import List


def is_valid_entity_name(name: str) -> bool:
    """
    检查实体名称是否有效，过滤掉明显无效的实体。
    
    Args:
        name: 实体名称
        
    Returns:
        bool: 是否是有效的实体名称
    
    Examples:
        >>> is_valid_entity_name("Innovation Platform")
        True
        >>> is_valid_entity_name("ab")
        False
        >>> is_valid_entity_name("123")
        False
        >>> is_valid_entity_name("null")
        False
    """
    if not name or not isinstance(name, str):
        return False
    
    # 过滤掉太短的名称
    if len(name.strip()) < 3:
        return False
    
    # 过滤掉只包含数字或特殊字符的名称
    if all(not c.isalpha() for c in name):
        return False
    
    # 过滤掉常见的占位符名称
    invalid_patterns = [
        'null', 'none', 'undefined', 'n/a', 'unknown', 
        'temp_', 'unknown', 'placeholder', 'example'
    ]
    
    name_lower = name.lower()
    for pattern in invalid_patterns:
        if pattern in name_lower:
            # 特殊处理：如果是以temp_开头但后面有有意义的内容，仍然保留
            if pattern == 'temp_' and len(name) > 10 and any(c.isalpha() for c in name[5:]):
                continue
            return False
    
    return True


def is_valid_relationship(innovation: str, organization: str, relation_type: str) -> bool:
    """
    检查关系是否有效，过滤掉明显无效的关系。
    
    Args:
        innovation: 创新名称
        organization: 组织名称
        relation_type: 关系类型
        
    Returns:
        bool: 是否是有效的关系
    
    Examples:
        >>> is_valid_relationship("AI Platform", "Tech Corp", "DEVELOPED_BY")
        True
        >>> is_valid_relationship("AI", "ab", "DEVELOPED_BY")
        False
        >>> is_valid_relationship("AI Platform", "Tech Corp", "INVALID")
        False
    """
    # 检查创新和组织名称是否有效
    if not is_valid_entity_name(innovation) or not is_valid_entity_name(organization):
        return False
    
    # 检查关系类型是否有效
    if relation_type not in ["DEVELOPED_BY", "COLLABORATION"]:
        return False
    
    # 过滤掉创新和组织相同的情况
    if innovation.lower() == organization.lower():
        return False
    
    return True


__all__ = [
    'is_valid_entity_name',
    'is_valid_relationship',
]
