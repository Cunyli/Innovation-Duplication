#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Extraction Module

This module provides functions for extracting entities and relationships
from graph documents.
"""

from typing import List, Dict, Optional
from .validators import is_valid_entity_name, is_valid_relationship


def extract_entities_from_document(doc, pred_entities: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract innovation and organization entities from document.
    Also accumulate extracted entities in the pred_entities list if provided.
    
    Args:
        doc: Source document with nodes attribute
        pred_entities: Optional list to accumulate extracted entities
        
    Returns:
        List of entity dictionaries in the format {"name": str, "type": str}
    
    Examples:
        >>> class MockNode:
        ...     def __init__(self, id, type, properties):
        ...         self.id = id
        ...         self.type = type
        ...         self.properties = properties
        >>> class MockDoc:
        ...     def __init__(self, nodes):
        ...         self.nodes = nodes
        >>> doc = MockDoc([MockNode("1", "Innovation", {"english_id": "AI Platform"})])
        >>> entities = extract_entities_from_document(doc)
        >>> len(entities)
        1
        >>> entities[0]["name"]
        'AI Platform'
    """
    entities = []
    
    # Extract entities from the document (assuming graph_doc format)
    if hasattr(doc, 'nodes'):
        for node in doc.nodes:
            # 获取实体名称，优先使用english_id
            entity_name = node.properties.get('english_id', node.id) if hasattr(node, 'properties') else node.id
            
            # 检查实体名称是否有效
            if not is_valid_entity_name(entity_name):
                continue
                
            entity = {
                "name": entity_name,
                "type": node.type
            }
            
            # 添加描述信息，便于后续过滤和评估
            if hasattr(node, 'properties') and 'description' in node.properties:
                entity["description"] = node.properties['description']
            
            entities.append(entity)
            
            # Accumulate to prediction list if provided
            if pred_entities is not None:
                pred_entities.append(entity)
    
    return entities


def extract_relationships_from_document(doc, pred_relations: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract relationships from document.
    Also accumulate extracted relationships in the pred_relations list if provided.
    
    Args:
        doc: Source document with relationships and nodes attributes
        pred_relations: Optional list to accumulate extracted relationships
        
    Returns:
        List of relationship dictionaries in the format 
        {"innovation": str, "organization": str, "relation": str}
    
    Examples:
        >>> class MockNode:
        ...     def __init__(self, id, properties):
        ...         self.id = id
        ...         self.properties = properties
        >>> class MockRel:
        ...     def __init__(self, source, target, type, source_type, target_type):
        ...         self.source = source
        ...         self.target = target
        ...         self.type = type
        ...         self.source_type = source_type
        ...         self.target_type = target_type
        ...         self.properties = {}
        >>> class MockDoc:
        ...     def __init__(self, nodes, relationships):
        ...         self.nodes = nodes
        ...         self.relationships = relationships
        >>> nodes = [MockNode("i1", {"english_id": "Innovation A"}),
        ...          MockNode("o1", {"english_id": "Org B"})]
        >>> rels = [MockRel("i1", "o1", "DEVELOPED_BY", "Innovation", "Organization")]
        >>> doc = MockDoc(nodes, rels)
        >>> relationships = extract_relationships_from_document(doc)
        >>> len(relationships)
        1
    """
    relationships = []
    
    # Extract relationships from the document (assuming graph_doc format)
    if hasattr(doc, 'relationships'):
        # 首先获取节点的english_id映射
        node_english_id = {}
        if hasattr(doc, 'nodes'):
            for node in doc.nodes:
                if hasattr(node, 'properties') and 'english_id' in node.properties:
                    node_english_id[node.id] = node.properties['english_id']
                else:
                    node_english_id[node.id] = node.id
        
        for rel in doc.relationships:
            # 只包含DEVELOPED_BY和COLLABORATION关系
            if rel.type in ["DEVELOPED_BY", "COLLABORATION"]:
                # 获取源和目标的名称，优先使用english_id
                source_name = node_english_id.get(rel.source, rel.source)
                target_name = node_english_id.get(rel.target, rel.target)
                
                # 确保source/target是正确的创新/组织映射
                if rel.source_type == "Innovation" and rel.target_type == "Organization":
                    innovation_name = source_name
                    organization_name = target_name
                elif rel.source_type == "Organization" and rel.target_type == "Innovation":
                    innovation_name = target_name
                    organization_name = source_name
                else:
                    # 如果关系不是创新-组织之间的关系，跳过
                    continue
                
                # 检查关系是否有效
                if not is_valid_relationship(innovation_name, organization_name, rel.type):
                    continue
                
                relationship = {
                    "innovation": innovation_name,
                    "organization": organization_name,
                    "relation": rel.type
                }
                
                # 添加描述信息，便于后续过滤和评估
                if hasattr(rel, 'properties') and 'description' in rel.properties:
                    relationship["description"] = rel.properties['description']
                
                relationships.append(relationship)
                
                # Accumulate to prediction list if provided
                if pred_relations is not None:
                    pred_relations.append(relationship)
    
    return relationships


__all__ = [
    'extract_entities_from_document',
    'extract_relationships_from_document',
]
