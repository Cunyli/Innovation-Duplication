#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
结果导出器测试模块

测试 ResultExporter 类及其相关组件的功能。
"""

import os
import json
import tempfile
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from innovation_platform.data_pipeline.processors.result_exporter import (
    DataSerializer,
    FileWriter,
    ResultExporter,
    export_analysis_results
)


class TestDataSerializer:
    """测试数据序列化器"""
    
    def test_serialize_canonical_mapping(self):
        """测试规范映射序列化"""
        mapping = {'inno1': 'canonical1', 'inno2': 'canonical1'}
        result = DataSerializer.serialize_canonical_mapping(mapping)
        assert result == mapping
        assert isinstance(result, dict)
    
    def test_serialize_consolidated_graph_with_sets(self):
        """测试图谱序列化（包含 set 对象）"""
        graph = {
            'innovations': {
                'inno1': {
                    'id': 'inno1',
                    'names': {'Name A', 'Name B'},
                    'descriptions': {'Desc 1'},
                    'developed_by': {'org1'},
                    'sources': {'source1'},
                    'source_ids': {'id1'},
                    'data_sources': {'company_website'}
                }
            },
            'organizations': {
                'org1': {'id': 'org1', 'name': 'Org A', 'description': 'Test org'}
            },
            'relationships': [
                {'source': 'inno1', 'target': 'org1', 'type': 'DEVELOPED_BY'}
            ]
        }
        
        result = DataSerializer.serialize_consolidated_graph(graph)
        
        # 验证 set 已转换为 list
        assert isinstance(result['innovations']['inno1']['names'], list)
        assert isinstance(result['innovations']['inno1']['descriptions'], list)
        assert len(result['innovations']['inno1']['names']) == 2
        
    def test_serialize_innovation_stats(self):
        """测试统计数据序列化"""
        analysis_results = {
            'stats': {
                'total': 100,
                'avg_sources': 2.5,
                'avg_developers': 1.8
            }
        }
        
        result = DataSerializer.serialize_innovation_stats(analysis_results)
        assert result == analysis_results['stats']
        assert result['total'] == 100
    
    def test_serialize_multi_source_innovations(self):
        """测试多源创新序列化"""
        analysis_results = {
            'multi_source': {
                'inno1': {
                    'names': {'Name A'},
                    'descriptions': {'Desc'},
                    'developed_by': {'org1'},
                    'sources': {'source1'},
                    'source_ids': {'id1'},
                    'data_sources': {'vtt_website'}
                }
            }
        }
        
        result = DataSerializer.serialize_multi_source_innovations(analysis_results)
        
        # 验证 set 已转换为 list
        assert isinstance(result['inno1']['names'], list)
        assert isinstance(result['inno1']['sources'], list)


class TestFileWriter:
    """测试文件写入器"""
    
    def test_write_json(self):
        """测试 JSON 文件写入"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            data = {'key': 'value', 'number': 42}
            FileWriter.write_json(data, temp_path)
            
            # 验证文件已创建并包含正确数据
            assert os.path.exists(temp_path)
            
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)
            
            assert loaded_data == data
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestResultExporter:
    """测试结果导出器"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录用于测试"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """准备测试数据"""
        return {
            'canonical_mapping': {'inno1': 'canonical1'},
            'consolidated_graph': {
                'innovations': {
                    'canonical1': {
                        'id': 'canonical1',
                        'names': {'Name A'},
                        'descriptions': {'Desc'},
                        'developed_by': {'org1'},
                        'sources': {'source1'},
                        'source_ids': {'id1'},
                        'data_sources': {'company_website'}
                    }
                },
                'organizations': {
                    'org1': {'id': 'org1', 'name': 'Org A', 'description': 'Test'}
                },
                'relationships': []
            },
            'analysis_results': {
                'stats': {'total': 1},
                'multi_source': {},
                'key_orgs': [('org1', 0.5)],
                'key_innovations': [('canonical1', 0.8)],
                'graph': type('Graph', (), {
                    'nodes': {'org1': {'name': 'Org A'}}
                })()
            }
        }
    
    def test_exporter_initialization(self, temp_dir):
        """测试导出器初始化"""
        exporter = ResultExporter(temp_dir)
        assert exporter.output_dir == temp_dir
        assert os.path.exists(temp_dir)
    
    def test_export_canonical_mapping(self, temp_dir, sample_data):
        """测试导出规范映射"""
        exporter = ResultExporter(temp_dir)
        exporter.export_canonical_mapping(sample_data['canonical_mapping'])
        
        filepath = os.path.join(temp_dir, 'canonical_mapping.json')
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        assert data == sample_data['canonical_mapping']
    
    def test_export_consolidated_graph(self, temp_dir, sample_data):
        """测试导出合并图谱"""
        exporter = ResultExporter(temp_dir)
        exporter.export_consolidated_graph(sample_data['consolidated_graph'])
        
        filepath = os.path.join(temp_dir, 'consolidated_graph.json')
        assert os.path.exists(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 验证数据结构
        assert 'innovations' in data
        assert 'organizations' in data
        assert 'relationships' in data
    
    def test_export_all(self, temp_dir, sample_data):
        """测试导出所有结果"""
        exporter = ResultExporter(temp_dir)
        exporter.export_all(
            sample_data['analysis_results'],
            sample_data['consolidated_graph'],
            sample_data['canonical_mapping']
        )
        
        # 验证所有文件都已创建
        expected_files = [
            'canonical_mapping.json',
            'consolidated_graph.json',
            'innovation_stats.json',
            'multi_source_innovations.json',
            'key_nodes.json'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(temp_dir, filename)
            assert os.path.exists(filepath), f"File {filename} was not created"


class TestExportAnalysisResults:
    """测试便捷函数"""
    
    def test_export_analysis_results(self):
        """测试顶层导出函数"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 准备测试数据
            canonical_mapping = {'inno1': 'canonical1'}
            consolidated_graph = {
                'innovations': {
                    'canonical1': {
                        'id': 'canonical1',
                        'names': {'Name A'},
                        'descriptions': set(),
                        'developed_by': set(),
                        'sources': set(),
                        'source_ids': set(),
                        'data_sources': set()
                    }
                },
                'organizations': {},
                'relationships': []
            }
            analysis_results = {
                'stats': {'total': 1},
                'multi_source': {},
                'key_orgs': [],
                'key_innovations': [],
                'graph': type('Graph', (), {'nodes': {}})()
            }
            
            # 执行导出
            export_analysis_results(
                analysis_results,
                consolidated_graph,
                canonical_mapping,
                temp_dir
            )
            
            # 验证文件已创建
            assert os.path.exists(os.path.join(temp_dir, 'canonical_mapping.json'))
            assert os.path.exists(os.path.join(temp_dir, 'consolidated_graph.json'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
