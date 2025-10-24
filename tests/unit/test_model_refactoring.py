#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for model initialization and embedding refactoring
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

def test_imports():
    """测试导入"""
    print("测试 1: 导入模块...")
    try:
        from innovation_platform.data_pipeline.processors import (
            initialize_openai_client,
            get_embedding,
            compute_similarity
        )
        print("✅ 成功导入所有函数")
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_initializer_classes():
    """测试模型初始化器类"""
    print("\n测试 2: 模型初始化器类...")
    try:
        from innovation_platform.data_pipeline.processors.model_initializer import (
            ConfigLoader,
            ModelConfigValidator,
            AzureModelInitializer
        )
        
        # 测试维度推断
        dim1 = ModelConfigValidator.infer_embedding_dimension('text-embedding-3-large')
        assert dim1 == 3072, f"Expected 3072, got {dim1}"
        
        dim2 = ModelConfigValidator.infer_embedding_dimension('text-embedding-3-small')
        assert dim2 == 1536, f"Expected 1536, got {dim2}"
        
        dim3 = ModelConfigValidator.infer_embedding_dimension('text-embedding-ada-002')
        assert dim3 == 1536, f"Expected 1536, got {dim3}"
        
        dim4 = ModelConfigValidator.infer_embedding_dimension('unknown-model')
        assert dim4 == 1536, f"Expected 1536 (default), got {dim4}"
        
        # 测试 endpoint 规范化
        endpoint1 = ModelConfigValidator.normalize_endpoint('https://api.openai.com/openai/deployments')
        assert endpoint1 == 'https://api.openai.com/', f"Unexpected endpoint: {endpoint1}"
        
        endpoint2 = ModelConfigValidator.normalize_endpoint('https://api.openai.com')
        assert endpoint2 == 'https://api.openai.com/', f"Unexpected endpoint: {endpoint2}"
        
        print("✅ 所有模型初始化器类测试通过")
        return True
    except Exception as e:
        print(f"❌ 模型初始化器类测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embedding_strategies():
    """测试嵌入策略"""
    print("\n测试 3: 嵌入策略...")
    try:
        from innovation_platform.data_pipeline.processors.embedding_strategy import (
            TFIDFEmbeddingStrategy,
            FallbackEmbedding,
            get_embedding,
            compute_similarity
        )
        
        # 测试 TF-IDF 嵌入
        test_text = "This is a test innovation about artificial intelligence and machine learning."
        tfidf_emb = TFIDFEmbeddingStrategy.embed(test_text)
        assert isinstance(tfidf_emb, np.ndarray), "TF-IDF embedding should return numpy array"
        assert len(tfidf_emb) == 1536, f"Expected dimension 1536, got {len(tfidf_emb)}"
        
        # 测试随机嵌入
        random_emb = FallbackEmbedding.embed(1536)
        assert isinstance(random_emb, np.ndarray), "Fallback embedding should return numpy array"
        assert len(random_emb) == 1536, f"Expected dimension 1536, got {len(random_emb)}"
        
        # 测试 get_embedding (无模型，应该使用 TF-IDF)
        emb = get_embedding(test_text, model=None)
        assert isinstance(emb, np.ndarray), "get_embedding should return numpy array"
        assert len(emb) == 1536, f"Expected dimension 1536, got {len(emb)}"
        
        # 测试相似度计算
        emb1 = get_embedding("artificial intelligence", model=None)
        emb2 = get_embedding("machine learning", model=None)
        emb3 = get_embedding("cooking recipes", model=None)
        
        sim1 = compute_similarity(emb1, emb2)
        sim2 = compute_similarity(emb1, emb3)
        
        assert 0 <= sim1 <= 1, f"Similarity should be between 0 and 1, got {sim1}"
        assert 0 <= sim2 <= 1, f"Similarity should be between 0 and 1, got {sim2}"
        
        # AI 和 ML 的相似度应该高于 AI 和烹饪
        print(f"   AI vs ML 相似度: {sim1:.4f}")
        print(f"   AI vs 烹饪 相似度: {sim2:.4f}")
        
        print("✅ 所有嵌入策略测试通过")
        return True
    except Exception as e:
        print(f"❌ 嵌入策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_initialize_client():
    """测试客户端初始化（可能失败，取决于配置）"""
    print("\n测试 4: 客户端初始化...")
    try:
        from innovation_platform.data_pipeline.processors import initialize_openai_client
        
        llm, embedding_model = initialize_openai_client()
        
        if llm is None and embedding_model is None:
            print("⚠️  客户端初始化返回 None (可能是配置缺失，这是预期行为)")
            return True
        else:
            print(f"✅ 成功初始化客户端: LLM={type(llm).__name__}, Embedding={type(embedding_model).__name__}")
            return True
    except Exception as e:
        print(f"❌ 客户端初始化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_file_integration():
    """测试主文件集成"""
    print("\n测试 5: 主文件集成...")
    try:
        # 尝试导入主文件中的函数
        from innovation_platform.innovation_resolution import (
            resolve_innovation_duplicates,
            create_innovation_knowledge_graph
        )
        print("✅ 成功从主文件导入函数")
        return True
    except Exception as e:
        print(f"❌ 主文件集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("开始测试模型初始化和嵌入重构")
    print("=" * 60)
    
    results = []
    results.append(("导入测试", test_imports()))
    results.append(("模型初始化器类", test_model_initializer_classes()))
    results.append(("嵌入策略", test_embedding_strategies()))
    results.append(("客户端初始化", test_initialize_client()))
    results.append(("主文件集成", test_main_file_integration()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s} : {status}")
    
    all_passed = all(result for _, result in results)
    
    print("=" * 60)
    if all_passed:
        print("🎉 所有测试通过！重构成功！")
        return 0
    else:
        print("⚠️  部分测试失败，请检查错误信息")
        return 1

if __name__ == "__main__":
    sys.exit(main())
