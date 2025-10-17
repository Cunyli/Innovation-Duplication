#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Initializer Module

Handles initialization of OpenAI/Azure models and configuration loading.
"""

import os
from typing import Tuple, Optional, Dict, Any
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


class ConfigLoader:
    """配置加载器，支持多种配置源"""
    
    @staticmethod
    def load_from_streamlit() -> Tuple[bool, Optional[Dict]]:
        """
        尝试从 Streamlit secrets 加载配置
        
        Returns:
            Tuple[bool, Optional[Dict]]: (是否成功, 配置字典)
        """
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and st.secrets and 'AZURE_OPENAI_API_KEY' in st.secrets:
                print("✅ 使用 Streamlit secrets 配置")
                return True, st.secrets
        except ImportError:
            pass
        return False, None
    
    @staticmethod
    def load_from_env_or_json() -> Dict:
        """
        从 .env 文件或 azure_config.json 加载配置
        
        Returns:
            Dict: 配置字典
        """
        from config.config_loader import load_config
        print("🔍 尝试从 .env 文件或 azure_config.json 加载配置...")
        return load_config(prefer_env=True)
    
    @staticmethod
    def extract_model_config(config: Dict, use_streamlit: bool) -> Dict[str, Any]:
        """
        从配置中提取模型配置
        
        Args:
            config: 原始配置字典
            use_streamlit: 是否使用 Streamlit 配置
        
        Returns:
            Dict: 模型配置
        """
        if use_streamlit:
            # Streamlit secrets 结构
            model_name = config.get('default_model', {}).get('name', 'gpt-4o-mini')
            if model_name in config:
                return config[model_name]
            else:
                # 直接使用根级配置
                return {
                    'api_key': config.get('AZURE_OPENAI_API_KEY'),
                    'api_base': config.get('AZURE_OPENAI_ENDPOINT'),
                    'api_version': config.get('AZURE_OPENAI_API_VERSION'),
                    'deployment': config.get('AZURE_OPENAI_DEPLOYMENT'),
                    'emb_deployment': config.get('AZURE_OPENAI_EMBEDDING_DEPLOYMENT')
                }
        else:
            # 从 .env 或 JSON 加载的配置结构
            model_name = 'gpt-4o-mini'
            return config.get(model_name, {})


class ModelConfigValidator:
    """模型配置验证器"""
    
    @staticmethod
    def validate(model_config: Dict) -> bool:
        """
        验证模型配置是否有效
        
        Args:
            model_config: 模型配置字典
        
        Returns:
            bool: 配置是否有效
        """
        if not model_config or not model_config.get('api_key'):
            print("❌ 错误: 未找到有效的 Azure OpenAI 配置")
            print("请确保以下任一配置方式:")
            print("  1. 在 .env 文件中设置 AZURE_OPENAI_API_KEY 等变量")
            print("  2. 在 data/keys/azure_config.json 中配置")
            print("  3. 使用 Streamlit secrets (仅限Web应用)")
            return False
        return True
    
    @staticmethod
    def normalize_endpoint(api_base: str) -> str:
        """
        规范化 API endpoint
        
        Args:
            api_base: 原始 API base URL
        
        Returns:
            str: 规范化后的 endpoint
        """
        base_endpoint = api_base.split('/openai')[0] if '/openai' in api_base else api_base
        if not base_endpoint.endswith('/'):
            base_endpoint += '/'
        return base_endpoint
    
    @staticmethod
    def infer_embedding_dimension(emb_deployment: str) -> int:
        """
        根据嵌入模型名称推断向量维度
        
        Args:
            emb_deployment: 嵌入模型部署名称
        
        Returns:
            int: 向量维度
        """
        if 'text-embedding-3-large' in emb_deployment:
            return 3072
        elif 'text-embedding-3-small' in emb_deployment:
            return 1536
        elif 'text-embedding-ada-002' in emb_deployment:
            return 1536
        else:
            # 默认使用 1536（最常见的维度）
            print(f"⚠️ 未识别的嵌入模型 '{emb_deployment}'，使用默认维度 1536")
            return 1536


class AzureModelInitializer:
    """Azure 模型初始化器"""
    
    @staticmethod
    def initialize_llm(model_config: Dict, base_endpoint: str) -> AzureChatOpenAI:
        """
        初始化 Azure Chat LLM
        
        Args:
            model_config: 模型配置
            base_endpoint: API endpoint
        
        Returns:
            AzureChatOpenAI: 初始化的 LLM
        """
        return AzureChatOpenAI(
            api_key=model_config.get('api_key'),
            azure_endpoint=base_endpoint,
            azure_deployment=model_config.get('deployment'),
            api_version=model_config.get('api_version'),
            temperature=0
        )
    
    @staticmethod
    def initialize_embedding_model(
        model_config: Dict, 
        base_endpoint: str, 
        dimension: int
    ) -> AzureOpenAIEmbeddings:
        """
        初始化 Azure 嵌入模型
        
        Args:
            model_config: 模型配置
            base_endpoint: API endpoint
            dimension: 嵌入维度
        
        Returns:
            AzureOpenAIEmbeddings: 初始化的嵌入模型
        """
        emb_api_version = model_config.get('emb_api_version', model_config.get('api_version'))
        return AzureOpenAIEmbeddings(
            api_key=model_config.get('api_key'),
            azure_endpoint=base_endpoint,
            azure_deployment=model_config.get('emb_deployment'),
            api_version=emb_api_version,
            dimensions=dimension
        )


def initialize_openai_client() -> Tuple[Optional[AzureChatOpenAI], Optional[AzureOpenAIEmbeddings]]:
    """
    初始化 OpenAI 客户端，优先使用 .env 配置
    
    Returns:
        Tuple[Optional[AzureChatOpenAI], Optional[AzureOpenAIEmbeddings]]: 
            (LLM 模型, 嵌入模型)
    """
    # 打印调试信息
    print("=" * 50)
    print("初始化OpenAI客户端...")
    print(f"当前工作目录: {os.getcwd()}")
    
    try:
        # 1. 加载配置
        use_streamlit, config = ConfigLoader.load_from_streamlit()
        if not use_streamlit:
            config = ConfigLoader.load_from_env_or_json()
        
        # 2. 提取模型配置
        model_config = ConfigLoader.extract_model_config(config, use_streamlit)
        
        # 3. 验证配置
        if not ModelConfigValidator.validate(model_config):
            return None, None
        
        # 4. 规范化 endpoint
        base_endpoint = ModelConfigValidator.normalize_endpoint(
            model_config.get('api_base', '')
        )
        
        # 5. 推断嵌入维度
        emb_deployment = model_config.get('emb_deployment', '')
        dimension = ModelConfigValidator.infer_embedding_dimension(emb_deployment)
        
        # 打印配置信息
        print(f"✅ 使用模型: {model_config.get('deployment')}")
        print(f"✅ 使用嵌入模型: {emb_deployment}")
        print(f"ℹ️  嵌入向量维度: {dimension}")
        
        # 6. 初始化模型
        llm = AzureModelInitializer.initialize_llm(model_config, base_endpoint)
        embedding_model = AzureModelInitializer.initialize_embedding_model(
            model_config, base_endpoint, dimension
        )
        
        return llm, embedding_model
        
    except Exception as e:
        print(f"❌ 初始化 OpenAI 客户端时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
