#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embedding Strategy Module

Provides different embedding strategies with fallback mechanisms.
"""

import numpy as np
from typing import Protocol, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel(Protocol):
    """嵌入模型协议"""
    
    def embed_query(self, text: str) -> list:
        """生成文本的嵌入向量"""
        ...


class OpenAIEmbeddingStrategy:
    """OpenAI 嵌入策略"""
    
    @staticmethod
    def embed(text: str, model: EmbeddingModel) -> Optional[np.ndarray]:
        """
        使用 OpenAI 模型生成嵌入
        
        Args:
            text: 输入文本
            model: OpenAI 嵌入模型
        
        Returns:
            Optional[np.ndarray]: 嵌入向量，失败返回 None
        """
        try:
            embedding = model.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            print(f"Error using OpenAI embedding: {e}")
            print("Falling back to TF-IDF embedding...")
            return None


class TFIDFEmbeddingStrategy:
    """TF-IDF 嵌入策略（作为后备方案）"""
    
    DEFAULT_DIMENSION = 1536
    TFIDF_MAX_FEATURES = 768
    
    @staticmethod
    def _prepare_sentences(text: str) -> list:
        """
        准备用于向量化的句子列表
        
        Args:
            text: 输入文本
        
        Returns:
            list: 句子列表
        """
        # 按句号分割句子
        sentences = [s.strip() for s in text.replace('\n', ' ').split('.') if s.strip()]
        
        # 如果句子太少，使用单词分割
        if len(sentences) < 2:
            sentences = text.split()
        
        # 确保至少有两个元素用于向量化
        if len(sentences) < 2:
            sentences = [text, "placeholder"]
        
        return sentences
    
    @staticmethod
    def _vectorize_sentences(sentences: list) -> np.ndarray:
        """
        使用 TF-IDF 向量化句子
        
        Args:
            sentences: 句子列表
        
        Returns:
            np.ndarray: 向量化结果
        """
        vectorizer = TfidfVectorizer(max_features=TFIDFEmbeddingStrategy.TFIDF_MAX_FEATURES)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # 使用平均值作为最终表示
        return tfidf_matrix.mean(axis=0).A[0]
    
    @staticmethod
    def _normalize_dimension(embedding: np.ndarray, target_dim: int = DEFAULT_DIMENSION) -> np.ndarray:
        """
        调整嵌入维度到目标维度
        
        Args:
            embedding: 原始嵌入
            target_dim: 目标维度
        
        Returns:
            np.ndarray: 调整后的嵌入
        """
        current_dim = len(embedding)
        
        if current_dim < target_dim:
            # 填充到目标维度
            return np.pad(embedding, (0, target_dim - current_dim), 'constant')
        elif current_dim > target_dim:
            # 截断到目标维度
            return embedding[:target_dim]
        else:
            return embedding
    
    @staticmethod
    def embed(text: str) -> np.ndarray:
        """
        使用 TF-IDF 生成嵌入
        
        Args:
            text: 输入文本
        
        Returns:
            np.ndarray: 嵌入向量
        """
        try:
            sentences = TFIDFEmbeddingStrategy._prepare_sentences(text)
            embedding = TFIDFEmbeddingStrategy._vectorize_sentences(sentences)
            return TFIDFEmbeddingStrategy._normalize_dimension(embedding)
        except Exception as e:
            print(f"Error creating TF-IDF embedding: {e}")
            # 返回随机嵌入作为最后手段
            return np.random.rand(TFIDFEmbeddingStrategy.DEFAULT_DIMENSION)


class FallbackEmbedding:
    """随机嵌入（最后的后备方案）"""
    
    @staticmethod
    def embed(dimension: int = 1536) -> np.ndarray:
        """
        生成随机嵌入向量
        
        Args:
            dimension: 向量维度
        
        Returns:
            np.ndarray: 随机嵌入向量
        """
        return np.random.rand(dimension)


def get_embedding(text: str, model: Optional[EmbeddingModel] = None) -> np.ndarray:
    """
    获取文本的嵌入向量，自动处理后备方案
    
    优先级：OpenAI 嵌入 > TF-IDF 嵌入 > 随机嵌入
    
    Args:
        text: 输入文本
        model: OpenAI 嵌入模型（可选）
    
    Returns:
        np.ndarray: 嵌入向量
    """
    # 优先尝试使用 OpenAI 模型
    if model is not None:
        openai_embedding = OpenAIEmbeddingStrategy.embed(text, model)
        if openai_embedding is not None:
            return openai_embedding
    
    # 后备到 TF-IDF
    return TFIDFEmbeddingStrategy.embed(text)


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    计算两个嵌入向量之间的余弦相似度
    
    Args:
        emb1: 第一个嵌入向量
        emb2: 第二个嵌入向量
    
    Returns:
        float: 余弦相似度分数
    """
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]
