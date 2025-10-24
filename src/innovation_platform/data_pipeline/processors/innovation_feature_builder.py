#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Innovation Feature Builder Module

Handles construction of textual features for innovation deduplication.
"""

import pandas as pd
from typing import Dict, Tuple
from tqdm import tqdm


class InnovationFeatureBuilder:
    """创新特征构建器 - 负责为创新构建用于相似度计算的文本特征"""
    
    @staticmethod
    def extract_basic_info(innovation_row: pd.Series) -> str:
        """
        提取创新的基本信息
        
        Args:
            innovation_row: 创新的 DataFrame 行
        
        Returns:
            str: 基本信息文本
        """
        source_name = str(innovation_row.get("source_english_id", "")).strip()
        source_desc = str(innovation_row.get("source_description", "")).strip()
        return f"Innovation name: {source_name}. Description: {source_desc}."
    
    @staticmethod
    def extract_developers(
        innovation_id: str, 
        df_relationships: pd.DataFrame
    ) -> str:
        """
        提取创新的开发者信息
        
        Args:
            innovation_id: 创新ID
            df_relationships: 关系数据框
        
        Returns:
            str: 开发者信息文本
        """
        developers = (
            df_relationships[
                (df_relationships["source_id"] == innovation_id) &
                (df_relationships["relationship_type"] == "DEVELOPED_BY")
            ]["target_english_id"]
            .dropna()
            .unique()
            .tolist()
        )
        
        if developers:
            return " Developed by: " + ", ".join(developers) + "."
        return ""
    
    @staticmethod
    def extract_relationships_context(
        innovation_id: str,
        df_relationships: pd.DataFrame
    ) -> str:
        """
        提取创新的关系上下文信息
        
        Args:
            innovation_id: 创新ID
            df_relationships: 关系数据框
        
        Returns:
            str: 关系上下文文本
        """
        context_parts = []
        
        related_rows = df_relationships[df_relationships["source_id"] == innovation_id]
        for _, rel_row in related_rows.iterrows():
            rel_desc = str(rel_row.get("relationship description", "")).strip()
            target_name = str(rel_row.get("target_english_id", "")).strip()
            target_desc = str(rel_row.get("target_description", "")).strip()
            
            if rel_desc and target_name and target_desc:
                context_parts.append(
                    f" {rel_desc} {target_name}, which is described as: {target_desc}."
                )
        
        return "".join(context_parts)
    
    @staticmethod
    def build_context(
        innovation_row: pd.Series,
        df_relationships: pd.DataFrame
    ) -> str:
        """
        为单个创新构建完整的文本上下文
        
        Args:
            innovation_row: 创新的 DataFrame 行
            df_relationships: 关系数据框
        
        Returns:
            str: 完整的文本上下文
        """
        innovation_id = innovation_row["source_id"]
        
        # 组合各部分信息
        context = InnovationFeatureBuilder.extract_basic_info(innovation_row)
        context += InnovationFeatureBuilder.extract_developers(innovation_id, df_relationships)
        context += InnovationFeatureBuilder.extract_relationships_context(innovation_id, df_relationships)
        
        return context
    
    @staticmethod
    def build_all_features(
        unique_innovations: pd.DataFrame,
        df_relationships: pd.DataFrame,
        show_progress: bool = True
    ) -> Dict[str, str]:
        """
        批量构建所有创新的特征
        
        Args:
            unique_innovations: 唯一创新的 DataFrame
            df_relationships: 关系数据框
            show_progress: 是否显示进度条
        
        Returns:
            Dict[str, str]: {innovation_id: context_text}
        """
        innovation_features = {}
        
        iterator = unique_innovations.iterrows()
        if show_progress:
            iterator = tqdm(
                iterator, 
                total=len(unique_innovations),
                desc="Creating innovation features"
            )
        
        for _, row in iterator:
            innovation_id = row["source_id"]
            
            # 避免重复处理
            if innovation_id in innovation_features:
                continue
            
            innovation_features[innovation_id] = InnovationFeatureBuilder.build_context(
                row, df_relationships
            )
        
        return innovation_features


class InnovationExtractor:
    """创新提取器 - 从 DataFrame 中提取唯一的创新记录"""
    
    @staticmethod
    def extract_unique_innovations(df_relationships: pd.DataFrame) -> pd.DataFrame:
        """
        从关系 DataFrame 中提取唯一的创新
        
        Args:
            df_relationships: 关系数据框
        
        Returns:
            pd.DataFrame: 唯一创新的 DataFrame
        """
        innovations = df_relationships[df_relationships["source_type"] == "Innovation"]
        unique_innovations = innovations.drop_duplicates(
            subset=["source_id"]
        ).reset_index(drop=True)
        
        print(f"Found {len(unique_innovations)} unique innovations.")
        return unique_innovations
    
    @staticmethod
    def validate_innovations(unique_innovations: pd.DataFrame) -> bool:
        """
        验证创新数据是否有效
        
        Args:
            unique_innovations: 唯一创新的 DataFrame
        
        Returns:
            bool: 是否有效
        """
        if unique_innovations.empty:
            print("⚠️  No innovations found in the dataset.")
            return False
        return True
