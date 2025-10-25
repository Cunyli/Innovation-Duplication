#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Innovation Resolution Challenge Solution

This script implements solutions for identifying duplicate innovations
and creating a consolidated view of innovation relationships.
"""

import json
import math
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional, Union, Protocol

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from .vis import visualize_network_tufte
from .query_engine import InnovationQueryEngine, QueryResult

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate


# Import local modules
from .innovation_utils import (
    compute_similarity_matrix,
    find_potential_duplicates,
    calculate_innovation_statistics
)
from .local_entity_processing import Node, Relationship

from .utils.cluster.cluster_algorithms import (
    cluster_hdbscan,
    cluster_kmeans,
    cluster_agglomerative,
    cluster_spectral,
    cluster_with_stats
)
from .utils.cluster.graph_clustering import (
    graph_threshold_clustering,
    graph_kcore_clustering
)

# Import cache module
from .core.cache import (
    CacheBackend,
    JsonFileCache,
    MemoryCache,
    EmbeddingCache,
    CacheFactory
)

# Import data loaders
from .data_pipeline.loaders import GraphDocumentLoader, NodeMapper

# Import data processors
from .data_pipeline.processors import (
    RelationshipProcessor,
    is_valid_entity_name,
    is_valid_relationship,
    extract_entities_from_document,
    extract_relationships_from_document,
    initialize_openai_client,
    get_embedding,
    compute_similarity,
    InnovationFeatureBuilder,
    InnovationExtractor,
    EmbeddingManager,
    ClusteringStrategyFactory
)

# Constants
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Set up paths
GRAPH_DOCS_COMPANY = DATA_DIR / 'graph_docs_names_resolved'
GRAPH_DOCS_VTT = DATA_DIR / 'graph_docs_vtt_domain_names_resolved'
DATAFRAMES_DIR = DATA_DIR / 'dataframes'

# Create results directory if it doesn't exist
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    RESULTS_DIR = Path(".")  # Use current directory as fallback

# Set up plotting style
sns.set_theme(style="whitegrid")


def load_and_combine_data() -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Load relationship data from both company websites and VTT domain,
    combine them into a single dataframe, and collect entities and relationships.
    
    Returns:
        Tuple of (combined dataframe, all_pred_entities, all_pred_relations)
    """
    from .data_pipeline.processors import DataSourceProcessor
    
    print("Loading data from company websites...")
    
    # Initialize lists to collect predicted entities and relationships
    all_pred_entities = []
    all_pred_relations = []
    
    # Load company domain data
    df_company = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'vtt_mentions_comp_domain.csv'))
    df_company = df_company[df_company['Website'].str.startswith('www.')]
    df_company['source_index'] = df_company.index

    # Create company data processor
    company_processor = DataSourceProcessor(
        graph_docs_dir=GRAPH_DOCS_COMPANY,
        data_source_name="company_website"
    )
    
    # Define metadata mapper for company data
    def company_metadata_mapper(row, idx):
        return {
            "Document number": row['source_index'],
            "Source Company": row["Company name"],
            "Link Source Text": row["Link"],
            "Source Text": row["text_content"],
            "data_source": "company_website"
        }
    
    # Process company data
    df_relationships_comp_url = company_processor.process(
        df=df_company,
        file_pattern="{Company name}_{index}.pkl",
        metadata_mapper=company_metadata_mapper,
        entity_extractor=extract_entities_from_document,
        relation_extractor=extract_relationships_from_document,
        pred_entities=all_pred_entities,
        pred_relations=all_pred_relations
    )
    
    # Load VTT domain data
    print("Loading data from VTT domain...")
    df_vtt_domain = pd.read_csv(os.path.join(DATAFRAMES_DIR, 'comp_mentions_vtt_domain.csv'))
    
    # Create VTT data processor
    vtt_processor = DataSourceProcessor(
        graph_docs_dir=GRAPH_DOCS_VTT,
        data_source_name="vtt_website"
    )
    
    # Define metadata mapper for VTT data
    def vtt_metadata_mapper(row, idx):
        return {
            "Document number": idx,
            "VAT id": row["Vat_id"],
            "Link Source Text": row["source_url"],
            "Source Text": row["main_body"],
            "data_source": "vtt_website"
        }
    
    # Process VTT data
    df_relationships_vtt_domain = vtt_processor.process(
        df=df_vtt_domain,
        file_pattern="{Vat_id}_{index}.pkl",
        metadata_mapper=vtt_metadata_mapper,
        entity_extractor=extract_entities_from_document,
        relation_extractor=extract_relationships_from_document,
        pred_entities=all_pred_entities,
        pred_relations=all_pred_relations
    )
    
    # Rename columns to align dataframes
    df_relationships_vtt_domain = df_relationships_vtt_domain.rename(columns={"VAT id": "Source Company"})
    
    # Combine dataframes
    combined_df = pd.concat([df_relationships_comp_url, df_relationships_vtt_domain], ignore_index=True)
    print(f"Combined dataframe contains {len(combined_df)} relationships")
    
    return combined_df, all_pred_entities, all_pred_relations


def resolve_innovation_duplicates(
    df_relationships: pd.DataFrame, 
    model=None,
    cache_config: Dict = None,
    method: str = "hdbscan",
    **method_kwargs) -> Dict[str, str]:
    """
    Identify and cluster duplicate innovations using semantic similarity from textual embeddings.
    
    Args:
        df_relationships (pd.DataFrame): A relationship dataset containing Innovation nodes.
        model (callable, optional): Embedding model function that converts text -> vector.
        cache_config (Dict, optional): 缓存配置，包含以下字段:
            - type: 缓存类型 ('embedding')
            - backend: 后端类型 ('json' or 'memory')
            - path: 缓存文件路径
            - use_cache: 是否启用缓存
        method (str, optional): Which clustering method to use. One of:
            - "hdbscan"
            - "kmeans"
            - "agglomerative"
            - "spectral"
            - "graph_threshold"
            - "graph_kcore"
          Default: "hdbscan".
        **method_kwargs: Additional keyword args to pass into the chosen clustering function.
          For example:
            - if method="hdbscan", you can pass min_cluster_size=2, metric="cosine", cluster_selection_method="eom".
            - if method="kmeans", you can pass n_clusters=450, random_state=42.
            - if method="agglomerative", you can pass n_clusters=450, affinity="cosine", linkage="average".
            - if method="spectral", you can pass n_clusters=450, affinity="nearest_neighbors", n_neighbors=10.
            - if method="graph_threshold", you can pass similarity_threshold=0.85, use_cosine=True.
            - if method="graph_kcore", you can pass similarity_threshold=0.85, k_core=15, use_cosine=True.

    Returns:
        Dict[str, str]: Mapping from each innovation ID -> its canonical cluster ID.
    """
    print("Resolving innovation duplicates...")
    
    # Step 1: 提取唯一的创新
    unique_innovations = InnovationExtractor.extract_unique_innovations(df_relationships)
    
    if not InnovationExtractor.validate_innovations(unique_innovations):
        return {}
    
    # Step 2: 构建创新特征
    innovation_features = InnovationFeatureBuilder.build_all_features(
        unique_innovations, df_relationships
    )
    
    # Step 3: 生成或加载嵌入向量
    embedding_manager = EmbeddingManager.create_from_config(
        cache_config=cache_config,
        embedding_function=get_embedding
    )
    
    innovation_ids, embedding_matrix = embedding_manager.get_embeddings(
        innovation_features, model
    )
    
    # Step 4: 使用策略模式执行聚类
    clustering_strategy = ClusteringStrategyFactory.create_strategy(method)
    canonical_mapping = clustering_strategy.cluster(
        embedding_matrix=embedding_matrix,
        innovation_ids=innovation_ids,
        **method_kwargs
    )
    
    # 打印结果摘要
    print(f"Found {len(set(canonical_mapping.values()))} unique innovation clusters "
          f"(reduced from {len(unique_innovations)}).")
    
    return canonical_mapping



def create_innovation_knowledge_graph(df_relationships: pd.DataFrame, canonical_mapping: Dict[str, str]) -> Dict:
    """
    Create a consolidated knowledge graph of innovations and their relationships.
    
    使用结构化的知识图谱构建器来创建图谱，提高代码可读性和可维护性。
    
    Args:
        df_relationships: DataFrame with innovation relationships
        canonical_mapping: Mapping from innovation IDs to canonical IDs
    
    Returns:
        Dict: Consolidated knowledge graph with the following structure:
            {
                'innovations': {
                    canonical_id: {
                        'id': str,
                        'names': set,
                        'descriptions': set,
                        'developed_by': set,
                        'sources': set,
                        'source_ids': set,
                        'data_sources': set
                    }
                },
                'organizations': {
                    org_id: {
                        'id': str,
                        'name': str,
                        'description': str
                    }
                },
                'relationships': [
                    {
                        'source': str,
                        'target': str,
                        'type': str  # 'DEVELOPED_BY' or 'COLLABORATION'
                    }
                ]
            }
    """
    from .data_pipeline.processors import create_consolidated_knowledge_graph
    return create_consolidated_knowledge_graph(df_relationships, canonical_mapping)


def create_query_engine(
    consolidated_graph: Dict,
    embedding_model=None,
    cache_config: Optional[Dict] = None,
) -> InnovationQueryEngine:
    """Create a semantic search engine on top of the consolidated graph."""

    return InnovationQueryEngine(
        consolidated_graph=consolidated_graph,
        embedding_model=embedding_model,
        cache_config=cache_config,
    )


def query_innovations(
    consolidated_graph: Dict,
    query: str,
    top_k: int = 5,
    embedding_model=None,
    cache_config: Optional[Dict] = None,
) -> List[QueryResult]:
    """Convenience wrapper to search innovations directly."""

    engine = create_query_engine(
        consolidated_graph=consolidated_graph,
        embedding_model=embedding_model,
        cache_config=cache_config,
    )
    return engine.search(query=query, top_k=top_k)


def analyze_innovation_network(consolidated_graph: Dict, top_n: int = 10, max_iter: int = 1000, print_summary: bool = False) -> Dict:
    """
    Analyze the innovation network to extract insights.
    
    使用结构化的网络分析器来分析创新网络，提高代码可读性和可维护性。
    
    Args:
        consolidated_graph: Consolidated knowledge graph
        top_n: Number of top nodes to return
        max_iter: Maximum iterations for centrality computation
        print_summary: Whether to print analysis summary
    
    Returns:
        Dict: Analysis results with the following structure:
            {
                'graph': NetworkX graph object,
                'stats': {
                    'total': int,
                    'avg_sources': float,
                    'avg_developers': float,
                    'multi_source_count': int,
                    'multi_developer_count': int
                },
                'multi_source': Dict of innovations with multiple sources,
                'top_orgs': List of (org_id, innovation_count) tuples,
                'key_orgs': List of (org_id, betweenness_centrality) tuples,
                'key_innovations': List of (innovation_id, eigenvector_centrality) tuples
            }
    """
    from .data_pipeline.processors import analyze_innovation_network as analyze_network
    return analyze_network(
        consolidated_graph,
        top_n=top_n,
        max_iter=max_iter,
        print_summary=print_summary,
    )


def export_results(analysis_results: Dict, consolidated_graph: Dict, canonical_mapping: Dict, output_dir: str = RESULTS_DIR):
    """
    Export analysis results and consolidated data to files.
    
    使用结构化的结果导出器来保存分析结果，提高代码可读性和可维护性。
    
    Args:
        analysis_results: Results from network analysis
        consolidated_graph: Consolidated knowledge graph
        canonical_mapping: Mapping from innovation IDs to canonical IDs
        output_dir: Directory to save results
    """
    from .data_pipeline.processors import export_analysis_results
    export_analysis_results(analysis_results, consolidated_graph, canonical_mapping, output_dir)



def main():
    """Main function to execute the innovation resolution workflow."""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="VTT Innovation Resolution Process - 创新重复识别与网络分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本运行（使用默认配置）
  PYTHONPATH=src python -m innovation_platform.innovation_resolution
  
  # 使用不同的聚类方法
  PYTHONPATH=src python -m innovation_platform.innovation_resolution --clustering-method kmeans --n-clusters 400
  
  # 禁用缓存并跳过评估
  PYTHONPATH=src python -m innovation_platform.innovation_resolution --no-cache --skip-eval
  
  # 自定义输出目录
  PYTHONPATH=src python -m innovation_platform.innovation_resolution --output-dir ./my_results
  
  # 完整配置示例
  PYTHONPATH=src python -m innovation_platform.innovation_resolution \\
    --clustering-method hdbscan \\
    --min-cluster-size 3 \\
    --top-n 15 \\
    --output-dir ./results \\
    --cache-path ./cache/embeddings.json \\
    --skip-visualization
        """
    )
    
    # ==================== 数据源配置 ====================
    data_group = parser.add_argument_group('数据源配置', '配置输入数据的路径和来源')
    data_group.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help=f"数据目录路径 (默认: {DATA_DIR})"
    )
    data_group.add_argument(
        "--graph-docs-company",
        default=GRAPH_DOCS_COMPANY,
        help=f"公司网站图谱文档路径 (默认: {GRAPH_DOCS_COMPANY})"
    )
    data_group.add_argument(
        "--graph-docs-vtt",
        default=GRAPH_DOCS_VTT,
        help=f"VTT领域图谱文档路径 (默认: {GRAPH_DOCS_VTT})"
    )
    
    # ==================== 聚类配置 ====================
    clustering_group = parser.add_argument_group('聚类配置', '配置创新重复识别的聚类算法')
    clustering_group.add_argument(
        "--clustering-method",
        default="hdbscan",
        choices=["hdbscan", "kmeans", "agglomerative", "spectral", "graph_threshold", "graph_kcore"],
        help="聚类算法选择 (默认: hdbscan)"
    )
    
    # HDBSCAN 参数
    clustering_group.add_argument(
        "--min-cluster-size",
        type=int,
        default=2,
        help="HDBSCAN: 最小聚类大小 (默认: 2)"
    )
    clustering_group.add_argument(
        "--metric",
        default="cosine",
        choices=["cosine", "euclidean", "manhattan"],
        help="HDBSCAN: 距离度量 (默认: cosine)"
    )
    clustering_group.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=["eom", "leaf"],
        help="HDBSCAN: 聚类选择方法 (默认: eom)"
    )
    
    # K-Means/Agglomerative/Spectral 参数
    clustering_group.add_argument(
        "--n-clusters",
        type=int,
        default=450,
        help="K-Means/Agglomerative/Spectral: 聚类数量 (默认: 450)"
    )
    
    # Graph-based 参数
    clustering_group.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Graph-based: 相似度阈值 (默认: 0.85)"
    )
    clustering_group.add_argument(
        "--k-core",
        type=int,
        default=15,
        help="Graph K-Core: K核数 (默认: 15)"
    )
    
    # ==================== 缓存配置 ====================
    cache_group = parser.add_argument_group('缓存配置', '配置嵌入向量缓存策略')
    cache_group.add_argument(
        "--cache-type",
        default="embedding",
        choices=["embedding"],
        help="缓存类型 (默认: embedding)"
    )
    cache_group.add_argument(
        "--cache-backend",
        default="json",
        choices=["json", "memory"],
        help="缓存后端类型 (默认: json)"
    )
    cache_group.add_argument(
        "--cache-path",
        default="./embedding_vectors.json",
        help="缓存文件路径 (默认: ./embedding_vectors.json)"
    )
    cache_group.add_argument(
        "--no-cache",
        action="store_true",
        help="禁用缓存（每次都重新计算嵌入）"
    )
    
    # ==================== 网络分析配置 ====================
    analysis_group = parser.add_argument_group('网络分析配置', '配置创新网络分析参数')
    analysis_group.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="返回前 N 个关键节点 (默认: 10)"
    )
    analysis_group.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="中心性计算的最大迭代次数 (默认: 1000)"
    )
    analysis_group.add_argument(
        "--print-summary",
        action="store_true",
        help="打印分析摘要到控制台"
    )
    
    # ==================== 输出配置 ====================
    output_group = parser.add_argument_group('输出配置', '配置结果输出路径和格式')
    output_group.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help=f"输出目录路径 (默认: {RESULTS_DIR})"
    )
    output_group.add_argument(
        "--skip-visualization",
        action="store_true",
        help="跳过网络可视化步骤"
    )
    output_group.add_argument(
        "--skip-export",
        action="store_true",
        help="跳过结果导出步骤"
    )
    
    # ==================== 评估配置 ====================
    eval_group = parser.add_argument_group('评估配置', '配置模型评估参数')
    eval_group.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过评估步骤"
    )
    eval_group.add_argument(
        "--auto-label",
        action="store_true",
        help="自动标注一致性样本并生成金标准文件"
    )
    eval_group.add_argument(
        "--eval-dir",
        default="evaluation",
        help="评估文件输出目录 (默认: evaluation)"
    )
    
    # ==================== 工作流控制 ====================
    workflow_group = parser.add_argument_group('工作流控制', '控制执行哪些步骤')
    workflow_group.add_argument(
        "--steps",
        nargs="+",
        choices=["load", "cluster", "graph", "analyze", "visualize", "export", "eval"],
        default=["load", "cluster", "graph", "analyze", "visualize", "export", "eval"],
        help="要执行的步骤（可多选，默认执行所有步骤）"
    )
    workflow_group.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )
    workflow_group.add_argument(
        "--quiet",
        action="store_true",
        help="安静模式（最小输出）"
    )
    
    args = parser.parse_args()
    
    # 构建配置字典
    config = {
        "data": {
            "data_dir": args.data_dir,
            "graph_docs_company": args.graph_docs_company,
            "graph_docs_vtt": args.graph_docs_vtt
        },
        "clustering": {
            "method": args.clustering_method,
            "min_cluster_size": args.min_cluster_size,
            "metric": args.metric,
            "cluster_selection_method": args.cluster_selection_method,
            "n_clusters": args.n_clusters,
            "similarity_threshold": args.similarity_threshold,
            "k_core": args.k_core
        },
        "cache": {
            "type": args.cache_type,
            "backend": args.cache_backend,
            "path": args.cache_path,
            "use_cache": not args.no_cache
        },
        "analysis": {
            "top_n": args.top_n,
            "max_iter": args.max_iter,
            "print_summary": args.print_summary
        },
        "output": {
            "output_dir": args.output_dir,
            "skip_visualization": args.skip_visualization,
            "skip_export": args.skip_export
        },
        "evaluation": {
            "skip_eval": args.skip_eval,
            "auto_label": args.auto_label,
            "eval_dir": args.eval_dir
        },
        "workflow": {
            "steps": set(args.steps),
            "verbose": args.verbose,
            "quiet": args.quiet
        }
    }
    
    # 打印配置信息
    if not args.quiet:
        print("=" * 70)
        print("🚀 VTT Innovation Resolution Process")
        print("=" * 70)
        print("\n📋 运行配置:")
        print(f"  聚类方法: {config['clustering']['method']}")
        print(f"  输出目录: {config['output']['output_dir']}")
        print(f"  缓存: {'启用' if config['cache']['use_cache'] else '禁用'}")
        print(f"  执行步骤: {', '.join(sorted(config['workflow']['steps']))}")
        print()
    
    # ==================== 执行工作流 ====================
    # ==================== 执行工作流 ====================
    
    try:
        # Step 1: Load and combine data
        if "load" in config['workflow']['steps']:
            if config['workflow']['verbose']:
                print("\n📂 Step 1: Loading and combining data...")
            df_relationships, all_pred_entities, all_pred_relations = load_and_combine_data()
            if config['workflow']['verbose']:
                print(f"✓ Loaded {len(df_relationships)} relationships")
        else:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping data loading step")
            return
        
        # Step 2: Initialize OpenAI client
        if config['workflow']['verbose']:
            print("\n🔧 Step 2: Initializing OpenAI client...")
        llm, embed_model = initialize_openai_client()
        
        if llm is None and config['workflow']['verbose']:
            print("⚠️  Warning: Language model not available. Some features may be limited.")
        
        if embed_model is None and config['workflow']['verbose']:
            print("⚠️  Warning: Embedding model not available. Using TF-IDF embeddings as fallback.")

        # Step 3: Resolve innovation duplicates
        if "cluster" in config['workflow']['steps']:
            if config['workflow']['verbose']:
                print(f"\n🔍 Step 3: Resolving innovation duplicates using {config['clustering']['method']}...")
            
            # 根据聚类方法选择参数
            clustering_kwargs = {}
            if config['clustering']['method'] == "hdbscan":
                clustering_kwargs = {
                    "min_cluster_size": config['clustering']['min_cluster_size'],
                    "metric": config['clustering']['metric'],
                    "cluster_selection_method": config['clustering']['cluster_selection_method']
                }
            elif config['clustering']['method'] in ["kmeans", "agglomerative", "spectral"]:
                clustering_kwargs = {
                    "n_clusters": config['clustering']['n_clusters']
                }
                if config['clustering']['method'] == "agglomerative":
                    clustering_kwargs.update({
                        "affinity": "cosine",
                        "linkage": "average"
                    })
                elif config['clustering']['method'] == "spectral":
                    clustering_kwargs.update({
                        "affinity": "nearest_neighbors",
                        "n_neighbors": 10
                    })
            elif config['clustering']['method'] in ["graph_threshold", "graph_kcore"]:
                clustering_kwargs = {
                    "similarity_threshold": config['clustering']['similarity_threshold'],
                    "use_cosine": True
                }
                if config['clustering']['method'] == "graph_kcore":
                    clustering_kwargs["k_core"] = config['clustering']['k_core']
            
            canonical_mapping = resolve_innovation_duplicates(
                df_relationships=df_relationships,
                model=embed_model,
                cache_config=config['cache'],
                method=config['clustering']['method'],
                **clustering_kwargs
            )
            
            if config['workflow']['verbose']:
                unique_clusters = len(set(canonical_mapping.values()))
                total_innovations = len(canonical_mapping)
                reduction = (1 - unique_clusters / total_innovations) * 100 if total_innovations > 0 else 0
                print(f"✓ Identified {unique_clusters} unique innovation clusters")
                print(f"  (reduced from {total_innovations} innovations, {reduction:.1f}% reduction)")
        else:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping clustering step")
            canonical_mapping = {}
            
        # Step 4: Create consolidated knowledge graph
        if "graph" in config['workflow']['steps']:
            if config['workflow']['verbose']:
                print("\n🕸️  Step 4: Building consolidated knowledge graph...")
            consolidated_graph = create_innovation_knowledge_graph(df_relationships, canonical_mapping)
            if config['workflow']['verbose']:
                print(f"✓ Graph built with {len(consolidated_graph['innovations'])} innovations")
                print(f"  and {len(consolidated_graph['organizations'])} organizations")
        else:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping graph building step")
            consolidated_graph = {'innovations': {}, 'organizations': {}, 'relationships': []}
        
        # Step 5: Analyze innovation network
        if "analyze" in config['workflow']['steps']:
            if config['workflow']['verbose']:
                print(f"\n📊 Step 5: Analyzing innovation network (top {config['analysis']['top_n']})...")
            analysis_results = analyze_innovation_network(
                consolidated_graph,
                top_n=config['analysis']['top_n'],
                max_iter=config['analysis']['max_iter'],
                print_summary=config['analysis']['print_summary']
            )
            if config['workflow']['verbose']:
                print(f"✓ Network analysis complete")
        else:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping network analysis step")
            analysis_results = None
        
        # Step 6: Visualize network
        if "visualize" in config['workflow']['steps'] and not config['output']['skip_visualization']:
            if config['workflow']['verbose']:
                print("\n🎨 Step 6: Visualizing network...")
            if analysis_results:
                visualize_network_tufte(analysis_results)
                if config['workflow']['verbose']:
                    print(f"✓ Visualization saved to {config['output']['output_dir']}")
            else:
                print("⚠️  Cannot visualize: analysis_results not available")
        elif config['output']['skip_visualization']:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping visualization step (--skip-visualization)")
        
        # Save predicted entities and relationships for evaluation
        os.makedirs(config['evaluation']['eval_dir'], exist_ok=True)
        
        # Remove duplicates from predicted entities and relations
        unique_pred_entities = []
        seen_entities = set()
        for entity in all_pred_entities:
            entity_key = (entity["name"].lower(), entity["type"])
            if entity_key not in seen_entities:
                seen_entities.add(entity_key)
                unique_pred_entities.append(entity)
        
        unique_pred_relations = []
        seen_relations = set()
        for relation in all_pred_relations:
            relation_key = (relation["innovation"].lower(), relation["organization"].lower(), relation["relation"])
            if relation_key not in seen_relations:
                seen_relations.add(relation_key)
                unique_pred_relations.append(relation)
        
        # Save to JSON files
        pred_entities_path = os.path.join(config['evaluation']['eval_dir'], "pred_entities.json")
        pred_relations_path = os.path.join(config['evaluation']['eval_dir'], "pred_relations.json")
        
        with open(pred_entities_path, "w", encoding="utf-8") as f:
            json.dump(unique_pred_entities, f, ensure_ascii=False, indent=2)
        
        with open(pred_relations_path, "w", encoding="utf-8") as f:
            json.dump(unique_pred_relations, f, ensure_ascii=False, indent=2)
        
        if config['workflow']['verbose']:
            print(f"\n💾 Saved predictions:")
            print(f"  {len(unique_pred_entities)} unique entities → {pred_entities_path}")
            print(f"  {len(unique_pred_relations)} unique relations → {pred_relations_path}")
        
        # Step 7: Export results
        if "export" in config['workflow']['steps'] and not config['output']['skip_export']:
            if config['workflow']['verbose']:
                print(f"\n📤 Step 7: Exporting results to {config['output']['output_dir']}...")
            if analysis_results:
                export_results(analysis_results, consolidated_graph, canonical_mapping, 
                             output_dir=config['output']['output_dir'])
                if config['workflow']['verbose']:
                    print("✓ Results exported successfully")
            else:
                print("⚠️  Cannot export: analysis_results not available")
        elif config['output']['skip_export']:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping export step (--skip-export)")
        
        # Step 8: Run evaluation
        if "eval" in config['workflow']['steps'] and not config['evaluation']['skip_eval']:
            if config['workflow']['verbose']:
                print("\n🎯 Step 8: Running evaluation...")
            
            # Convert consolidated_graph to Node and Relationship objects for evaluation
            from .evaluation import run_all_evaluations
            
            # Create Node objects for merged innovations
            merged_innovations = []
            for inno_id, inno_data in consolidated_graph['innovations'].items():
                node = Node(
                    id=inno_id,
                    type="Innovation",
                    properties={
                        "aliases": "|".join(list(inno_data['names'])) if inno_data['names'] else "",
                        "source_docs": "|".join(list(inno_data['descriptions'])) if inno_data['descriptions'] else "",
                        "developed_by": "|".join(list(inno_data['developed_by'])) if inno_data['developed_by'] else "",
                        "sources": "|".join(list(inno_data['sources'])) if inno_data['sources'] else ""
                    }
                )
                merged_innovations.append(node)
            
            # Create all nodes
            all_nodes = []
            
            # Add innovations
            for inno_id, inno_data in consolidated_graph['innovations'].items():
                node = Node(
                    id=inno_id,
                    type="Innovation",
                    properties={
                        "name": list(inno_data['names'])[0] if inno_data['names'] else inno_id,
                        "description": list(inno_data['descriptions'])[0] if inno_data['descriptions'] else ""
                    }
                )
                all_nodes.append(node)
            
            # Add organizations
            for org_id, org_data in consolidated_graph['organizations'].items():
                node = Node(
                    id=org_id,
                    type="Organization",
                    properties={
                        "name": str(org_data['name']) if org_data['name'] is not None else "",
                        "description": str(org_data['description']) if org_data['description'] is not None else ""
                    }
                )
                all_nodes.append(node)
            
            # Create relationships
            all_rels = []
            for rel_data in consolidated_graph['relationships']:
                rel = Relationship(
                    source=rel_data['source'],
                    source_type="Innovation" if rel_data['source'] in consolidated_graph['innovations'] else "Organization",
                    target=rel_data['target'],
                    target_type="Organization" if rel_data['target'] in consolidated_graph['organizations'] else "Innovation",
                    type=rel_data['type'],
                    properties={}
                )
                all_rels.append(rel)
            
            # Run evaluation
            evaluation_results = run_all_evaluations(
                merged_innovations=merged_innovations,
                all_nodes=all_nodes,
                all_rels=all_rels,
                data_dir=config['data']['data_dir'],
                results_dir=config['output']['output_dir'],
                eval_dir=config['evaluation']['eval_dir'],
                auto_label=config['evaluation']['auto_label'],
                llm=llm
            )
            
            if config['workflow']['verbose']:
                print("✓ Evaluation complete")
        elif config['evaluation']['skip_eval']:
            if not config['workflow']['quiet']:
                print("⏭️  Skipping evaluation step (--skip-eval)")
        
        # 完成
        if not config['workflow']['quiet']:
            print("\n" + "=" * 70)
            print("✅ Innovation Resolution process completed successfully!")
            print(f"📁 Results saved to: {config['output']['output_dir']}")
            print("=" * 70)
    
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ Error: {e}")
        if config['workflow']['verbose']:
            import traceback
            traceback.print_exc()
        print("=" * 70)
        return 1
    
    return 0



if __name__ == "__main__":
    main() 
