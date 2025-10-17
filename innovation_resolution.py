#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Innovation Resolution Challenge Solution

This script implements solutions for identifying duplicate innovations
and creating a consolidated view of innovation relationships.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional, Union, Protocol
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
from vis import visualize_network_tufte

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate


import math


# Import local modules
from innovation_utils import (
    compute_similarity_matrix,
    find_potential_duplicates,
    calculate_innovation_statistics
)
from local_entity_processing import Node, Relationship

from utils.cluster.cluster_algorithms import (
    cluster_hdbscan,
    cluster_kmeans,
    cluster_agglomerative,
    cluster_spectral,
    cluster_with_stats
)
from utils.cluster.graph_clustering import (
    graph_threshold_clustering,
    graph_kcore_clustering
)

# Import cache module
from core.cache import (
    CacheBackend,
    JsonFileCache,
    MemoryCache,
    EmbeddingCache,
    CacheFactory
)

# Import data loaders
from data.loaders import GraphDocumentLoader, NodeMapper

# Import data processors
from data.processors import (
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
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Set up paths
GRAPH_DOCS_COMPANY = os.path.join(DATA_DIR, 'graph_docs_names_resolved')
GRAPH_DOCS_VTT = os.path.join(DATA_DIR, 'graph_docs_vtt_domain_names_resolved')
DATAFRAMES_DIR = os.path.join(DATA_DIR, 'dataframes')

# Create results directory if it doesn't exist
try:
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    RESULTS_DIR = '.'  # Use current directory as fallback

# Set up plotting style
sns.set_theme(style="whitegrid")


def load_and_combine_data() -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Load relationship data from both company websites and VTT domain,
    combine them into a single dataframe, and collect entities and relationships.
    
    Returns:
        Tuple of (combined dataframe, all_pred_entities, all_pred_relations)
    """
    from data.processors import DataSourceProcessor
    
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
    
    Args:
        df_relationships: DataFrame with innovation relationships
        canonical_mapping: Mapping from innovation IDs to canonical IDs
    
    Returns:
        Dict: Consolidated knowledge graph
    """
    print("Creating innovation knowledge graph...")
    
    # Step 1: Create consolidated innovations
    consolidated_innovations = {}
    
    for _, row in tqdm(df_relationships[df_relationships['source_type'] == 'Innovation'].iterrows(), 
                      desc="Consolidating innovations"):
        innovation_id = row['source_id']
        canonical_id = canonical_mapping.get(innovation_id, innovation_id)
        
        if canonical_id not in consolidated_innovations:
            consolidated_innovations[canonical_id] = {
                'id': canonical_id,
                'names': set(),
                'descriptions': set(),
                'developed_by': set(),
                'sources': set(),
                'source_ids': set([innovation_id]),
                'data_sources': set()
            }
        else:
            consolidated_innovations[canonical_id]['source_ids'].add(innovation_id)
        
        consolidated_innovations[canonical_id]['names'].add(str(row['source_english_id']))
        consolidated_innovations[canonical_id]['descriptions'].add(str(row['source_description']))
        consolidated_innovations[canonical_id]['sources'].add(str(row['Link Source Text']))
        consolidated_innovations[canonical_id]['data_sources'].add(str(row['data_source']))
        
        # Add relationship
        if row['relationship_type'] == 'DEVELOPED_BY':
            consolidated_innovations[canonical_id]['developed_by'].add(row['target_id'])
    
    # Step 2: Build consolidated graph
    consolidated_graph = {
        'innovations': consolidated_innovations,
        'organizations': {},
        'relationships': []
    }
    
    # Add organizations
    for _, row in tqdm(df_relationships[df_relationships['target_type'] == 'Organization'].drop_duplicates(subset=['target_id']).iterrows(),
                      desc="Adding organizations"):
        org_id = row['target_id']
        if org_id not in consolidated_graph['organizations']:
            consolidated_graph['organizations'][org_id] = {
                'id': org_id,
                'name': row['target_english_id'],
                'description': row['target_description']
            }
    
    # Add relationships
    for canonical_id, innovation in tqdm(consolidated_innovations.items(), desc="Adding relationships"):
        for org_id in innovation['developed_by']:
            consolidated_graph['relationships'].append({
                'source': canonical_id,
                'target': org_id,
                'type': 'DEVELOPED_BY'
            })
    
    # Add collaboration relationships
    for _, row in tqdm(df_relationships[
        (df_relationships['source_type'] == 'Organization') & 
        (df_relationships['relationship_type'] == 'COLLABORATION')
    ].iterrows(), desc="Adding collaborations"):
        consolidated_graph['relationships'].append({
            'source': row['source_id'],
            'target': row['target_id'],
            'type': 'COLLABORATION'
        })
    
    print(f"Created knowledge graph with {len(consolidated_graph['innovations'])} innovations, " 
          f"{len(consolidated_graph['organizations'])} organizations, and "
          f"{len(consolidated_graph['relationships'])} relationships")
    
    return consolidated_graph


def analyze_innovation_network(consolidated_graph: Dict) -> Dict:
    """
    Analyze the innovation network to extract insights.
    
    Args:
        consolidated_graph: Consolidated knowledge graph
    
    Returns:
        Dict: Analysis results
    """
    print("Analyzing innovation network...")
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for innovation_id, innovation in consolidated_graph['innovations'].items():
        G.add_node(innovation_id, 
                   type='Innovation', 
                   names=', '.join(innovation['names']),
                   sources=len(innovation['sources']),
                   developed_by=len(innovation['developed_by']))
    
    for org_id, org in consolidated_graph['organizations'].items():
        G.add_node(org_id, 
                   type='Organization', 
                   name=org['name'])
    
    # Add edges
    for rel in consolidated_graph['relationships']:
        G.add_edge(rel['source'], rel['target'], type=rel['type'])
    
    # Basic statistics
    innovation_stats = {
        'total': len(consolidated_graph['innovations']),
        'avg_sources': sum(len(i['sources']) for i in consolidated_graph['innovations'].values()) / max(1, len(consolidated_graph['innovations'])),
        'avg_developers': sum(len(i['developed_by']) for i in consolidated_graph['innovations'].values()) / max(1, len(consolidated_graph['innovations'])),
        'multi_source_count': sum(1 for i in consolidated_graph['innovations'].values() if len(i['sources']) > 1),
        'multi_developer_count': sum(1 for i in consolidated_graph['innovations'].values() if len(i['developed_by']) > 1)
    }
    
    # Find innovations with multiple sources
    multi_source_innovations = {
        k: v for k, v in consolidated_graph['innovations'].items() 
        if len(v['sources']) > 1
    }
    
    # Find organizations with most innovations
    org_innovation_counts = {}
    for rel in consolidated_graph['relationships']:
        if rel['type'] == 'DEVELOPED_BY' and rel['target'] in consolidated_graph['organizations']:
            org_id = rel['target']
            if org_id not in org_innovation_counts:
                org_innovation_counts[org_id] = 0
            org_innovation_counts[org_id] += 1
    
    top_orgs = sorted(
        [(org_id, count) for org_id, count in org_innovation_counts.items()], 
        key=lambda x: x[1], 
        reverse=True
    )[:10]
    
    # Centrality analysis
    try:
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Find key players
        key_orgs = sorted(
            [(node, betweenness_centrality[node]) 
             for node in G.nodes if G.nodes[node].get('type') == 'Organization'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        key_innovations = sorted(
            [(node, eigenvector_centrality[node]) 
             for node in G.nodes if G.nodes[node].get('type') == 'Innovation'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
    except:
        # If centrality algorithms fail, provide empty lists
        key_orgs = []
        key_innovations = []
    
    return {
        'graph': G,
        'stats': innovation_stats,
        'multi_source': multi_source_innovations,
        'top_orgs': top_orgs,
        'key_orgs': key_orgs,
        'key_innovations': key_innovations
    }


def export_results(analysis_results: Dict, consolidated_graph: Dict, canonical_mapping: Dict, output_dir: str = RESULTS_DIR):
    """
    Export analysis results and consolidated data to files.
    
    Args:
        analysis_results: Results from network analysis
        consolidated_graph: Consolidated knowledge graph
        canonical_mapping: Mapping from innovation IDs to canonical IDs
        output_dir: Directory to save results
    """
    print("Exporting results...")
    
    # Save canonical mapping
    with open(os.path.join(output_dir, 'canonical_mapping.json'), 'w') as f:
        # Convert sets to lists for JSON serialization
        mapping_for_json = {k: v for k, v in canonical_mapping.items()}
        json.dump(mapping_for_json, f, indent=2)
    
    # Save consolidated graph (need to convert sets to lists for JSON serialization)
    graph_for_json = {
        'innovations': {},
        'organizations': consolidated_graph['organizations'],
        'relationships': consolidated_graph['relationships']
    }
    
    for k, v in consolidated_graph['innovations'].items():
        graph_for_json['innovations'][k] = {
            'id': v['id'],
            'names': list(v['names']),
            'descriptions': list(v['descriptions']),
            'developed_by': list(v['developed_by']),
            'sources': list(v['sources']),
            'source_ids': list(v['source_ids']),
            'data_sources': list(v['data_sources'])
        }
    
    with open(os.path.join(output_dir, 'consolidated_graph.json'), 'w') as f:
        json.dump(graph_for_json, f, indent=2)
    
    # Save innovation statistics
    with open(os.path.join(output_dir, 'innovation_stats.json'), 'w') as f:
        stats_for_json = analysis_results['stats']
        json.dump(stats_for_json, f, indent=2)
    
    # Save multi-source innovations information
    multi_source_for_json = {}
    for k, v in analysis_results['multi_source'].items():
        multi_source_for_json[k] = {
            'names': list(v['names']),
            'descriptions': list(v['descriptions']),
            'developed_by': list(v['developed_by']),
            'sources': list(v['sources']),
            'source_ids': list(v['source_ids']),
            'data_sources': list(v['data_sources'])
        }
    
    with open(os.path.join(output_dir, 'multi_source_innovations.json'), 'w') as f:
        json.dump(multi_source_for_json, f, indent=2)
    
    # Save key organizations and innovations
    key_nodes = {
        'key_organizations': [{
            'id': org_id,
            'centrality': centrality,
            'name': analysis_results['graph'].nodes[org_id].get('name', org_id)
        } for org_id, centrality in analysis_results['key_orgs']],
        'key_innovations': [{
            'id': inno_id,
            'centrality': centrality,
            'names': list(consolidated_graph['innovations'][inno_id]['names'])
        } for inno_id, centrality in analysis_results['key_innovations'] if inno_id in consolidated_graph['innovations']]
    }
    
    with open(os.path.join(output_dir, 'key_nodes.json'), 'w') as f:
        json.dump(key_nodes, f, indent=2)
    
    print(f"Results exported to {output_dir}")


def main():
    """Main function to execute the innovation resolution workflow."""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="VTT Innovation Resolution process")
    parser.add_argument("--cache-type", default="embedding", choices=["embedding"],
                       help="Cache type to use")
    parser.add_argument("--cache-backend", default="json", choices=["json", "memory"],
                       help="Cache backend type")
    parser.add_argument("--cache-path", default="./embedding_vectors.json",
                       help="Path to cache file (for file-based backends)")
    parser.add_argument("--no-cache", action="store_true", 
                       help="Disable caching")

    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--auto-label", action="store_true",
                       help="Automatically label consistency samples and generate gold standard files")
    
    args = parser.parse_args()
    
    # 缓存配置
    cache_config = {
        "type": args.cache_type,
        "backend": args.cache_backend,
        "path": args.cache_path,
        "use_cache": not args.no_cache
    }
    
    print("Starting VTT Innovation Resolution process...")
    print(f"Cache configuration: {cache_config}")
    
    # Step 1: Load and combine data (modified to also collect predictions)
    df_relationships, all_pred_entities, all_pred_relations = load_and_combine_data()
    
    # Step 2: Initialize OpenAI client
    llm, embed_model = initialize_openai_client()
    
    if llm is None:
        print("Warning: Language model not available. Some features may be limited.")
    
    if embed_model is None:
        print("Warning: Embedding model not available. Using TF-IDF embeddings as fallback.")

    # Step 3: Resolve innovation duplicates
    canonical_mapping = resolve_innovation_duplicates(
        df_relationships=df_relationships,
        model=embed_model,
        cache_config=cache_config,
        method="hdbscan",  # 默认使用hdbscan
        min_cluster_size=2,  # 可配置参数
        metric="cosine",
        cluster_selection_method="eom"
    )
        
    # Step 4: Create consolidated knowledge graph
    consolidated_graph = create_innovation_knowledge_graph(df_relationships, canonical_mapping)
    print(consolidated_graph['organizations'].get("FI01120389"))
    
    # Step 5: Analyze innovation network
    analysis_results = analyze_innovation_network(consolidated_graph)
    
    # Step 6: Visualize network

    visualize_network_tufte(analysis_results)
    
    # Save predicted entities and relationships for evaluation
    os.makedirs("evaluation", exist_ok=True)
    
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
    pred_entities_path = os.path.join("evaluation", "pred_entities.json")
    pred_relations_path = os.path.join("evaluation", "pred_relations.json")
    
    with open(pred_entities_path, "w", encoding="utf-8") as f:
        json.dump(unique_pred_entities, f, ensure_ascii=False, indent=2)
    
    with open(pred_relations_path, "w", encoding="utf-8") as f:
        json.dump(unique_pred_relations, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(unique_pred_entities)} unique predicted entities to {pred_entities_path}")
    print(f"Saved {len(unique_pred_relations)} unique predicted relations to {pred_relations_path}")
    
    # Step 7: Export results
    export_results(analysis_results, consolidated_graph, canonical_mapping)
    
    # Step 8: Run evaluation if not skipped
    if not args.skip_eval:
        # Convert consolidated_graph to Node and Relationship objects for evaluation
        from evaluation import run_all_evaluations
        
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
            data_dir=DATA_DIR,
            results_dir=RESULTS_DIR,
            eval_dir="evaluation",
            auto_label=args.auto_label,
            llm=llm  # 传递语言模型以便自动标注
        )
    
    print("Innovation Resolution process completed successfully!")
    print(f"Results and visualizations saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main() 
