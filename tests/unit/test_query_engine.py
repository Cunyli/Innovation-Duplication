#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unit tests / smoke tests for the semantic query engine."""

import json
from pathlib import Path

from innovation_platform.query_engine import InnovationQueryEngine


def _load_real_graph_or_sample():
    """Prefer the consolidated graph emitted by the main pipeline."""
    graph_path = Path("results/consolidated_graph.json")
    if graph_path.exists():
        with graph_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback sample if the real graph is not available yet
    return {
        "innovations": {
            "inno_ai": {
                "id": "inno_ai",
                "names": {"AI Platform"},
                "descriptions": {"Deep learning platform for industrial quality control."},
                "developed_by": {"org_tech"},
                "sources": {"https://example.com/ai"},
                "source_ids": {"I001"},
                "data_sources": {"company_website"},
            },
        },
        "organizations": {
            "org_tech": {
                "id": "org_tech",
                "name": "Tech Labs",
                "description": "AI specialists.",
            },
        },
        "relationships": [
            {"source": "inno_ai", "target": "org_tech", "type": "DEVELOPED_BY"},
        ],
    }


def test_query_returns_relevant_innovations_and_prints_sample():
    graph = _load_real_graph_or_sample()
    engine = InnovationQueryEngine(graph, embedding_model=None, cache_config={"use_cache": False})

    query = "renewable energy storage" if len(graph["innovations"]) > 1 else "industrial AI quality control"
    results = engine.search(query, top_k=3)

    assert results, "Expected at least one search hit"

    print("\n--- Query Engine Sample Output ---")
    print(f"Query: {query}")
    for hit in results:
        print(
            f"- {hit.innovation_id} | score={hit.score:.3f} | "
            f"names={hit.names[:1]} | organisations={hit.organizations[:2]}"
        )
