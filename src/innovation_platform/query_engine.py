#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Innovation Query Engine

Provides lightweight semantic search over the consolidated innovation graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data_pipeline.processors import EmbeddingManager, get_embedding


def _join(values: Sequence[str], separator: str = "; ") -> str:
    """Join a sequence while skipping empty items."""
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    return separator.join(cleaned)


def _render_innovation_text(
    innovation: Dict,
    organizations: Dict[str, Dict],
) -> str:
    """Create a descriptive text used for embedding a canonical innovation."""
    names = _join(list(innovation.get("names", []))) or innovation["id"]
    descriptions = _join(list(innovation.get("descriptions", [])), " ")
    data_sources = _join(list(innovation.get("data_sources", [])))

    org_ids = innovation.get("developed_by", set())
    org_names = [
        organizations[org_id]["name"]
        if org_id in organizations and organizations[org_id]["name"]
        else org_id
        for org_id in org_ids
    ]
    developed_by = _join(org_names)

    parts = [
        f"Innovation: {names}.",
    ]
    if descriptions:
        parts.append(f"Description: {descriptions}.")
    if developed_by:
        parts.append(f"Developed by: {developed_by}.")
    if data_sources:
        parts.append(f"Data sources: {data_sources}.")
    return " ".join(parts)


@dataclass
class QueryResult:
    """Container for a single search hit."""

    innovation_id: str
    score: float
    names: List[str]
    organizations: List[str]
    descriptions: List[str]


class InnovationQueryEngine:
    """
    Semantic search over consolidated innovations.

    Builds text representations for each canonical innovation, embeds them (with
    caching), and provides cosine-similarity search.
    """

    def __init__(
        self,
        consolidated_graph: Dict,
        embedding_model=None,
        cache_config: Optional[Dict] = None,
    ) -> None:
        self.consolidated_graph = consolidated_graph
        self.embedding_model = embedding_model
        self.cache_config = cache_config or {
            "type": "embedding",
            "backend": "json",
            "path": "./query_embeddings.json",
            "use_cache": True,
        }

        self._innovation_texts = self._build_innovation_texts()
        self._ids, self._matrix = self._embed_corpus()
        self._norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        self._norms[self._norms == 0] = 1.0  # avoid divide-by-zero

    def _build_innovation_texts(self) -> Dict[str, str]:
        """Construct textual representations for each canonical innovation."""
        innovations = self.consolidated_graph.get("innovations", {})
        organizations = self.consolidated_graph.get("organizations", {})

        texts: Dict[str, str] = {}
        for canonical_id, innovation in innovations.items():
            texts[canonical_id] = _render_innovation_text(innovation, organizations)
        return texts

    def _embed_corpus(self) -> Tuple[List[str], np.ndarray]:
        """Embed every innovation text with caching support."""
        if not self._innovation_texts:
            return [], np.empty((0, 1536))

        manager = EmbeddingManager.create_from_config(
            cache_config=self.cache_config,
            embedding_function=get_embedding,
        )
        ids, matrix = manager.get_embeddings(self._innovation_texts, self.embedding_model)
        return ids, matrix

    def search(self, query: str, top_k: int = 5) -> List[QueryResult]:
        """
        Search for the most relevant innovations for a free-text query.

        Args:
            query: User query text.
            top_k: Number of hits to return.

        Returns:
            A list of QueryResult objects ordered by similarity (highest first).
        """
        if not query or not self._ids:
            return []

        query_embedding = get_embedding(query, self.embedding_model).reshape(1, -1)
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        scores = (self._matrix @ query_embedding.T).reshape(-1, 1) / (
            self._norms * query_norm
        )
        scores = scores.ravel()

        top_indices = np.argsort(scores)[::-1][:top_k]
        innovations = self.consolidated_graph.get("innovations", {})
        organizations = self.consolidated_graph.get("organizations", {})

        results: List[QueryResult] = []
        for idx in top_indices:
            canonical_id = self._ids[idx]
            score = float(scores[idx])
            innovation = innovations.get(canonical_id, {})

            org_ids = list(innovation.get("developed_by", []))
            org_names = [
                organizations[oid]["name"]
                if oid in organizations and organizations[oid]["name"]
                else oid
                for oid in org_ids
            ]

            results.append(
                QueryResult(
                    innovation_id=canonical_id,
                    score=score,
                    names=sorted(innovation.get("names", [])),
                    organizations=org_names,
                    descriptions=sorted(innovation.get("descriptions", [])),
                )
            )
        return results


__all__ = ["InnovationQueryEngine", "QueryResult"]
