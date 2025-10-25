"""
Core package for the Innovation Platform application.

This package exposes the primary pipeline modules so downstream code can use
them via imports like ``from innovation_platform import innovation_resolution``.
"""

from . import innovation_resolution  # re-export for convenience
from . import innovation_utils
from . import local_entity_processing
from .query_engine import InnovationQueryEngine, QueryResult
from .pipeline_runner import PipelineRunner

__all__ = [
    "innovation_resolution",
    "innovation_utils",
    "local_entity_processing",
    "InnovationQueryEngine",
    "QueryResult",
    "PipelineRunner",
]
