"""
Data pipeline utilities including loaders and processors.

These modules were previously under the top-level ``data`` namespace and are
now colocated with the rest of the application code.
"""

from . import loaders, processors

__all__ = ["loaders", "processors"]
