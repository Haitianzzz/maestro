"""Maestro scheduler package (spec 04).

Only the DAG primitives are wired up at this module layer. The live
``Scheduler`` class (spec 04 §3) is built in module 9.
"""

from .dag import (
    DAGError,
    defer_lower_priority_on_conflicts,
    detect_write_conflicts,
    topological_batches,
)

__all__ = [
    "DAGError",
    "defer_lower_priority_on_conflicts",
    "detect_write_conflicts",
    "topological_batches",
]
