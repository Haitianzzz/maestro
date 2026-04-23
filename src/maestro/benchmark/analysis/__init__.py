"""Benchmark analysis helpers (spec 10 §5)."""

from .analyze import (
    ExperimentAnalyzer,
    compute_pareto_frontier,
    compute_speedups,
    render_comparison_markdown,
)

__all__ = [
    "ExperimentAnalyzer",
    "compute_pareto_frontier",
    "compute_speedups",
    "render_comparison_markdown",
]
