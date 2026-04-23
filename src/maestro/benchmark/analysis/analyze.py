"""Cross-run analysis for benchmark results (spec 10 §5).

Given a set of :class:`BenchmarkReport` objects (one per config), this
module produces the summary artifacts the REPORT.md consumes:

* :func:`compute_speedups` — wall-clock speedup vs the declared baseline.
* :func:`compute_pareto_frontier` — ``(config, cost, resolve_rate)`` for
  the Pareto plot (spec 10 §5.2 Figure 3).
* :func:`task_win_matrix` — ``task x config`` resolve matrix.
* :func:`render_comparison_markdown` — emits the Markdown block
  ``benchmark/results/COMPARISON.md`` includes.

The heavy visualisation (matplotlib PNGs) lives in downstream scripts
Week 6 Day 2 writes; this module stays plot-free so it can run inside
`pytest` without a display dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import BenchmarkReport, TaskBenchmarkResult


@dataclass(frozen=True)
class ParetoPoint:
    """One point on the cost-vs-resolve Pareto scatter."""

    config_name: str
    avg_cost: float
    resolve_rate: float
    dominated: bool


class ExperimentAnalyzer:
    """Convenience wrapper holding ``{config_name: report}`` in one place."""

    def __init__(self, reports: dict[str, BenchmarkReport]) -> None:
        self._reports = reports

    @property
    def reports(self) -> dict[str, BenchmarkReport]:
        return self._reports

    def compute_speedups(self, baseline: str = "baseline") -> dict[str, float]:
        return compute_speedups(self._reports, baseline=baseline)

    def compute_pareto(self) -> list[ParetoPoint]:
        return compute_pareto_frontier(self._reports)

    def task_win_matrix(self) -> dict[str, dict[str, bool]]:
        """Return ``{task_id: {config_name: resolved}}``."""
        matrix: dict[str, dict[str, bool]] = {}
        for config_name, report in self._reports.items():
            for r in report.per_task_results:
                matrix.setdefault(r.task_id, {})[config_name] = r.resolved
        return matrix

    def render_comparison(self, *, baseline: str = "baseline") -> str:
        return render_comparison_markdown(self._reports, baseline=baseline)


# ---------------------------------------------------------------------------
# Pure functions (exposed so callers can compose with their own data)
# ---------------------------------------------------------------------------


def compute_speedups(
    reports: dict[str, BenchmarkReport], *, baseline: str = "baseline"
) -> dict[str, float]:
    """Return ``{config_name: wall_clock_speedup_vs_baseline}``.

    Missing baseline (or zero wall-clock) yields an empty dict rather than
    a ZeroDivisionError — this keeps the analysis robust when the baseline
    run was skipped or aborted.
    """
    if baseline not in reports:
        return {}
    base_ms = reports[baseline].avg_wall_clock_ms
    if base_ms <= 0:
        return {name: 1.0 for name in reports}
    return {
        name: (base_ms / r.avg_wall_clock_ms) if r.avg_wall_clock_ms > 0 else 0.0
        for name, r in reports.items()
    }


def compute_pareto_frontier(
    reports: dict[str, BenchmarkReport],
) -> list[ParetoPoint]:
    """Classify each report as on / off the cost↔resolve Pareto frontier.

    A point ``P`` is dominated if there exists another ``Q`` with
    ``Q.cost <= P.cost`` AND ``Q.resolve_rate >= P.resolve_rate`` AND at
    least one of those inequalities is strict. Sorted by cost ascending
    for stable plotting.
    """
    points = [
        ParetoPoint(
            config_name=name,
            avg_cost=r.avg_cost,
            resolve_rate=r.resolve_rate,
            dominated=False,
        )
        for name, r in reports.items()
    ]
    # Dominance check — quadratic but N <= ~10 configs, negligible.
    classified: list[ParetoPoint] = []
    for p in points:
        dominated = any(
            (q.avg_cost <= p.avg_cost and q.resolve_rate >= p.resolve_rate)
            and (q.avg_cost < p.avg_cost or q.resolve_rate > p.resolve_rate)
            for q in points
            if q.config_name != p.config_name
        )
        classified.append(
            ParetoPoint(
                config_name=p.config_name,
                avg_cost=p.avg_cost,
                resolve_rate=p.resolve_rate,
                dominated=dominated,
            )
        )
    classified.sort(key=lambda p: p.avg_cost)
    return classified


def render_comparison_markdown(
    reports: dict[str, BenchmarkReport], *, baseline: str = "baseline"
) -> str:
    """Markdown digest: resolve / cost / speedup / Pareto per config."""
    speedups = compute_speedups(reports, baseline=baseline)
    pareto = {p.config_name: p for p in compute_pareto_frontier(reports)}

    lines = [
        "# Maestro benchmark — cross-config comparison",
        "",
        f"Baseline for speedup: `{baseline}`",
        "",
        "| Config | Resolve rate | Avg cost | Avg tokens (in/out) | "
        "Avg wall-clock (s) | Speedup | Pareto |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for name in sorted(reports):
        r = reports[name]
        speedup = speedups.get(name)
        speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
        badge = "frontier" if not pareto[name].dominated else "dominated"
        lines.append(
            f"| {name} | {r.resolve_rate:.1%} | {r.avg_cost:.4f} | "
            f"{r.avg_tokens_input:.0f}/{r.avg_tokens_output:.0f} | "
            f"{r.avg_wall_clock_ms / 1000:.1f} | {speedup_str} | {badge} |"
        )
    lines.append("")
    lines.append(_task_win_matrix_markdown(reports))
    return "\n".join(lines) + "\n"


def _task_win_matrix_markdown(reports: dict[str, BenchmarkReport]) -> str:
    all_tasks: list[str] = sorted(
        {r.task_id for report in reports.values() for r in report.per_task_results}
    )
    if not all_tasks:
        return ""
    header = ["| task | " + " | ".join(sorted(reports)) + " |"]
    sep = ["|---|" + "---|" * len(reports)]
    rows: list[str] = []
    per_task: dict[str, dict[str, TaskBenchmarkResult]] = {}
    for name, report in reports.items():
        for r in report.per_task_results:
            per_task.setdefault(r.task_id, {})[name] = r
    for task in all_tasks:
        cells = []
        for cfg in sorted(reports):
            entry = per_task.get(task, {}).get(cfg)
            cells.append("✔" if entry and entry.resolved else "✘")
        rows.append(f"| {task} | " + " | ".join(cells) + " |")
    return "\n".join(["## Task x config win matrix", "", *header, *sep, *rows])


__all__ = [
    "ExperimentAnalyzer",
    "ParetoPoint",
    "compute_pareto_frontier",
    "compute_speedups",
    "render_comparison_markdown",
]
