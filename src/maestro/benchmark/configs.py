"""Benchmark configurations — the ablation matrix (spec 09 §4, spec 10 §2).

A config is a pure-data description of how to construct the Maestro
pipeline for one run: ``max_parallel``, which verifier tiers are enabled,
and any parameter overrides the experiment needs. The harness then
builds an ``OrchestratorDeps`` and runs the same LangGraph.

The enumeration here corresponds 1:1 to the experiment IDs in spec 10 §2:

* ``baseline``        — E-ref: sequential, no verify
* ``verify_t1``       — E3:   sequential, deterministic only
* ``verify_t12``      — E4:   sequential, T1 + T2
* ``verify_all``      — E5:   sequential, T1 + T2 + T3
* ``parallel_only``   — E2:   parallel, no verify
* ``parallel_verify_t1``  — bridge
* ``parallel_verify_t12`` — bridge
* ``full``            — E6:   parallel + full verify (Maestro's headline)

Judge-sensitivity sweep (J1..J4) is represented as ``JudgeOverrides``
layered on top of ``full``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ConfigName = Literal[
    "baseline",
    "verify_t1",
    "verify_t12",
    "verify_all",
    "parallel_only",
    "parallel_verify_t1",
    "parallel_verify_t12",
    "full",
]


@dataclass(frozen=True)
class JudgeOverrides:
    """Override knobs for the LLM-Judge (spec 10 §2.2)."""

    samples: int = 3
    disagreement_threshold: float = 0.3


@dataclass(frozen=True)
class BenchConfig:
    """Declarative description of one benchmark config."""

    name: ConfigName
    max_parallel: int
    enabled_tiers: frozenset[str]
    max_retries: int = 2
    judge: JudgeOverrides = field(default_factory=JudgeOverrides)

    @property
    def is_parallel(self) -> bool:
        return self.max_parallel > 1


_BASE_TIERS: frozenset[str] = frozenset({"deterministic", "test_based", "llm_judge"})


CONFIGS: dict[ConfigName, BenchConfig] = {
    "baseline": BenchConfig(
        name="baseline",
        max_parallel=1,
        enabled_tiers=frozenset(),
    ),
    "verify_t1": BenchConfig(
        name="verify_t1",
        max_parallel=1,
        enabled_tiers=frozenset({"deterministic"}),
    ),
    "verify_t12": BenchConfig(
        name="verify_t12",
        max_parallel=1,
        enabled_tiers=frozenset({"deterministic", "test_based"}),
    ),
    "verify_all": BenchConfig(
        name="verify_all",
        max_parallel=1,
        enabled_tiers=_BASE_TIERS,
    ),
    "parallel_only": BenchConfig(
        name="parallel_only",
        max_parallel=4,
        enabled_tiers=frozenset(),
    ),
    "parallel_verify_t1": BenchConfig(
        name="parallel_verify_t1",
        max_parallel=4,
        enabled_tiers=frozenset({"deterministic"}),
    ),
    "parallel_verify_t12": BenchConfig(
        name="parallel_verify_t12",
        max_parallel=4,
        enabled_tiers=frozenset({"deterministic", "test_based"}),
    ),
    "full": BenchConfig(
        name="full",
        max_parallel=4,
        enabled_tiers=_BASE_TIERS,
    ),
}


def get_config(name: str) -> BenchConfig:
    """Return the :class:`BenchConfig` for ``name``. Raises ``KeyError`` if unknown."""
    cfg: BenchConfig | None = CONFIGS.get(name)  # type: ignore[call-overload]
    if cfg is None:
        raise KeyError(f"Unknown benchmark config {name!r}. Known: {sorted(CONFIGS)}")
    return cfg


def all_config_names() -> list[str]:
    return sorted(CONFIGS)


__all__ = [
    "CONFIGS",
    "BenchConfig",
    "ConfigName",
    "JudgeOverrides",
    "all_config_names",
    "get_config",
]
