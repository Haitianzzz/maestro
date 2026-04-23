"""Unit tests for ``benchmark.configs`` (spec 09 §4, spec 10 §2)."""

from __future__ import annotations

import pytest

from maestro.benchmark.configs import (
    CONFIGS,
    JudgeOverrides,
    all_config_names,
    get_config,
)


def test_known_configs_cover_ablation_matrix() -> None:
    # Spec 10 §2 requires these exact names to be present.
    required = {
        "baseline",
        "verify_t1",
        "verify_t12",
        "verify_all",
        "parallel_only",
        "full",
    }
    assert required.issubset(CONFIGS.keys())


def test_full_is_parallel_with_all_tiers() -> None:
    cfg = get_config("full")
    assert cfg.is_parallel is True
    assert cfg.enabled_tiers == frozenset({"deterministic", "test_based", "llm_judge"})


def test_baseline_is_serial_with_no_verify() -> None:
    cfg = get_config("baseline")
    assert cfg.max_parallel == 1
    assert cfg.is_parallel is False
    assert cfg.enabled_tiers == frozenset()


def test_parallel_only_enables_parallel_without_verify() -> None:
    cfg = get_config("parallel_only")
    assert cfg.is_parallel is True
    assert cfg.enabled_tiers == frozenset()


def test_get_config_unknown_raises() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        get_config("no_such_config")


def test_all_config_names_is_sorted() -> None:
    names = all_config_names()
    assert names == sorted(names)


def test_judge_overrides_defaults_match_spec() -> None:
    j = JudgeOverrides()
    assert j.samples == 3
    assert j.disagreement_threshold == 0.3
