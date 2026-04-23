"""Tests for ``maestro.llm.config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from maestro.llm.config import ClientConfig, load_config
from maestro.llm.errors import LLMConfigError


def _write_config(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_load_config_expands_env_vars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-123")
    cfg_path = tmp_path / "config.yaml"
    _write_config(
        cfg_path,
        """
base_url: "https://example.com/v1"
api_key: "${TEST_API_KEY}"
currency: "RMB"
models:
  planner:
    name: qwen3-max
    display_name: Qwen3-Max
    price_input_per_mtok: 2.8
    price_output_per_mtok: 8.4
  judge:
    name: deepseek-v3
    display_name: DeepSeek-V3
    price_input_per_mtok: 0.28
    price_output_per_mtok: 1.12
""",
    )
    cfg = load_config(cfg_path)
    assert cfg.api_key == "secret-123"
    assert cfg.currency == "RMB"
    assert cfg.models["planner"].name == "qwen3-max"


def test_load_config_missing_env_var_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("NEVER_SET_VAR", raising=False)
    cfg_path = tmp_path / "config.yaml"
    _write_config(
        cfg_path,
        """
base_url: "https://example.com"
api_key: "${NEVER_SET_VAR}"
models: {}
""",
    )
    with pytest.raises(LLMConfigError, match="NEVER_SET_VAR"):
        load_config(cfg_path)


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write_config(cfg_path, "- just\n- a\n- list\n")
    with pytest.raises(LLMConfigError, match="mapping at root"):
        load_config(cfg_path)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_get_model_raises_on_unknown_role() -> None:
    cfg = ClientConfig(
        base_url="https://x",
        api_key="k",
        models={},
    )
    with pytest.raises(LLMConfigError, match="No model configured"):
        cfg.get_model("planner")


def test_default_currency_is_rmb() -> None:
    cfg = ClientConfig(base_url="https://x", api_key="k", models={})
    # M5: benchmark/experiments are budgeted in RMB.
    assert cfg.currency == "RMB"
