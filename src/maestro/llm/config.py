"""LLM client configuration (spec 02 §2).

Config is loaded from a YAML file with ``${ENV_VAR}`` substitution. The default
location is ``~/.maestro/config.yaml`` and can be overridden via the
``MAESTRO_CONFIG`` environment variable.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import LLMConfigError

Currency = Literal["USD", "RMB"]
Role = str  # "planner" / "subagent" / "judge" — kept open for future roles.

DEFAULT_CONFIG_PATH = Path.home() / ".maestro" / "config.yaml"
_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ModelConfig(BaseModel):
    """Config for a single LLM model used under a given role.

    ``price_input_per_mtok`` / ``price_output_per_mtok`` are in the currency
    specified by ``ClientConfig.currency``. The fields intentionally carry the
    ``_mtok`` suffix (price per million tokens) regardless of currency — this
    is the unit most pricing pages publish in.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    display_name: str
    price_input_per_mtok: float = Field(
        ge=0.0,
        description="Price per million input tokens, in the currency specified by ClientConfig.currency.",
    )
    price_output_per_mtok: float = Field(
        ge=0.0,
        description="Price per million output tokens, in the currency specified by ClientConfig.currency.",
    )
    max_tokens_output: int = Field(default=4096, ge=1)
    supports_structured_output: bool = True


class ClientConfig(BaseModel):
    """Top-level configuration for the Maestro LLM client."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key: str
    models: dict[Role, ModelConfig]
    default_timeout_seconds: int = Field(default=60, ge=1)
    max_retries: int = Field(default=3, ge=0)
    global_semaphore_limit: int = Field(default=10, ge=1)
    currency: Currency = "RMB"

    def get_model(self, role: Role) -> ModelConfig:
        """Return the model bound to ``role``, raising ``LLMConfigError`` if absent."""
        try:
            return self.models[role]
        except KeyError as exc:
            raise LLMConfigError(
                f"No model configured for role {role!r}. Configured roles: {sorted(self.models)}"
            ) from exc


def _expand_env_vars(value: object) -> object:
    """Recursively expand ``${VAR}`` patterns against os.environ."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            try:
                return os.environ[name]
            except KeyError as exc:
                raise LLMConfigError(
                    f"Environment variable {name!r} referenced in config is not set"
                ) from exc

        return _ENV_VAR_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def resolve_config_path(override: Path | None = None) -> Path:
    """Resolve which config file to load (CLI arg > env var > default)."""
    if override is not None:
        return override
    env = os.getenv("MAESTRO_CONFIG")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path | None = None) -> ClientConfig:
    """Load a ``ClientConfig`` from YAML with env-var substitution."""
    resolved = resolve_config_path(path)
    if not resolved.exists():
        raise LLMConfigError(f"Config file not found: {resolved}")

    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LLMConfigError(f"Config file {resolved} must contain a mapping at root")

    expanded = _expand_env_vars(raw)
    try:
        return ClientConfig.model_validate(expanded)
    except Exception as exc:
        raise LLMConfigError(f"Invalid Maestro config in {resolved}: {exc}") from exc


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ClientConfig",
    "Currency",
    "ModelConfig",
    "Role",
    "load_config",
    "resolve_config_path",
]
