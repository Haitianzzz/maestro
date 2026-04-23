"""Unit tests for ``maestro.planner.planner`` (spec 03 §7)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from maestro.llm.client import LLMCallMetadata, LLMClient
from maestro.llm.config import ClientConfig, ModelConfig
from maestro.models import (
    PlannerLLMOutput,
    PlannerLLMSubTask,
    TaskDAG,
    TaskSpec,
    generate_subtask_id,
    generate_task_id,
)
from maestro.planner.planner import Planner, PlanningError

_FIXTURE_REPO = Path(__file__).parent.parent / "fixtures" / "tiny_flask_app"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> LLMClient:
    cfg = ClientConfig(
        base_url="https://example.com/v1",
        api_key="fake",
        models={
            "planner": ModelConfig(
                name="qwen3-max",
                display_name="Qwen3-Max",
                price_input_per_mtok=2.8,
                price_output_per_mtok=8.4,
            ),
        },
    )
    return LLMClient(cfg)


def _fake_metadata() -> LLMCallMetadata:
    return LLMCallMetadata(
        model_name="qwen3-max",
        role="planner",
        tokens_input=10,
        tokens_output=10,
        latency_ms=100,
        cost=0.01,
        currency="RMB",
        called_at=datetime(2026, 4, 23, tzinfo=UTC),
        success=True,
        http_retry_count=0,
    )


def _install_responder(
    client: LLMClient, responder: Callable[..., Awaitable[tuple[Any, LLMCallMetadata]]]
) -> AsyncMock:
    mock = AsyncMock(side_effect=responder)
    client.call_structured = mock  # type: ignore[assignment,method-assign]
    return mock


def _task_spec(
    description: str = "Add user authentication",
    target_files_hint: list[str] | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=generate_task_id(),
        description=description,
        repo_path=_FIXTURE_REPO,
        target_files_hint=target_files_hint,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


async def test_plan_single_subtask() -> None:
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Tweak the landing page copy",
                    reads=["src/app.py"],
                    writes=["src/app.py"],
                    depends_on_indices=[],
                    priority=1,
                ),
            ],
            global_context="Small tweak.",
            planning_rationale="Only one change needed.",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)

    spec = _task_spec("Tweak the landing copy")
    planner = Planner(client)
    dag = await planner.plan(spec)
    assert isinstance(dag, TaskDAG)
    assert dag.task_id == spec.task_id
    assert len(dag.subtasks) == 1
    assert dag.subtasks[0].subtask_id == generate_subtask_id(spec.task_id, 0)
    assert dag.subtasks[0].writes == ["src/app.py"]


async def test_post_process_handles_non_consecutive_indices() -> None:
    """Protect the index→ordinal mapping in ``_post_process``.

    The LLM may emit any set of integers for ``index`` / ``depends_on_indices``
    (``[0, 5, 100]`` here). After post-processing, subtask ids must be the
    canonical consecutive ``{task_id}-NNN`` sequence and ``depends_on`` must
    be translated to those canonical ids, never carry through the raw integer.
    """
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="First",
                    writes=["src/app.py"],
                ),
                PlannerLLMSubTask(
                    index=5,
                    description="Second",
                    writes=["src/db.py"],
                ),
                PlannerLLMSubTask(
                    index=100,
                    description="Third",
                    writes=["src/models/user.py"],
                    depends_on_indices=[5],
                ),
            ],
            global_context="non-consecutive indices",
            planning_rationale="",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    spec = _task_spec()
    dag = await Planner(client).plan(spec)

    assert len(dag.subtasks) == 3
    assert dag.subtasks[0].subtask_id == generate_subtask_id(spec.task_id, 0)
    assert dag.subtasks[1].subtask_id == generate_subtask_id(spec.task_id, 1)
    assert dag.subtasks[2].subtask_id == generate_subtask_id(spec.task_id, 2)
    # depends_on references ordinal 1 (== LLM index 5), not the literal "5".
    assert dag.subtasks[2].depends_on == [generate_subtask_id(spec.task_id, 1)]


async def test_plan_multi_subtask_with_dependencies() -> None:
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Implement signup endpoint",
                    reads=["src/models/user.py", "src/db.py"],
                    writes=["src/auth/signup.py"],
                    priority=2,
                ),
                PlannerLLMSubTask(
                    index=1,
                    description="Implement login endpoint",
                    reads=["src/models/user.py", "src/db.py"],
                    writes=["src/auth/login.py"],
                    priority=2,
                ),
                PlannerLLMSubTask(
                    index=2,
                    description="Register auth routes",
                    reads=["src/app.py"],
                    writes=["src/app.py"],
                    depends_on_indices=[0, 1],
                    priority=3,
                ),
            ],
            global_context="Add auth",
            planning_rationale="Two independent endpoints + one registrar.",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    spec = _task_spec()
    dag = await Planner(client).plan(spec)
    assert len(dag.subtasks) == 3
    # IDs are assigned in index order and depends_on is rewritten.
    assert dag.subtasks[2].depends_on == [
        generate_subtask_id(spec.task_id, 0),
        generate_subtask_id(spec.task_id, 1),
    ]
    # DAG validates.
    dag.validate_dag()


# ---------------------------------------------------------------------------
# Retry + error handling
# ---------------------------------------------------------------------------


async def test_plan_retries_when_path_missing() -> None:
    client = _make_client()

    attempts: list[int] = []

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        attempts.append(1)
        if len(attempts) == 1:
            out = PlannerLLMOutput(
                subtasks=[
                    PlannerLLMSubTask(
                        index=0,
                        description="Edit missing file",
                        reads=["src/does_not_exist.py"],
                        writes=["src/app.py"],
                    ),
                ],
                global_context="bad plan",
                planning_rationale="nope",
            )
        else:
            out = PlannerLLMOutput(
                subtasks=[
                    PlannerLLMSubTask(
                        index=0,
                        description="Edit existing",
                        reads=["src/app.py"],
                        writes=["src/app.py"],
                    ),
                ],
                global_context="good plan",
                planning_rationale="fixed path",
            )
        return out, _fake_metadata()

    mock = _install_responder(client, responder)
    dag = await Planner(client).plan(_task_spec())
    assert mock.call_count == 2
    assert dag.subtasks[0].reads == ["src/app.py"]


async def test_plan_retries_when_cycle_detected() -> None:
    client = _make_client()

    attempts: list[int] = []

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        attempts.append(1)
        if len(attempts) == 1:
            out = PlannerLLMOutput(
                subtasks=[
                    PlannerLLMSubTask(
                        index=0,
                        description="a",
                        writes=["src/app.py"],
                        depends_on_indices=[1],
                    ),
                    PlannerLLMSubTask(
                        index=1,
                        description="b",
                        writes=["src/db.py"],
                        depends_on_indices=[0],
                    ),
                ],
                global_context="cycle",
                planning_rationale="",
            )
        else:
            out = PlannerLLMOutput(
                subtasks=[
                    PlannerLLMSubTask(index=0, description="a", writes=["src/app.py"]),
                    PlannerLLMSubTask(
                        index=1,
                        description="b",
                        writes=["src/db.py"],
                        depends_on_indices=[0],
                    ),
                ],
                global_context="ok",
                planning_rationale="",
            )
        return out, _fake_metadata()

    _install_responder(client, responder)
    dag = await Planner(client).plan(_task_spec())
    dag.validate_dag()


async def test_plan_raises_after_retry_exhaustion() -> None:
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Edit nonexistent",
                    reads=["src/ghost.py"],
                    writes=["src/ghost.py"],
                    depends_on_indices=[1],  # also references missing index
                ),
            ],
            global_context="bad",
            planning_rationale="bad",
        )
        return out, _fake_metadata()

    mock = _install_responder(client, responder)
    with pytest.raises(PlanningError):
        await Planner(client, max_retries=2).plan(_task_spec())
    assert mock.call_count == 3  # 1 initial + 2 retries


async def test_plan_rejects_empty_subtasks() -> None:
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[],
            global_context="nothing",
            planning_rationale="nothing",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    with pytest.raises(PlanningError):
        await Planner(client, max_retries=0).plan(_task_spec())


async def test_plan_rejects_forbidden_write_path() -> None:
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Pwn git",
                    writes=[".git/config"],
                ),
            ],
            global_context="bad",
            planning_rationale="bad",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    with pytest.raises(PlanningError):
        await Planner(client, max_retries=0).plan(_task_spec())


async def test_plan_allows_writes_not_yet_existing() -> None:
    """``writes`` paths don't need to exist yet — sub-agent may create them."""
    client = _make_client()

    async def responder(**_: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Create new auth module",
                    reads=["src/models/user.py"],
                    writes=["src/auth/signup.py"],
                ),
            ],
            global_context="new module",
            planning_rationale="create it",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    dag = await Planner(client).plan(_task_spec())
    assert dag.subtasks[0].writes == ["src/auth/signup.py"]


async def test_plan_target_files_hint_is_passed_into_prompt() -> None:
    """The planner prompt must include the user's target_files_hint."""
    client = _make_client()
    captured: dict[str, Any] = {}

    async def responder(**kwargs: Any) -> tuple[PlannerLLMOutput, LLMCallMetadata]:
        captured["messages"] = kwargs["messages"]
        out = PlannerLLMOutput(
            subtasks=[
                PlannerLLMSubTask(
                    index=0,
                    description="Edit hinted file",
                    reads=["src/app.py"],
                    writes=["src/app.py"],
                ),
            ],
            global_context="ok",
            planning_rationale="",
        )
        return out, _fake_metadata()

    _install_responder(client, responder)
    await Planner(client).plan(_task_spec(target_files_hint=["src/app.py"]))
    user_msg = captured["messages"][-1]["content"]
    assert "Files the user suggests you focus on" in user_msg
    assert "src/app.py" in user_msg
