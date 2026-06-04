"""Level 3: LLM-as-judge evals against the real Anthropic API.

These tests grade the *semantic* properties of the agent's output -- things
Level 2's structural assertions can't see: factual hallucination, topic drift,
duplicative findings, and editorialising. A separate, more capable model
(``claude-sonnet-4-6``) judges the Haiku agent against a four-dimension
pass/fail rubric.

Search is stubbed with the same curated payloads Level 2 uses, so a judge fail
points squarely at the agent's model + prompt, not at day-to-day Tavily
variability. Both agent and judge run at temperature=0 to keep the signal as
deterministic as the underlying API permits.

The whole module skips when ``ANTHROPIC_API_KEY`` is unset so attendees can
run Levels 1 and 2 without burning Sonnet credits.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
import pytest

from agent.agent import summarize
from agent.models import SummaryResult
from agent.tools import SearchResult, StubSearchTool
from tests.conftest import LEVEL3_USAGE
from tests.test_level2 import _GREAT_FIRE_RESULTS, _PHOTOSYNTHESIS_RESULTS


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; Level 3 requires a live Anthropic key",
)


JUDGE_MODEL = "claude-sonnet-4-6"
_JUDGE_TOOL_NAME = "submit_verdicts"
_JUDGE_MAX_TOKENS = 1024

_DIMENSIONS: tuple[str, ...] = (
    "factual_accuracy",
    "on_topic_relevance",
    "finding_distinctness",
    "no_opinions_or_recommendations",
)

_JUDGE_SYSTEM_PROMPT = """You are an evaluation judge for a research summarizer agent. You will be given:
- A research topic
- The exact search results that were provided to the agent
- The agent's structured output (topic, synopsis, key findings, citations)

Grade the agent's output on four semantic dimensions, each as "pass" or "fail".
Return your verdicts by calling the submit_verdicts tool. Do not return text."""

_JUDGE_RUBRIC = """Grade the following four dimensions:

1. factual_accuracy
   Every claim in the synopsis and key findings must be supported by, or
   consistent with, the provided search-result snippets. Inventing facts or
   contradicting the snippets is a fail.

2. on_topic_relevance
   The synopsis and findings must address the requested topic, not a
   tangentially related one. Drift to a neighbouring subject is a fail.

3. finding_distinctness
   The 2-5 key findings must be genuinely distinct points. Findings that
   rephrase each other, or that trivially restate the synopsis, are a fail.

4. no_opinions_or_recommendations
   The output must stay descriptive. Editorialising, advice, or value
   judgements (e.g. "this is important", "you should...") are a fail.

For each dimension, include one short sentence of justification in the
comments field."""


def _verdict_tool_schema() -> dict[str, Any]:
    verdict = {"type": "string", "enum": ["pass", "fail"]}
    return {
        "type": "object",
        "properties": {
            **{dim: verdict for dim in _DIMENSIONS},
            "comments": {
                "type": "string",
                "description": "One short sentence per dimension explaining the verdict.",
            },
        },
        "required": [*_DIMENSIONS, "comments"],
    }


@pytest.fixture(autouse=True)
def _pin_temperature_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMARIZER_TEMPERATURE", "0")


@pytest.fixture(autouse=True)
def _capture_anthropic_usage(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Wrap ``anthropic.Anthropic`` so we record token usage for every call,
    tagging the row as ``"agent"`` (Haiku) or ``"judge"`` (Sonnet) based on
    the requested model. Both code paths import ``anthropic`` independently,
    so we patch the symbol on both modules.
    """
    real_anthropic_cls = anthropic.Anthropic
    test_id = request.node.nodeid

    class _CapturingMessages:
        def __init__(self, real: Any) -> None:
            self._real = real

        def create(self, **kwargs: Any) -> Any:
            response = self._real.create(**kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                role = "judge" if kwargs.get("model") == JUDGE_MODEL else "agent"
                LEVEL3_USAGE.append(
                    (test_id, role, int(usage.input_tokens), int(usage.output_tokens))
                )
            return response

    class _CapturingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._real = real_anthropic_cls(*args, **kwargs)
            self.messages = _CapturingMessages(self._real.messages)

    monkeypatch.setattr("agent.agent.anthropic.Anthropic", _CapturingClient)
    # The judge in this module instantiates anthropic.Anthropic directly,
    # so patch the module symbol too.
    monkeypatch.setattr("anthropic.Anthropic", _CapturingClient)


def _build_judge_user_message(
    topic: str, search_results: list[SearchResult], agent_result: SummaryResult
) -> str:
    return (
        f"Topic: {topic}\n\n"
        f"Search results provided to the agent:\n"
        f"{json.dumps([r.model_dump() for r in search_results], indent=2)}\n\n"
        f"Agent output:\n"
        f"{json.dumps(agent_result.model_dump(), indent=2)}\n\n"
        f"{_JUDGE_RUBRIC}"
    )


def _grade(
    topic: str, search_results: list[SearchResult], agent_result: SummaryResult
) -> tuple[dict[str, str], str]:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=_JUDGE_MAX_TOKENS,
        temperature=0,
        system=_JUDGE_SYSTEM_PROMPT,
        tools=[
            {
                "name": _JUDGE_TOOL_NAME,
                "description": "Submit the four pass/fail verdicts and a short justification.",
                "input_schema": _verdict_tool_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": _JUDGE_TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": _build_judge_user_message(topic, search_results, agent_result),
            }
        ],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == _JUDGE_TOOL_NAME:
            payload = block.input
            verdicts = {dim: payload[dim] for dim in _DIMENSIONS}
            comments = payload.get("comments", "")
            return verdicts, comments
    raise RuntimeError(
        f"Judge response did not include a {_JUDGE_TOOL_NAME} tool_use block"
    )


_TOPIC_FIXTURES: list[tuple[str, list[SearchResult]]] = [
    ("photosynthesis", _PHOTOSYNTHESIS_RESULTS),
    ("the Great Fire of London", _GREAT_FIRE_RESULTS),
]


@pytest.mark.parametrize(
    ("topic", "stub_results"),
    _TOPIC_FIXTURES,
    ids=[t for t, _ in _TOPIC_FIXTURES],
)
def test_level3_judge_grades_agent_output(
    topic: str, stub_results: list[SearchResult]
) -> None:
    agent_result = summarize(
        topic, search_tool=StubSearchTool(results=list(stub_results))
    )
    assert isinstance(agent_result, SummaryResult), (
        f"[{topic}]: expected SummaryResult, got {type(agent_result).__name__}"
    )

    verdicts, comments = _grade(topic, stub_results, agent_result)

    for dim in _DIMENSIONS:
        assert verdicts[dim] == "pass", (
            f"[{topic}] {dim} = {verdicts[dim]!r}. "
            f"Judge comments: {comments}. "
            f"Agent output: {agent_result.model_dump()}"
        )
