"""Level 1: deterministic unit tests for the agent's orchestration.

The search tool is injected (a real seam in production code). The Anthropic
client is monkeypatched at the module path the agent imports it through. These
tests run offline, make no API calls, and exercise plumbing only — input
validation, tool-call orchestration, error propagation, and response parsing.
Semantic quality belongs to Levels 2+.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from agent.agent import summarize
from agent.models import SummaryResult
from agent.tools import SearchResult, SearchToolError, StubSearchTool


def _tool_use_block(name: str, input_: dict[str, Any]) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="tool_use", name=name, input=input_)


def _text_block(text: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(type="text", text=text)


def _response(*blocks: types.SimpleNamespace) -> types.SimpleNamespace:
    return types.SimpleNamespace(content=list(blocks))


class _FakeMessages:
    def __init__(self, response: types.SimpleNamespace) -> None:
        self._response = response

    def create(self, **_: Any) -> types.SimpleNamespace:
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response: types.SimpleNamespace) -> None:
        self.messages = _FakeMessages(response)


def _patch_anthropic(
    monkeypatch: pytest.MonkeyPatch, response: types.SimpleNamespace
) -> None:
    def factory(*_args: Any, **_kwargs: Any) -> _FakeAnthropicClient:
        return _FakeAnthropicClient(response)

    monkeypatch.setattr("agent.agent.anthropic.Anthropic", factory)


@pytest.fixture(autouse=True)
def _block_unstubbed_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard against silent real-API calls. Tests that need an LLM response
    override this by calling ``_patch_anthropic()``."""

    def factory(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError(
            "anthropic.Anthropic was instantiated without a test stub; "
            "the test reached the LLM call path unexpectedly."
        )

    monkeypatch.setattr("agent.agent.anthropic.Anthropic", factory)


VALID_SUMMARY_INPUT: dict[str, Any] = {
    "topic": "photosynthesis",
    "synopsis": "Plants convert sunlight into chemical energy.",
    "key_findings": [
        "Chlorophyll absorbs light",
        "Produces oxygen as a byproduct",
    ],
    "citations": [
        {
            "title": "Wikipedia: Photosynthesis",
            "url": "https://en.wikipedia.org/wiki/Photosynthesis",
            "snippet": "Photosynthesis is a process used by plants...",
        }
    ],
}


@pytest.fixture
def one_search_result() -> list[SearchResult]:
    return [
        SearchResult(
            title="Wikipedia: Photosynthesis",
            url="https://en.wikipedia.org/wiki/Photosynthesis",
            snippet="Photosynthesis is a process used by plants...",
        )
    ]


@pytest.mark.parametrize("topic", ["", "   ", "\t\n"])
def test_empty_topic_raises_value_error(topic: str) -> None:
    with pytest.raises(ValueError, match="topic"):
        summarize(topic, search_tool=StubSearchTool(results=[]))


def test_search_tool_called_with_topic_and_max_results(
    monkeypatch: pytest.MonkeyPatch, one_search_result: list[SearchResult]
) -> None:
    calls: list[tuple[str, int]] = []

    class RecordingSearchTool:
        def search(
            self, query: str, max_results: int = 5
        ) -> list[SearchResult]:
            calls.append((query, max_results))
            return one_search_result

    _patch_anthropic(
        monkeypatch,
        _response(_tool_use_block("return_summary", VALID_SUMMARY_INPUT)),
    )

    summarize("  photosynthesis  ", search_tool=RecordingSearchTool())

    assert calls == [("photosynthesis", 5)]


def test_search_tool_error_propagates() -> None:
    class FailingSearchTool:
        def search(
            self, query: str, max_results: int = 5
        ) -> list[SearchResult]:
            raise SearchToolError("Tavily exploded")

    with pytest.raises(SearchToolError, match="Tavily exploded"):
        summarize("photosynthesis", search_tool=FailingSearchTool())


def test_tool_use_block_parsed_into_summary_result(
    monkeypatch: pytest.MonkeyPatch, one_search_result: list[SearchResult]
) -> None:
    _patch_anthropic(
        monkeypatch,
        _response(_tool_use_block("return_summary", VALID_SUMMARY_INPUT)),
    )

    result = summarize(
        "photosynthesis", search_tool=StubSearchTool(results=one_search_result)
    )

    assert isinstance(result, SummaryResult)
    assert result.topic == "photosynthesis"
    assert result.synopsis == "Plants convert sunlight into chemical energy."
    assert result.key_findings == [
        "Chlorophyll absorbs light",
        "Produces oxygen as a byproduct",
    ]
    assert len(result.citations) == 1
    assert result.citations[0].url == "https://en.wikipedia.org/wiki/Photosynthesis"


def test_missing_tool_use_block_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, one_search_result: list[SearchResult]
) -> None:
    _patch_anthropic(
        monkeypatch,
        _response(_text_block("Sorry, I refuse to call the tool.")),
    )

    with pytest.raises(RuntimeError, match="return_summary"):
        summarize(
            "photosynthesis",
            search_tool=StubSearchTool(results=one_search_result),
        )
