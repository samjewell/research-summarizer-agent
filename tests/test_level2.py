"""Level 2: constrained model tests against the real Anthropic API.

These tests exercise prompt effectiveness and structured-output compliance --
the bounds the spec enforces in the prompt only (synopsis sentence count,
key_findings count, citation provenance) and so are invisible to Level 1.

Search is stubbed with a curated payload so that a failure points squarely at
the model + prompt, not at day-to-day Tavily variability. Temperature is pinned
to 0 via the ``SUMMARIZER_TEMPERATURE`` env override to minimise nondeterminism.

Each topic runs three times to give a crude reliability signal: at temperature=0
the model is near-deterministic but not strictly so (server-side tokenisation
and tool-use sampling still introduce slight variance).

The whole module skips when ``ANTHROPIC_API_KEY`` is unset so attendees can run
Level 1 freely without burning Anthropic credits.
"""

from __future__ import annotations

import os
import re

import pytest

from agent.agent import summarize
from agent.models import SummaryResult
from agent.tools import SearchResult, StubSearchTool


pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; Level 2 requires a live Anthropic key",
)


_PHOTOSYNTHESIS_RESULTS: list[SearchResult] = [
    SearchResult(
        title="Photosynthesis - Wikipedia",
        url="https://en.wikipedia.org/wiki/Photosynthesis",
        snippet=(
            "Photosynthesis is the biological process by which plants, algae, "
            "and some bacteria use light energy to convert carbon dioxide and "
            "water into glucose and oxygen. The pigment chlorophyll absorbs "
            "light primarily in the blue and red portions of the spectrum."
        ),
    ),
    SearchResult(
        title="Photosynthesis | Definition, Equation, Steps | Britannica",
        url="https://www.britannica.com/science/photosynthesis",
        snippet=(
            "Photosynthesis takes place in the chloroplasts of plant cells. "
            "The light-dependent reactions split water molecules, releasing "
            "oxygen as a byproduct, while the Calvin cycle fixes carbon "
            "dioxide into sugars."
        ),
    ),
    SearchResult(
        title="What Is Photosynthesis? - NASA Climate Kids",
        url="https://climatekids.nasa.gov/photosynthesis/",
        snippet=(
            "Through photosynthesis, plants take in sunlight, water, and "
            "carbon dioxide and produce energy-rich sugars along with the "
            "oxygen that animals breathe."
        ),
    ),
]


_GREAT_FIRE_RESULTS: list[SearchResult] = [
    SearchResult(
        title="Great Fire of London - Wikipedia",
        url="https://en.wikipedia.org/wiki/Great_Fire_of_London",
        snippet=(
            "The Great Fire of London was a major conflagration that swept "
            "through the central parts of the English city from Sunday 2 "
            "September to Thursday 6 September 1666. The fire began in a "
            "bakery on Pudding Lane belonging to Thomas Farriner and spread "
            "rapidly through the timber-framed buildings of the medieval city."
        ),
    ),
    SearchResult(
        title="Great Fire of London | Summary, Cause, Damage | Britannica",
        url="https://www.britannica.com/event/Great-Fire-of-London",
        snippet=(
            "The fire destroyed an estimated 13,200 houses, 87 parish "
            "churches, and the original St Paul's Cathedral. Although fewer "
            "than ten deaths were officially recorded, the true toll is "
            "thought to have been considerably higher, with the poor and "
            "middle classes hardest hit."
        ),
    ),
    SearchResult(
        title="The Great Fire of London - Museum of London",
        url="https://www.museumoflondon.org.uk/discover/great-fire-london",
        snippet=(
            "Strong easterly winds and a long summer drought turned a small "
            "bakery fire into a city-wide disaster. The rebuilding effort "
            "that followed reshaped London with wider streets, brick "
            "buildings, and Sir Christopher Wren's redesigned St Paul's "
            "Cathedral, and prompted the first organised fire insurance."
        ),
    ),
]


_TOPIC_FIXTURES: list[tuple[str, list[SearchResult]]] = [
    ("photosynthesis", _PHOTOSYNTHESIS_RESULTS),
    ("the Great Fire of London", _GREAT_FIRE_RESULTS),
]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@pytest.fixture(autouse=True)
def _pin_temperature_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUMMARIZER_TEMPERATURE", "0")


def _count_sentences(text: str) -> int:
    pieces = [p for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return len(pieces)


@pytest.mark.parametrize("run_index", [0, 1, 2])
@pytest.mark.parametrize(
    ("topic", "stub_results"),
    _TOPIC_FIXTURES,
    ids=[t for t, _ in _TOPIC_FIXTURES],
)
def test_level2_summarize(
    topic: str, stub_results: list[SearchResult], run_index: int
) -> None:
    stub_urls = {r.url for r in stub_results}

    result = summarize(topic, search_tool=StubSearchTool(results=list(stub_results)))

    # 1. Structured-output compliance.
    assert isinstance(result, SummaryResult), (
        f"run {run_index} [{topic}]: expected SummaryResult, "
        f"got {type(result).__name__}"
    )

    # 2. Synopsis sentence count: prompt mandates 2-4.
    sentence_count = _count_sentences(result.synopsis)
    assert 2 <= sentence_count <= 4, (
        f"run {run_index} [{topic}]: synopsis has {sentence_count} sentences, "
        f"expected 2-4. Synopsis: {result.synopsis!r}"
    )

    # 3. Key findings count: prompt mandates 2-5.
    finding_count = len(result.key_findings)
    assert 2 <= finding_count <= 5, (
        f"run {run_index} [{topic}]: key_findings has {finding_count} items, "
        f"expected 2-5. Findings: {result.key_findings!r}"
    )

    # 4. Citation provenance: every cited URL came from the stubbed payload.
    #    This is the Option B defect canary -- a prompt that fails to constrain
    #    URLs will fabricate plausible-looking ones and trip this assertion.
    for citation in result.citations:
        assert citation.url in stub_urls, (
            f"run {run_index} [{topic}]: citation URL {citation.url!r} was not "
            f"in the provided search results {stub_urls}"
        )

    # 5. Citation existence: at least one citation must be returned. Per-finding
    #    mapping is a semantic question and lives at Level 3.
    assert len(result.citations) >= 1, (
        f"run {run_index} [{topic}]: no citations returned"
    )

    # 6. Topic fidelity: the returned topic matches the input.
    assert result.topic.strip().lower() == topic.lower(), (
        f"run {run_index} [{topic}]: expected topic {topic!r}, "
        f"got {result.topic!r}"
    )
