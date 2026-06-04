"""Test-suite scaffolding for the tests directory.

In addition to the project-root conftest (which fixes ``sys.path``), this file
collects per-test Anthropic API usage for Level 2 and Level 3 runs and prints
a cost summary at the end of the session. The summary doubles as a teaching
aid for the workshop's "how expensive is this test?" question.

Prices below reflect Anthropic public list pricing for the pinned models at
the workshop pin date. Refresh from https://www.anthropic.com/pricing if stale.
"""

from __future__ import annotations

# claude-haiku-4-5 list price, USD per million tokens (agent model).
_HAIKU_INPUT_PER_MTOK = 1.00
_HAIKU_OUTPUT_PER_MTOK = 5.00

# claude-sonnet-4-6 list price, USD per million tokens (Level 3 judge model).
_SONNET_INPUT_PER_MTOK = 3.00
_SONNET_OUTPUT_PER_MTOK = 15.00


# Populated by the capture fixture in ``tests/test_level2.py``. Each entry is
# ``(test_nodeid, input_tokens, output_tokens)``.
LEVEL2_USAGE: list[tuple[str, int, int]] = []


# Populated by the capture fixture in ``tests/test_level3.py``. Each entry is
# ``(test_nodeid, role, input_tokens, output_tokens)`` where ``role`` is
# ``"agent"`` (Haiku-priced) or ``"judge"`` (Sonnet-priced).
LEVEL3_USAGE: list[tuple[str, str, int, int]] = []


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    _summarise_level2(terminalreporter)
    _summarise_level3(terminalreporter)


def _summarise_level2(tr) -> None:
    if not LEVEL2_USAGE:
        return

    total_in = sum(entry[1] for entry in LEVEL2_USAGE)
    total_out = sum(entry[2] for entry in LEVEL2_USAGE)
    cost = (
        total_in / 1_000_000 * _HAIKU_INPUT_PER_MTOK
        + total_out / 1_000_000 * _HAIKU_OUTPUT_PER_MTOK
    )

    tr.write_sep("=", "Level 2 Anthropic usage")
    tr.write_line(f"{'test':<55} {'in':>8} {'out':>8}")
    for test_id, ti, to in LEVEL2_USAGE:
        tr.write_line(f"{test_id:<55} {ti:>8} {to:>8}")
    tr.write_line("-" * 73)
    tr.write_line(f"{'TOTAL':<55} {total_in:>8} {total_out:>8}")
    tr.write_line(
        f"Estimated cost @ ${_HAIKU_INPUT_PER_MTOK}/MTok in, "
        f"${_HAIKU_OUTPUT_PER_MTOK}/MTok out: ${cost:.4f}"
    )


def _summarise_level3(tr) -> None:
    if not LEVEL3_USAGE:
        return

    tr.write_sep("=", "Level 3 Anthropic usage")
    tr.write_line(f"{'test':<55} {'role':<6} {'in':>8} {'out':>8}")

    total_agent_in = total_agent_out = 0
    total_judge_in = total_judge_out = 0
    for test_id, role, ti, to in LEVEL3_USAGE:
        tr.write_line(f"{test_id:<55} {role:<6} {ti:>8} {to:>8}")
        if role == "agent":
            total_agent_in += ti
            total_agent_out += to
        elif role == "judge":
            total_judge_in += ti
            total_judge_out += to

    tr.write_line("-" * 80)
    tr.write_line(
        f"{'AGENT TOTAL (Haiku)':<55} {'':<6} "
        f"{total_agent_in:>8} {total_agent_out:>8}"
    )
    tr.write_line(
        f"{'JUDGE TOTAL (Sonnet)':<55} {'':<6} "
        f"{total_judge_in:>8} {total_judge_out:>8}"
    )

    agent_cost = (
        total_agent_in / 1_000_000 * _HAIKU_INPUT_PER_MTOK
        + total_agent_out / 1_000_000 * _HAIKU_OUTPUT_PER_MTOK
    )
    judge_cost = (
        total_judge_in / 1_000_000 * _SONNET_INPUT_PER_MTOK
        + total_judge_out / 1_000_000 * _SONNET_OUTPUT_PER_MTOK
    )
    tr.write_line(
        f"Estimated cost: agent ${agent_cost:.4f} + "
        f"judge ${judge_cost:.4f} = ${agent_cost + judge_cost:.4f}"
    )
