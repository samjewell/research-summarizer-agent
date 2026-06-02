"""Test-suite scaffolding for the tests directory.

In addition to the project-root conftest (which fixes ``sys.path``), this file
collects per-test Anthropic API usage for Level 2 runs and prints a cost
summary at the end of the session. The summary doubles as a teaching aid for
the workshop's "how expensive is this test?" question.

Prices below reflect Anthropic public list pricing for the pinned model at the
workshop pin date. Refresh from https://www.anthropic.com/pricing if stale.
"""

from __future__ import annotations

# claude-haiku-4-5 list price, USD per million tokens.
_PRICE_INPUT_PER_MTOK = 1.00
_PRICE_OUTPUT_PER_MTOK = 5.00


# Populated by the capture fixture in ``tests/test_level2.py``. Each entry is
# ``(test_nodeid, input_tokens, output_tokens)``.
LEVEL2_USAGE: list[tuple[str, int, int]] = []


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    if not LEVEL2_USAGE:
        return

    total_in = sum(entry[1] for entry in LEVEL2_USAGE)
    total_out = sum(entry[2] for entry in LEVEL2_USAGE)
    cost = (
        total_in / 1_000_000 * _PRICE_INPUT_PER_MTOK
        + total_out / 1_000_000 * _PRICE_OUTPUT_PER_MTOK
    )

    tr = terminalreporter
    tr.write_sep("=", "Level 2 Anthropic usage")
    tr.write_line(f"{'test':<55} {'in':>8} {'out':>8}")
    for test_id, ti, to in LEVEL2_USAGE:
        tr.write_line(f"{test_id:<55} {ti:>8} {to:>8}")
    tr.write_line("-" * 73)
    tr.write_line(f"{'TOTAL':<55} {total_in:>8} {total_out:>8}")
    tr.write_line(
        f"Estimated cost @ ${_PRICE_INPUT_PER_MTOK}/MTok in, "
        f"${_PRICE_OUTPUT_PER_MTOK}/MTok out: ${cost:.4f}"
    )
