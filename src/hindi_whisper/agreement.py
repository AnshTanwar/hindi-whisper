"""LocalAgreement prefix commit. No model imports, so tests stay light."""

from __future__ import annotations


def longest_common_prefix(previous: list[str], current: list[str]) -> list[str]:
    """Shared leading tokens of two consecutive hypotheses."""
    prefix: list[str] = []
    for left, right in zip(previous, current):
        if left != right:
            break
        prefix.append(left)
    return prefix
