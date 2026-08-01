"""Label definitions and robust label parsing for RumourEval SDQC stance.

The stance task has four classes (subtask A, English): support, deny, query,
comment. This module is the single source of truth for the label set, the
string <-> integer mapping, and for turning free-form model output into a
label.

Design notes / bug fixes relative to the original notebooks
-----------------------------------------------------------
The original evaluation parsed a model's generated text with::

    VALID_LABELS = {"support", "deny", "query", "comment"}   # a *set*
    for label in VALID_LABELS:
        if label in raw_output:
            return label

Two problems with that:

1. Iterating a ``set`` has an arbitrary, hash-seed-dependent order, so when a
   generation happened to contain more than one label word the returned label
   was not deterministic.
2. Plain substring matching is greedy: "supportive" contains "support",
   "no comment" contains "comment", etc.

``parse_label`` below fixes both: it scans in a fixed, documented priority
order and matches on word boundaries, falling back to ``INVALID`` only when no
label word is present.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Canonical order. Index == integer id used everywhere (golds/preds, F1, ...).
LABELS: List[str] = ["support", "deny", "query", "comment"]

LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
ID_TO_LABEL = {i: name for i, name in enumerate(LABELS)}

INVALID = "invalid"

# Match a label only as a whole word (so "supportive" / "no comment" do not
# spuriously trigger). Priority order is fixed and explicit.
_LABEL_PATTERNS = [(name, re.compile(rf"\b{name}\b", re.IGNORECASE)) for name in LABELS]


def parse_label(raw_output: str) -> str:
    """Extract a stance label from free-form model output.

    Returns one of ``LABELS`` or :data:`INVALID` if no label word is found.
    Deterministic: labels are checked in the fixed order of ``LABELS``.
    """
    if not raw_output:
        return INVALID
    text = raw_output.strip().lower()
    for name, pattern in _LABEL_PATTERNS:
        if pattern.search(text):
            return name
    return INVALID


def to_id(label: str) -> Optional[int]:
    """String label -> integer id, or ``None`` if unknown."""
    return LABEL_TO_ID.get(label)
