"""Strict parsing of one-token judge answers.

A reply counts as an answer only if its first line consists of the answer alone, optionally
wrapped in Markdown, quotes, HTML tags or punctuation ("**B**", "A.", "yes</code>") or preceded
by "Solution"/"Answer:". Anything else (refusals such as "Sorry, I can't help with that.",
explanations, degenerate text) is None and is excluded from the analysis. An earlier parser took
the first A/B character anywhere in the reply, which read the "a" of "can't" as answer A.
"""
from __future__ import annotations

import re
from typing import Optional

_TAGS = re.compile(r"</?[A-Za-z][^>]*>")
_PREFIX = re.compile(r"^(?:solution|answer)\b\s*:?\s*", re.IGNORECASE)
_WRAP = "*_`\"'()[]{}.,:;!?"


def _first_line(text: Optional[str]) -> str:
    s = _TAGS.sub(" ", text or "").strip()
    line = s.splitlines()[0] if s else ""
    return line.strip().strip(_WRAP).strip()


def parse_choice(text: Optional[str]) -> Optional[int]:
    """1 for "A" (or "1"), 2 for "B" (or "2"), None for anything else."""
    s = _PREFIX.sub("", _first_line(text)).strip(_WRAP).strip()
    return {"A": 1, "1": 1, "B": 2, "2": 2}.get(s.upper())


def parse_yes_no(text: Optional[str]) -> Optional[str]:
    """"yes" or "no", None for anything else."""
    s = _first_line(text).lower()
    return s if s in ("yes", "no") else None
