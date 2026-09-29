"""Checks that the model's evidence quotes really appear in the document.

A value the model can't point to in the text is the most common kind of extraction hallucination.
Quotes are compared after normalising case, whitespace and quote characters, so line breaks and
column spacing in the source don't cause false alarms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from doc_extraction.schemas import Evidence

_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


@dataclass(frozen=True)
class GroundingReport:
    grounded: list[str]  # required fields with a quote found in the document
    ungrounded: list[str]  # required fields whose quote is not in the document
    missing: list[str]  # required fields with no quote at all

    @property
    def ok(self) -> bool:
        return not self.ungrounded and not self.missing


def check_evidence(document: str, evidence: list[Evidence], required: tuple[str, ...]) -> GroundingReport:
    text = normalise(document)
    by_field: dict[str, list[str]] = {}
    for item in evidence:
        by_field.setdefault(item.field, []).append(item.quote)

    grounded, ungrounded, missing = [], [], []
    for field in required:
        quotes = [q for q in by_field.get(field, []) if q.strip()]
        if not quotes:
            missing.append(field)
        elif any(normalise(q) in text for q in quotes):
            grounded.append(field)
        else:
            ungrounded.append(field)
    return GroundingReport(grounded, ungrounded, missing)
