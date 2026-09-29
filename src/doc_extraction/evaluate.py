"""Measures extraction quality against labelled documents.

Reports, per split (tuned = documents the rules were written against, held_out = unseen layouts):
classification and routing accuracy, field-level accuracy, and unsafe auto-approvals: documents approved
without review even though at least one field was wrong. That last number is the one that matters when
you let the system post invoices on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from doc_extraction.loaders import load
from doc_extraction.pipeline import ExtractionPipeline, ExtractionResult, Status


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.split()).casefold()
    return value


def field_matches(name: str, expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual in (None, "", [])
    if name == "line_items":
        return isinstance(actual, list) and len(actual) == expected
    if name == "parties":
        return {_norm(n) for n in expected} == {_norm(p.get("name", "")) for p in (actual or [])}
    if name == "iban" and actual is not None:
        return expected.replace(" ", "") == str(actual).replace(" ", "")
    if isinstance(expected, list):  # several acceptable answers
        return _norm(actual) in {_norm(e) for e in expected}
    if isinstance(expected, bool | int):
        return actual == expected
    if actual is None:
        return False
    try:
        return Decimal(str(expected)) == Decimal(str(actual))
    except InvalidOperation:
        return _norm(expected) == _norm(actual)


@dataclass
class DocumentScore:
    source: str
    split: str
    result: ExtractionResult
    classified_ok: bool
    routed_ok: bool
    fields_total: int
    fields_correct: int
    wrong_fields: list[str] = field(default_factory=list)

    @property
    def unsafe_auto_approval(self) -> bool:
        return self.result.status is Status.AUTO_APPROVED and bool(self.wrong_fields)


def evaluate(pipeline: ExtractionPipeline, labels_path: Path, samples_dir: Path) -> list[DocumentScore]:
    scores = []
    for label in json.loads(labels_path.read_text(encoding="utf-8")):
        result = pipeline.run(load(samples_dir / label["source"]), label["source"])
        data = result.data or {}
        wrong = [
            name for name, expected in label["fields"].items() if not field_matches(name, expected, data.get(name))
        ]
        scores.append(
            DocumentScore(
                source=label["source"],
                split=label["split"],
                result=result,
                classified_ok=result.document_type.value == label["document_type"],
                routed_ok=result.status.value == label["expected_status"],
                fields_total=len(label["fields"]),
                fields_correct=len(label["fields"]) - len(wrong),
                wrong_fields=wrong,
            )
        )
    return scores


def report(scores: list[DocumentScore]) -> str:
    lines = [
        f"{'Split':<10} {'Docs':>5} {'Classified':>11} {'Routed':>7} {'Fields':>8} {'Unsafe auto-approvals':>22}",
        "-" * 68,
    ]
    for split in ("tuned", "held_out", "all"):
        subset = [s for s in scores if split == "all" or s.split == split]
        if not subset:
            continue
        n = len(subset)
        fields_total = sum(s.fields_total for s in subset)
        fields_ok = sum(s.fields_correct for s in subset)
        field_acc = f"{fields_ok / fields_total:.0%}" if fields_total else "-"
        lines.append(
            f"{split:<10} {n:>5} {sum(s.classified_ok for s in subset) / n:>11.0%} "
            f"{sum(s.routed_ok for s in subset) / n:>7.0%} {field_acc:>8} "
            f"{sum(s.unsafe_auto_approval for s in subset):>22}"
        )
    lines.append("")
    for s in scores:
        if s.wrong_fields or not s.routed_ok:
            lines.append(
                f"{s.source}: status {s.result.status.value}, wrong fields: {', '.join(s.wrong_fields) or 'none'}"
            )
            lines.extend(f"    reason: {r}" for r in s.result.review_reasons[:2])
    return "\n".join(lines)
