"""Stores every result in SQLite and lets a reviewer correct, approve or reject queued documents."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from doc_extraction.pipeline import ExtractionPipeline, ExtractionResult, IssueOut, Status
from doc_extraction.schemas import SCHEMAS, DocumentType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    document_type TEXT NOT NULL,
    status TEXT NOT NULL,
    data TEXT,
    issues TEXT NOT NULL,
    review_reasons TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed_by TEXT,
    reviewed_at TEXT,
    review_note TEXT
);
CREATE INDEX IF NOT EXISTS documents_status ON documents(status);
"""


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class NotReviewableError(ValueError):
    pass


class StoredResult(ExtractionResult):
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None


class ReviewStore:
    def __init__(self, path: str | Path = "extraction.db") -> None:
        self._path = str(path)
        with closing(self._connect()) as db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path)
        db.row_factory = sqlite3.Row
        return db

    def save(self, result: ExtractionResult, text: str) -> StoredResult:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO documents (id, source, document_type, status, data, issues, review_reasons, text,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    result.id,
                    result.source,
                    result.document_type.value,
                    result.status.value,
                    json.dumps(result.data),
                    json.dumps([i.model_dump() for i in result.issues]),
                    json.dumps(result.review_reasons),
                    text,
                    result.created_at.isoformat(),
                ),
            )
        return self.get(result.id)

    def get(self, document_id: str) -> StoredResult | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _to_result(row) if row else None

    def text(self, document_id: str) -> str | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT text FROM documents WHERE id = ?", (document_id,)).fetchone()
        return row["text"] if row else None

    def find_duplicate(self, vendor_name: str, invoice_number: str) -> StoredResult | None:
        """An earlier, not rejected invoice with the same vendor and invoice number."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT * FROM documents WHERE document_type = 'invoice' AND status != 'rejected'"
                " AND lower(json_extract(data, '$.invoice_number')) = lower(?)",
                (invoice_number,),
            ).fetchall()
        wanted = " ".join(vendor_name.split()).casefold()
        for row in rows:
            data = json.loads(row["data"])
            if " ".join(str(data.get("vendor_name", "")).split()).casefold() == wanted:
                return _to_result(row)
        return None

    def list(self, status: Status | None = None, limit: int = 100) -> list[StoredResult]:
        query, args = "SELECT * FROM documents", []
        if status:
            query, args = query + " WHERE status = ?", [status.value]
        with closing(self._connect()) as db:
            rows = db.execute(query + " ORDER BY created_at LIMIT ?", [*args, limit]).fetchall()
        return [_to_result(r) for r in rows]

    def review(
        self,
        document_id: str,
        action: ReviewAction,
        reviewer: str,
        corrections: dict | None = None,
        note: str | None = None,
    ) -> StoredResult:
        """Applies a reviewer's decision. Corrections are merged into the data and re-validated first."""
        current = self.get(document_id)
        if current is None:
            raise KeyError(document_id)
        if current.status is not Status.NEEDS_REVIEW:
            raise NotReviewableError(f"Document is '{current.status.value}', not waiting for review.")

        data, issues, reasons = current.data, current.issues, current.review_reasons
        if action is ReviewAction.APPROVE:
            schema = SCHEMAS.get(current.document_type)
            if schema is None:
                raise NotReviewableError("Only invoices and contracts can be approved.")
            merged = {**(current.data or {}), **(corrections or {})}
            extracted = schema.model_validate(merged)  # raises ValidationError on bad corrections
            reassessed = ExtractionPipeline.assess(
                self.text(document_id) or "", current.source, current.document_type, extracted
            )
            data, issues, reasons = reassessed.data, reassessed.issues, reassessed.review_reasons
            if reasons:
                note = (note + " " if note else "") + f"Approved with {len(reasons)} open issue(s)."

        status = Status.APPROVED if action is ReviewAction.APPROVE else Status.REJECTED
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE documents SET status = ?, data = ?, issues = ?, review_reasons = ?,"
                " reviewed_by = ?, reviewed_at = ?, review_note = ? WHERE id = ?",
                (
                    status.value,
                    json.dumps(data),
                    json.dumps([i.model_dump() for i in issues]),
                    json.dumps(reasons),
                    reviewer,
                    datetime.now(UTC).isoformat(),
                    note,
                    document_id,
                ),
            )
        return self.get(document_id)


def _to_result(row: sqlite3.Row) -> StoredResult:
    return StoredResult(
        id=row["id"],
        source=row["source"],
        document_type=DocumentType(row["document_type"]),
        status=Status(row["status"]),
        data=json.loads(row["data"]) if row["data"] else None,
        issues=[IssueOut(**i) for i in json.loads(row["issues"])],
        review_reasons=json.loads(row["review_reasons"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        reviewed_by=row["reviewed_by"],
        reviewed_at=datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None,
        review_note=row["review_note"],
    )
