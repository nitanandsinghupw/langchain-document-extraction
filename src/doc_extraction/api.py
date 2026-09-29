"""HTTP API for extraction and human review.

Run: uvicorn doc_extraction.api:app --reload
"""

from __future__ import annotations

import csv
import io
import os
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from doc_extraction.loaders import MAX_CHARS, UnsupportedDocumentError, text_from_bytes
from doc_extraction.models import get_model
from doc_extraction.pipeline import ExtractionPipeline, Status
from doc_extraction.review_store import NotReviewableError, ReviewAction, ReviewStore, StoredResult
from doc_extraction.schemas import DocumentType


class TextRequest(BaseModel):
    text: str = Field(min_length=20, max_length=MAX_CHARS)
    source: str = Field(default="inline", max_length=200)


class ReviewRequest(BaseModel):
    action: ReviewAction
    reviewer: str = Field(min_length=1, max_length=100)
    corrections: dict | None = Field(default=None, description="Field values to overwrite before approving")
    note: str | None = Field(default=None, max_length=1000)


def create_app(pipeline: ExtractionPipeline | None = None, store: ReviewStore | None = None) -> FastAPI:
    pipeline = pipeline or ExtractionPipeline(get_model())
    store = store or ReviewStore(os.getenv("EXTRACTION_DB", "extraction.db"))
    app = FastAPI(title="Document extraction API", version="0.1.0")

    def process(text: str, source: str) -> StoredResult:
        result = pipeline.run(text, source)
        data = result.data or {}
        is_invoice = (
            result.document_type is DocumentType.INVOICE and data.get("invoice_number") and data.get("vendor_name")
        )
        duplicate = store.find_duplicate(data["vendor_name"], data["invoice_number"]) if is_invoice else None
        if duplicate:
            result = result.model_copy(
                update={
                    "status": Status.NEEDS_REVIEW,
                    "review_reasons": [
                        *result.review_reasons,
                        f"Possible duplicate of document {duplicate.id} ({duplicate.source}).",
                    ],
                }
            )
        return store.save(result, text)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/extract", response_model=StoredResult)
    async def extract_file(file: Annotated[UploadFile, File()]) -> StoredResult:
        data = await file.read()
        try:
            text = text_from_bytes(data, file.filename or "upload.txt")
        except UnsupportedDocumentError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        return process(text, file.filename or "upload")

    @app.post("/extract/text", response_model=StoredResult)
    def extract_text(request: TextRequest) -> StoredResult:
        return process(request.text, request.source)

    @app.get("/documents", response_model=list[StoredResult])
    def list_documents(status: Status | None = None, limit: int = Query(100, ge=1, le=500)) -> list[StoredResult]:
        return store.list(status, limit)

    @app.get("/documents/{document_id}", response_model=StoredResult)
    def get_document(document_id: str) -> StoredResult:
        if (result := store.get(document_id)) is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return result

    @app.post("/documents/{document_id}/review", response_model=StoredResult)
    def review(document_id: str, request: ReviewRequest) -> StoredResult:
        try:
            return store.review(document_id, request.action, request.reviewer, request.corrections, request.note)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Document not found.") from exc
        except NotReviewableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Corrections are invalid: {exc.error_count()} errors") from exc

    @app.get("/export/invoices.csv", response_class=PlainTextResponse)
    def export_invoices() -> str:
        """Approved invoices (automatically or by a reviewer), one row each, ready for accounting import."""
        columns = [
            "id",
            "invoice_number",
            "invoice_date",
            "due_date",
            "vendor_name",
            "vendor_vat_id",
            "currency",
            "subtotal",
            "tax_amount",
            "total",
            "iban",
            "status",
            "reviewed_by",
        ]
        out = io.StringIO()
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for status in (Status.AUTO_APPROVED, Status.APPROVED):
            for doc in store.list(status, limit=500):
                if doc.document_type is DocumentType.INVOICE and doc.data:
                    writer.writerow(
                        {**doc.data, "id": doc.id, "status": doc.status.value, "reviewed_by": doc.reviewed_by}
                    )
        return out.getvalue()

    return app


app = create_app()
