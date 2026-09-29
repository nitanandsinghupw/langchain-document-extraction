"""classify -> extract -> business rules + evidence check -> auto-approve or send to review."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from doc_extraction import prompts
from doc_extraction.grounding import check_evidence
from doc_extraction.schemas import EVIDENCE_REQUIRED, SCHEMAS, Classification, Contract, DocumentType, Invoice
from doc_extraction.structured import StructuredOutputError, structured_call
from doc_extraction.validation import Issue, Severity, validate_contract, validate_invoice


class Status(StrEnum):
    AUTO_APPROVED = "auto_approved"  # passed every check, no human needed
    NEEDS_REVIEW = "needs_review"  # waiting in the review queue
    APPROVED = "approved"  # approved by a reviewer (possibly after corrections)
    REJECTED = "rejected"  # rejected by a reviewer
    UNSUPPORTED = "unsupported"  # not an invoice or a contract


class IssueOut(BaseModel):
    code: str
    severity: str
    message: str
    field: str | None = None

    @classmethod
    def of(cls, issue: Issue) -> IssueOut:
        return cls(code=issue.code, severity=issue.severity.value, message=issue.message, field=issue.field)


class ExtractionResult(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str
    document_type: DocumentType
    status: Status
    data: dict | None = None
    issues: list[IssueOut] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_EXTRACT_PROMPTS = {DocumentType.INVOICE: prompts.EXTRACT_INVOICE, DocumentType.CONTRACT: prompts.EXTRACT_CONTRACT}


class ExtractionPipeline:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def classify(self, text: str) -> DocumentType:
        system, user = prompts.render(prompts.CLASSIFY, text)
        return structured_call(self._model, Classification, system, user).document_type

    def run(self, text: str, source: str = "inline") -> ExtractionResult:
        doc_type = self.classify(text)
        if doc_type is DocumentType.OTHER:
            return ExtractionResult(
                source=source,
                document_type=doc_type,
                status=Status.UNSUPPORTED,
                review_reasons=["Not an invoice or a contract."],
            )

        system, user = prompts.render(_EXTRACT_PROMPTS[doc_type], text)
        try:
            extracted = structured_call(self._model, SCHEMAS[doc_type], system, user)
        except StructuredOutputError as exc:
            return ExtractionResult(
                source=source,
                document_type=doc_type,
                status=Status.NEEDS_REVIEW,
                review_reasons=[f"Extraction failed: {exc}"],
            )
        return self.assess(text, source, doc_type, extracted)

    @staticmethod
    def assess(text: str, source: str, doc_type: DocumentType, extracted: Invoice | Contract) -> ExtractionResult:
        """Applies business rules and the evidence check. Also used after a reviewer corrects the data."""
        issues = validate_invoice(extracted) if isinstance(extracted, Invoice) else validate_contract(extracted)
        grounding = check_evidence(text, extracted.evidence, EVIDENCE_REQUIRED[doc_type])

        reasons = [i.message for i in issues if i.severity is Severity.ERROR]
        reasons += [f"No evidence quote for '{f}'." for f in grounding.missing]
        reasons += [f"The quote given for '{f}' does not appear in the document." for f in grounding.ungrounded]

        return ExtractionResult(
            source=source,
            document_type=doc_type,
            status=Status.NEEDS_REVIEW if reasons else Status.AUTO_APPROVED,
            data=extracted.model_dump(mode="json"),
            issues=[IssueOut.of(i) for i in issues],
            review_reasons=reasons,
        )
