import json

import pytest
from conftest import DATA
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from doc_extraction.loaders import UnsupportedDocumentError, load, text_from_bytes
from doc_extraction.pipeline import ExtractionPipeline, Status
from doc_extraction.schemas import Classification
from doc_extraction.structured import StructuredOutputError, structured_call


@pytest.mark.parametrize(
    "name, status",
    [
        ("inv-001.txt", Status.AUTO_APPROVED),
        ("inv-003.txt", Status.NEEDS_REVIEW),  # total doesn't add up
        ("inv-004.txt", Status.NEEDS_REVIEW),  # IBAN checksum
        ("inv-005.txt", Status.NEEDS_REVIEW),  # no due date
        ("ctr-001.txt", Status.AUTO_APPROVED),
        ("ctr-002.txt", Status.NEEDS_REVIEW),  # ends before it starts
        ("memo-001.txt", Status.UNSUPPORTED),
    ],
)
def test_documents_are_routed_correctly(pipeline, sample, name, status):
    assert pipeline.run(sample(name), name).status is status


def test_extracted_values_are_typed(pipeline, sample):
    data = pipeline.run(sample("inv-001.txt")).data

    assert data["invoice_number"] == "BL-2025-0417"
    assert data["invoice_date"] == "2025-09-02"
    assert data["total"] == "2814.35"
    assert len(data["line_items"]) == 3


def test_pdfs_are_read_through_their_text_layer(pipeline):
    result = pipeline.run(load(DATA / "pdf" / "inv-001.pdf"), "inv-001.pdf")

    assert result.status is Status.AUTO_APPROVED
    assert result.data["total"] == "2814.35"


def test_hallucinated_values_go_to_review(sample):
    """A model that returns a plausible total with a quote that isn't in the document."""
    invoice = {
        "invoice_number": "BL-2025-0417",
        "invoice_date": "2025-09-02",
        "due_date": "2025-10-02",
        "vendor_name": "Brightline Office Supplies GmbH",
        "currency": "EUR",
        "subtotal": "2365.00",
        "tax_amount": "449.35",
        "total": "2814.35",
        "iban": "DE89370400440532013000",
        "line_items": [{"description": "Chairs", "quantity": 1, "unit_price": "2365.00", "amount": "2365.00"}],
        "evidence": [
            {"field": "invoice_number", "quote": "Invoice No: BL-2025-0417"},
            {"field": "invoice_date", "quote": "Invoice Date: 2025-09-02"},
            {"field": "vendor_name", "quote": "BRIGHTLINE OFFICE SUPPLIES GmbH"},
            {"field": "total", "quote": "Grand total 2814.35"},
        ],
    }
    model = FakeListChatModel(responses=[json.dumps({"document_type": "invoice", "reason": "r"}), json.dumps(invoice)])

    result = ExtractionPipeline(model).run(sample("inv-001.txt"))

    assert result.status is Status.NEEDS_REVIEW
    assert any("'total' does not appear" in r for r in result.review_reasons)


def test_invalid_model_output_is_repaired_once():
    model = FakeListChatModel(responses=["sure, it's an invoice", '{"document_type": "invoice", "reason": "r"}'])
    assert structured_call(model, Classification, "ROLE: x", "doc").document_type == "invoice"


def test_unrepairable_output_raises():
    model = FakeListChatModel(responses=["no", "still no"])
    with pytest.raises(StructuredOutputError):
        structured_call(model, Classification, "ROLE: x", "doc")


def test_unsupported_files_are_rejected():
    with pytest.raises(UnsupportedDocumentError):
        text_from_bytes(b"...", "sheet.xlsx")
