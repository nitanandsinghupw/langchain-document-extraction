from doc_extraction.grounding import check_evidence
from doc_extraction.schemas import Evidence

DOC = "Invoice No: BL-2025-0417\nTotal:      2814.35 EUR"


def test_quotes_match_despite_different_spacing_and_case():
    report = check_evidence(DOC, [Evidence(field="total", quote="total: 2814.35 eur")], ("total",))
    assert report.ok


def test_invented_quotes_are_flagged():
    report = check_evidence(DOC, [Evidence(field="total", quote="Total: 9999.00 EUR")], ("total",))
    assert report.ungrounded == ["total"]


def test_missing_evidence_is_flagged():
    report = check_evidence(DOC, [], ("invoice_number", "total"))
    assert report.missing == ["invoice_number", "total"]
