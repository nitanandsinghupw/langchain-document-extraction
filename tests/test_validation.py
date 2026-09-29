from datetime import date
from decimal import Decimal

from doc_extraction.schemas import Contract, Invoice, LineItem, Party
from doc_extraction.validation import iban_is_valid, implausible_name, validate_contract, validate_invoice


def invoice(**overrides) -> Invoice:
    fields = dict(
        invoice_number="A-1",
        invoice_date=date(2025, 9, 1),
        due_date=date(2025, 10, 1),
        vendor_name="Acme GmbH",
        currency="EUR",
        subtotal=Decimal("100.00"),
        tax_amount=Decimal("19.00"),
        total=Decimal("119.00"),
        iban="DE89370400440532013000",
        line_items=[
            LineItem(description="x", quantity=Decimal(2), unit_price=Decimal("50.00"), amount=Decimal("100.00"))
        ],
    )
    return Invoice(**{**fields, **overrides})


def codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_a_consistent_invoice_has_no_errors():
    assert not [i for i in validate_invoice(invoice()) if i.severity == "error"]


def test_arithmetic_errors_are_caught():
    assert "total_mismatch" in codes(validate_invoice(invoice(total=Decimal("120.00"))))
    assert "subtotal_mismatch" in codes(validate_invoice(invoice(subtotal=Decimal("90.00"), total=Decimal("109.00"))))
    bad_line = LineItem(description="x", quantity=Decimal(3), unit_price=Decimal("50.00"), amount=Decimal("100.00"))
    assert "line_amount_mismatch" in codes(validate_invoice(invoice(line_items=[bad_line])))


def test_rounding_within_one_cent_is_accepted():
    assert "total_mismatch" not in codes(validate_invoice(invoice(total=Decimal("119.01"))))


def test_dates_and_currency_are_checked():
    assert "due_date_missing" in codes(validate_invoice(invoice(due_date=None)))
    assert "due_before_issue" in codes(validate_invoice(invoice(due_date=date(2025, 8, 1))))
    assert "unknown_currency" in codes(validate_invoice(invoice(currency="XYZ")))


def test_iban_checksum():
    assert iban_is_valid("DE89 3704 0044 0532 0130 00")
    assert iban_is_valid("GB33BUKB20201555555555")
    assert not iban_is_valid("DE89 3704 0044 0532 0130 01")
    assert not iban_is_valid("not an iban")
    assert "invalid_iban" in codes(validate_invoice(invoice(iban="DE89370400440532013001")))


def test_sentence_fragments_are_not_accepted_as_names():
    assert implausible_name("Voltwerk Components BV") is None
    assert implausible_name("Kettle & Pine Design Ltd") is None
    assert implausible_name("This Supply Agreement is entered into by Voltwerk Components BV") is not None


def test_contract_rules():
    base = dict(title="NDA", parties=[Party(name="A GmbH"), Party(name="B BV")], effective_date=date(2025, 6, 1))
    assert "end_before_start" in codes(validate_contract(Contract(**base, end_date=date(2025, 5, 31))))
    assert "renewal_notice_missing" in codes(validate_contract(Contract(**base, auto_renew=True)))
    assert "implausible_days" in codes(validate_contract(Contract(**base, payment_days=900)))
