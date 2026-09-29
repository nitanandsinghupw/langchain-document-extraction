"""Business rules. Deterministic checks catch what a model gets wrong: arithmetic, checksums, date order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from doc_extraction.schemas import Contract, Invoice

TOLERANCE = Decimal("0.01")
KNOWN_CURRENCIES = {"EUR", "USD", "GBP", "CHF", "SEK", "DKK", "NOK", "PLN", "CZK", "JPY", "INR", "CAD", "AUD"}
_VAT_ID = re.compile(r"^[A-Z]{2}[0-9A-Z]{8,12}$")


class Severity(StrEnum):
    ERROR = "error"  # blocks automatic approval
    WARNING = "warning"  # shown to the reviewer, doesn't block


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    message: str
    field: str | None = None


def iban_is_valid(iban: str) -> bool:
    """ISO 13616 check: move the first four characters to the end, letters to numbers, mod 97 must be 1."""
    compact = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}", compact):
        return False
    rearranged = compact[4:] + compact[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(digits) % 97 == 1


def _differs(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) > TOLERANCE


def validate_invoice(invoice: Invoice) -> list[Issue]:
    issues: list[Issue] = []
    err, warn = Severity.ERROR, Severity.WARNING

    if reason := implausible_name(invoice.vendor_name):
        issues.append(
            Issue("implausible_vendor_name", err, f"Vendor name '{invoice.vendor_name[:60]}' {reason}.", "vendor_name")
        )

    for i, item in enumerate(invoice.line_items, start=1):
        if _differs(item.quantity * item.unit_price, item.amount):
            issues.append(
                Issue(
                    "line_amount_mismatch",
                    err,
                    f"Line {i}: {item.quantity} x {item.unit_price} is not {item.amount}.",
                    "line_items",
                )
            )

    if invoice.line_items:
        lines_total = sum((item.amount for item in invoice.line_items), Decimal("0"))
        if _differs(lines_total, invoice.subtotal):
            issues.append(
                Issue(
                    "subtotal_mismatch",
                    err,
                    f"Line items add up to {lines_total}, but the subtotal is {invoice.subtotal}.",
                    "subtotal",
                )
            )
    else:
        issues.append(Issue("no_line_items", warn, "No line items were found.", "line_items"))

    expected_total = invoice.subtotal + invoice.tax_amount
    if _differs(expected_total, invoice.total):
        issues.append(
            Issue(
                "total_mismatch",
                err,
                f"Subtotal {invoice.subtotal} + tax {invoice.tax_amount} = {expected_total}, "
                f"but the total is {invoice.total}.",
                "total",
            )
        )

    if invoice.due_date is None:
        issues.append(Issue("due_date_missing", err, "The invoice has no due date.", "due_date"))
    elif invoice.due_date < invoice.invoice_date:
        issues.append(Issue("due_before_issue", err, "The due date is before the invoice date.", "due_date"))

    if invoice.currency.upper() not in KNOWN_CURRENCIES:
        issues.append(Issue("unknown_currency", err, f"'{invoice.currency}' is not a supported currency.", "currency"))

    if invoice.iban and not iban_is_valid(invoice.iban):
        issues.append(Issue("invalid_iban", err, f"IBAN {invoice.iban} fails the checksum.", "iban"))
    if not invoice.iban:
        issues.append(Issue("no_iban", warn, "No IBAN found; check the payment details manually.", "iban"))

    if invoice.vendor_vat_id and not _VAT_ID.fullmatch(invoice.vendor_vat_id.replace(" ", "").upper()):
        issues.append(
            Issue("vat_id_format", warn, f"VAT ID '{invoice.vendor_vat_id}' has an unusual format.", "vendor_vat_id")
        )
    return issues


def implausible_name(name: str) -> str | None:
    """Catches values that are clearly a sentence fragment rather than a name.

    A quote can be found in the document and still be the wrong span of text, so evidence checks
    alone don't catch this. Returns the reason, or None if the name looks fine.
    """
    words = name.split()
    if not words:
        return "is empty"
    if len(words) > 8:
        return f"has {len(words)} words"
    lowercase = [w for w in words if w.islower() and w not in {"&", "and", "of", "de", "van", "der", "und", "the"}]
    if len(lowercase) >= 2:
        return f"contains ordinary words ({', '.join(lowercase[:3])})"
    return None


def validate_contract(contract: Contract) -> list[Issue]:
    issues: list[Issue] = []
    err, warn = Severity.ERROR, Severity.WARNING

    for party in contract.parties:
        if reason := implausible_name(party.name):
            issues.append(Issue("implausible_party_name", err, f"Party name '{party.name[:60]}' {reason}.", "parties"))

    if contract.end_date and contract.end_date <= contract.effective_date:
        issues.append(
            Issue(
                "end_before_start",
                err,
                f"The end date {contract.end_date} is not after the effective date {contract.effective_date}.",
                "end_date",
            )
        )
    if contract.auto_renew and contract.renewal_notice_days is None:
        issues.append(
            Issue(
                "renewal_notice_missing",
                warn,
                "The contract renews automatically but no notice period was found.",
                "renewal_notice_days",
            )
        )
    for name in ("renewal_notice_days", "termination_notice_days", "payment_days"):
        value = getattr(contract, name)
        if value is not None and not 0 <= value <= 365:
            issues.append(Issue("implausible_days", err, f"{name} = {value} is outside 0-365.", name))
    if contract.liability_cap is not None and contract.liability_cap <= 0:
        issues.append(Issue("liability_cap_not_positive", err, "The liability cap must be positive.", "liability_cap"))
    if contract.governing_law is None:
        issues.append(Issue("governing_law_missing", warn, "No governing law was found.", "governing_law"))
    return issues
