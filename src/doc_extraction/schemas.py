"""What we extract. Money is Decimal (never float), dates are ISO dates, and every key field needs evidence."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    INVOICE = "invoice"
    CONTRACT = "contract"
    OTHER = "other"


class Classification(BaseModel):
    document_type: DocumentType
    reason: str = Field(description="One short sentence explaining the choice")


class Evidence(BaseModel):
    """A verbatim quote from the document that supports one extracted field."""

    field: str = Field(description="Name of the field, e.g. 'total' or 'effective_date'")
    quote: str = Field(description="Exact text copied from the document, 3-15 words")


class LineItem(BaseModel):
    description: str
    quantity: Decimal
    unit_price: Decimal
    amount: Decimal


class Invoice(BaseModel):
    invoice_number: str
    invoice_date: date
    due_date: date | None = None
    vendor_name: str
    vendor_vat_id: str | None = None
    customer_name: str | None = None
    currency: str = Field(description="ISO 4217 code, e.g. EUR, GBP, USD")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal
    tax_amount: Decimal = Decimal("0")
    total: Decimal
    iban: str | None = Field(default=None, description="IBAN without spaces")
    payment_terms: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Party(BaseModel):
    name: str
    role: str | None = Field(default=None, description="e.g. Customer, Provider, Supplier, Buyer")


class Contract(BaseModel):
    title: str
    parties: list[Party] = Field(min_length=2)
    effective_date: date
    end_date: date | None = None
    auto_renew: bool = False
    renewal_notice_days: int | None = None
    termination_notice_days: int | None = None
    payment_days: int | None = Field(default=None, description="Days allowed to pay invoices")
    liability_cap: Decimal | None = None
    liability_cap_currency: str | None = None
    governing_law: str | None = Field(default=None, description="Country whose law applies")
    evidence: list[Evidence] = Field(default_factory=list)


# Fields that must be backed by a verbatim quote. A value without grounded evidence goes to review.
EVIDENCE_REQUIRED: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.INVOICE: ("invoice_number", "invoice_date", "total", "vendor_name"),
    DocumentType.CONTRACT: ("effective_date", "parties", "governing_law"),
}

SCHEMAS: dict[DocumentType, type[BaseModel]] = {DocumentType.INVOICE: Invoice, DocumentType.CONTRACT: Contract}
