"""A rule-based stand-in for an LLM, so the pipeline, API and tests run without API keys.

It reads the ROLE line of the system prompt and extracts fields with regular expressions, returning
the same JSON (including evidence quotes) a model would. It is also a useful baseline: the evaluation
shows exactly which layouts and phrasings simple rules miss, which is the gap an LLM closes.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_DOC = re.compile(r"<document>\n(.*)\n</document>", re.DOTALL)
_AMOUNT = r"(\d[\d,]*\.\d{2})"
_CURRENCIES = r"\b(EUR|USD|GBP|CHF|SEK|DKK|NOK|PLN)\b"
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%d.%m.%Y")
# One to six capitalised words (or "&") followed by a legal form. Case-sensitive on purpose, so a
# sentence like "This agreement is entered into by X GmbH" doesn't become part of the name.
_COMPANY = r"(?:(?:[A-Z][\w.'\-]*|&)\s+){1,6}(?:GmbH|AB|BV|Ltd|LTD|Inc\.?|INC\.?|LLC|SA|AG)\b\.?"


def _date(value: str) -> str | None:
    value = value.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _money(value: str) -> str:
    return value.replace(",", "")


class _Extractor:
    """Collects values and the exact text each one came from."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.data: dict[str, Any] = {}
        self.evidence: list[dict[str, str]] = []

    def find(self, field: str, pattern: str, convert=lambda v: v.strip(), flags: int = re.IGNORECASE | re.MULTILINE):
        match = re.search(pattern, self.text, flags)
        if not match:
            return None
        value = convert(match.group(1))
        if value is not None:
            self.data[field] = value
            self.evidence.append({"field": field, "quote": match.group(0).strip()})
        return value


class OfflineExtractionModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "offline-extraction-model"

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        role = str(messages[0].content).splitlines()[0].removeprefix("ROLE: ").strip()
        match = _DOC.search(str(messages[-1].content))
        document = (match.group(1) if match else "").replace("&lt;", "<").replace("&gt;", ">")
        handler = {
            "classifier": self._classify,
            "invoice-extractor": self._invoice,
            "contract-extractor": self._contract,
        }.get(role, lambda _: "{}")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(handler(document))))])

    @staticmethod
    def _classify(text: str) -> dict:
        lower = text.lower()
        invoice = sum(
            signal in lower
            for signal in (
                "invoice no",
                "invoice number",
                "subtotal",
                "net amount",
                "amount due",
                "bill to",
                "unit price",
            )
        )
        contract = sum(
            signal in lower
            for signal in (
                "agreement",
                "governing law",
                "governed by",
                "parties",
                "termination",
                "terminate",
                "effective",
            )
        )
        first_line = next((line for line in lower.splitlines() if line.strip()), "")
        if "agreement" in first_line:
            contract += 2
        if "invoice" in first_line:
            invoice += 2
        if max(invoice, contract) < 2:
            return {"document_type": "other", "reason": "Too few invoice or contract signals."}
        if contract > invoice:
            return {"document_type": "contract", "reason": f"{contract} contract signals vs {invoice} invoice signals."}
        return {"document_type": "invoice", "reason": f"{invoice} invoice signals vs {contract} contract signals."}

    @staticmethod
    def _invoice(text: str) -> dict:
        x = _Extractor(text)
        x.find("invoice_number", r"Invoice\s*(?:No\.?|number|#)[:\s]+([A-Z0-9][A-Z0-9-]+)")
        x.find("invoice_date", r"^(?:Invoice Date|Date of issue|Date)[:\s]+(.+)$", _date)
        x.find("due_date", r"^(?:Due Date|Payment due)[:\s]+(.+)$", _date)
        if not x.find("vendor_name", rf"^From:\s+({_COMPANY})", flags=re.MULTILINE):
            x.find("vendor_name", rf"\A\s*({_COMPANY})", flags=re.MULTILINE)
        x.find("vendor_vat_id", r"VAT (?:ID|Reg\.?)[:\s]+([A-Z]{2}[A-Z0-9]{8,12})")
        x.find("customer_name", r"^(?:Bill to|To):\s+([^,\n]+)")
        x.find("currency", _CURRENCIES, lambda v: v.upper(), flags=0)
        x.find("subtotal", rf"^(?:Subtotal|Net amount)[:\s]+(?:[A-Z]{{3}}\s+)?{_AMOUNT}", _money)
        x.find("tax_amount", rf"^(?:VAT|Tax)\b(?!\s*(?:ID|Reg))[^\n]*?{_AMOUNT}", _money)
        x.find("total", rf"^(?:Total|Amount due)[:\s]+(?:[A-Z]{{3}}\s+)?{_AMOUNT}", _money)
        x.find("iban", r"IBAN[:\s]+([A-Z]{2}\d{2}[A-Z0-9 ]{11,40}?)\s*$", lambda v: v.replace(" ", ""))
        x.find("payment_terms", r"^(?:Payment terms|Terms):\s*(.+)$")

        x.data["line_items"] = [
            {"description": d.strip(), "quantity": q, "unit_price": _money(p), "amount": _money(a)}
            for d, q, p, a in re.findall(rf"^(\S.*?)\s{{2,}}(\d+(?:\.\d+)?)\s+{_AMOUNT}\s+{_AMOUNT}\s*$", text, re.M)
        ]
        x.data["evidence"] = x.evidence
        return x.data

    @staticmethod
    def _contract(text: str) -> dict:
        x = _Extractor(text)
        x.find("title", r"\A\s*([A-Z][A-Z \-]{5,})$")
        parties = re.findall(rf"({_COMPANY})\s*\(\"([^\"]+)\"\)", text)
        x.data["parties"] = [{"name": name.strip(), "role": role} for name, role in parties]
        if parties:
            first = re.search(rf"{re.escape(parties[0][0])}\s*\(\"", text)
            x.evidence.append({"field": "parties", "quote": text[first.start() : first.end() + 20] if first else ""})
        x.find("effective_date", r"(?:takes effect on|effective from|Effective date:)\s*(\d{4}-\d{2}-\d{2})", _date)
        x.find("end_date", r"(?:ending on|Expiry date:|expires on)\s*(\d{4}-\d{2}-\d{2})", _date)
        x.data["auto_renew"] = bool(re.search(r"renews automatically", text, re.I)) and not re.search(
            r"does not renew", text, re.I
        )
        x.find("renewal_notice_days", r"(\d+) days(?:'|\u2019)?(?: written notice)? before the end", int)
        x.find("termination_notice_days", r"terminat\w*[^.]*?(\d+) days(?:'|\u2019)? written notice(?! before)", int)
        x.find("payment_days", r"(?:payable|pay each invoice) within (\d+) days", int)
        x.find("liability_cap", r"liability[^.]*?limited to ([\d,]+) [A-Z]{3}", _money)
        x.find("liability_cap_currency", r"liability[^.]*?limited to [\d,]+ ([A-Z]{3})")
        x.find("governing_law", r"(?:governed by the laws? of|Governing law:)\s*(?:the\s+)?([A-Z][\w ]+?)\.", str.strip)
        x.data["evidence"] = x.evidence
        return x.data
