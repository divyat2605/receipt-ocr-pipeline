"""Gemini-first receipt parser with a deterministic regex fallback."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from google import genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_MODEL = "gemini-2.5-flash-lite"
_PRICE_RE = re.compile(r"(?<!\d)(?:[$EURGBPINR])?\s*(-?\d{1,5}(?:[.,]\d{2}))(?!\d)")
_DATE_RE = re.compile(
    r"\b("
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}"
    r")\b",
    re.IGNORECASE,
)
_TOTAL_RE = re.compile(r"\b(grand\s+total|total|amount\s+due|balance\s+due|due)\b", re.I)
_SUBTOTAL_RE = re.compile(r"\b(sub\s*total|subtotal)\b", re.I)
_EXCLUDE_ITEM_RE = re.compile(
    r"\b(total|subtotal|tax|change|cash|card|visa|mastercard|amex|balance|amount|"
    r"tender|payment|receipt|invoice|due)\b",
    re.IGNORECASE,
)
_CURRENCY_SYMBOLS = {"$": "USD", "EUR": "EUR", "GBP": "GBP", "INR": "INR", "RM": "MYR", "R": "ZAR"}
_CURRENCY_CODES = {
    "USD", "GBP", "EUR", "INR", "CAD", "AUD", "JPY", "CHF", "CNY", "MXN", "SGD", "NZD",
    "MYR", "ZAR", "SAR",
}
_CURRENCY_ALIASES = {
    "RM": "MYR",
    "MYM": "MYR",
    "R": "ZAR",
    "SR": "SAR",
}
_NULL_FIELDS: dict[str, Any] = {
    "store_name": None,
    "date": None,
    "items": [],
    "subtotal": None,
    "total_amount": None,
    "currency": None,
}

_PROMPT = """You are a receipt parser. Extract fields from OCR text.

Return ONLY valid JSON with these keys:
- store_name: string or null
- date: YYYY-MM-DD string or null
- items: array of objects, each with name and price strings
- subtotal: string or null
- total_amount: string or null
- currency: ISO 4217 code such as USD, INR, EUR, or null

Do not include markdown, comments, confidence scores, or extra keys.

OCR text:
{raw_text}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Return a Gemini client configured from ``GEMINI_API_KEY``."""
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")
    if _client is None:
        _client = genai.Client(api_key=api_key)
    return _client


def _strip_code_fence(text: str) -> str:
    """Remove optional markdown code fences around model JSON."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _normalise_price(value: str | None) -> str | None:
    """Normalize a matched price-like string to decimal text."""
    if value is None:
        return None
    cleaned = re.sub(r"[^\d,.\-]", "", value).replace(",", ".")
    if cleaned.count(".") > 1:
        left, right = cleaned.rsplit(".", 1)
        cleaned = left.replace(".", "") + "." + right
    return cleaned or None


def _normalise_date(value: str | None) -> str | None:
    """Best-effort normalization of common receipt date formats to YYYY-MM-DD."""
    if not value:
        return None
    candidate = value.strip().replace(".", "/")
    formats = [
        "%Y-%m-%d", "%Y/%m/%d",
        "%d/%m/%Y", "%m/%d/%Y",
        "%d/%m/%y", "%m/%d/%y",
        "%d-%m-%Y", "%m-%d-%Y",
        "%d-%m-%y", "%m-%d-%y",
        "%B %d %Y", "%B %d, %Y",
        "%b %d %Y", "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return value.strip()


def _detect_currency(text: str) -> str | None:
    """Infer an ISO currency code from symbols or explicit codes."""
    upper = text.upper()
    if "$" in text:
        return "USD"
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in upper:
            return code
    for code in sorted(_CURRENCY_CODES):
        if re.search(rf"\b{code}\b", upper):
            return code
    return None


def _normalise_currency(value: Any) -> str | None:
    """Normalize common currency symbols/aliases to ISO 4217 codes."""
    if value is None:
        return None
    currency = str(value).strip().upper()
    if not currency:
        return None
    if currency in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[currency]
    if currency in _CURRENCY_CODES:
        return currency
    if currency in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[currency]
    return None


def _first_price(line: str) -> str | None:
    """Return the last price-like token in a line."""
    matches = _PRICE_RE.findall(line)
    return _normalise_price(matches[-1]) if matches else None


def _parse_items(lines: list[str]) -> list[dict[str, str]]:
    """Extract likely item rows from lines ending with a price."""
    items: list[dict[str, str]] = []
    for line in lines:
        price = _first_price(line)
        if not price or _EXCLUDE_ITEM_RE.search(line):
            continue
        name = _PRICE_RE.sub("", line).strip(" -:\t")
        name = re.sub(r"\s{2,}", " ", name)
        if len(name) >= 2:
            items.append({"name": name, "price": price})
    return items


def _parse_by_regex(raw_text: str) -> dict[str, Any]:
    """Parse receipt fields with regex and line heuristics."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return {**_NULL_FIELDS, "flags": ["parser_failed"]}

    date_match = _DATE_RE.search(raw_text)
    subtotal = None
    total = None
    total_candidates: list[str] = []

    for line in lines:
        price = _first_price(line)
        if not price:
            continue
        if _SUBTOTAL_RE.search(line):
            subtotal = price
        if _TOTAL_RE.search(line):
            total_candidates.append(price)

    if total_candidates:
        total = total_candidates[-1]

    store_name = None
    for line in lines[: min(6, len(lines))]:
        if _PRICE_RE.search(line) or _DATE_RE.search(line):
            continue
        if len(line) >= 3:
            store_name = line
            break

    return {
        "store_name": store_name,
        "date": _normalise_date(date_match.group(1) if date_match else None),
        "items": _parse_items(lines),
        "subtotal": subtotal,
        "total_amount": total,
        "currency": _detect_currency(raw_text),
    }


def _parse_with_gemini(raw_text: str, retries: int) -> dict[str, Any]:
    client = _get_client()
    prompt = _PROMPT.format(raw_text=raw_text)

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=_MODEL,
                contents=prompt,
                config={
                    "temperature": 0,
                    "max_output_tokens": 1200,
                    "response_mime_type": "application/json",
                },
            )
            parsed = json.loads(_strip_code_fence(response.text or ""))
            return _coerce_schema(parsed)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini returned non-JSON: {exc}") from exc
        except Exception as exc:
            is_retryable = any(
                token in str(exc).lower()
                for token in ("429", "rate", "quota", "timeout", "temporarily")
            )
            if is_retryable and attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("Gemini call retrying in %ds after: %s", wait, exc)
                time.sleep(wait)
                continue
            raise

    raise RuntimeError("Gemini parsing failed after retries.")


def _coerce_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    """Ensure parser output has the exact expected keys and simple value types."""
    result = {**_NULL_FIELDS, **{k: parsed.get(k) for k in _NULL_FIELDS}}
    result["date"] = _normalise_date(result.get("date"))
    result["subtotal"] = _normalise_price(str(result["subtotal"])) if result.get("subtotal") is not None else None
    result["total_amount"] = (
        _normalise_price(str(result["total_amount"])) if result.get("total_amount") is not None else None
    )
    result["currency"] = _normalise_currency(result.get("currency"))

    items: list[dict[str, str | None]] = []
    for item in result.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        price = _normalise_price(str(item.get("price"))) if item.get("price") is not None else None
        if name or price:
            items.append({"name": name or None, "price": price})
    result["items"] = items
    return result


def parse_receipt(raw_text: str, retries: int = 3, use_api: bool = True) -> dict[str, Any]:
    """
    Parse raw OCR text into receipt fields.

    Gemini is used when ``GEMINI_API_KEY`` is available. If the API is
    unavailable or returns invalid JSON, a regex parser emits a partial result
    so downstream JSON generation still succeeds.
    """
    if not raw_text.strip():
        return {**_NULL_FIELDS, "flags": ["parser_failed"]}

    if use_api:
        try:
            return _parse_with_gemini(raw_text, retries)
        except Exception as exc:
            logger.warning("Gemini parser unavailable; using regex fallback: %s", exc)

    parsed = _parse_by_regex(raw_text)
    result = _coerce_schema(parsed)
    if use_api:
        result["flags"] = [*result.get("flags", []), "parser_fallback"]
    return result
