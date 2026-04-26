"""Per-field confidence scoring from OCR confidence, geometry, and heuristics."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from ocr import OCRResult

logger = logging.getLogger(__name__)

KNOWN_CURRENCIES = frozenset(
    {
        "USD", "GBP", "EUR", "INR", "CAD", "AUD", "JPY", "CHF", "CNY",
        "MXN", "SGD", "NZD", "MYR", "ZAR", "SAR",
    }
)
_DATE_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})"
    r"|(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})"
    r"|([A-Za-z]+ \d{1,2},?\s*\d{2,4})",
)
_TOTAL_KW_RE = re.compile(
    r"\b(TOTAL|AMOUNT|GRAND\s+TOTAL|BALANCE\s+DUE|AMOUNT\s+DUE)\b",
    re.IGNORECASE,
)
_LOW_CONF_THRESHOLD = 0.70
_MISSING_CONFIDENCE = 0.0


def _norm(text: Any) -> str:
    """Normalize text for fuzzy matching."""
    return re.sub(r"[^a-z0-9.]+", " ", str(text).lower()).strip()


def _similarity(left: str, right: str) -> float:
    """Return fuzzy text similarity in [0, 1]."""
    left_n = _norm(left)
    right_n = _norm(right)
    if not left_n or not right_n:
        return 0.0
    if left_n in right_n or right_n in left_n:
        return 1.0
    return SequenceMatcher(None, left_n, right_n).ratio()


def _line_text(results: list[OCRResult], line: int) -> str:
    return " ".join(r["text"] for r in results if r["line"] == line)


def _line_confidence(results: list[OCRResult], line: int) -> float | None:
    confs = [r["confidence"] for r in results if r["line"] == line]
    return float(sum(confs) / len(confs)) if confs else None


def _best_ocr_conf_for_value(value: Any, results: list[OCRResult]) -> float | None:
    """
    Estimate confidence for a field value from OCR detections.

    It first matches the full visual line, then individual detections. This is a
    practical approximation of field-region confidence without needing the LLM
    to return coordinates.
    """
    if value is None or value == "" or not results:
        return None

    value_text = str(value)
    best_score = 0.0
    best_conf: float | None = None

    for line in sorted({r["line"] for r in results}):
        text = _line_text(results, line)
        score = _similarity(value_text, text)
        conf = _line_confidence(results, line)
        if conf is not None and score > best_score:
            best_score = score
            best_conf = conf

    matched = [
        r["confidence"]
        for r in results
        if _similarity(value_text, r["text"]) >= 0.72
    ]
    if matched:
        token_conf = float(sum(matched) / len(matched))
        if best_conf is None or token_conf > best_conf:
            best_conf = token_conf

    if best_conf is None:
        return None
    return best_conf if best_score >= 0.35 or matched else None


def _line_of_value(value: Any, results: list[OCRResult]) -> int | None:
    """Return the most likely line containing a value."""
    if value is None:
        return None
    best_line = None
    best_score = 0.0
    for line in sorted({r["line"] for r in results}):
        score = _similarity(str(value), _line_text(results, line))
        if score > best_score:
            best_line = line
            best_score = score
    return best_line if best_score >= 0.35 else None


def _total_keyword_near_value(total_value: Any, results: list[OCRResult]) -> bool:
    """Return True when TOTAL-family keywords appear within two visual lines."""
    line = _line_of_value(total_value, results)
    if line is None:
        return False
    for candidate in range(line - 2, line + 3):
        if _TOTAL_KW_RE.search(_line_text(results, candidate)):
            return True
    return False


def _store_in_top_20_percent(store_value: Any, results: list[OCRResult]) -> bool:
    """Return True when the store name appears near the top of the receipt."""
    line = _line_of_value(store_value, results)
    if line is None:
        return False
    max_line = max((r["line"] for r in results), default=0)
    return line <= max(1, int((max_line + 1) * 0.20))


def compute_item_confidence(
    item: dict[str, Any],
    ocr_results: list[OCRResult],
    avg_ocr_confidence: float,
) -> float:
    """Estimate an item row confidence from name/price OCR matches."""
    name_conf = _best_ocr_conf_for_value(item.get("name"), ocr_results)
    price_conf = _best_ocr_conf_for_value(item.get("price"), ocr_results)
    confs = [c for c in (name_conf, price_conf) if c is not None]
    if not confs:
        return round(avg_ocr_confidence * 0.55, 4)
    return round(sum(confs) / len(confs), 4)


def compute_confidence(
    parsed: dict[str, Any],
    ocr_results: list[OCRResult],
    raw_text: str,
    avg_ocr_confidence: float,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """
    Compute per-field confidence and flags.

    Base confidence comes from OCR detections that align with the parsed value.
    Date, total, store, and currency fields receive small validation boosts.
    Missing or low-confidence fields are flagged.
    """
    del raw_text
    flags: list[str] = []

    def _score(field_name: str, field_value: Any, boost: float = 0.0) -> float:
        if field_value is None or field_value == "":
            flags.append(f"missing:{field_name}")
            return _MISSING_CONFIDENCE
        base = _best_ocr_conf_for_value(field_value, ocr_results)
        if base is None:
            base = avg_ocr_confidence * 0.55
        return min(1.0, base + boost)

    store_val = parsed.get("store_name")
    store_boost = 0.05 if _store_in_top_20_percent(store_val, ocr_results) else 0.0
    store_conf = _score("store_name", store_val, store_boost)

    date_val = parsed.get("date")
    date_boost = 0.10 if (date_val and _DATE_RE.search(str(date_val))) else 0.0
    date_conf = _score("date", date_val, date_boost)

    subtotal_val = parsed.get("subtotal")
    subtotal_conf = _score("subtotal", subtotal_val)

    total_val = parsed.get("total_amount")
    if not total_val and subtotal_val:
        total_val = subtotal_val
        flags.append("total_from_subtotal")
    total_boost = 0.10 if _total_keyword_near_value(total_val, ocr_results) else 0.0
    total_conf = _score("total_amount", total_val, total_boost)

    currency_val = parsed.get("currency")
    currency_str = currency_val.upper() if isinstance(currency_val, str) else None
    currency_boost = 0.10 if currency_str in KNOWN_CURRENCIES else 0.0
    currency_conf = _score("currency", currency_val, currency_boost)

    fields: dict[str, dict[str, Any]] = {
        "store_name": {"value": store_val, "confidence": round(store_conf, 4)},
        "date": {"value": date_val, "confidence": round(date_conf, 4)},
        "subtotal": {"value": subtotal_val, "confidence": round(subtotal_conf, 4)},
        "total_amount": {"value": total_val, "confidence": round(total_conf, 4)},
        "currency": {"value": currency_val, "confidence": round(currency_conf, 4)},
    }

    for field_name, data in fields.items():
        if data["confidence"] < _LOW_CONF_THRESHOLD:
            flags.append(f"low_confidence:{field_name}")

    return fields, flags
