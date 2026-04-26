"""Assemble the final per-receipt JSON from parsed and confidence-scored fields."""

from __future__ import annotations

from typing import Any

from confidence import compute_confidence, compute_item_confidence
from ocr import OCRResult


def build_receipt_json(
    receipt_id: str,
    parsed: dict[str, Any],
    word_results: list[OCRResult],
    raw_text: str,
    avg_ocr_confidence: float,
    extra_flags: list[str] | None = None,
) -> dict[str, Any]:
    """
    Produce the final structured JSON dict for one receipt.

    Combines parsed fields, EasyOCR geometry/confidence, and regex boosts into
    the canonical output schema.
    """
    extra_flags = extra_flags or []

    fields, conf_flags = compute_confidence(
        parsed, word_results, raw_text, avg_ocr_confidence
    )

    parser_flags: list[str] = parsed.get("flags", [])
    seen: set[str] = set()
    flags: list[str] = []
    for flag in extra_flags + conf_flags + parser_flags:
        if flag not in seen:
            seen.add(flag)
            flags.append(flag)

    items: list[dict[str, Any]] = []
    for item in parsed.get("items") or []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        price = item.get("price")
        if name or price:
            items.append({
                "name": name,
                "price": price,
                "confidence": compute_item_confidence(
                    item, word_results, avg_ocr_confidence
                ),
            })

    currency_value = fields.get("currency", {}).get("value")

    return {
        "receipt_id": receipt_id,
        "store_name": fields["store_name"],
        "date": fields["date"],
        "items": items,
        "subtotal": fields["subtotal"],
        "total_amount": fields["total_amount"],
        "currency": currency_value,
        "flags": flags,
        "ocr_avg_confidence": round(avg_ocr_confidence, 4),
    }
