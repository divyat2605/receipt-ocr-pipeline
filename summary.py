"""Aggregates per-receipt JSONs into a single summary.json."""

import glob
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def aggregate(output_dir: str) -> dict[str, Any]:
    """Read per-receipt JSON files and compute summary stats."""
    pattern = os.path.join(output_dir, "*.json")
    receipt_files = [
        p for p in glob.glob(pattern)
        if os.path.basename(p) != "summary.json"
    ]

    total_spend = 0.0
    currencies: set[str] = set()
    store_stats: dict[str, dict[str, Any]] = {}
    low_conf_receipts: list[str] = []
    flagged_fields_count = 0

    for path in receipt_files:
        try:
            with open(path, "r", encoding="utf-8-sig") as fh:
                receipt = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
            continue

        receipt_id = receipt.get("receipt_id", os.path.splitext(os.path.basename(path))[0])

        if "error" in receipt:
            continue

        currency = receipt.get("currency")
        if isinstance(currency, str) and currency:
            currencies.add(currency.upper())

        total_field = receipt.get("total_amount", {})
        total_value = total_field.get("value") if isinstance(total_field, dict) else None
        receipt_total = _safe_float(total_value)
        total_spend += receipt_total

        store_field = receipt.get("store_name", {})
        store_name = store_field.get("value") if isinstance(store_field, dict) else None
        if store_name:
            if store_name not in store_stats:
                store_stats[store_name] = {"count": 0, "total": 0.0}
            store_stats[store_name]["count"] += 1
            store_stats[store_name]["total"] = round(
                store_stats[store_name]["total"] + receipt_total, 2
            )

        ocr_conf = receipt.get("ocr_avg_confidence", 1.0)
        if _safe_float(ocr_conf) < 0.70:
            low_conf_receipts.append(receipt_id)

        flags: list[str] = receipt.get("flags", [])
        flagged_fields_count += sum(1 for f in flags if f.startswith("low_confidence:"))

    return {
        "total_receipts": len(receipt_files),
        "total_spend": round(total_spend, 2),
        "currencies_detected": sorted(currencies),
        "transactions_by_store": store_stats,
        "low_confidence_receipts": low_conf_receipts,
        "flagged_fields_count": flagged_fields_count,
    }


def write_summary(output_dir: str) -> str:
    summary = aggregate(output_dir)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary written -> %s", summary_path)
    return summary_path
