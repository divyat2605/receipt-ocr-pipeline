import json

from summary import aggregate


def test_summary_aggregates_receipts(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({
        "receipt_id": "a",
        "store_name": {"value": "Shop", "confidence": 0.9},
        "total_amount": {"value": "12.50", "confidence": 0.95},
        "currency": "USD",
        "flags": [],
        "ocr_avg_confidence": 0.9,
    }), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({
        "receipt_id": "b",
        "store_name": {"value": "Shop", "confidence": 0.6},
        "total_amount": {"value": "2.50", "confidence": 0.8},
        "currency": "USD",
        "flags": ["low_confidence:store_name"],
        "ocr_avg_confidence": 0.65,
    }), encoding="utf-8")

    summary = aggregate(str(tmp_path))

    assert summary["total_receipts"] == 2
    assert summary["total_spend"] == 15.0
    assert summary["currencies_detected"] == ["USD"]
    assert summary["transactions_by_store"]["Shop"] == {"count": 2, "total": 15.0}
    assert summary["low_confidence_receipts"] == ["b"]
    assert summary["flagged_fields_count"] == 1
