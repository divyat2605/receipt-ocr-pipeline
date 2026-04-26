from confidence import compute_confidence
from structurer import build_receipt_json


OCR_RESULTS = [
    {"text": "TRADER JOES", "bbox": [[0, 0], [100, 0], [100, 20], [0, 20]], "confidence": 0.95, "line": 0},
    {"text": "2024-06-28", "bbox": [[0, 30], [90, 30], [90, 50], [0, 50]], "confidence": 0.9, "line": 1},
    {"text": "CARROTS", "bbox": [[0, 60], [80, 60], [80, 80], [0, 80]], "confidence": 0.86, "line": 2},
    {"text": "1.29", "bbox": [[100, 60], [140, 60], [140, 80], [100, 80]], "confidence": 0.88, "line": 2},
    {"text": "TOTAL", "bbox": [[0, 90], [60, 90], [60, 110], [0, 110]], "confidence": 0.92, "line": 3},
    {"text": "1.29", "bbox": [[100, 90], [140, 90], [140, 110], [100, 110]], "confidence": 0.9, "line": 3},
    {"text": "USD", "bbox": [[0, 120], [40, 120], [40, 140], [0, 140]], "confidence": 0.93, "line": 4},
]


def test_confidence_scores_fields_and_flags_missing_subtotal():
    parsed = {
        "store_name": "TRADER JOES",
        "date": "2024-06-28",
        "items": [{"name": "CARROTS", "price": "1.29"}],
        "subtotal": None,
        "total_amount": "1.29",
        "currency": "USD",
    }

    fields, flags = compute_confidence(parsed, OCR_RESULTS, "", 0.9)

    assert fields["store_name"]["confidence"] >= 0.95
    assert fields["date"]["confidence"] == 1.0
    assert fields["subtotal"]["value"] is None
    assert "missing:subtotal" in flags
    assert "low_confidence:subtotal" in flags


def test_structurer_adds_item_confidence_from_ocr_matches():
    parsed = {
        "store_name": "TRADER JOES",
        "date": "2024-06-28",
        "items": [{"name": "CARROTS", "price": "1.29"}],
        "subtotal": None,
        "total_amount": "1.29",
        "currency": "USD",
    }

    result = build_receipt_json("receipt_1", parsed, OCR_RESULTS, "", 0.9)

    assert result["receipt_id"] == "receipt_1"
    assert result["items"][0]["confidence"] >= 0.86
    assert result["currency"] == "USD"
