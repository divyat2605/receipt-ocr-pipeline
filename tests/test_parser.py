from parser import parse_receipt


def test_regex_parser_extracts_core_fields_without_api():
    raw_text = """TRADER JOES
123 MAIN ST
06/28/2024
CARROTS 1.29
MILK $3.50
SUBTOTAL $4.79
TOTAL $4.79"""

    parsed = parse_receipt(raw_text, use_api=False)

    assert parsed["store_name"] == "TRADER JOES"
    assert parsed["date"] == "2024-06-28"
    assert parsed["items"] == [
        {"name": "CARROTS", "price": "1.29"},
        {"name": "MILK", "price": "3.50"},
    ]
    assert parsed["subtotal"] == "4.79"
    assert parsed["total_amount"] == "4.79"
    assert parsed["currency"] == "USD"


def test_regex_parser_handles_empty_ocr_text():
    parsed = parse_receipt("", use_api=False)

    assert parsed["store_name"] is None
    assert "parser_failed" in parsed["flags"]


def test_parser_normalises_currency_aliases():
    raw_text = """SHOP
TOTAL RM 12.40"""

    parsed = parse_receipt(raw_text, use_api=False)

    assert parsed["currency"] == "MYR"
