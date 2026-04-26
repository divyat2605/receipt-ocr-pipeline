# Receipt OCR Pipeline

Batch pipeline that converts receipt images into structured JSON with confidence scores.

The pipeline follows:

```text
image -> OpenCV preprocessing -> EasyOCR text + bbox + confidence
      -> Gemini parser, with regex fallback
      -> structured JSON + field confidence
      -> aggregate summary.json
```

## Features

- OpenCV preprocessing: grayscale, blank-image detection, denoise, deskew, CLAHE, and small-image upscaling.
- EasyOCR with GPU support when available.
- OCR detections preserve text, bounding boxes, confidence, and visual line numbers.
- Gemini parser using `gemini-2.5-flash-lite`.
- Local regex parser fallback for offline runs or API failures.
- Per-field confidence from OCR matches plus validation boosts.
- Missing and low-confidence fields are flagged.
- Per-receipt JSON plus aggregate `summary.json`.
- Unit tests for parser fallback, confidence scoring, structuring, and summary aggregation.

## Project Structure

```text
receipt_ocr/
├── main.py
├── preprocess.py
├── ocr.py
├── parser.py
├── confidence.py
├── structurer.py
├── summary.py
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Requirements

- Python 3.10 or 3.11 recommended
- NVIDIA GPU with CUDA 12.1 drivers recommended
- Gemini API key for Gemini parsing

The pipeline can run without an API key by using `--no-api`, but Gemini parsing is the configured API path.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env`:

```text
GEMINI_API_KEY=your_key_here
```

## Usage

Put receipt images in `dataset/`:

```text
dataset/
  receipt_001.jpg
  receipt_002.png
```

Run with Gemini parsing:

```powershell
python main.py --input .\dataset --output .\results --workers 4
```

Run offline with the regex fallback:

```powershell
python main.py --input .\dataset --output .\results --workers 4 --no-api
```

Useful debug flags:

```powershell
python main.py --input .\dataset --output .\results --verbose --cache-ocr
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`.

## Output Schema

Each receipt produces `results/<receipt_id>.json`:

```json
{
  "receipt_id": "receipt_001",
  "store_name": { "value": "TRADER JOES", "confidence": 0.95 },
  "date": { "value": "2024-06-28", "confidence": 1.0 },
  "items": [
    { "name": "CARROTS", "price": "1.29", "confidence": 0.88 }
  ],
  "subtotal": { "value": "4.79", "confidence": 0.91 },
  "total_amount": { "value": "4.79", "confidence": 0.98 },
  "currency": "USD",
  "flags": [],
  "ocr_avg_confidence": 0.89
}
```

The aggregate `results/summary.json` includes:

```json
{
  "total_receipts": 371,
  "total_spend": 4821.3,
  "currencies_detected": ["USD"],
  "transactions_by_store": {
    "TRADER JOES": { "count": 5, "total": 120.45 }
  },
  "low_confidence_receipts": ["receipt_003"],
  "flagged_fields_count": 12
}
```

## Confidence Logic

- Base confidence comes from EasyOCR detections that align with each parsed field value.
- Store confidence receives a small boost when the store appears near the top of the receipt.
- Date confidence receives a boost when the value matches a supported date pattern.
- Total confidence receives a boost when a total-related keyword appears within two visual lines.
- Currency confidence receives a boost for known ISO 4217 codes.
- Item confidence uses OCR matches for the item name and item price.
- Missing fields get `missing:<field>` and low scores.
- Scores below `0.70` get `low_confidence:<field>`.

## Edge Cases

- Skewed images: corrected with Hough-line deskew and `cv2.warpAffine`.
- Low contrast: enhanced with CLAHE.
- Blank or unreadable images: emit an error JSON with flags.
- Missing fields: emit `null` values with confidence `0.0` and flags.
- Partial receipts: emit whatever can be extracted.
- Multi-currency: supports common symbols and ISO codes such as USD, INR, EUR, and GBP.

## Results

Full pipeline results on the 371-receipt dataset are documented in [Results.MD](Results.MD) — covering success rate, financial extraction, OCR confidence, missing fields, currency distribution, item extraction, and top stores.

## Tests

```powershell
pytest
```

## Challenges and Improvements

The hard part is mapping semantic fields back to OCR evidence. The implementation keeps EasyOCR bounding boxes and line indexes, then approximates field confidence by matching parsed values back to OCR lines and tokens. A stronger production version would ask Gemini to return evidence spans or line IDs for every field, which would make confidence more exact.

The regex fallback is intentionally conservative. It is useful for offline grading and API outages, but Gemini parsing should produce better item grouping and store/date extraction on messy receipts.
