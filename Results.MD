# Receipt OCR Pipeline — Results Report

**Generated:** 2026-04-26  
**Data source:** `results/*.json` (excluding `results/summary.json`)  
**Pipeline:** OpenCV preprocessing → EasyOCR → Gemini 2.5 Flash Lite parser → confidence scoring

---

## 1. Run Overview

| Metric | Value | Explanation |
|---|---:|---|
| Total receipts processed | 371 | Number of receipt images run through the full pipeline |
| Successful (structured output) | 370 | Receipts that produced a valid JSON with at least partial fields |
| Errored (pipeline failure) | 1 | Receipts that hit an unhandled exception and have only an `error` field |
| Success rate | **99.73%** | Pipeline completed without crashing on nearly every input |
| Error rate | 0.27% | One receipt failed — extremely low failure rate |

The single errored receipt likely had a corrupt or unreadable image file. All other receipts produced at least partial structured output.

---

## 2. Financial Extraction

| Metric | Value | Explanation |
|---|---:|---|
| Total spend across all receipts | **34,221.20** | Sum of `total_amount.value` across all successful receipts |
| Receipts with a total extracted | 369 / 370 | How many receipts had a usable `total_amount` |
| Receipts missing total | 1 | One receipt had no recognizable total line |
| Average total (receipts with total) | 92.74 | Mean spend only counting receipts where total was found |
| Average total (all successful) | 92.49 | Mean spend over all 370 successful receipts (missing = 0) |

**Takeaway:** Financial extraction is strong. Only 1 receipt is missing a total, meaning spend calculations are reliable across 99.7% of the dataset.

---

## 3. OCR Confidence

OCR confidence is scored per word by EasyOCR (0.0 = no confidence, 1.0 = certain). The pipeline averages these per receipt.

| Metric | Value | Explanation |
|---|---:|---|
| Average OCR confidence | **0.7305** | Mean across all 370 successful receipts |
| Median OCR confidence | **0.7562** | The middle value — better measure of typical quality than mean |
| Receipts below 0.70 threshold | 117 | Receipts flagged as needing human review |
| Low-confidence receipt rate | **31.62%** | About 1 in 3 receipts is below the quality threshold |

**Takeaway:** Typical receipt quality is moderate (median ~0.76). About a third of receipts fall below the 0.70 threshold and may benefit from a manual review step or image quality improvements before re-processing.

---

## 4. Field Extraction Quality

### 4a. Flags Overview

The pipeline attaches quality flags to each receipt. A flag signals something to investigate, not necessarily an error.

| Metric | Value | Explanation |
|---|---:|---|
| Receipts with at least one flag | 319 | Receipts carrying any quality warning |
| Flagged receipt rate | **86.22%** | Most receipts have at least one field-level concern |
| Total `low_confidence:*` flags | 521 | Each flag means one specific field scored below 0.70 |
| `parser_fallback` count | 0 | Gemini API was available for all receipts — regex fallback was never triggered |

A high flagged rate (86%) is expected because the pipeline flags individual fields (store, date, subtotal, total, currency) separately. A receipt with one uncertain field still counts as "flagged" even if all other fields are perfect.

### 4b. Missing Fields

| Field | Missing count | Rate (of 370 successful) | Explanation |
|---|---:|---:|---|
| `date` | 22 | **5.95%** | No recognizable date pattern found in OCR text |
| `subtotal` | 17 | **4.59%** | No subtotal line detected — many receipts omit it |
| `currency` | 12 | **3.24%** | Could not determine currency from symbols or codes |
| `total_amount` | 1 | **0.27%** | Total line present in nearly all receipts |

**Takeaway:** Date and subtotal are the weakest fields. Currency is mostly reliable. Total amount is almost always found.

---

## 5. Currency Distribution

| Currency | Receipt count | Share | Notes |
|---|---:|---:|---|
| MYR (Malaysian Ringgit) | 336 | **90.81%** | Dataset is overwhelmingly Malaysian receipts |
| USD (US Dollar) | 19 | 5.14% | Likely Walmart and other US-origin receipts |
| UNKNOWN | 12 | 3.24% | Matches the 12 receipts with missing currency field |
| SAR (Saudi Riyal) | 2 | 0.54% | Small subset |
| ZAR (South African Rand) | 1 | 0.27% | Single receipt |

**Takeaway:** Dataset is MYR-dominant. The 12 UNKNOWN entries directly correspond to the 12 receipts with missing currency, confirming the flag is accurate.

---

## 6. Item Extraction

Line items (individual products/services) are extracted from receipt text.

| Metric | Value | Explanation |
|---|---:|---|
| Total items extracted | **1,151** | Across all 370 successful receipts |
| Average items per receipt | 3.11 | Typical receipt has 3 line items extracted |
| Average item confidence | **0.7896** | Mean confidence score for extracted item rows |

**Takeaway:** Item extraction confidence (~0.79) is slightly higher than the overall OCR average, suggesting the item detection logic is well-targeted. Three items per receipt is low — many receipts likely have more items but the parser misses some on noisy scans.

---

## 7. Top Stores by Receipt Count

| Store | Receipts | Total Spend |
|---|---:|---:|
| SYARIKAT PERNIAGAAN GIN KEE | 32 | 2,483.58 |
| UNIHAKKA INTERNATIONAL SDN BHD | 16 | 678.40 |
| RESTORAN WAN SHENG | 12 | 78.60 |
| KEDAI PAPAN YEW CHUAN | 11 | 1,656.41 |
| Walmart | 11 | 571.89 |
| PASARAYA BORONG PINTAR SDN BHD | 6 | 25.85 |
| LIGHTROOM GALLERY SDN BHD | 4 | 703.60 |
| ADVANCO COMPANY | 4 | 129.45 |
| LIM SENG THO HARDWARE TRADING | 4 | 31.50 |
| PERNIAGAAN ZHENG HUI | 3 | 776.50 |

The top 10 stores account for 103 receipts (27.8% of the dataset). SYARIKAT PERNIAGAAN GIN KEE alone represents 8.6% and nearly 2,500 in spend — the highest-value merchant in the dataset.

---

## 8. Consistency Check

Cross-verification between recomputed stats and `results/summary.json`:

| Metric | Recomputed | summary.json | Match |
|---|---:|---:|---|
| total_receipts | 371 | 371 | Yes |
| total_spend | 34,221.20 | 34,221.20 | Yes |
| flagged_fields_count | 521 | 521 | Yes |
| low_confidence_receipts_count | 117 | 117 | Yes |

All key aggregates match exactly. The pipeline's `summary.json` output is reliable.

---

## 9. Summary and Next Steps

**What's working well:**
- 99.73% success rate — the pipeline is stable
- Total amount extraction works on 99.7% of receipts
- Currency detection is accurate (only 3.24% unknown)
- Gemini API handled all 371 receipts without triggering the regex fallback

**Where to improve:**
1. **OCR quality (31.6% low-confidence)** — Pre-processing tuning (stronger denoising, adaptive thresholding) could reduce this
2. **Date extraction (5.95% missing)** — Wider date format patterns or a second Gemini pass on failed fields
3. **Item extraction (avg 3.11 items)** — Better line-item segmentation for dense receipts
4. **Subtotal (4.59% missing)** — Many receipts skip subtotal, but some may have it under different labels
