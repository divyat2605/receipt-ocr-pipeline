"""EasyOCR wrapper that preserves text, bounding boxes, and confidence."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import Any, TypedDict

import easyocr
import numpy as np

logger = logging.getLogger(__name__)


class OCRResult(TypedDict):
    """One EasyOCR detection with geometry and confidence."""

    text: str
    bbox: list[list[float]]
    confidence: float
    line: int


_reader: easyocr.Reader | None = None
_MODEL_DIR = Path(__file__).resolve().parent / ".easyocr_models"
_reader_lock = threading.Lock()
_ocr_lock = threading.Lock()


def _get_reader() -> easyocr.Reader:
    """Return the shared EasyOCR reader, using a project-local model cache."""
    global _reader
    with _reader_lock:
        if _reader is None:
            logger.info("Initialising EasyOCR reader with project-local model cache.")
            _MODEL_DIR.mkdir(exist_ok=True)
            try:
                _reader = easyocr.Reader(
                    ["en"],
                    gpu=True,
                    model_storage_directory=str(_MODEL_DIR),
                    user_network_directory=str(_MODEL_DIR),
                    verbose=False,
                )
            except RuntimeError as exc:
                if "file name in directory" not in str(exc).lower():
                    raise
                logger.warning("EasyOCR model cache looked corrupt; clearing local cache.")
                shutil.rmtree(_MODEL_DIR, ignore_errors=True)
                _MODEL_DIR.mkdir(exist_ok=True)
                _reader = easyocr.Reader(
                    ["en"],
                    gpu=True,
                    model_storage_directory=str(_MODEL_DIR),
                    user_network_directory=str(_MODEL_DIR),
                    verbose=False,
                )
    return _reader


def _bbox_top(bbox: list[list[float]]) -> float:
    return min(point[1] for point in bbox)


def _bbox_left(bbox: list[list[float]]) -> float:
    return min(point[0] for point in bbox)


def _assign_lines(results: list[OCRResult]) -> None:
    """Assign approximate line indexes by grouping detections with similar y values."""
    if not results:
        return

    heights = [
        max(point[1] for point in item["bbox"]) - min(point[1] for point in item["bbox"])
        for item in results
    ]
    tolerance = max(8.0, float(np.median(heights)) * 0.65)

    current_line = -1
    current_y: float | None = None
    for item in sorted(results, key=lambda r: (_bbox_top(r["bbox"]), _bbox_left(r["bbox"]))):
        y = _bbox_top(item["bbox"])
        if current_y is None or abs(y - current_y) > tolerance:
            current_line += 1
            current_y = y
        item["line"] = current_line


def _raw_text(results: list[OCRResult]) -> str:
    """Return OCR text ordered by line, with each visual line joined by spaces."""
    lines: dict[int, list[OCRResult]] = {}
    for item in results:
        lines.setdefault(item["line"], []).append(item)
    return "\n".join(
        " ".join(item["text"] for item in sorted(items, key=lambda r: _bbox_left(r["bbox"])))
        for _, items in sorted(lines.items())
    )


def run_ocr(image: np.ndarray) -> tuple[list[OCRResult], str, float]:
    """
    Run EasyOCR on a preprocessed image.

    Returns:
        ``(ocr_results, raw_text, avg_confidence)`` where every OCR result
        includes text, bounding box, confidence, and approximate line number.
    """
    reader = _get_reader()
    with _ocr_lock:
        detections: list[tuple[Any, str, float]] = reader.readtext(
            image, detail=1, paragraph=False
        )

    results: list[OCRResult] = []
    for bbox, text, conf in detections:
        stripped = text.strip()
        if not stripped:
            continue
        results.append({
            "text": stripped,
            "bbox": [[float(x), float(y)] for x, y in bbox],
            "confidence": float(conf),
            "line": 0,
        })

    if not results:
        logger.warning("EasyOCR returned no text for this image.")
        return [], "", 0.0

    _assign_lines(results)
    results.sort(key=lambda r: (r["line"], _bbox_left(r["bbox"])))
    avg_confidence = float(np.mean([r["confidence"] for r in results]))

    return results, _raw_text(results), avg_confidence
