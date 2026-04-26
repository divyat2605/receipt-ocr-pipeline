"""Receipt OCR Pipeline entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from ocr import run_ocr
from parser import parse_receipt
from preprocess import preprocess
from structurer import build_receipt_json
from summary import write_summary

load_dotenv()

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


def _write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _process_one(
    image_path: str,
    output_dir: str,
    use_api: bool,
    cache_ocr: bool,
) -> str:
    """Process one receipt image end-to-end and write its JSON output."""
    receipt_id = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{receipt_id}.json")

    try:
        processed_img, pre_flags = preprocess(image_path)
        if processed_img is None:
            _write_json(out_path, {
                "receipt_id": receipt_id,
                "error": "; ".join(pre_flags),
                "flags": pre_flags,
            })
            return out_path

        word_results, raw_text, avg_conf = run_ocr(processed_img)
        if cache_ocr:
            _write_text(os.path.join(output_dir, f"{receipt_id}.ocr.txt"), raw_text)

        parsed = parse_receipt(raw_text, use_api=use_api)
        receipt_json = build_receipt_json(
            receipt_id=receipt_id,
            parsed=parsed,
            word_results=word_results,
            raw_text=raw_text,
            avg_ocr_confidence=avg_conf,
            extra_flags=pre_flags,
        )
        _write_json(out_path, receipt_json)

    except Exception as exc:
        logger.error("Unhandled error processing %s: %s", image_path, exc, exc_info=True)
        _write_json(out_path, {
            "receipt_id": receipt_id,
            "error": str(exc),
            "flags": ["processing_failed"],
        })

    return out_path


def _collect_images(input_dir: str) -> list[str]:
    """Recursively find supported image files in input_dir."""
    images: list[str] = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if Path(fname).suffix.lower() in _SUPPORTED_EXTENSIONS:
                images.append(os.path.join(root, fname))
    return sorted(images)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch receipt OCR pipeline (OpenCV + EasyOCR + Gemini).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing receipt images; searched recursively.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where per-receipt JSONs and summary.json are written.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel processing threads.",
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Skip Gemini calls and use the local regex parser only.",
    )
    parser.add_argument(
        "--cache-ocr",
        action="store_true",
        help="Write raw OCR text files next to JSON outputs for debugging.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    """Run the full batch pipeline."""
    args = _parse_args()
    _configure_logging(args.verbose)

    if not os.path.isdir(args.input):
        logger.error("Input directory not found: %s", args.input)
        raise SystemExit(1)

    os.makedirs(args.output, exist_ok=True)
    images = _collect_images(args.input)
    if not images:
        logger.warning("No supported images found in %s", args.input)
        return

    logger.info("Found %d image(s). Processing with %d worker(s).", len(images), args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                img,
                args.output,
                not args.no_api,
                args.cache_ocr,
            ): img
            for img in images
        }
        with tqdm(total=len(images), desc="Processing receipts", unit="img") as pbar:
            for future in as_completed(futures):
                img_path = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error("Future raised for %s: %s", img_path, exc)
                finally:
                    pbar.update(1)

    summary_path = write_summary(args.output)
    logger.info("Done. Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
