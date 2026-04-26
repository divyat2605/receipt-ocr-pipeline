"""OpenCV preprocessing: load -> grayscale -> denoise -> deskew -> CLAHE."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_WHITE_THRESHOLD = 240
_BLANK_RATIO = 0.99
_MIN_DIMENSION_FOR_OCR = 1200


def _load_gray(image_path: str) -> np.ndarray | None:
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Failed to load image: %s", image_path)
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _is_blank(gray: np.ndarray) -> bool:
    white_ratio = np.sum(gray > _WHITE_THRESHOLD) / gray.size
    return white_ratio > _BLANK_RATIO


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Reduce image noise before skew detection and OCR."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect a dominant skew angle with Hough lines and rotate to correct it."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        if lines is None:
            return gray

        angles: list[float] = []
        for line in lines:
            _, theta = line[0]
            angle = float(theta * 180.0 / np.pi) - 90.0
            if abs(angle) < 45.0:
                angles.append(angle)
        if not angles:
            return gray

        median_angle = float(np.median(angles))
        if abs(median_angle) < 0.5:
            return gray

        h, w = gray.shape
        transform = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
        return cv2.warpAffine(
            gray,
            transform,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception as exc:
        logger.warning("Deskew failed, skipping: %s", exc)
        return gray


def _clahe(gray: np.ndarray) -> np.ndarray:
    """Enhance local contrast with CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _resize_if_small(gray: np.ndarray) -> np.ndarray:
    """Scale small receipts up so OCR sees clearer text."""
    h, w = gray.shape
    if min(h, w) >= _MIN_DIMENSION_FOR_OCR:
        return gray
    scale = _MIN_DIMENSION_FOR_OCR / max(1, min(h, w))
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess(image_path: str) -> tuple[np.ndarray | None, list[str]]:
    """
    Run the full preprocessing pipeline for one receipt image.

    Returns ``(processed_image, flags)``. The image is ``None`` when it cannot
    be used, and flags describe the reason.
    """
    gray = _load_gray(image_path)
    if gray is None:
        return None, ["image_load_failed"]
    if _is_blank(gray):
        return None, ["blank_image"]

    denoised = _denoise(gray)
    deskewed = _deskew(denoised)
    enhanced = _clahe(deskewed)
    resized = _resize_if_small(enhanced)
    return resized, []
