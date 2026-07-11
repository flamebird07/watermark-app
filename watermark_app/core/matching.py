"""Template extraction and matching utilities."""

import numpy as np
import cv2
from typing import Optional
from pathlib import Path
import logging

from .io import cv2_read, cv2_read_gray

logger = logging.getLogger(__name__)


def extract_template(reference_path: str,
                     x: Optional[int] = None,
                     y: Optional[int] = None,
                     w: Optional[int] = None,
                     h: Optional[int] = None) -> Optional[np.ndarray]:
    """Extract a template from a reference image.

    If no coordinates given, returns the full image as template.
    """
    template = cv2_read_gray(reference_path)
    if template is None:
        return None

    if x is not None and y is not None and w is not None and h is not None:
        template = template[y:y + h, x:x + w]

    logger.info(f"Extracted template: {template.shape[1]}x{template.shape[0]} from {reference_path}")
    return template


def match_template(gray: np.ndarray, template: np.ndarray,
                   threshold: float = 0.6) -> Optional[dict]:
    """Match template against grayscale image.

    Returns dict with match info or None.
    """
    th, tw = template.shape[:2]
    h, w = gray.shape[:2]

    if tw > w or th > h:
        return None

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    return {
        'confidence': max_val,
        'location': max_loc,
        'size': (tw, th),
        'bbox': (max_loc[0], max_loc[1], max_loc[0] + tw, max_loc[1] + th)
    }
