"""Unified watermark-removal pipeline."""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from .inpainting import get_lama_inpainter, opencv_inpaint
from .io import cv2_read, cv2_read_gray, cv2_write_with_size_limit, save_exif
from .masks import (auto_detect_watermark, corner_mask, detect_doubao_watermark,
                    dilate_mask, external_mask, multi_scale_template_mask,
                    region_mask, _refine_mask_to_watermark_edges)


@dataclass
class ProcessResult:
    status: str = 'pending'
    output_path: str = ''
    watermarks_found: int = 0
    confidence: float = 0.0
    backend_used: str = ''
    elapsed: float = 0.0
    warnings: list = field(default_factory=list)
    error_code: str = ''
    mask_generated: Optional[np.ndarray] = None


def _detect_residual_watermark(output, mask, original_gray):
    """Check if watermark strokes remain after inpainting.

    Compares edge content in the repaired region against the original.
    Returns a mask of residual candidates, or None if clean.
    """
    output_gray = cv2.cvtColor(np.ascontiguousarray(output), cv2.COLOR_BGR2GRAY)
    # Dilate original mask slightly to capture surrounding area
    check_mask = cv2.dilate(mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                            iterations=1)
    ys, xs = np.where(check_mask > 0)
    if len(xs) < 10:
        return None

    y1, y2 = max(0, ys.min()), min(output.shape[0], ys.max() + 1)
    x1, x2 = max(0, xs.min()), min(output.shape[1], xs.max() + 1)

    roi_out = output_gray[y1:y2, x1:x2]
    roi_orig = original_gray[y1:y2, x1:x2]

    if roi_out.size < 9:
        return None

    # Extract candidates from repaired region
    from .masks import _extract_watermark_candidates
    candidates = _extract_watermark_candidates(roi_out)
    if candidates is None:
        return None

    # Compare: if repaired region has significantly more candidates than
    # a smooth inpainting would produce, those are residuals.
    cand_ratio = cv2.countNonZero(candidates) / max(1, candidates.size)
    if cand_ratio < 0.02:
        return None  # Clean enough

    # Also check original: if original had similar edge density in this
    # region, it's likely background texture, not residual watermark.
    orig_edges = cv2.Canny(roi_orig, 50, 150)
    orig_ratio = cv2.countNonZero(orig_edges) / max(1, orig_edges.size)
    if cand_ratio < orig_ratio * 0.5:
        return None  # Residual is less than original texture

    # Build residual mask in full-image coordinates
    result = np.zeros_like(mask)
    result[y1:y2, x1:x2] = candidates
    # Only keep pixels that were NOT in the original mask (true residuals)
    result = cv2.bitwise_and(result, cv2.bitwise_not(mask))
    return result if cv2.countNonZero(result) > 0 else None


def process(input_path: str, output_path: str = '', mode: str = 'corner',
            reference_path: str = '', corner: str = 'bottom-right', region=None,
            mask_path: str = '', backend: str = 'lama', scan_pct: float = .18,
            padding: int = 15, max_size_mb: float = 5,
            progress_callback: Optional[Callable] = None,
            cancel_token: Optional[Any] = None, fixed_position=None,
            debug_mask_path: str = '') -> ProcessResult:
    """Process one image.

    A debug mask is written only when ``debug_mask_path`` is explicitly supplied;
    normal processing never leaves diagnostic files in the ``cleaned`` folder.
    """
    result, started = ProcessResult(), time.time()

    def report(value, message):
        if progress_callback:
            progress_callback(value, message)

    try:
        if cancel_token and cancel_token.cancelled:
            result.status = 'cancelled'
            return result
        report(.05, 'Reading image...')
        image = cv2_read(input_path)
        if image is None:
            result.status, result.error_code = 'failed', 'READ_ERROR'
            return result
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if not output_path:
            src = Path(input_path)
            output_path = str(src.parent / 'cleaned' / f'cleaned_{src.name}')
        try:
            if os.path.samefile(input_path, output_path):
                src = Path(input_path)
                output_path = str(src.parent / 'cleaned' / f'cleaned_{src.name}')
        except (OSError, FileNotFoundError):
            pass

        report(.2, 'Generating mask...')
        if mode == 'template':
            template = cv2_read_gray(reference_path) if reference_path else None
            if template is None:
                result.status, result.error_code = 'failed', 'NO_REFERENCE'
                return result
            mask = multi_scale_template_mask(gray, template)
        elif mode == 'corner':
            mask = None
            ocr_bbox = None
            # Step 1: Try OCR-based detection for precise positioning
            if image is not None:
                ocr_result = detect_doubao_watermark(image, corner, scan_pct)
                if ocr_result is not None:
                    if len(ocr_result) == 3:
                        _, _, ocr_bbox = ocr_result
                    result.warnings.append(f'OCR watermark detected')
            # Step 2: Use fixed_position if available (more precise than OCR bbox)
            # OCR bbox often has extra padding; fixed_position is tighter
            if fixed_position is not None:
                mask = corner_mask(h, w, corner, scan_pct, fixed_position=fixed_position)
            elif ocr_bbox is not None:
                mask = corner_mask(h, w, corner, scan_pct, ocr_bbox=ocr_bbox)
            else:
                # Auto corner rectangle
                mask = corner_mask(h, w, corner, scan_pct)
        elif mode == 'region':
            if region is None:
                result.status, result.error_code = 'failed', 'NO_REGION'
                return result
            mask = region_mask(h, w, region)
        elif mode == 'mask':
            mask = external_mask(mask_path, h, w) if mask_path else None
        elif mode == 'auto':
            mask = auto_detect_watermark(gray, scan_pct)
        else:
            result.status, result.error_code = 'failed', 'INVALID_MODE'
            return result

        if mask is None or not np.any(mask):
            result.status = 'no_watermark'
            return result
        # Stroke masks already include a small safety dilation. Cap user
        # padding for refined fixed-position masks so a precise mask cannot
        # grow back into a destructive full-corner repair.
        effective_padding = min(padding, 5) if mode == 'corner' and fixed_position is not None else padding
        mask = dilate_mask(mask, effective_padding)
        result.mask_generated = mask.copy()

        if debug_mask_path:
            from .io import cv2_write
            if not cv2_write(debug_mask_path, mask):
                result.warnings.append(f'Could not save debug mask: {debug_mask_path}')

        mask_area = int(cv2.countNonZero(mask))
        result.confidence = mask_area / float(h * w)
        result.watermarks_found = 1
        if cancel_token and cancel_token.cancelled:
            result.status = 'cancelled'
            return result

        report(.5, f'Inpainting with {backend}...')
        if backend == 'lama':
            output = get_lama_inpainter().inpaint(image, mask)
        elif backend == 'opencv':
            output = opencv_inpaint(image, mask)
        else:
            result.status, result.error_code = 'failed', 'INVALID_BACKEND'
            return result
        result.backend_used = backend

        # --- Residual detection and secondary repair ---
        # After inpainting, check if watermark candidates remain in the
        # repaired region. If so, expand mask slightly and re-inpaint.
        if mode == 'corner' and output is not None:
            residual_mask = _detect_residual_watermark(output, mask, gray)
            if residual_mask is not None and cv2.countNonZero(residual_mask) > 0:
                result.warnings.append('Residual watermark detected; running secondary repair')
                combined_mask = cv2.bitwise_or(mask, residual_mask)
                combined_mask = cv2.dilate(combined_mask,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                    iterations=1)
                if backend == 'lama':
                    output = get_lama_inpainter().inpaint(image, combined_mask)
                elif backend == 'opencv':
                    output = opencv_inpaint(image, combined_mask)
                result.mask_generated = combined_mask

        report(.8, 'Saving result...')
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2_write_with_size_limit(output_path, output, max_size_mb):
            result.status, result.error_code = 'failed', 'WRITE_ERROR'
            return result
        save_exif(input_path, output_path)
        result.output_path, result.status = output_path, 'success'
        report(1., 'Complete')
        return result
    except Exception as exc:
        result.status, result.error_code = 'failed', 'PROCESS_ERROR'
        result.warnings.append(str(exc))
        return result
    finally:
        result.elapsed = time.time() - started

