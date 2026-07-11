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
from .masks import (auto_detect_watermark, corner_mask, dilate_mask,
                    external_mask, has_watermark_edges,
                    multi_scale_template_mask, region_mask)


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
            if not has_watermark_edges(gray, corner, scan_pct, fixed_position=fixed_position):
                result.warnings.append(f'No strong watermark edges detected in {corner}; processing requested area')
            mask = corner_mask(h, w, corner, scan_pct, fixed_position, gray)
            if fixed_position is not None and mask is None:
                result.warnings.append('No watermark-like strokes found in fixed_position; source left unchanged')
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

