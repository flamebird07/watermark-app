"""Mask creation utilities."""

import cv2
import numpy as np

from .io import cv2_read_gray


def _gray_u8(gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(gray)


def _canny(gray, low=50, high=150):
    """Canny with the exact dtype/layout required by OpenCV."""
    return cv2.Canny(_gray_u8(gray), low, high)


def _refine_mask_to_watermark_edges(gray, x1, y1, x2, y2):
    """Build a stroke-level repair mask inside a user supplied search box.

    ``fixed_position`` is a search constraint, not permission to repaint the
    complete rectangle.  Keeping this distinction here prevents a 80--100%
    corner preset from destroying one fifth of the source image.
    """
    gray = _gray_u8(gray)
    h, w = gray.shape
    x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
    y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
    roi = gray[y1:y2, x1:x2]
    if roi.size < 9 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None

    # Canny finds translucent lettering more reliably after local contrast
    # normalization. Otsu contributes filled glyph pixels instead of only the
    # two outlines produced by Canny.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(roi)
    edges = cv2.Canny(enhanced, 30, 100)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Only threshold pixels close to an edge are candidates. This avoids
    # selecting an entire bright/dark background half of the ROI.
    near_edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    candidates = edges | ((bright | dark) & near_edges)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    refined = np.zeros_like(roi)
    roi_area = roi.size
    for label in range(1, count):
        bx, by, bw, bh, area = stats[label]
        # Reject isolated noise and components large enough to be scenery or
        # the complete fixed-position rectangle.
        if area < 3 or area > roi_area * .35:
            continue
        if bw > roi.shape[1] * .9 and bh > roi.shape[0] * .9:
            continue
        refined[labels == label] = 255

    if not np.any(refined):
        return None
    refined = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if cv2.countNonZero(refined) > roi_area * .55:
        return None
    result = np.zeros((h, w), np.uint8)
    result[y1:y2, x1:x2] = refined
    return result


def multi_scale_template_mask(gray, template, threshold=.55, scales=None):
    gray, template = _gray_u8(gray), _gray_u8(template)
    h, w = gray.shape
    th, tw = template.shape
    mask = np.zeros((h, w), np.uint8)
    for scale in scales or np.linspace(.3, 2, 20):
        sw, sh = int(tw * scale), int(th * scale)
        if sw < 3 or sh < 3 or sw > w or sh > h:
            continue
        resized = cv2.resize(template, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED))
        if score >= threshold:
            x, y = loc
            mask[y:y + sh, x:x + sw] = 255
    return mask if np.any(mask) else None


template_mask = multi_scale_template_mask


def corner_mask(h, w, corner='bottom-right', scan_pct=.18, fixed_position=None, gray=None):
    if fixed_position is not None:
        x1, y1, x2, y2 = fixed_position
        if all(0 <= v <= 100 for v in fixed_position):
            x1, x2 = int(w*x1/100), int(w*x2/100)
            y1, y2 = int(h*y1/100), int(h*y2/100)
        x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
        if gray is not None:
            return _refine_mask_to_watermark_edges(gray, x1, y1, x2, y2)
    else:
        sw, sh = max(1, int(w*scan_pct)), max(1, int(h*scan_pct))
        boxes = {'top-left': (0,0,sw,sh), 'top-right': (w-sw,0,w,sh),
                 'bottom-left': (0,h-sh,sw,h), 'bottom-right': (w-sw,h-sh,w,h)}
        x1, y1, x2, y2 = boxes.get(corner, boxes['bottom-right'])
    mask = np.zeros((h, w), np.uint8)
    mask[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 255
    return mask


def region_mask(h, w, region):
    x, y, rw, rh = region
    mask = np.zeros((h, w), np.uint8)
    mask[max(0,y):min(h,y+rh), max(0,x):min(w,x+rw)] = 255
    return mask


def external_mask(path, h, w):
    mask = cv2_read_gray(path)
    if mask is None:
        return None
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray((mask > 127).astype(np.uint8) * 255)


def dilate_mask(mask, padding=15):
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    if padding <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding*2+1, padding*2+1))
    return cv2.dilate(mask, kernel)


def has_watermark_edges(gray, corner='bottom-right', scan_pct=.18, threshold=.02, fixed_position=None):
    gray = _gray_u8(gray)
    mask = corner_mask(*gray.shape, corner, scan_pct, fixed_position)
    # Crop before Canny so the artificial black boundary of a masked full image
    # is not counted as watermark detail.
    if mask is None:
        return False
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return False
    roi = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return cv2.countNonZero(_canny(roi)) / max(1, roi.size) > threshold


def auto_detect_watermark(gray, scan_pct=.18):
    gray = _gray_u8(gray)
    scored = []
    for corner in ('bottom-right','bottom-left','top-right','top-left'):
        mask = corner_mask(*gray.shape, corner, scan_pct)
        ys, xs = np.where(mask > 0)
        roi = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
        scored.append((cv2.countNonZero(_canny(roi))/max(1,roi.size), mask))
    score, mask = max(scored, key=lambda item: item[0])
    return mask if score > .02 else None

