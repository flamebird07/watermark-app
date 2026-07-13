"""Mask creation utilities."""

import cv2
import numpy as np
import logging

_logger = logging.getLogger(__name__)

# Lazy OCR reader singleton
_ocr_reader = None


def _get_ocr_reader():
    """Get or create a lazy EasyOCR reader (Chinese + English)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
        except Exception as e:
            _logger.warning("EasyOCR init failed: %s", e)
            return None
    return _ocr_reader


def detect_doubao_watermark(image, corner='bottom-right', scan_pct=.18):
    """Detect '豆包 AI 生成' watermark text using OCR.

    Scans the corner region of *image* for Chinese/English text that matches
    the Doubao AI watermark pattern.  Uses EasyOCR with geometric consistency
    checks (height uniformity, baseline alignment, reasonable spacing) to
    distinguish real watermark text from background texture.

    Returns ``(mask, confidence)`` where *mask* is a full-image uint8 mask
    (255 = watermark) and *confidence* is in [0, 1], or ``None`` if no
    watermark text is detected.
    """
    if image is None or image.size == 0:
        return None

    gray = _gray_u8(image)
    h, w = gray.shape
    sw, sh = max(1, int(w * scan_pct)), max(1, int(h * scan_pct))
    boxes = {
        'top-left': (0, 0, sw, sh),
        'top-right': (w - sw, 0, w, sh),
        'bottom-left': (0, h - sh, sw, h),
        'bottom-right': (w - sw, h - sh, w, h),
    }
    cx1, cy1, cx2, cy2 = boxes.get(corner, boxes['bottom-right'])

    # Crop corner region (use color image for OCR)
    roi = image[cy1:cy2, cx1:cx2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        return None

    reader = _get_ocr_reader()
    if reader is None:
        return None

    try:
        results = reader.readtext(roi, paragraph=False)
    except Exception as e:
        _logger.debug("OCR readtext failed: %s", e)
        return None

    # Also try enhanced image for semi-transparent watermarks
    enhanced_roi = None
    if not results or all(len(r) >= 3 and r[2] < 0.1 for r in results):
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray_roi)
        enhanced_roi = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        try:
            results2 = reader.readtext(enhanced_roi, paragraph=False)
            if results2:
                results = results2
        except Exception:
            pass

    if not results:
        return None

    # --- Filter for Doubao watermark keywords ---
    doubao_keywords = ['豆包', '生成', 'doubao', 'Doubao', 'DOUBAO', 'AI']
    doubao_chars = set('豆包生成')  # individual characters for fuzzy matching
    matched_boxes = []

    for item in results:
        if len(item) == 3:
            bbox, text, conf = item
        elif len(item) == 2:
            bbox, text = item
            conf = 0.5
        else:
            continue

        text_clean = text.strip()
        if not text_clean:
            continue

        # Check if text matches watermark pattern (exact or fuzzy)
        matched = False
        for kw in doubao_keywords:
            if kw in text_clean:
                matched = True
                break

        # Fuzzy: if OCR text contains enough watermark characters
        if not matched:
            text_chars = set(text_clean)
            overlap = len(text_chars & doubao_chars)
            if overlap >= 1:
                matched = True

        if matched and conf > 0.0001:
            matched_boxes.append((bbox, text_clean, conf))

    if not matched_boxes:
        return None

    # --- Geometric consistency check ---
    # Watermark text should have consistent height, baseline alignment,
    # and reasonable inter-character spacing.
    heights = []
    baselines = []
    centers_x = []

    for bbox, text, conf in matched_boxes:
        pts = np.array(bbox, dtype=np.float32)
        y_top = pts[:, 1].min()
        y_bot = pts[:, 1].max()
        x_center = pts[:, 0].mean()
        heights.append(y_bot - y_top)
        baselines.append(y_bot)
        centers_x.append(x_center)

    if len(heights) >= 2:
        h_arr = np.array(heights)
        b_arr = np.array(baselines)
        # Height consistency: std/mean < 0.4
        h_mean = np.mean(h_arr)
        h_std = np.std(h_arr)
        if h_mean > 0 and h_std / h_mean > 0.4:
            return None
        # Baseline consistency: std < 0.3 * mean_height
        b_std = np.std(b_arr)
        if h_mean > 0 and b_std > 0.3 * h_mean:
            return None
        # Spacing: sort by x, check gaps are reasonable
        cx_sorted = sorted(centers_x)
        if len(cx_sorted) >= 2:
            gaps = np.diff(cx_sorted)
            if len(gaps) > 0:
                gap_mean = np.mean(gaps)
                # Reject if gaps are too large (scattered background text)
                if gap_mean > h_mean * 8:
                    return None

    # --- Build mask from matched text polygons ---
    text_mask = np.zeros((roi.shape[0], roi.shape[1]), np.uint8)
    total_conf = 0.0
    all_pts = []

    for bbox, text, conf in matched_boxes:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(text_mask, [pts], 255)
        total_conf += conf
        all_pts.append(pts)

    # Light dilation (1-2px) to cover stroke edges
    text_mask = cv2.dilate(text_mask,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                           iterations=1)

    # Compute OCR bbox in full-image coordinates (union of all matched boxes)
    if all_pts:
        all_pts_arr = np.concatenate(all_pts, axis=0)
        ocr_x1 = int(all_pts_arr[:, 0].min()) + cx1
        ocr_y1 = int(all_pts_arr[:, 1].min()) + cy1
        ocr_x2 = int(all_pts_arr[:, 0].max()) + cx1
        ocr_y2 = int(all_pts_arr[:, 1].max()) + cy1
        # Add small padding
        pad = max(3, int(min(h, w) * 0.005))
        ocr_bbox = (max(0, ocr_x1 - pad), max(0, ocr_y1 - pad),
                    min(w, ocr_x2 + pad), min(h, ocr_y2 + pad))
    else:
        ocr_bbox = None

    if cv2.countNonZero(text_mask) == 0:
        # Even if mask is empty, return OCR bbox if available
        return (None, 0.0, ocr_bbox) if ocr_bbox else None

    # Place back in full-image coordinates
    full_mask = np.zeros((h, w), np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = text_mask

    confidence = min(1.0, total_conf / max(1, len(matched_boxes)))
    return full_mask, confidence, ocr_bbox


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


def _extract_watermark_candidates(roi):
    """Extract candidate watermark pixels from an ROI using background residual.

    Returns a binary mask of candidate pixels, or None if the ROI is too small.

    Uses morphological top-hat / black-hat to isolate bright-on-dark and
    dark-on-bright features (watermark strokes) from the background.
    Also combines Canny edges for thin stroke outlines.
    """
    if roi.size < 9 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None

    roi_h, roi_w = roi.shape[:2]
    # Kernel size for background estimation — should be larger than stroke width
    # but smaller than typical watermark extent.
    k = max(3, int(min(roi_h, roi_w) * 0.06))
    if k % 2 == 0:
        k += 1
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Top-hat: bright features on dark background (white watermark on image)
    tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel_bg)
    # Black-hat: dark features on bright background (dark watermark on image)
    blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, kernel_bg)

    # Combine: any stroke will appear in one of the two
    residual = cv2.max(tophat, blackhat)

    # Adaptive threshold on residual to get candidate pixels
    # Use Otsu on the residual — strokes should stand out
    _, candidates = cv2.threshold(residual, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If Otsu picks too few or too many pixels, fall back to a fixed percentile
    nz_ratio = cv2.countNonZero(candidates) / max(1, roi.size)
    if nz_ratio < 0.002 or nz_ratio > 0.3:
        # Fallback: use a moderate threshold
        thresh_val = max(15, int(np.percentile(residual, 85)))
        _, candidates = cv2.threshold(residual, thresh_val, 255,
                                      cv2.THRESH_BINARY)

    # Second-pass cleanup: if candidates are still too dense after threshold,
    # raise the threshold progressively to reduce background clutter.
    # This handles textured backgrounds where top-hat/black-hat produce
    # widespread residual even after Otsu.
    nz_ratio = cv2.countNonZero(candidates) / max(1, roi.size)
    if nz_ratio > 0.20:
        thresh_val = max(20, int(np.percentile(residual, 92)))
        _, candidates = cv2.threshold(residual, thresh_val, 255,
                                      cv2.THRESH_BINARY)
        nz_ratio = cv2.countNonZero(candidates) / max(1, roi.size)
        if nz_ratio > 0.20:
            thresh_val = max(25, int(np.percentile(residual, 96)))
            _, candidates = cv2.threshold(residual, thresh_val, 255,
                                          cv2.THRESH_BINARY)

    # Also add Canny edges (catches thin strokes the residual may miss)
    # Only keep edges that overlap with residual candidates to avoid
    # background texture edges from polluting the candidate set.
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(roi)
    edges = cv2.Canny(enhanced, 30, 100)

    # Dilate candidates slightly to catch nearby edges
    dilated = cv2.dilate(candidates, np.ones((3, 3), np.uint8), iterations=1)
    # Only add edges that overlap with dilated candidates
    candidates = candidates | (edges & dilated)

    # Light morphological close to connect nearby stroke fragments
    # Use a smaller kernel when candidates are dense to prevent
    # merging background texture into large blobs.
    # Skip closing entirely when candidates are very dense (>25%).
    nz_ratio = cv2.countNonZero(candidates) / max(1, roi.size)
    if nz_ratio <= 0.25:
        if nz_ratio > 0.15:
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        else:
            close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, close_k)

    return candidates


def _refine_mask_to_watermark_edges(gray, x1, y1, x2, y2):
    """Build a stroke-level repair mask inside a user supplied search box.

    ``fixed_position`` is a search constraint, not permission to repaint the
    complete rectangle.  Keeping this distinction here prevents a 80--100%
    corner preset from destroying one fifth of the source image.

    Pipeline: single candidate extraction → merge components → select
    watermark group → generate mask from that group.
    """
    gray = _gray_u8(gray)
    h, w = gray.shape
    x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
    y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
    roi = gray[y1:y2, x1:x2]
    if roi.size < 9 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w

    # --- Single candidate extraction using background residual ---
    candidates = _extract_watermark_candidates(roi)
    if candidates is None or cv2.countNonZero(candidates) == 0:
        return None

    # --- Connected components on candidates ---
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates, connectivity=8)

    min_pixels = max(3, int(roi_area * 0.0005))
    comps = []
    for lab in range(1, n_labels):
        bx = int(stats[lab, cv2.CC_STAT_LEFT])
        by = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        px = int(stats[lab, cv2.CC_STAT_AREA])

        if px < min_pixels:
            continue
        if bw > roi_w * 0.9 and bh > roi_h * 0.9:
            continue
        comps.append({'label': lab, 'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
                      'px': px, 'cx': bx + bw / 2, 'cy': by + bh / 2})

    if not comps:
        return None

    # --- Spatial clustering: group nearby components using distance ---
    # Instead of morphological closing (which merges background texture),
    # use distance-based clustering to find groups of related components.
    avg_char_w = np.median([c['bw'] for c in comps])
    avg_char_h = np.median([c['bh'] for c in comps])
    # Merge distance: ~2 character widths horizontally, ~1 character height vertically
    merge_dist_x = max(5, int(avg_char_w * 2.5))
    merge_dist_y = max(3, int(avg_char_h * 1.5))

    used = [False] * len(comps)
    groups = []
    for i, c in enumerate(comps):
        if used[i]:
            continue
        # BFS to find all nearby components
        queue = [i]
        used[i] = True
        group_comps = [c]
        while queue:
            curr = queue.pop(0)
            cx1 = comps[curr]['cx']
            cy1 = comps[curr]['cy']
            for j, d in enumerate(comps):
                if used[j]:
                    continue
                dx = abs(d['cx'] - cx1)
                dy = abs(d['cy'] - cy1)
                if dx <= merge_dist_x and dy <= merge_dist_y:
                    used[j] = True
                    queue.append(j)
                    group_comps.append(d)

        # Compute group bounding box
        min_x = min(c['bx'] for c in group_comps)
        min_y = min(c['by'] for c in group_comps)
        max_x = max(c['bx'] + c['bw'] for c in group_comps)
        max_y = max(c['by'] + c['bh'] for c in group_comps)
        gw = max_x - min_x
        gh = max_y - min_y

        groups.append({
            'comps': group_comps,
            'bx': min_x, 'by': min_y, 'bw': gw, 'bh': gh,
            'area': sum(c['px'] for c in group_comps),
            'orig_count': len(group_comps),
            'cx': min_x + gw / 2, 'cy': min_y + gh / 2,
        })

    if not groups:
        return None

    # --- Select best watermark group ---
    min_coverage = 0.01
    best = None
    best_score = -1.0

    for g in groups:
        gw, gh = g['bw'], g['bh']
        if gw < roi_w * min_coverage and gh < roi_h * min_coverage:
            continue
        # Allow larger groups (up to 80% of ROI) since search area is constrained
        if g['area'] > roi_area * 0.8:
            continue

        count_score = min(g['orig_count'] / 3.0, 1.0)

        density = g['area'] / max(1, gw * gh)
        density_score = 1.0 - abs(density - 0.3) / 0.7
        density_score = max(0.0, density_score)

        aspect = gw / max(1, gh)
        aspect_score = min(aspect / 2.0, 1.0) if aspect > 0.5 else 0.3

        # Text-characteristic: prefer wide, short groups
        height_frac = gh / max(1, roi_h)
        if height_frac > 0.6 and gw / max(1, roi_w) > 0.6:
            text_fit = 0.1
        elif height_frac > 0.4:
            text_fit = 0.3
        elif aspect > 1.5 and height_frac < 0.3:
            text_fit = 1.0
        else:
            text_fit = 0.5

        score = (0.30 * count_score + 0.20 * density_score +
                 0.20 * aspect_score + 0.30 * text_fit)

        if score > best_score:
            best_score = score
            best = g

    # Fallback: pick the largest group
    if best is None or best_score < 0.15:
        valid = [g for g in groups
                 if g['area'] <= roi_area * 0.8
                 and (g['bw'] >= roi_w * min_coverage
                      or g['bh'] >= roi_h * min_coverage)]
        if valid:
            best = max(valid, key=lambda g: g['area'])
            best_score = 0.25
        else:
            return None

    # --- Generate mask from selected group's components only ---
    # Create mask from individual component pixels, not merged/closed blobs
    refined = np.zeros_like(roi)
    for c in best['comps']:
        refined[c['by']:c['by']+c['bh'], c['bx']:c['bx']+c['bw']] = \
            candidates[c['by']:c['by']+c['bh'], c['bx']:c['bx']+c['bw']]

    # Light dilation (3x3) for stroke coverage
    refined = cv2.dilate(refined,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    if not np.any(refined):
        return None

    # Allow up to 60% of ROI for the final mask
    if cv2.countNonZero(refined) > roi_area * 0.60:
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


def detect_watermark_size(gray, corner='bottom-right', scan_pct=.18):
    """Detect the actual watermark size and position within a corner region.

    Uses background residual (top-hat / black-hat) + connected-component
    analysis to find the watermark's true bounding box instead of assuming
    a fixed square area.

    Returns (x1, y1, x2, y2) in full-image coordinates, or None if detection
    fails or confidence is too low.
    """
    gray = _gray_u8(gray)
    h, w = gray.shape
    sw, sh = max(1, int(w * scan_pct)), max(1, int(h * scan_pct))
    boxes = {
        'top-left': (0, 0, sw, sh), 'top-right': (w - sw, 0, w, sh),
        'bottom-left': (0, h - sh, sw, h), 'bottom-right': (w - sw, h - sh, w, h),
    }
    cx1, cy1, cx2, cy2 = boxes.get(corner, boxes['bottom-right'])
    roi = gray[cy1:cy2, cx1:cx2]
    if roi.size < 9 or roi.shape[0] < 3 or roi.shape[1] < 3:
        return None

    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w

    # Proportional thresholds (relative to ROI size, not fixed pixels).
    min_side = max(2, int(min(roi_h, roi_w) * 0.02))
    min_pixels = max(3, int(roi_area * 0.0005))
    # Minimum coverage: watermark must span at least this fraction of ROI
    min_coverage = 0.04  # 4% of ROI width or height

    # ---- Step 1: Single candidate extraction (shared with refine) ----
    candidates = _extract_watermark_candidates(roi)
    if candidates is None or cv2.countNonZero(candidates) == 0:
        return None

    # ---- Step 2: Connected-component analysis ----
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidates, connectivity=8)

    comps = []
    for lab in range(1, n_labels):                    # 0 = background
        bx = int(stats[lab, cv2.CC_STAT_LEFT])
        by = int(stats[lab, cv2.CC_STAT_TOP])
        bw = int(stats[lab, cv2.CC_STAT_WIDTH])
        bh = int(stats[lab, cv2.CC_STAT_HEIGHT])
        px = int(stats[lab, cv2.CC_STAT_AREA])

        if px < min_pixels:
            continue
        if bw > roi_w * 0.9 and bh > roi_h * 0.9:
            continue
        if bw < min_side and bh < min_side:
            continue

        comps.append({
            'label': lab,
            'bx': bx, 'by': by, 'bw': bw, 'bh': bh,
            'px': px,
            'cx': bx + bw / 2.0, 'cy': by + bh / 2.0,
        })

    if not comps:
        return None

    # ---- Step 3: Spatial clustering (distance-based, not morphological) ----
    avg_char_w = np.median([c['bw'] for c in comps])
    avg_char_h = np.median([c['bh'] for c in comps])
    merge_dist_x = max(5, int(avg_char_w * 2.5))
    merge_dist_y = max(3, int(avg_char_h * 1.5))

    used = [False] * len(comps)
    groups = []
    for i, c in enumerate(comps):
        if used[i]:
            continue
        queue = [i]
        used[i] = True
        group_comps = [c]
        while queue:
            curr = queue.pop(0)
            cx1 = comps[curr]['cx']
            cy1 = comps[curr]['cy']
            for j, d in enumerate(comps):
                if used[j]:
                    continue
                dx = abs(d['cx'] - cx1)
                dy = abs(d['cy'] - cy1)
                if dx <= merge_dist_x and dy <= merge_dist_y:
                    used[j] = True
                    queue.append(j)
                    group_comps.append(d)

        min_x = min(c['bx'] for c in group_comps)
        min_y = min(c['by'] for c in group_comps)
        max_x = max(c['bx'] + c['bw'] for c in group_comps)
        max_y = max(c['by'] + c['bh'] for c in group_comps)
        gw = max_x - min_x
        gh = max_y - min_y

        groups.append({
            'comps': group_comps,
            'bx': min_x, 'by': min_y, 'bw': gw, 'bh': gh,
            'area': sum(c['px'] for c in group_comps),
            'orig_count': len(group_comps),
            'cx': min_x + gw / 2, 'cy': min_y + gh / 2,
        })

    if not groups:
        return None

    # ---- Step 4: Score and pick the best group ----
    # Scoring formula (density is NOT dominant):
    #   score = count_score * arrangement * corner_proximity * size_fit
    # where:
    #   count_score        — bonus for multi-component groups (watermarks)
    #   arrangement        — alignment consistency (horizontal spread >> vertical)
    #   corner_proximity   — distance to expected corner
    #   size_fit           — penalise too-small or too-large groups

    best = None
    best_score = -1.0

    def _corner_dist(cx, cy):
        if corner == 'bottom-right':
            return ((roi_w - cx) ** 2 + (roi_h - cy) ** 2) ** 0.5
        elif corner == 'bottom-left':
            return (cx ** 2 + (roi_h - cy) ** 2) ** 0.5
        elif corner == 'top-right':
            return ((roi_w - cx) ** 2 + cy ** 2) ** 0.5
        else:  # top-left
            return (cx ** 2 + cy ** 2) ** 0.5

    max_dist = (roi_w ** 2 + roi_h ** 2) ** 0.5

    for grp in groups:
        gw, gh = grp['bw'], grp['bh']

        # Hard constraints
        if gw < min_side or gh < min_side:
            continue
        if gw * gh > roi_area * 0.45:
            continue
        # Must cover at least min_coverage of ROI in one dimension
        if gw < roi_w * min_coverage and gh < roi_h * min_coverage:
            continue

        # 1) Multi-component score (watermarks have 2+ characters)
        count_score = min(grp['orig_count'] / 5.0, 1.0)

        # 2) Arrangement consistency
        grp_comps = grp['comps']
        if len(grp_comps) >= 2:
            cy_vals = [c['cy'] for c in grp_comps]
            cx_vals = [c['cx'] for c in grp_comps]
            v_spread = max(cy_vals) - min(cy_vals)
            h_spread = max(cx_vals) - min(cx_vals)
            # Good: h_spread >> v_spread (horizontal text)
            if h_spread > 0:
                alignment = min(v_spread / max(1, h_spread), 1.0)
                arrangement = 1.0 - alignment * 0.5  # best when v_spread is small
            else:
                arrangement = 0.5
        else:
            arrangement = 0.6  # single component, neutral

        # 3) Corner proximity
        dist = _corner_dist(grp['cx'], grp['cy'])
        corner_bonus = max(0.0, 1.0 - dist / max(1, max_dist))

        # 4) Size fit: prefer groups that are 5-40% of ROI area
        area_frac = (gw * gh) / max(1, roi_area)
        if area_frac < 0.01:
            size_fit = 0.2
        elif area_frac < 0.05:
            size_fit = 0.6
        elif area_frac < 0.25:
            size_fit = 1.0
        elif area_frac < 0.45:
            size_fit = 0.7
        else:
            size_fit = 0.1

        # 5) Text-characteristic fit: watermark text is typically
        #    low-height relative to ROI, with high width-to-height ratio.
        #    Penalize groups that span a large fraction of ROI height
        #    (background textures tend to fill the whole region).
        height_frac = gh / max(1, roi_h)
        width_frac = gw / max(1, roi_w)
        group_aspect = gw / max(1, gh)
        if height_frac > 0.6 and width_frac > 0.6:
            text_fit = 0.1   # likely a big texture block, not watermark
        elif height_frac > 0.4:
            text_fit = 0.3
        elif group_aspect > 1.5 and height_frac < 0.3:
            text_fit = 1.0   # ideal: wide, short — typical watermark text
        elif group_aspect > 1.0:
            text_fit = 0.7
        else:
            text_fit = 0.4   # taller-than-wide, less typical

        score = (0.25 * count_score
                 + 0.20 * arrangement
                 + 0.20 * corner_bonus
                 + 0.15 * size_fit
                 + 0.20 * text_fit)

        if score > best_score:
            best_score = score
            best = (grp['bx'], grp['by'], gw, gh)

    # Fallback: pick the largest group by area (not the first one)
    if best is None:
        valid = [g for g in groups
                 if g['bw'] * g['bh'] <= roi_area * 0.45
                 and (g['bw'] >= roi_w * min_coverage
                      or g['bh'] >= roi_h * min_coverage)]
        if valid:
            largest = max(valid, key=lambda g: g['area'])
            best = (largest['bx'], largest['by'], largest['bw'], largest['bh'])
            # Sync best_score so the confidence check below doesn't reject
            # a valid fallback selection. Use a neutral score (0.25) that
            # reflects "reliable enough but unranked".
            best_score = 0.25

    if best is None:
        return None

    # Low confidence check: if best_score is very low, return None
    # to let the caller fall back to fixed corner rectangle
    if best_score < 0.20:
        return None

    rx, ry, rw, rh = best

    # Convert back to full-image coordinates with a small proportional padding.
    pad = max(3, int(min(roi_h, roi_w) * 0.01))
    x1 = max(0, cx1 + rx - pad)
    y1 = max(0, cy1 + ry - pad)
    x2 = min(w, cx1 + rx + rw + pad)
    y2 = min(h, cy1 + ry + rh + pad)
    return (x1, y1, x2, y2)


def corner_mask(h, w, corner='bottom-right', scan_pct=.18, fixed_position=None,
                gray=None, rectangle_fallback=False, ocr_bbox=None):
    # Priority: OCR bbox > fixed_position > auto-detection
    if ocr_bbox is not None:
        # OCR found text — use OCR bbox directly as mask
        x1, y1, x2, y2 = ocr_bbox
        x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask
    elif fixed_position is not None:
        x1, y1, x2, y2 = fixed_position
        if all(0 <= v <= 100 for v in fixed_position):
            x1, x2 = int(w*x1/100), int(w*x2/100)
            y1, y2 = int(h*y1/100), int(h*y2/100)
        x1, x2 = sorted((max(0, int(x1)), min(w, int(x2))))
        y1, y2 = sorted((max(0, int(y1)), min(h, int(y2))))
        # Use rectangle mask for fixed_position
        mask = np.zeros((h, w), np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask
    else:
        # Auto-detection: return corner rectangle
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


def _estimate_stroke_width(mask):
    """Estimate median stroke width from a binary mask using distance transform.

    Returns the full stroke width in pixels, or None if the mask is too
    dense (likely a filled rectangle) or empty.
    """
    binary = (mask > 0).astype(np.uint8)
    nz = cv2.countNonZero(binary)
    if nz == 0:
        return None
    # If mask is very dense (filled rectangle), stroke width estimation
    # is meaningless — distances would reflect rectangle size, not strokes.
    density = nz / max(1, binary.size)
    if density > 0.5:
        return None
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    nonzero = dist[dist > 0]
    if len(nonzero) == 0:
        return None
    # median distance ≈ half stroke width
    half_width = float(np.median(nonzero))
    return max(1.0, half_width * 2.0)


def dilate_mask(mask, padding=15):
    mask = np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)
    if padding <= 0:
        return mask
    # Check if mask is a solid rectangle (high density within bounding box)
    # Distance transform gives wrong stroke width for filled regions
    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        bbox_area = (xs.max()-xs.min()+1) * (ys.max()-ys.min()+1)
        mask_area = cv2.countNonZero(mask)
        bbox_density = mask_area / max(1, bbox_area)
        # If mask fills >80% of its bounding box, it's solid - skip adaptive
        if bbox_density < 0.8:
            sw = _estimate_stroke_width(mask)
            if sw is not None and sw > 2:
                padding = max(2, int(sw * 0.75))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (padding*2+1, padding*2+1))
    return cv2.dilate(mask, kernel)


def has_watermark_edges(gray, corner='bottom-right', scan_pct=.18, threshold=.02, fixed_position=None):
    gray = _gray_u8(gray)
    mask = corner_mask(*gray.shape, corner, scan_pct, fixed_position, gray=gray)
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
        mask = corner_mask(*gray.shape, corner, scan_pct, gray=gray)
        if mask is None:
            scored.append((0.0, None))
            continue
        ys, xs = np.where(mask > 0)
        if not len(xs):
            scored.append((0.0, mask))
            continue
        roi = gray[ys.min():ys.max()+1, xs.min():xs.max()+1]
        scored.append((cv2.countNonZero(_canny(roi))/max(1,roi.size), mask))
    score, mask = max(scored, key=lambda item: item[0])
    return mask if score > .02 and mask is not None else None

