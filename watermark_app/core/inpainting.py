"""Inpainting backends with normalized OpenCV-compatible inputs."""

import logging
import threading

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_lama_instance = None
_lama_lock = threading.Lock()


def _normalise_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an HxWx3 NumPy array")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _normalise_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a NumPy array")
    if mask.ndim == 3:
        mask = cv2.cvtColor(np.ascontiguousarray(mask), cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2 or mask.shape != shape:
        raise ValueError(f"mask must have shape {shape}")
    return np.ascontiguousarray((mask > 0).astype(np.uint8) * 255)


class LaMaInpainter:
    """CPU LaMa wrapper using a contextual crop and feathered compositing."""

    def __init__(self):
        self._model = None
        self._initialized = False

    def _init_model(self) -> bool:
        if self._initialized:
            return True
        with _lama_lock:
            if self._initialized:
                return True
            try:
                import inspect
                import torch
                from simple_lama_inpainting import SimpleLama

                # Explicit CPU selection avoids CUDA/device mismatches on machines
                # with an unavailable or partially configured GPU runtime.
                if "device" in inspect.signature(SimpleLama).parameters:
                    self._model = SimpleLama(device=torch.device("cpu"))
                else:
                    # Older simple-lama releases choose the device by calling
                    # torch.cuda.is_available() and expose no device argument.
                    is_available = torch.cuda.is_available
                    try:
                        torch.cuda.is_available = lambda: False
                        self._model = SimpleLama()
                    finally:
                        torch.cuda.is_available = is_available
                self._initialized = True
                logger.info("LaMa model initialized on CPU")
                return True
            except ImportError as exc:
                logger.warning("CPU LaMa is unavailable: %s", exc)
                return False
            except Exception as exc:
                logger.exception("Failed to initialize CPU LaMa: %s", exc)
                return False

    def inpaint(self, image: np.ndarray, mask: np.ndarray, feather: int = 5) -> np.ndarray:
        image = _normalise_image(image)
        mask = _normalise_mask(mask, image.shape[:2])
        if not np.any(mask):
            return image.copy()
        if not self._init_model():
            return opencv_inpaint(image, mask)

        h, w = image.shape[:2]
        x, y, bw, bh = cv2.boundingRect(cv2.findNonZero(mask))
        pad = 50
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

        min_size = 256
        if x2 - x1 < min_size:
            cx = (x1 + x2) // 2
            x1, x2 = max(0, cx - min_size // 2), min(w, cx + (min_size + 1) // 2)
            x1 = max(0, x2 - min_size)
        if y2 - y1 < min_size:
            cy = (y1 + y2) // 2
            y1, y2 = max(0, cy - min_size // 2), min(h, cy + (min_size + 1) // 2)
            y1 = max(0, y2 - min_size)

        crop = image[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        from PIL import Image

        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(crop_mask, mode="L")
        result = np.asarray(self._model(pil_image, pil_mask))
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        result = cv2.cvtColor(result[:, :, :3], cv2.COLOR_RGB2BGR)
        if result.shape[:2] != crop.shape[:2]:
            result = cv2.resize(result, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        binary_alpha = crop_mask.astype(np.float32) / 255.0
        alpha = binary_alpha
        if feather > 0:
            size = feather * 2 + 1
            blurred = cv2.GaussianBlur(binary_alpha, (size, size), max(feather / 3, 0.1))
            # The entire requested mask remains fully repaired; feathering only
            # softens pixels outside its edge.
            alpha = np.maximum(binary_alpha, blurred)
        alpha = alpha[:, :, None]
        blended = crop.astype(np.float32) * (1.0 - alpha) + result.astype(np.float32) * alpha
        output = image.copy()
        output[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return output


def opencv_inpaint(image: np.ndarray, mask: np.ndarray, radius: int = 5,
                   method: int = cv2.INPAINT_TELEA) -> np.ndarray:
    image = _normalise_image(image)
    mask = _normalise_mask(mask, image.shape[:2])
    if not np.any(mask):
        return image.copy()
    return cv2.inpaint(image, mask, float(radius), method)


def get_lama_inpainter() -> LaMaInpainter:
    global _lama_instance
    with _lama_lock:
        if _lama_instance is None:
            _lama_instance = LaMaInpainter()
        return _lama_instance

