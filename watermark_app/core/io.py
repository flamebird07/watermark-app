from pathlib import Path
import cv2
import numpy as np


def cv2_read(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_read_gray(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def cv2_write(path, image, quality=95):
    ext = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, data = cv2.imencode(ext, image, params)
    if ok:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.tofile(str(path))
    return bool(ok)


def cv2_write_with_size_limit(path, image, max_size_mb=5, initial_quality=95):
    ext = Path(path).suffix.lower()
    qualities = [initial_quality, 90, 80, 70, 60, 50, 40, 30]
    for quality in dict.fromkeys(qualities):
        if not cv2_write(path, image, quality):
            return False
        if ext == '.png' or Path(path).stat().st_size <= max_size_mb * 1024 * 1024:
            return True
    return False


def load_exif(path):
    return {}


def save_exif(src_path, dst_path):
    # Metadata copying is best-effort and deliberately does not make processing fail.
    try:
        from PIL import Image
        with Image.open(src_path) as src:
            exif = src.info.get('exif')
        if exif:
            with Image.open(dst_path) as dst:
                dst.save(dst_path, exif=exif)
    except Exception:
        pass

