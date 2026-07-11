"""Image preview canvas with zoom, pan, and comparison slider."""

from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, QPoint, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QWheelEvent, QMouseEvent
import numpy as np
from typing import Optional
import cv2


class PreviewCanvas(QWidget):
    """Image preview widget with zoom, drag, and comparison modes."""

    comparison_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._original_pixmap: Optional[QPixmap] = None
        self._mask_pixmap: Optional[QPixmap] = None
        self._result_pixmap: Optional[QPixmap] = None

        self._zoom = 1.0
        self._offset = QPoint(0, 0)
        self._dragging = False
        self._drag_start = QPoint(0, 0)
        self._mode = 'original'  # original, mask, result, compare
        self._comparison_pos = 0.5  # 0-1

        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555;")

    def set_image(self, image: np.ndarray):
        """Set image from numpy array (BGR)."""
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._original_pixmap = QPixmap.fromImage(qimg.copy())
        self._pixmap = self._original_pixmap
        self._fit_to_view()

    def set_mask(self, mask: np.ndarray):
        """Set mask overlay."""
        if mask is None:
            self._mask_pixmap = None
            return
        h, w = mask.shape
        # Create red semi-transparent overlay
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[mask > 0] = [255, 0, 0]
        qimg = QImage(overlay.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self._mask_pixmap = QPixmap.fromImage(qimg.copy())

    def set_result(self, image: np.ndarray):
        """Set result image."""
        if image is None:
            self._result_pixmap = None
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._result_pixmap = QPixmap.fromImage(qimg.copy())

    def set_mode(self, mode: str):
        """Switch display mode: original, mask, result, compare."""
        self._mode = mode
        if mode == 'original':
            self._pixmap = self._original_pixmap
        elif mode == 'result' and self._result_pixmap:
            self._pixmap = self._result_pixmap
        elif mode == 'compare':
            self._pixmap = self._original_pixmap
        self.update()

    def set_comparison(self, pos: float):
        """Set comparison slider position (0-1)."""
        self._comparison_pos = max(0.0, min(1.0, pos))
        self.comparison_changed.emit(self._comparison_pos)
        self.update()

    def _fit_to_view(self):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        if pw == 0 or ph == 0:
            return
        scale_x = vw / pw
        scale_y = vh / ph
        self._zoom = min(scale_x, scale_y) * 0.95
        self._offset = QPoint(
            int((vw - pw * self._zoom) / 2),
            int((vh - ph * self._zoom) / 2)
        )
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._pixmap is None:
            painter.setPen(Qt.GlobalColor.gray)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "拖入图片或点击\"添加图片\"")
            return

        if self._mode == 'compare' and self._original_pixmap and self._result_pixmap:
            # Draw original on left, result on right with divider
            painter.save()
            painter.translate(self._offset)
            painter.scale(self._zoom, self._zoom)

            # Left side (original)
            painter.setClipRect(0, 0,
                                int(self._comparison_pos * self._pixmap.width()),
                                self._pixmap.height())
            painter.drawPixmap(0, 0, self._original_pixmap)

            painter.setClipping(False)

            # Right side (result)
            if self._result_pixmap:
                painter.setClipRect(
                    int(self._comparison_pos * self._pixmap.width()), 0,
                    self._result_pixmap.width(), self._result_pixmap.height())
                painter.drawPixmap(0, 0, self._result_pixmap)
                painter.setClipping(False)

            # Divider line
            divider_x = int(self._comparison_pos * self._pixmap.width())
            painter.setPen(Qt.GlobalColor.white)
            painter.drawLine(divider_x, 0, divider_x, self._pixmap.height())

            painter.restore()
        else:
            # Normal mode
            painter.translate(self._offset)
            painter.scale(self._zoom, self._zoom)
            painter.drawPixmap(0, 0, self._pixmap)

            # Overlay mask if in mask mode
            if self._mode == 'mask' and self._mask_pixmap:
                painter.setOpacity(0.4)
                painter.drawPixmap(0, 0, self._mask_pixmap)
                painter.setOpacity(1.0)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.1 if delta > 0 else 0.9
        self._zoom *= factor
        self._zoom = max(0.05, min(20.0, self._zoom))
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.pos() - self._drag_start
            self._offset += delta
            self._drag_start = event.pos()
            self.update()

    def resizeEvent(self, event):
        if self._pixmap:
            self._fit_to_view()

    def clear(self):
        self._pixmap = None
        self._original_pixmap = None
        self._mask_pixmap = None
        self._result_pixmap = None
        self.update()
