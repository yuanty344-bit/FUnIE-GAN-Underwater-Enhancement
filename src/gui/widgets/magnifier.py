"""
局部放大镜组件

鼠标悬停时显示局部放大预览，查看增强细节。
"""
from typing import Optional
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QPoint, QRect
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QPixmap


class Magnifier(QWidget):
    """局部放大镜 — 悬浮显示光标周围区域的放大视图"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(None)  # No parent = top-level tooltip window
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        if hasattr(Qt, 'WA_TransientForParent'):
            self.setAttribute(Qt.WA_TransientForParent)
        self.setFixedSize(150, 150)
        self._zoom = 3.0
        self._source_pixmap: Optional[QPixmap] = None
        self._source_offset = QPoint(0, 0)
        self._source_zoom = 1.0
        self._image_size = (0, 0)
        self._cross_size = 5
        self._enabled = True
        self.hide()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.hide()

    def set_source(self, pixmap: QPixmap, offset: QPoint, view_zoom: float,
                   image_size: tuple) -> None:
        self._source_pixmap = pixmap
        self._source_offset = offset
        self._source_zoom = view_zoom
        self._image_size = image_size

    def show_at(self, global_pos: QPoint, widget_pos: QPoint) -> None:
        if not self._enabled or self._source_pixmap is None:
            return
        self._center_point = widget_pos
        x = global_pos.x() + 18
        y = global_pos.y() - self.height() - 18
        if y < 0:
            y = global_pos.y() + 18
        self.move(x, y)
        if not self.isVisible():
            self.show()
        self.update()

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(1.5, min(10.0, zoom))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        r = self.rect()
        painter.fillRect(r, QColor(50, 50, 50))

        if self._source_pixmap is None or self._source_pixmap.isNull():
            painter.setPen(Qt.gray)
            font = QFont()
            font.setPixelSize(11)
            painter.setFont(font)
            painter.drawText(r, Qt.AlignCenter, "No image")
            painter.end()
            return

        half_s = int((r.width() / 2) / self._zoom)
        cx = int((self._center_point.x() - self._source_offset.x()) / self._source_zoom)
        cy = int((self._center_point.y() - self._source_offset.y()) / self._source_zoom)

        src_rect = QRect(cx - half_s, cy - half_s, half_s * 2, half_s * 2)
        painter.drawPixmap(r, self._source_pixmap, src_rect)

        # Crosshair
        pen = QPen(QColor(255, 255, 0, 180))
        pen.setWidth(1)
        painter.setPen(pen)
        ch = r.center()
        cs = self._cross_size
        painter.drawLine(ch.x() - cs, ch.y(), ch.x() + cs, ch.y())
        painter.drawLine(ch.x(), ch.y() - cs, ch.x(), ch.y() + cs)

        # Border
        painter.setPen(QPen(QColor(80, 80, 80), 2))
        painter.drawRect(r.adjusted(0, 0, -1, -1))
        painter.end()
