"""
RGB 直方图组件

显示图像的 RGB 通道直方图，支持增强前后对比。
"""
from typing import Optional
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt

from ...i18n import tr


class HistogramWidget(QWidget):
    """RGB 直方图绘制组件"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMinimumWidth(200)
        self._hist_before: Optional[list] = None  # [r_hist, g_hist, b_hist]
        self._hist_after: Optional[list] = None
        self._before_label = tr("增强前")
        self._after_label = tr("增强后")

    def set_histograms(self, image_before: Optional[np.ndarray] = None,
                       image_after: Optional[np.ndarray] = None) -> None:
        hist_before = self._compute_hist(image_before) if image_before is not None else None
        hist_after = self._compute_hist(image_after) if image_after is not None else None
        self._hist_before = [self._smooth(h) for h in hist_before] if hist_before else None
        self._hist_after = [self._smooth(h) for h in hist_after] if hist_after else None
        self.update()

    def _compute_hist(self, image: np.ndarray) -> list:
        if image is None or image.size == 0:
            return []
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        hists = []
        for c in range(3):
            hist = cv2_compute_hist(image[:, :, c])
            hists.append(hist)
        return hists

    def _smooth(self, hist: np.ndarray, kernel: int = 3) -> np.ndarray:
        if kernel < 2:
            return hist
        pad = kernel // 2
        padded = np.pad(hist.astype(float), pad, mode='edge')
        smoothed = np.convolve(padded, np.ones(kernel) / kernel, mode='valid')
        return smoothed.astype(float)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), QColor(250, 250, 250))

        if self._hist_before is None and self._hist_after is None:
            painter.setPen(QColor(150, 150, 150))
            font = QFont()
            font.setPixelSize(12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, tr("无直方图数据"))
            painter.end()
            return

        colors = [QColor(220, 60, 60), QColor(60, 180, 60), QColor(60, 60, 220)]
        label_y = 12
        graph_y = 18
        graph_h = h - 24
        graph_w = w - 12
        left = 6

        if self._hist_before:
            painter.setFont(QFont("", 9))
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(left, label_y, self._before_label)
            self._draw_hist_set(painter, self._hist_before, left, graph_y, graph_w, graph_h, colors, alpha=160)

        if self._hist_after:
            label_x = left + graph_w // 2 if self._hist_before else left
            painter.setPen(QColor(50, 50, 180))
            painter.drawText(label_x, label_y, self._after_label)
            gx = left + graph_w // 2 if self._hist_before else left
            gw = graph_w // 2 if self._hist_before else graph_w
            self._draw_hist_set(painter, self._hist_after, gx, graph_y, gw, graph_h, colors, alpha=200)

        painter.end()

    def _draw_hist_set(self, painter, hists, left, top, w, h, colors, alpha):
        if not hists or len(hists) < 3:
            return
        all_max = max(max(hist) for hist in hists[:3]) if hists else 1
        if all_max == 0:
            all_max = 1
        scale = h / all_max

        # 背景网格
        pen = QPen(QColor(220, 220, 220))
        pen.setWidth(1)
        painter.setPen(pen)
        for frac in [0.25, 0.5, 0.75]:
            y = int(top + h * (1 - frac))
            painter.drawLine(left, y, left + w, y)

        bin_w = max(1, w / 256)

        for ch in range(3):
            c = QColor(colors[ch])
            c.setAlpha(alpha)
            painter.setPen(QPen(c, 1))
            hist = hists[ch]
            points = []
            for i in range(256):
                x = left + int(i * bin_w)
                y = top + int(h - hist[i] * scale)
                points.append((x, y))
            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])

    def retranslate_ui(self) -> None:
        self._before_label = tr("增强前")
        self._after_label = tr("增强后")
        self.update()


def cv2_compute_hist(channel: np.ndarray, bins: int = 256) -> np.ndarray:
    hist, _ = np.histogram(channel.ravel(), bins=bins, range=(0, 256))
    return hist.astype(np.float32)
