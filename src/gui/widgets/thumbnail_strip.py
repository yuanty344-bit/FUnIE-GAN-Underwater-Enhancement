"""
缩略图浏览器

底部或侧边栏缩略图条，显示文件夹内所有图片，点击快速切换。
"""
import logging
from typing import Optional, List
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QLabel, QPushButton, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont

from ...i18n import tr

logger = logging.getLogger(__name__)

THUMB_SIZE = 80
THUMB_SPACING = 4


class ThumbnailLabel(QLabel):
    """单个缩略图标签"""
    clicked = pyqtSignal(str)

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid transparent; border-radius: 4px;")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(Path(file_path).name)

    def mousePressEvent(self, event):
        self.clicked.emit(self.file_path)

    def enterEvent(self, event):
        self.setStyleSheet("border: 2px solid #0078d4; border-radius: 4px;")

    def leaveEvent(self, event):
        self.setStyleSheet("border: 2px solid transparent; border-radius: 4px;")


class ThumbnailStrip(QWidget):
    """图像缩略图浏览器"""
    file_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._files: List[Path] = []
        self._thumbnails: List[ThumbnailLabel] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFixedHeight(THUMB_SIZE + THUMB_SPACING + 12)

        self.container = QWidget()
        self.strip_layout = QHBoxLayout(self.container)
        self.strip_layout.setContentsMargins(4, 2, 4, 2)
        self.strip_layout.setSpacing(THUMB_SPACING)
        self.strip_layout.addStretch()

        self.scroll.setWidget(self.container)
        layout.addWidget(self.scroll)

    def load_folder(self, folder: str) -> None:
        """加载文件夹中的所有图像"""
        self.clear()
        folder_path = Path(folder)
        exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        self._files = sorted(
            [f for f in folder_path.iterdir() if f.suffix.lower() in exts],
            key=lambda f: f.name.lower()
        )

        for file_path in self._files:
            thumb = self._create_thumbnail(str(file_path))
            label = ThumbnailLabel(str(file_path))
            if thumb:
                label.setPixmap(thumb.scaled(THUMB_SIZE - 8, THUMB_SIZE - 8,
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                label.setText(Path(file_path).name[:8])
            label.clicked.connect(self._on_thumbnail_clicked)
            self._thumbnails.append(label)
            # 在 stretch 之前插入
            self.strip_layout.insertWidget(self.strip_layout.count() - 1, label)

    def _create_thumbnail(self, file_path: str) -> Optional[QPixmap]:
        try:
            img = cv2.imread(file_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)
        except Exception:
            return None

    def _on_thumbnail_clicked(self, file_path: str):
        self.file_selected.emit(file_path)
        # 高亮选中
        for t in self._thumbnails:
            if t.file_path == file_path:
                t.setStyleSheet("border: 2px solid #0078d4; border-radius: 4px; "
                                "background-color: rgba(0,120,212,30);")
            else:
                t.setStyleSheet("border: 2px solid transparent; border-radius: 4px;")

    def clear(self) -> None:
        """清空所有缩略图"""
        for t in self._thumbnails:
            self.strip_layout.removeWidget(t)
            t.deleteLater()
        self._thumbnails.clear()
        self._files.clear()

    def navigate_next(self) -> Optional[str]:
        """选择下一个文件"""
        if not self._thumbnails:
            return None
        for i, t in enumerate(self._thumbnails):
            if "background-color" in t.styleSheet():
                next_idx = (i + 1) % len(self._thumbnails)
                self._thumbnails[next_idx].clicked.emit(self._thumbnails[next_idx].file_path)
                return self._thumbnails[next_idx].file_path
        # none selected, pick first
        self._thumbnails[0].clicked.emit(self._thumbnails[0].file_path)
        return self._thumbnails[0].file_path

    def navigate_prev(self) -> Optional[str]:
        """选择上一个文件"""
        if not self._thumbnails:
            return None
        for i, t in enumerate(self._thumbnails):
            if "background-color" in t.styleSheet():
                prev_idx = (i - 1) % len(self._thumbnails)
                self._thumbnails[prev_idx].clicked.emit(self._thumbnails[prev_idx].file_path)
                return self._thumbnails[prev_idx].file_path
        self._thumbnails[0].clicked.emit(self._thumbnails[0].file_path)
        return self._thumbnails[0].file_path
