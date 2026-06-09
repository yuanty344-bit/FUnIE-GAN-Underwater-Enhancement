"""
图像属性面板

显示文件信息、RGB直方图和质量指标。
"""
from typing import Optional
import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTextEdit
)

from ...i18n import tr
from ..widgets.histogram_widget import HistogramWidget


class PropertyPanel(QWidget):
    """图像属性侧边面板"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # 文件信息
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        self.text_info.setMaximumHeight(130)
        self.label_file_info = QLabel(tr("文件信息:"))
        layout.addWidget(self.label_file_info)
        layout.addWidget(self.text_info)

        # 直方图
        self.label_histogram = QLabel(tr("RGB直方图:"))
        layout.addWidget(self.label_histogram)
        self.histogram = HistogramWidget()
        layout.addWidget(self.histogram)

        # 质量指标
        self.text_metrics = QTextEdit()
        self.text_metrics.setReadOnly(True)
        self.label_quality_metrics = QLabel(tr("质量指标:"))
        layout.addWidget(self.label_quality_metrics)
        layout.addWidget(self.text_metrics)

        layout.addStretch()

    def set_file_info(self, text: str) -> None:
        self.text_info.setPlainText(text)

    def set_metrics(self, text: str) -> None:
        self.text_metrics.setPlainText(text)

    def set_histograms(self, before: Optional[np.ndarray] = None,
                       after: Optional[np.ndarray] = None) -> None:
        self.histogram.set_histograms(before, after)

    def retranslate_ui(self) -> None:
        self.label_file_info.setText(tr("文件信息:"))
        self.label_histogram.setText(tr("RGB直方图:"))
        self.label_quality_metrics.setText(tr("质量指标:"))
        self.histogram.retranslate_ui()
