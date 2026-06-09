"""
对比图导出对话框

导出选项：布局选择、格式、质量设置。
"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout,
    QComboBox, QSpinBox, QPushButton,
    QDialogButtonBox, QFileDialog, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt
import numpy as np

from ...i18n import tr
from ...processors.export_tools import export_comparison


class ExportDialog(QDialog):
    """对比图导出对话框"""

    def __init__(self, parent=None,
                 original: np.ndarray = None,
                 enhanced: np.ndarray = None):
        super().__init__(parent)
        self.setWindowTitle(tr("导出对比图"))
        self.setMinimumWidth(360)
        self._original = original
        self._enhanced = enhanced
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        group = QGroupBox(tr("拼接布局"))
        form = QFormLayout()
        self.combo_layout = QComboBox()
        self.combo_layout.addItem(tr("左右对比 (推荐)"), "horizontal")
        self.combo_layout.addItem(tr("上下对比"), "vertical")
        self.combo_layout.addItem(tr("左右分割线"), "split")
        form.addRow(tr("布局:"), self.combo_layout)
        group.setLayout(form)
        layout.addWidget(group)

        fmt_group = QGroupBox(tr("导出格式"))
        fmt_form = QFormLayout()
        self.combo_format = QComboBox()
        self.combo_format.addItem("PNG", "png")
        self.combo_format.addItem("JPEG", "jpg")
        fmt_form.addRow(tr("格式:"), self.combo_format)

        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(10, 100)
        self.spin_quality.setValue(95)
        fmt_form.addRow(tr("JPEG质量:"), self.spin_quality)
        fmt_group.setLayout(fmt_form)
        layout.addWidget(fmt_group)

        self.btn_export = QPushButton(tr("导出..."))
        self.btn_export.clicked.connect(self._on_export)
        layout.addWidget(self.btn_export)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

    def _on_export(self):
        if self._original is None or self._enhanced is None:
            QMessageBox.information(self, tr("提示"), tr("请先加载并增强图像"))
            return

        layout_key = self.combo_layout.currentData()
        fmt = self.combo_format.currentData()
        path, _ = QFileDialog.getSaveFileName(
            self, tr("保存对比图"),
            f"comparison.{fmt}",
            f"{fmt.upper()} (*.{fmt})"
        )
        if not path:
            return

        success = export_comparison(self._original, self._enhanced,
                                    Path(path), layout=layout_key)
        if success:
            QMessageBox.information(self, tr("完成"), tr("对比图已导出:\n") + path)
        else:
            QMessageBox.critical(self, tr("错误"), tr("导出失败"))

    def set_images(self, original, enhanced):
        self._original = original
        self._enhanced = enhanced

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("导出对比图"))
