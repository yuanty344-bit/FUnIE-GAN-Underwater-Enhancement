"""
预设管理对话框

用户自定义滤镜预设的保存、加载、管理。
"""
import logging
from typing import Optional, List, Dict

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QLabel, QMessageBox,
    QFileDialog, QInputDialog, QDialogButtonBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ...i18n import tr
from ...utils.preset_io import (
    load_user_presets, save_user_presets,
    export_preset, import_preset,
)

logger = logging.getLogger(__name__)


class PresetDialog(QDialog):
    """预设管理对话框"""
    preset_applied = pyqtSignal(dict)
    presets_changed = pyqtSignal(list)

    def __init__(self, parent=None, current_params: Optional[Dict] = None):
        super().__init__(parent)
        self.setWindowTitle(tr("预设管理"))
        self.setMinimumSize(420, 380)
        self._current_params = current_params or {}
        self._presets: List[Dict] = []
        self._init_ui()
        self._load_presets()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel(tr("管理您的自定义滤镜预设"))
        layout.addWidget(label)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_apply)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayout()

        self.btn_save = QPushButton(tr("保存当前"))
        self.btn_save.clicked.connect(self._on_save_current)
        btn_layout.addWidget(self.btn_save)

        self.btn_apply = QPushButton(tr("应用"))
        self.btn_apply.clicked.connect(self._on_apply)
        btn_layout.addWidget(self.btn_apply)

        self.btn_export = QPushButton(tr("导出..."))
        self.btn_export.clicked.connect(self._on_export)
        btn_layout.addWidget(self.btn_export)

        self.btn_import = QPushButton(tr("导入..."))
        self.btn_import.clicked.connect(self._on_import)
        btn_layout.addWidget(self.btn_import)

        self.btn_delete = QPushButton(tr("删除"))
        self.btn_delete.clicked.connect(self._on_delete)
        btn_layout.addWidget(self.btn_delete)

        layout.addLayout(btn_layout)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.close)
        layout.addWidget(btn_box)

    def _load_presets(self):
        self._presets = load_user_presets()
        self._refresh_list()

    def _refresh_list(self):
        self.list_widget.clear()
        for p in self._presets:
            name = p.get("name", tr("未命名"))
            desc = f"B:{p.get('brightness',0):.2f} C:{p.get('contrast',1):.2f} S:{p.get('saturation',1):.2f}"
            item = QListWidgetItem(f"{name}  [{desc}]")
            item.setData(Qt.UserRole, p)
            self.list_widget.addItem(item)

    def _on_save_current(self):
        if not self._current_params:
            QMessageBox.information(self, tr("提示"), tr("没有可保存的滤镜参数"))
            return
        name, ok = QInputDialog.getText(self, tr("保存预设"), tr("预设名称:"))
        if not ok or not name.strip():
            return
        preset = dict(self._current_params)
        preset["name"] = name.strip()
        self._presets.append(preset)
        save_user_presets(self._presets)
        self._refresh_list()
        self.presets_changed.emit(list(self._presets))

    def _on_apply(self):
        items = self.list_widget.selectedItems()
        if items:
            preset = items[0].data(Qt.UserRole)
            self.preset_applied.emit(preset)
        elif self.list_widget.currentItem():
            preset = self.list_widget.currentItem().data(Qt.UserRole)
            self.preset_applied.emit(preset)

    def _on_export(self):
        items = self.list_widget.selectedItems()
        if not items:
            QMessageBox.information(self, tr("提示"), tr("请先选择一个预设"))
            return
        preset = items[0].data(Qt.UserRole)
        path, _ = QFileDialog.getSaveFileName(
            self, tr("导出预设"), f"{preset.get('name','preset')}.json",
            "JSON (*.json)"
        )
        if path:
            export_preset(preset, path)

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("导入预设"), "", "JSON (*.json)"
        )
        if path:
            preset = import_preset(path)
            if preset:
                self._presets.append(preset)
                save_user_presets(self._presets)
                self._refresh_list()
                self.presets_changed.emit(list(self._presets))
            else:
                QMessageBox.warning(self, tr("错误"), tr("无法导入预设，文件格式不正确"))

    def _on_delete(self):
        items = self.list_widget.selectedItems()
        if not items:
            return
        preset = items[0].data(Qt.UserRole)
        name = preset.get("name", "")
        reply = QMessageBox.question(self, tr("确认删除"),
            tr("确定要删除预设 \"") + name + tr("\" 吗？"))
        if reply == QMessageBox.Yes:
            self._presets = [p for p in self._presets if p != preset]
            save_user_presets(self._presets)
            self._refresh_list()
            self.presets_changed.emit(list(self._presets))

    def retranslate_ui(self) -> None:
        self.setWindowTitle(tr("预设管理"))
