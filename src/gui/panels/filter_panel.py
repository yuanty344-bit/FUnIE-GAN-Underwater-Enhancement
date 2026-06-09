"""
Filter adjustment panel.
"""
from typing import Optional, Callable, Dict
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QSlider,
    QComboBox, QStackedWidget, QHBoxLayout, QLabel, QSpinBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, pyqtSignal

from ...i18n import tr


class FilterPanel(QWidget):
    """Filter adjustment side panel."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Preset group ──
        self.group_preset = QGroupBox(tr("预设"))
        preset_layout = QVBoxLayout()
        self.btn_preset_original = QPushButton(tr("原始"))
        preset_layout.addWidget(self.btn_preset_original)
        self.btn_preset_auto = QPushButton(tr("自动增强"))
        preset_layout.addWidget(self.btn_preset_auto)
        self.btn_preset_vivid = QPushButton(tr("鲜艳"))
        preset_layout.addWidget(self.btn_preset_vivid)
        self.btn_preset_soft = QPushButton(tr("柔和"))
        preset_layout.addWidget(self.btn_preset_soft)
        self.group_preset.setLayout(preset_layout)
        layout.addWidget(self.group_preset)

        # ── Filter sliders ──
        slider_defs = [
            ("brightness", tr("亮度")),
            ("contrast", tr("对比度")),
            ("saturation", tr("饱和度")),
            ("sharpness", tr("锐度")),
            ("wb", tr("白平衡")),
        ]
        self._groups: Dict[str, QGroupBox] = {}
        self._sliders: Dict[str, QSlider] = {}

        for key, title in slider_defs:
            group = QGroupBox(title)
            group_layout = QVBoxLayout()
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            group_layout.addWidget(slider)
            group.setLayout(group_layout)
            layout.addWidget(group)
            self._groups[key] = group
            self._sliders[key] = slider

        # Reset button
        self.btn_reset_filters = QPushButton(tr("重置所有"))
        layout.addWidget(self.btn_reset_filters)

        layout.addStretch()

    # -- alias properties for backward compatibility --
    @property
    def group_brightness(self):
        return self._groups["brightness"]

    @property
    def group_contrast(self):
        return self._groups["contrast"]

    @property
    def group_saturation(self):
        return self._groups["saturation"]

    @property
    def group_sharpness(self):
        return self._groups["sharpness"]

    @property
    def group_wb(self):
        return self._groups["wb"]

    @property
    def slider_brightness(self):
        return self._sliders["brightness"]

    @property
    def slider_contrast(self):
        return self._sliders["contrast"]

    @property
    def slider_saturation(self):
        return self._sliders["saturation"]

    @property
    def slider_sharpness(self):
        return self._sliders["sharpness"]

    @property
    def slider_wb(self):
        return self._sliders["wb"]

    def connect_filter_changed(self, slot: Callable) -> None:
        for slider in self._sliders.values():
            slider.valueChanged.connect(slot)

    def connect_preset(self, original: Callable, auto: Callable,
                       vivid: Callable, soft: Callable) -> None:
        self.btn_preset_original.clicked.connect(original)
        self.btn_preset_auto.clicked.connect(auto)
        self.btn_preset_vivid.clicked.connect(vivid)
        self.btn_preset_soft.clicked.connect(soft)

    def connect_reset(self, slot: Callable) -> None:
        self.btn_reset_filters.clicked.connect(slot)

    def retranslate_ui(self) -> None:
        self.group_preset.setTitle(tr("预设"))
        self.btn_preset_original.setText(tr("原始"))
        self.btn_preset_auto.setText(tr("自动增强"))
        self.btn_preset_vivid.setText(tr("鲜艳"))
        self.btn_preset_soft.setText(tr("柔和"))
        title_map = {
            "brightness": tr("亮度"), "contrast": tr("对比度"),
            "saturation": tr("饱和度"), "sharpness": tr("锐度"),
            "wb": tr("白平衡"),
        }
        for key, grp in self._groups.items():
            grp.setTitle(title_map.get(key, key))
        self.btn_reset_filters.setText(tr("重置所有"))
