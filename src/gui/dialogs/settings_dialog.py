"""
设置对话框模块

提供应用程序设置界面，
包括模型配置、处理选项和界面偏好设置。
"""

from typing import Optional

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QWidget, QFormLayout, QLabel, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QPushButton, QDialogButtonBox,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QFileDialog, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal

from ...i18n import tr


class SettingsDialog(QDialog):
    """
    设置对话框
    
    提供分类设置的选项卡界面，
    包括模型、处理、界面和滤镜等设置。
    """
    
    # 设置已应用信号
    settings_applied = pyqtSignal()
    
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        config=None
    ):
        """
        初始化设置对话框
        
        Args:
            parent: 父窗口
            config: 配置管理器实例
        """
        super().__init__(parent)
        
        self.config = config
        
        self.setWindowTitle(tr("设置"))
        self.setMinimumSize(600, 500)
        self.setModal(True)

        self._init_ui()
        self._load_settings()
        self.retranslate_ui()

    def retranslate_ui(self) -> None:
        """重新翻译所有UI字符串"""
        self.setWindowTitle(tr("设置"))
        self.tabs.setTabText(0, tr("模型"))
        self.tabs.setTabText(1, tr("处理"))
        self.tabs.setTabText(2, tr("界面"))
        self.tabs.setTabText(3, tr("批量处理"))
        self.tabs.setTabText(4, tr("视频处理"))

        # 模型 tab
        self.label_model_path.setText(tr("模型路径:"))
        self.btn_browse.setText(tr("浏览..."))
        self.label_device.setText(tr("计算设备:"))
        self.label_framework.setText(tr("深度学习框架:"))
        self.label_batch_size.setText(tr("批处理大小:"))
        self.label_precision.setText(tr("计算精度:"))
        self.btn_check.setText(tr("检查模型"))
        self._update_model_status_label()

        # 处理 tab
        self.label_output_format.setText(tr("输出格式:"))
        self.label_quality.setText(tr("JPEG质量:"))
        self.label_preserve_meta.setText(tr("保留元数据:"))
        self.label_auto_enhance.setText(tr("加载时自动增强:"))
        self.label_default_mode.setText(tr("默认增强模式:"))

        # 界面 tab
        self.label_theme.setText(tr("界面主题:"))
        self.label_language.setText(tr("界面语言:"))
        self.label_show_toolbar.setText(tr("显示工具栏:"))
        self.label_show_statusbar.setText(tr("显示状态栏:"))
        self.label_max_recent.setText(tr("最近文件数量:"))
        self.btn_clear_recent.setText(tr("清空最近文件"))

        # 批量 tab
        self.label_parallel.setText(tr("启用并行处理:"))
        self.label_max_workers.setText(tr("最大工作线程:"))
        self.label_skip_existing.setText(tr("跳过已处理文件:"))
        self.label_output_subdir.setText(tr("输出子目录:"))
        self.label_naming_pattern.setText(tr("命名模板:"))
        self.label_save_metrics.setText(tr("保存处理指标:"))

        # 视频 tab
        self.label_output_fps.setText(tr("输出帧率:"))
        self.edit_fps.setPlaceholderText(tr("留空则保持原帧率"))
        self.label_codec.setText(tr("视频编码器:"))
        self.label_frame_interval.setText(tr("处理帧间隔:"))
        self.label_video_format.setText(tr("输出格式:"))

        # 数字后缀
        self.spin_batch_size.setSuffix(tr(" 张"))
        self.spin_quality.setSpecialValueText(tr("最高质量"))
        self.spin_max_workers.setSuffix(tr(" 个"))
        self.spin_frame_interval.setSuffix(tr(" 帧"))

        # 重新填充下拉框（保留 itemData 键）
        self._reload_combo_texts()

    def _reload_combo_texts(self) -> None:
        """重新加载下拉框显示文本（保留当前选择的 itemData 键）"""
        # (device, key): [display_texts...]
        combo_defs = [
            (self.combo_device, [("auto", "Auto"), ("cpu", "CPU"), ("gpu", "GPU")]),
            (self.combo_framework, [("tensorflow", "TensorFlow"), ("pytorch", "PyTorch")]),
            (self.combo_precision, [("float32", "float32"), ("float16", "float16")]),
            (self.combo_output_format, [("png", "PNG"), ("jpeg", "JPEG"), ("bmp", "BMP"), ("tiff", "TIFF")]),
            (self.combo_default_mode, [("auto", tr("自动")), ("standard", tr("标准")), ("light", tr("轻度")), ("strong", tr("强力"))]),
            (self.combo_theme, [("light", tr("浅色 (Light)")), ("dark", tr("深色 (Dark)")), ("fusion", "Fusion"), ("windows", "Windows"), ("windowsvista", "WindowsVista")]),
            (self.combo_language, [("zh_CN", tr("简体中文")), ("en", "English")]),
            (self.combo_codec, [("mp4v", "mp4v (H.264)"), ("xvid", "XVID"), ("divx", "DIVX"), ("wmv1", "WMV1")]),
            (self.combo_video_format, [("mp4", "MP4"), ("avi", "AVI"), ("mkv", "MKV"), ("mov", "MOV")]),
        ]
        for combo, defs in combo_defs:
            current_key = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for key, display in defs:
                combo.addItem(display, key)
            idx = combo.findData(current_key)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _init_ui(self) -> None:
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 创建选项卡
        self.tabs = QTabWidget()

        # 模型设置
        self.tabs.addTab(self._create_model_tab(), tr("模型"))

        # 处理设置
        self.tabs.addTab(self._create_processing_tab(), tr("处理"))

        # 界面设置
        self.tabs.addTab(self._create_ui_tab(), tr("界面"))

        # 批量处理设置
        self.tabs.addTab(self._create_batch_tab(), tr("批量处理"))

        # 视频处理设置
        self.tabs.addTab(self._create_video_tab(), tr("视频处理"))

        layout.addWidget(self.tabs)
        
        # 按钮栏
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply
        )
        
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self._on_apply)
        
        layout.addWidget(button_box)
    
    def _create_model_tab(self) -> QWidget:
        """创建模型设置选项卡"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # 模型路径
        self.edit_model_path = QLineEdit()
        self.edit_model_path.setReadOnly(True)
        self.btn_browse = QPushButton(tr("浏览..."))
        self.btn_browse.clicked.connect(self._browse_model_path)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.edit_model_path)
        path_layout.addWidget(self.btn_browse)
        self.label_model_path = QLabel(tr("模型路径:"))
        layout.addRow(self.label_model_path, path_layout)

        # 设备选择 (itemData: auto/cpu/gpu)
        self.combo_device = QComboBox()
        self.combo_device.addItem(tr("自动"), "auto")
        self.combo_device.addItem("CPU", "cpu")
        self.combo_device.addItem("GPU", "gpu")
        self.label_device = QLabel(tr("计算设备:"))
        layout.addRow(self.label_device, self.combo_device)

        # 框架选择 (itemData: tensorflow/pytorch)
        self.combo_framework = QComboBox()
        self.combo_framework.addItem("TensorFlow", "tensorflow")
        self.combo_framework.addItem("PyTorch", "pytorch")
        self.label_framework = QLabel(tr("深度学习框架:"))
        layout.addRow(self.label_framework, self.combo_framework)

        # 批处理大小
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setRange(1, 32)
        self.spin_batch_size.setSuffix(tr(" 张"))
        self.label_batch_size = QLabel(tr("批处理大小:"))
        layout.addRow(self.label_batch_size, self.spin_batch_size)

        # 精度选择 (itemData: float32/float16)
        self.combo_precision = QComboBox()
        self.combo_precision.addItem("float32", "float32")
        self.combo_precision.addItem("float16", "float16")
        self.label_precision = QLabel(tr("计算精度:"))
        layout.addRow(self.label_precision, self.combo_precision)

        layout.addRow("", QLabel())  # 空白行

        # 模型信息
        self.label_model_info = QLabel(tr("模型状态: 未检查"))
        layout.addRow("", self.label_model_info)

        self.btn_check = QPushButton(tr("检查模型"))
        self.btn_check.clicked.connect(self._check_model)
        layout.addRow("", self.btn_check)

        return widget
    
    def _create_processing_tab(self) -> QWidget:
        """创建处理设置选项卡"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # 输出格式 (itemData: png/jpeg/bmp/tiff)
        self.combo_output_format = QComboBox()
        self.combo_output_format.addItem("PNG", "png")
        self.combo_output_format.addItem("JPEG", "jpeg")
        self.combo_output_format.addItem("BMP", "bmp")
        self.combo_output_format.addItem("TIFF", "tiff")
        self.label_output_format = QLabel(tr("输出格式:"))
        layout.addRow(self.label_output_format, self.combo_output_format)

        # 质量
        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(1, 100)
        self.spin_quality.setSuffix("%")
        self.spin_quality.setSpecialValueText(tr("最高质量"))
        self.label_quality = QLabel(tr("JPEG质量:"))
        layout.addRow(self.label_quality, self.spin_quality)

        # 保留元数据
        self.check_preserve_metadata = QCheckBox()
        self.check_preserve_metadata.setChecked(True)
        self.label_preserve_meta = QLabel(tr("保留元数据:"))
        layout.addRow(self.label_preserve_meta, self.check_preserve_metadata)

        # 自动增强
        self.check_auto_enhance = QCheckBox()
        self.label_auto_enhance = QLabel(tr("加载时自动增强:"))
        layout.addRow(self.label_auto_enhance, self.check_auto_enhance)

        # 默认增强模式 (itemData: auto/standard/light/strong)
        self.combo_default_mode = QComboBox()
        self.combo_default_mode.addItem(tr("自动"), "auto")
        self.combo_default_mode.addItem(tr("标准"), "standard")
        self.combo_default_mode.addItem(tr("轻度"), "light")
        self.combo_default_mode.addItem(tr("强力"), "strong")
        self.label_default_mode = QLabel(tr("默认增强模式:"))
        layout.addRow(self.label_default_mode, self.combo_default_mode)

        return widget
    
    def _create_ui_tab(self) -> QWidget:
        """创建界面设置选项卡"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # 主题 (itemData: fusion/windows/windowsvista)
        self.combo_theme = QComboBox()
        self.combo_theme.addItem(tr("浅色 (Light)"), "light")
        self.combo_theme.addItem(tr("深色 (Dark)"), "dark")
        self.combo_theme.addItem("Fusion", "fusion")
        self.combo_theme.addItem("Windows", "windows")
        self.combo_theme.addItem("WindowsVista", "windowsvista")
        self.label_theme = QLabel(tr("界面主题:"))
        layout.addRow(self.label_theme, self.combo_theme)

        # 语言 (itemData: zh_CN/en)
        self.combo_language = QComboBox()
        self.combo_language.addItem(tr("简体中文"), "zh_CN")
        self.combo_language.addItem("English", "en")
        self.label_language = QLabel(tr("界面语言:"))
        layout.addRow(self.label_language, self.combo_language)

        # 显示工具栏
        self.check_show_toolbar = QCheckBox()
        self.check_show_toolbar.setChecked(True)
        self.label_show_toolbar = QLabel(tr("显示工具栏:"))
        layout.addRow(self.label_show_toolbar, self.check_show_toolbar)

        # 显示状态栏
        self.check_show_statusbar = QCheckBox()
        self.check_show_statusbar.setChecked(True)
        self.label_show_statusbar = QLabel(tr("显示状态栏:"))
        layout.addRow(self.label_show_statusbar, self.check_show_statusbar)

        # 最大最近文件数
        self.spin_max_recent = QSpinBox()
        self.spin_max_recent.setRange(0, 50)
        self.label_max_recent = QLabel(tr("最近文件数量:"))
        layout.addRow(self.label_max_recent, self.spin_max_recent)

        # 清空最近文件
        self.btn_clear_recent = QPushButton(tr("清空最近文件"))
        self.btn_clear_recent.clicked.connect(self._clear_recent_files)
        layout.addRow("", self.btn_clear_recent)

        return widget
    
    def _create_batch_tab(self) -> QWidget:
        """创建批量处理设置选项卡"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # 并行处理
        self.check_parallel = QCheckBox()
        self.check_parallel.setChecked(True)
        self.label_parallel = QLabel(tr("启用并行处理:"))
        layout.addRow(self.label_parallel, self.check_parallel)

        # 最大工作线程数
        self.spin_max_workers = QSpinBox()
        self.spin_max_workers.setRange(1, 16)
        self.spin_max_workers.setSuffix(tr(" 个"))
        self.label_max_workers = QLabel(tr("最大工作线程:"))
        layout.addRow(self.label_max_workers, self.spin_max_workers)

        # 跳过已处理文件
        self.check_skip_existing = QCheckBox()
        self.check_skip_existing.setChecked(True)
        self.label_skip_existing = QLabel(tr("跳过已处理文件:"))
        layout.addRow(self.label_skip_existing, self.check_skip_existing)

        # 输出子目录
        self.edit_output_subdir = QLineEdit()
        self.edit_output_subdir.setPlaceholderText("enhanced")
        self.label_output_subdir = QLabel(tr("输出子目录:"))
        layout.addRow(self.label_output_subdir, self.edit_output_subdir)

        # 命名规则
        self.edit_naming_pattern = QLineEdit()
        self.edit_naming_pattern.setPlaceholderText("{name}_enhanced")
        self.label_naming_pattern = QLabel(tr("命名模板:"))
        layout.addRow(self.label_naming_pattern, self.edit_naming_pattern)
        hint = QLabel(tr("可用变量: {name} {date} {time} {ext}"))
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addRow("", hint)

        # 保存指标
        self.check_save_metrics = QCheckBox()
        self.label_save_metrics = QLabel(tr("保存处理指标:"))
        layout.addRow(self.label_save_metrics, self.check_save_metrics)

        return widget
    
    def _create_video_tab(self) -> QWidget:
        """创建视频处理设置选项卡"""
        widget = QWidget()
        layout = QFormLayout(widget)

        # 输出帧率
        self.edit_fps = QLineEdit()
        self.edit_fps.setPlaceholderText(tr("留空则保持原帧率"))
        self.label_output_fps = QLabel(tr("输出帧率:"))
        layout.addRow(self.label_output_fps, self.edit_fps)

        # 编码器 (itemData: mp4v/xvid/divx/wmv1)
        self.combo_codec = QComboBox()
        self.combo_codec.addItem("mp4v (H.264)", "mp4v")
        self.combo_codec.addItem("XVID", "xvid")
        self.combo_codec.addItem("DIVX", "divx")
        self.combo_codec.addItem("WMV1", "wmv1")
        self.label_codec = QLabel(tr("视频编码器:"))
        layout.addRow(self.label_codec, self.combo_codec)

        # 帧间隔
        self.spin_frame_interval = QSpinBox()
        self.spin_frame_interval.setRange(1, 10)
        self.spin_frame_interval.setSuffix(tr(" 帧"))
        self.label_frame_interval = QLabel(tr("处理帧间隔:"))
        layout.addRow(self.label_frame_interval, self.spin_frame_interval)

        # 输出格式 (itemData: mp4/avi/mkv/mov)
        self.combo_video_format = QComboBox()
        self.combo_video_format.addItem("MP4", "mp4")
        self.combo_video_format.addItem("AVI", "avi")
        self.combo_video_format.addItem("MKV", "mkv")
        self.combo_video_format.addItem("MOV", "mov")
        self.label_video_format = QLabel(tr("输出格式:"))
        layout.addRow(self.label_video_format, self.combo_video_format)

        return widget
    
    def _load_settings(self) -> None:
        """加载当前设置 (使用 findData 而非硬编码索引)"""
        if self.config is None:
            return

        # 模型设置
        self.edit_model_path.setText(self.config.get('model.path', ''))

        def _select_by_data(combo, key, default=None):
            idx = combo.findData(key)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif default is not None:
                idx = combo.findData(default)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        _select_by_data(self.combo_device, self.config.get('model.device', 'auto'), 'auto')
        _select_by_data(self.combo_framework, self.config.get('model.framework', 'pytorch'), 'pytorch')
        self.spin_batch_size.setValue(self.config.get('model.batch_size', 1))
        _select_by_data(self.combo_precision, self.config.get('model.precision', 'float32'), 'float32')

        # 处理设置
        fmt = self.config.get('processing.output_format', 'png').lower()
        _select_by_data(self.combo_output_format, fmt, 'png')
        self.spin_quality.setValue(self.config.get('processing.quality', 95))
        self.check_preserve_metadata.setChecked(self.config.get('processing.preserve_metadata', True))
        self.check_auto_enhance.setChecked(self.config.get('processing.auto_enhance', False))
        _select_by_data(self.combo_default_mode, self.config.get('processing.default_mode', 'auto'), 'auto')

        # 界面设置
        _select_by_data(self.combo_theme, self.config.get('ui.theme', 'light'), 'light')
        _select_by_data(self.combo_language, self.config.get('ui.language', 'zh_CN'), 'zh_CN')
        self.check_show_toolbar.setChecked(self.config.get('ui.show_toolbar', True))
        self.check_show_statusbar.setChecked(self.config.get('ui.show_statusbar', True))
        self.spin_max_recent.setValue(self.config.get('ui.max_recent_files', 10))

        # 批量处理设置
        self.check_parallel.setChecked(self.config.get('batch.parallel', True))
        self.spin_max_workers.setValue(self.config.get('batch.max_workers', 4))
        self.check_skip_existing.setChecked(self.config.get('batch.skip_existing', True))
        self.edit_output_subdir.setText(self.config.get('batch.output_subdir', 'enhanced'))
        self.edit_naming_pattern.setText(self.config.get('batch.naming_pattern', '{name}_enhanced'))
        self.check_save_metrics.setChecked(self.config.get('batch.save_metrics', False))

        # 视频处理设置
        fps = self.config.get('video.output_fps')
        self.edit_fps.setText(str(fps) if fps else "")
        _select_by_data(self.combo_codec, self.config.get('video.output_codec', 'mp4v'), 'mp4v')
        self.spin_frame_interval.setValue(self.config.get('video.frame_interval', 1))
        _select_by_data(self.combo_video_format, self.config.get('video.output_format', 'mp4'), 'mp4')
    
    def _save_settings(self) -> None:
        """保存设置 (使用 currentData 而非索引映射)"""
        if self.config is None:
            return

        # 模型设置
        self.config.set('model.path', self.edit_model_path.text())
        self.config.set('model.device', self.combo_device.currentData() or 'auto')
        self.config.set('model.framework', self.combo_framework.currentData() or 'pytorch')
        self.config.set('model.batch_size', self.spin_batch_size.value())
        self.config.set('model.precision', self.combo_precision.currentData() or 'float32')

        # 处理设置
        self.config.set('processing.output_format', self.combo_output_format.currentData() or 'png')
        self.config.set('processing.quality', self.spin_quality.value())
        self.config.set('processing.preserve_metadata', self.check_preserve_metadata.isChecked())
        self.config.set('processing.auto_enhance', self.check_auto_enhance.isChecked())
        self.config.set('processing.default_mode', self.combo_default_mode.currentData() or 'auto')

        # 界面设置
        self.config.set('ui.theme', self.combo_theme.currentData() or 'light')
        self.config.set('ui.language', self.combo_language.currentData() or 'zh_CN')
        self.config.set('ui.show_toolbar', self.check_show_toolbar.isChecked())
        self.config.set('ui.show_statusbar', self.check_show_statusbar.isChecked())
        self.config.set('ui.max_recent_files', self.spin_max_recent.value())

        # 批量处理设置
        self.config.set('batch.parallel', self.check_parallel.isChecked())
        self.config.set('batch.max_workers', self.spin_max_workers.value())
        self.config.set('batch.skip_existing', self.check_skip_existing.isChecked())
        self.config.set('batch.output_subdir', self.edit_output_subdir.text())
        self.config.set('batch.naming_pattern', self.edit_naming_pattern.text() or '{name}_enhanced')
        self.config.set('batch.save_metrics', self.check_save_metrics.isChecked())

        # 视频处理设置
        fps_text = self.edit_fps.text().strip()
        self.config.set('video.output_fps', float(fps_text) if fps_text else None)
        self.config.set('video.output_codec', self.combo_codec.currentData() or 'mp4v')
        self.config.set('video.frame_interval', self.spin_frame_interval.value())
        self.config.set('video.output_format', self.combo_video_format.currentData() or 'mp4')

        # 保存到文件
        self.config.save()
    
    def _update_model_status_label(self) -> None:
        """更新模型状态标签（语言切换时使用）"""
        current = self.label_model_info.text()
        # 保留当前状态文本，只在 retranslate_ui 时被调用

    def _browse_model_path(self) -> None:
        """浏览模型路径"""
        path = QFileDialog.getExistingDirectory(
            self,
            tr("选择模型目录"),
            self.edit_model_path.text() or "./models"
        )

        if path:
            self.edit_model_path.setText(path)

    def _check_model(self) -> None:
        """检查模型状态"""
        import os
        from pathlib import Path

        path = self.edit_model_path.text()

        if not path:
            self.label_model_info.setText(tr("模型状态: 未指定路径"))
            self.label_model_info.setStyleSheet("color: orange")
            return

        model_files = list(Path(path).glob("*.h5")) + list(Path(path).glob("*.pth"))

        if model_files:
            self.label_model_info.setText(tr("模型状态: 已找到 ") + str(len(model_files)) + tr(" 个模型文件"))
            self.label_model_info.setStyleSheet("color: green")
        else:
            self.label_model_info.setText(tr("模型状态: 未找到模型文件"))
            self.label_model_info.setStyleSheet("color: red")

    def _clear_recent_files(self) -> None:
        """清空最近文件"""
        reply = QMessageBox.question(
            self,
            tr("确认"),
            tr("确定要清空最近文件列表吗？"),
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.config.clear_recent_files()
            QMessageBox.information(self, tr("完成"), tr("最近文件列表已清空"))
    
    def _on_apply(self) -> None:
        """应用设置"""
        self._save_settings()
        self.settings_applied.emit()
    
    def _on_accept(self) -> None:
        """确认"""
        self._save_settings()
        self.settings_applied.emit()
        self.accept()
