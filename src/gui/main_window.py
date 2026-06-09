"""
主窗口模块

提供应用程序主窗口，包含菜单栏、工具栏、状态栏，
以及图像显示区域和对比面板。
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from enum import Enum
import datetime

from typing import Union, Optional, Tuple, List, Any  # 加上 Union
from pathlib import Path

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QAction, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QDockWidget, QProgressDialog,
    QLabel, QShortcut, QApplication, QSizePolicy,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider,
    QGroupBox, QCheckBox, QPushButton, QTextEdit
)
from PyQt5.QtCore import (
    Qt, QTimer, QSize, QPoint, QSettings, QByteArray,
    pyqtSignal, pyqtSlot, QThread
)
from PyQt5.QtGui import (
    QIcon, QKeySequence, QPixmap, QImage,
    QImageReader, QImageWriter
)

import numpy as np
import cv2

from ..core.enhancer import ImageEnhancer, EnhancementMode
from ..core.metrics import ImageMetrics
from ..core.model_manager import ModelManager
from ..core.dehaze_wrapper import deblur, remove_haze_pipeline
from ..utils.file_io import (
    read_image, write_image, get_image_files,
    ensure_directories_exist
)
from ..utils.config import ConfigManager
from ..utils.session import SessionManager
from .image_viewer import ImageViewer
from .compare_panel import ComparePanel
from .dialogs.settings_dialog import SettingsDialog
from .dialogs.batch_dialog import BatchProcessDialog
from .dialogs.video_dialog import VideoProcessDialog
from .dialogs.preset_dialog import PresetDialog
from .dialogs.export_dialog import ExportDialog
from .panels.filter_panel import FilterPanel
from .panels.property_panel import PropertyPanel
from .widgets.thumbnail_strip import ThumbnailStrip
from .widgets.magnifier import Magnifier
from .icons import get_icon
from ..i18n import tr, set_language, get_language
from ..processors.batch_processor import BatchProcessor
from ..processors.video_processor import VideoProcessor
from ..processors.adjust_filters import AdjustFilters
from .theme import apply_theme, toggle_theme, current_theme

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    应用程序主窗口类
    
    提供完整的GUI界面，包括：
    - 菜单栏（文件、编辑、视图、增强、工具、帮助）
    - 工具栏（快捷操作按钮）
    - 左侧面板（滤镜调节）
    - 中央区域（图像显示/对比面板）
    - 状态栏（显示当前状态和提示信息）
    
    Signals:
        image_loaded: 图像加载完成信号
        image_enhanced: 图像增强完成信号
        processing_started: 处理开始信号
        processing_finished: 处理完成信号
        progress_updated: 进度更新信号
    """
    
    # 信号定义
    image_loaded = pyqtSignal(object)  # numpy array
    image_enhanced = pyqtSignal(object, object)  # original, enhanced
    processing_started = pyqtSignal(str)  # message
    processing_finished = pyqtSignal(bool, str)  # success, message
    progress_updated = pyqtSignal(int, int, str)  # current, total, message
    
    def __init__(
        self,
        config: Optional[ConfigManager] = None,
        model_dir: str = './models',
        output_dir: str = './output',
        initial_image: Optional[str] = None,
        batch_mode: Optional[str] = None,
        video_file: Optional[str] = None
    ):
        """
        初始化主窗口
        
        Args:
            config: 配置管理器实例
            model_dir: 模型目录路径
            output_dir: 输出目录路径
            initial_image: 初始加载的图像路径
            batch_mode: 批量处理模式目录
            video_file: 视频文件路径
        """
        super().__init__()
        
        # 保存配置
        self.config = config or ConfigManager(auto_load=True)
        self.model_dir = Path(model_dir)
        self.output_dir = Path(output_dir)
        
        # 确保目录存在
        ensure_directories_exist([self.output_dir, self.model_dir])
        
        # 初始化组件
        self.enhancer: Optional[ImageEnhancer] = None
        self.metrics = ImageMetrics()
        
        # 图像数据
        self.original_image: Optional[np.ndarray] = None
        self.enhanced_image: Optional[np.ndarray] = None
        self.current_file_path: Optional[Path] = None
        
        # 滤镜调节器
        self.adjust_filters = AdjustFilters()

        # 模型管理器
        self.model_manager = ModelManager(self.model_dir)
        self.model_manager.set_on_switch(self._on_model_switched)

        # 会话管理器
        self.session_manager = SessionManager(session_dir=str(Path(self.output_dir).parent / "temp"))

        # 批量处理器
        self.batch_processor: Optional[BatchProcessor] = None

        # 视频处理器
        self.video_processor: Optional[VideoProcessor] = None

        # 撤销/重做栈
        self._undo_stack: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self._max_history = 50

        # UI状态
        self.is_enhanced = False
        self.show_comparison = False
        
        # 初始化UI
        self._init_ui()
        
        # 连接信号
        self._connect_signals()
        
        # 初始化增强器
        self._init_enhancer()
        
        # 加载初始内容
        self._load_initial_content(
            initial_image=initial_image,
            batch_mode=batch_mode,
            video_file=video_file
        )
        
        logger.info("主窗口初始化完成")
    
    def _init_ui(self) -> None:
        """初始化用户界面"""
        # 设置窗口属性
        self.setWindowTitle(tr("水下图像增强软件 - Underwater Image Enhancement"))
        self.setMinimumSize(1200, 800)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_toolbar()

        # 分配图标
        self._assign_menu_icons()

        # 创建状态栏
        self._create_statusbar()
        
        # 创建中央部件
        self._create_central_widget()
        
        # 创建左侧面板
        self._create_dock_widgets()
        
        # 应用配置
        self._apply_config()
    
    def _create_menu_bar(self) -> None:
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        self.menu_file = menubar.addMenu(tr("文件(&F)"))

        self.action_open = QAction(tr("打开图像..."), self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.setStatusTip(tr("打开图像文件"))
        self.menu_file.addAction(self.action_open)

        self.action_open_folder = QAction(tr("打开文件夹..."), self)
        self.action_open_folder.setStatusTip(tr("打开文件夹进行批量处理"))
        self.menu_file.addAction(self.action_open_folder)

        self.menu_file.addSeparator()

        self.action_save = QAction(tr("保存"), self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.setEnabled(False)
        self.menu_file.addAction(self.action_save)

        self.action_save_as = QAction(tr("另存为..."), self)
        self.action_save_as.setShortcut(QKeySequence.SaveAs)
        self.action_save_as.setEnabled(False)
        self.menu_file.addAction(self.action_save_as)

        self.menu_file.addSeparator()

        # 最近文件
        self.menu_recent = QMenu(tr("最近文件"), self)
        self.menu_file.addMenu(self.menu_recent)
        self._update_recent_files_menu()

        self.menu_file.addSeparator()

        self.action_save_session = QAction(tr("保存会话"), self)
        self.menu_file.addAction(self.action_save_session)

        self.action_restore_session = QAction(tr("恢复上次会话"), self)
        self.menu_file.addAction(self.action_restore_session)

        self.menu_file.addSeparator()

        self.action_exit = QAction(tr("退出"), self)
        self.action_exit.setShortcut(QKeySequence.Quit)
        self.menu_file.addAction(self.action_exit)

        # 编辑菜单
        self.menu_edit = menubar.addMenu(tr("编辑(&E)"))

        self.action_undo = QAction(tr("撤销"), self)
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_undo.setEnabled(False)
        self.menu_edit.addAction(self.action_undo)

        self.action_redo = QAction(tr("重做"), self)
        self.action_redo.setShortcut(QKeySequence.Redo)
        self.action_redo.setEnabled(False)
        self.menu_edit.addAction(self.action_redo)

        self.menu_edit.addSeparator()

        self.action_reset = QAction(tr("重置图像"), self)
        self.action_reset.setShortcut(QKeySequence("Ctrl+R"))
        self.action_reset.setEnabled(False)
        self.menu_edit.addAction(self.action_reset)

        self.action_reset_all = QAction(tr("重置所有"), self)
        self.menu_edit.addAction(self.action_reset_all)

        # 视图菜单
        self.menu_view = menubar.addMenu(tr("视图(&V)"))

        self.action_zoom_in = QAction(tr("放大"), self)
        self.action_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.menu_view.addAction(self.action_zoom_in)

        self.action_zoom_out = QAction(tr("缩小"), self)
        self.action_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.menu_view.addAction(self.action_zoom_out)

        self.action_zoom_fit = QAction(tr("适应窗口"), self)
        self.action_zoom_fit.setShortcut(QKeySequence("Ctrl+0"))
        self.menu_view.addAction(self.action_zoom_fit)

        self.action_zoom_100 = QAction(tr("实际大小 (100%)"), self)
        self.action_zoom_100.setShortcut(QKeySequence("Ctrl+1"))
        self.menu_view.addAction(self.action_zoom_100)

        self.menu_view.addSeparator()

        self.action_show_comparison = QAction(tr("对比模式"), self)
        self.action_show_comparison.setCheckable(True)
        self.action_show_comparison.setEnabled(False)
        self.menu_view.addAction(self.action_show_comparison)

        self.menu_view.addSeparator()

        self.action_toggle_filter = QAction(tr("滤镜调节面板"), self)
        self.action_toggle_filter.setCheckable(True)
        self.action_toggle_filter.setChecked(True)
        self.menu_view.addAction(self.action_toggle_filter)

        self.action_toggle_property = QAction(tr("图像属性面板"), self)
        self.action_toggle_property.setCheckable(True)
        self.action_toggle_property.setChecked(True)
        self.menu_view.addAction(self.action_toggle_property)

        self.action_toggle_thumbnail = QAction(tr("缩略图浏览器"), self)
        self.action_toggle_thumbnail.setCheckable(True)
        self.action_toggle_thumbnail.setChecked(False)
        self.menu_view.addAction(self.action_toggle_thumbnail)

        # 增强菜单
        self.menu_enhance = menubar.addMenu(tr("增强(&A)"))

        self.action_enhance = QAction(tr("自动增强"), self)
        self.action_enhance.setShortcut(QKeySequence("Ctrl+E"))
        self.action_enhance.setEnabled(False)
        self.menu_enhance.addAction(self.action_enhance)

        self.menu_enhance_mode = QMenu(tr("增强模式"), self)
        self.menu_enhance.addMenu(self.menu_enhance_mode)

        self.action_mode_auto = QAction(tr("自动"), self)
        self.action_mode_auto.setCheckable(True)
        self.action_mode_auto.setChecked(True)
        self.menu_enhance_mode.addAction(self.action_mode_auto)

        self.action_mode_standard = QAction(tr("标准"), self)
        self.action_mode_standard.setCheckable(True)
        self.menu_enhance_mode.addAction(self.action_mode_standard)

        self.action_mode_light = QAction(tr("轻度"), self)
        self.action_mode_light.setCheckable(True)
        self.menu_enhance_mode.addAction(self.action_mode_light)

        self.action_mode_strong = QAction(tr("强力"), self)
        self.action_mode_strong.setCheckable(True)
        self.menu_enhance_mode.addAction(self.action_mode_strong)

        self.menu_enhance.addSeparator()

        self.action_deblur = QAction(tr("去模糊"), self)
        self.action_deblur.setEnabled(False)
        self.menu_enhance.addAction(self.action_deblur)

        # 将模式菜单项设为互斥单选组
        from PyQt5.QtWidgets import QActionGroup
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        for action in [self.action_mode_auto, self.action_mode_standard,
                       self.action_mode_light, self.action_mode_strong]:
            mode_group.addAction(action)

        # 工具菜单
        self.menu_tools = menubar.addMenu(tr("工具(&T)"))

        self.action_batch = QAction(tr("批量处理..."), self)
        self.menu_tools.addAction(self.action_batch)

        self.action_video = QAction(tr("视频处理..."), self)
        self.menu_tools.addAction(self.action_video)

        self.menu_tools.addSeparator()

        self.action_preset_manager = QAction(tr("预设管理..."), self)
        self.menu_tools.addAction(self.action_preset_manager)

        self.action_export_comparison = QAction(tr("导出对比图..."), self)
        self.action_export_comparison.setEnabled(False)
        self.menu_tools.addAction(self.action_export_comparison)

        self.menu_tools.addSeparator()

        self.action_settings = QAction(tr("设置..."), self)
        self.action_settings.setShortcut(QKeySequence("Ctrl+,"))
        self.menu_tools.addAction(self.action_settings)

        # 帮助菜单
        self.menu_help = menubar.addMenu(tr("帮助(&H)"))

        self.action_about = QAction(tr("关于"), self)
        self.menu_help.addAction(self.action_about)

        self.action_metrics = QAction(tr("质量指标"), self)
        self.action_metrics.setEnabled(False)
        self.menu_help.addAction(self.action_metrics)
    
    def _create_toolbar(self) -> None:
        """创建工具栏"""
        # 主工具栏
        self.toolbar = QToolBar(tr("主工具栏"))
        self.toolbar.setObjectName("main_toolbar")
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(self.toolbar)

        # 添加工具栏按钮
        self.action_open.setIcon(get_icon('open'))
        self.toolbar.addAction(self.action_open)

        self.action_save.setIcon(get_icon('save'))
        self.toolbar.addAction(self.action_save)

        self.toolbar.addSeparator()

        self.action_enhance.setIcon(get_icon('enhance'))
        self.toolbar.addAction(self.action_enhance)

        self.toolbar.addSeparator()

        self.action_zoom_in.setIcon(get_icon('zoom_in'))
        self.toolbar.addAction(self.action_zoom_in)

        self.action_zoom_out.setIcon(get_icon('zoom_out'))
        self.toolbar.addAction(self.action_zoom_out)

        self.action_zoom_fit.setIcon(get_icon('zoom_fit'))
        self.toolbar.addAction(self.action_zoom_fit)

        # 增强模式选择器
        self.toolbar.addSeparator()
        self.label_enhance_mode = QLabel(tr("增强模式:"))
        self.toolbar.addWidget(self.label_enhance_mode)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem(tr("自动"), "auto")
        self.combo_mode.addItem(tr("标准"), "standard")
        self.combo_mode.addItem(tr("轻度"), "light")
        self.combo_mode.addItem(tr("强力"), "strong")
        self.combo_mode.setCurrentIndex(0)
        self.toolbar.addWidget(self.combo_mode)

        # 模型选择器
        self.toolbar.addSeparator()
        self.label_model_selector = QLabel(tr("模型:"))
        self.toolbar.addWidget(self.label_model_selector)

        self.combo_model = QComboBox()
        self.combo_model.setMinimumWidth(120)
        self._refresh_model_combo()
        self.toolbar.addWidget(self.combo_model)
    
    def _create_statusbar(self) -> None:
        """创建状态栏"""
        statusbar = QStatusBar()
        self.setStatusBar(statusbar)

        # 状态标签
        self.label_status = QLabel(tr("就绪"))
        statusbar.addWidget(self.label_status)

        # 缩放比例
        self.status_label_zoom = QLabel(tr("  缩放:"))
        statusbar.addPermanentWidget(self.status_label_zoom)
        self.label_zoom = QLabel("100%")
        statusbar.addPermanentWidget(self.label_zoom)

        # 图像尺寸
        self.status_label_size = QLabel(tr("  尺寸:"))
        statusbar.addPermanentWidget(self.status_label_size)
        self.label_size = QLabel("-")
        statusbar.addPermanentWidget(self.label_size)

        # 模型状态
        self.status_label_model = QLabel(tr("  模型:"))
        statusbar.addPermanentWidget(self.status_label_model)
        self.label_model = QLabel(tr("未加载"))
        statusbar.addPermanentWidget(self.label_model)
    
    def _create_central_widget(self) -> None:
        """创建中央部件"""
        # 堆叠窗口用于切换显示模式
        self.stack_widget = QWidget()
        self.stack_layout = QVBoxLayout(self.stack_widget)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        
        # 图像查看器
        self.image_viewer = ImageViewer()
        
        # 对比面板
        self.compare_panel = ComparePanel()
        self.compare_panel.hide()
        
        self.stack_layout.addWidget(self.image_viewer)
        self.stack_layout.addWidget(self.compare_panel)
        
        self.setCentralWidget(self.stack_widget)
    
    def _create_dock_widgets(self) -> None:
        """创建停靠面板 — 左:滤镜+变换, 右:属性+历史, 底部:缩略图"""
        # ── 左侧：滤镜调节（主面板）─ 始终可见 ──
        self.filter_panel = FilterPanel()
        self.filter_panel.connect_preset(
            original=lambda: self._apply_preset("original"),
            auto=lambda: self._apply_preset("auto"),
            vivid=lambda: self._apply_preset("vivid"),
            soft=lambda: self._apply_preset("soft"),
        )
        self.filter_panel.connect_filter_changed(self._on_filter_changed)
        self.filter_panel.connect_reset(self._reset_filters)

        self.filter_dock = QDockWidget(tr("滤镜调节"), self)
        self.filter_dock.setObjectName("filter_dock")
        self.filter_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.filter_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.filter_dock.setWidget(self.filter_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.filter_dock)

        # ── 右侧：图像属性（主面板）─ 始终可见 ──
        self.property_panel = PropertyPanel()
        self.property_dock = QDockWidget(tr("图像属性"), self)
        self.property_dock.setObjectName("property_dock")
        self.property_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.property_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        self.property_dock.setWidget(self.property_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.property_dock)

        # ── 底部：缩略图浏览器（默认隐藏）──
        self.thumbnail_strip = ThumbnailStrip()
        self.thumbnail_dock = QDockWidget(tr("缩略图"), self)
        self.thumbnail_dock.setObjectName("thumbnail_dock")
        self.thumbnail_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.thumbnail_dock.setWidget(self.thumbnail_strip)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.thumbnail_dock)
        self.thumbnail_dock.hide()

        # ── 放大镜（随图像悬停显示）──
        self.magnifier = Magnifier()
        self.magnifier.set_enabled(False)
        self.image_viewer.set_magnifier(self.magnifier)
    

    def _assign_menu_icons(self) -> None:
        """为菜单项分配图标"""
        self.action_open.setIcon(get_icon('open'))
        self.action_save.setIcon(get_icon('save'))
        self.action_open_folder.setIcon(get_icon('open'))
        self.action_batch.setIcon(get_icon('batch'))
        self.action_video.setIcon(get_icon('video'))
        self.action_enhance.setIcon(get_icon('enhance'))
        self.action_zoom_in.setIcon(get_icon('zoom_in'))
        self.action_zoom_out.setIcon(get_icon('zoom_out'))
        self.action_zoom_fit.setIcon(get_icon('zoom_fit'))
        self.action_reset.setIcon(get_icon('reset'))
        self.action_settings.setIcon(get_icon('settings'))
        self.action_about.setIcon(get_icon('about'))
        self.action_metrics.setIcon(get_icon('compare'))
        self.action_show_comparison.setIcon(get_icon('compare'))

    def _connect_signals(self) -> None:
        """连接信号槽"""
        # 文件操作
        self.action_open.triggered.connect(self._on_open_image)
        self.action_open_folder.triggered.connect(self._on_open_folder)
        self.action_save.triggered.connect(self._on_save)
        self.action_save_as.triggered.connect(self._on_save_as)
        self.action_save_session.triggered.connect(self._on_save_session)
        self.action_restore_session.triggered.connect(self._on_restore_session)
        self.action_exit.triggered.connect(self.close)

        # 编辑操作
        self.action_undo.triggered.connect(self._on_undo)
        self.action_redo.triggered.connect(self._on_redo)
        self.action_reset.triggered.connect(self._on_reset)
        self.action_reset_all.triggered.connect(self._reset_filters)

        # 视图操作
        self.action_zoom_in.triggered.connect(self._on_zoom_in)
        self.action_zoom_out.triggered.connect(self._on_zoom_out)
        self.action_zoom_fit.triggered.connect(self._on_zoom_fit)
        self.action_zoom_100.triggered.connect(self._on_zoom_100)
        self.action_show_comparison.toggled.connect(self._on_toggle_comparison)
        self.action_toggle_filter.toggled.connect(self.filter_dock.setVisible)
        self.action_toggle_property.toggled.connect(self.property_dock.setVisible)
        self.action_toggle_thumbnail.toggled.connect(self.thumbnail_dock.setVisible)
        self.filter_dock.visibilityChanged.connect(self.action_toggle_filter.setChecked)
        self.property_dock.visibilityChanged.connect(self.action_toggle_property.setChecked)
        self.thumbnail_dock.visibilityChanged.connect(self.action_toggle_thumbnail.setChecked)

        # 增强操作
        self.action_enhance.triggered.connect(self._on_enhance)
        self.action_deblur.triggered.connect(self._on_deblur)
        self.combo_mode.currentIndexChanged.connect(
            lambda: self._on_mode_changed(self.combo_mode.currentData()))
        self.combo_model.currentIndexChanged.connect(self._on_model_combo_changed)

        # 菜单模式 → 下拉框 同步 (use index to avoid text dependency)
        self.action_mode_auto.triggered.connect(lambda: self.combo_mode.setCurrentIndex(0))
        self.action_mode_standard.triggered.connect(lambda: self.combo_mode.setCurrentIndex(1))
        self.action_mode_light.triggered.connect(lambda: self.combo_mode.setCurrentIndex(2))
        self.action_mode_strong.triggered.connect(lambda: self.combo_mode.setCurrentIndex(3))

        # 工具操作
        self.action_batch.triggered.connect(self._on_batch_processing)
        self.action_video.triggered.connect(self._on_video_processing)
        self.action_preset_manager.triggered.connect(self._on_preset_manager)
        self.action_export_comparison.triggered.connect(self._on_export_comparison)
        self.action_settings.triggered.connect(self._on_settings)

        # 帮助操作
        self.action_about.triggered.connect(self._on_about)
        self.action_metrics.triggered.connect(self._on_show_metrics)

        # 图像查看器信号
        self.image_viewer.zoom_changed.connect(self._on_zoom_changed)
        self.image_viewer.file_dropped.connect(self.load_image)
        # 缩略图
        self.thumbnail_strip.file_selected.connect(self.load_image)

        # 键盘快捷键
        QShortcut(QKeySequence("Ctrl+W"), self, self._on_open_image)
        QShortcut(QKeySequence("Ctrl+S"), self, self._on_save)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.thumbnail_strip.navigate_prev())
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.thumbnail_strip.navigate_next())
    
    def _init_enhancer(self) -> None:
        """初始化增强器"""
        try:
            # 查找模型文件
            model_files = list(self.model_dir.glob("*.h5")) + list(self.model_dir.glob("*.pth"))
            
            model_path = None
            if model_files:
                model_path = str(model_files[0])
                logger.info(f"找到模型文件: {model_path}")
            
            # 创建增强器
            self.enhancer = ImageEnhancer(
                model_path=model_path,
                auto_load=True
            )
            
            if self.enhancer.is_model_loaded:
                self.label_model.setText(tr("已加载"))
                self.label_model.setStyleSheet("color: green")
            else:
                self.label_model.setText(tr("未加载"))
                self.label_model.setStyleSheet("color: orange")

            # 设置进度回调
            self.enhancer.set_progress_callback(self._on_progress)

        except Exception as e:
            logger.error(f"增强器初始化失败: {e}")
            self.label_model.setText(tr("错误"))
            self.label_model.setStyleSheet("color: red")
    
    def _load_initial_content(
        self,
        initial_image: Optional[str] = None,
        batch_mode: Optional[str] = None,
        video_file: Optional[str] = None
    ) -> None:
        """加载初始内容"""
        if video_file:
            self._on_video_processing(video_file)
        elif batch_mode:
            self._on_batch_processing(batch_mode)
        elif initial_image:
            self.load_image(initial_image)
    
    def load_image(self, path: Union[str, Path]) -> bool:
        """
        加载图像
        
        Args:
            path: 图像文件路径
            
        Returns:
            加载是否成功
        """
        path = Path(path)
        
        if not path.exists():
            self.statusBar().showMessage(tr("文件不存在: ") + str(path), 3000)
            return False

        try:
            self.label_status.setText(tr("正在加载..."))
            self.setCursor(Qt.WaitCursor)

            # 读取图像
            self.original_image = read_image(path)
            self.enhanced_image = None
            self.current_file_path = path
            self.is_enhanced = False

            # 显示图像
            self.image_viewer.set_image(self.original_image)

            # 重置UI状态
            self._reset_ui_state()

            # 更新属性面板和直方图
            self._update_property_panel()
            self.property_panel.set_histograms(self.original_image)

            # 启用放大镜
            self.magnifier.set_enabled(True)

            # 添加到最近文件
            self.config.add_recent_file(str(path))
            self._update_recent_files_menu()

            # 更新状态栏
            h, w = self.original_image.shape[:2]
            self.label_status.setText(tr("已加载: ") + path.name)
            self.label_size.setText(f"{w} × {h}")

            # 启用菜单项
            self._enable_image_actions(True)

            self.unsetCursor()

            logger.info(f"图像已加载: {path}")
            return True

        except Exception as e:
            self.unsetCursor()
            self.label_status.setText(tr("加载失败"))
            logger.error(f"图像加载失败: {e}")
            QMessageBox.critical(self, tr("错误"), tr("图像加载失败:\n") + str(e))
            return False
    
    def _enable_image_actions(self, enabled: bool) -> None:
        """启用/禁用图像相关操作"""
        self.action_save.setEnabled(enabled)
        self.action_save_as.setEnabled(enabled)
        self.action_reset.setEnabled(enabled)
        self.action_enhance.setEnabled(enabled and self.enhancer is not None)
        self.action_deblur.setEnabled(enabled)
        self.action_show_comparison.setEnabled(enabled and self.is_enhanced)
        self.action_metrics.setEnabled(enabled and self.is_enhanced)
        self.action_export_comparison.setEnabled(enabled and self.is_enhanced)
    
    def _reset_ui_state(self) -> None:
        """重置UI状态"""
        self.show_comparison = False
        self.action_show_comparison.setChecked(False)
        self.image_viewer.show()
        self.compare_panel.hide()
        self._reset_filters()
    
    def _update_property_panel(self) -> None:
        """更新属性面板"""
        if self.current_file_path is None:
            self.property_panel.set_file_info(tr("无"))
            return

        info = []
        info.append(tr("文件名: ") + self.current_file_path.name)
        info.append(tr("路径: ") + str(self.current_file_path.parent))
        info.append(tr("大小: ") + f"{self.current_file_path.stat().st_size / 1024:.1f} KB")

        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            info.append(tr("尺寸: ") + f"{w} × {h}")
            info.append(tr("通道: ") + str(self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1))

        self.property_panel.set_file_info("\n".join(info))
    
    def _update_metrics_panel(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        """更新质量指标面板"""
        if original is None or enhanced is None:
            self.property_panel.set_metrics(tr("无"))
            return

        metrics = self.metrics.evaluate(original, enhanced)

        info = []
        info.append(tr("【增强后质量指标】"))
        info.append(f"UIQM: {metrics.uiqm:.4f}")
        info.append(f"UISM: {metrics.uism:.4f}")
        info.append(f"UICONM: {metrics.uiconm:.4f}")
        info.append(f"SSIM: {metrics.ssim:.4f}")
        info.append(f"PSNR: {metrics.psnr:.2f} dB")
        info.append(f"MAE: {metrics.mae:.4f}")
        info.append(f"UCIQE: {metrics.uciqe:.4f}")

        self.property_panel.set_metrics("\n".join(info))
    
    def _update_recent_files_menu(self) -> None:
        """更新最近文件菜单"""
        self.menu_recent.clear()
        
        recent_files = self.config.get('ui.recent_files', [])
        
        if not recent_files:
            action = self.menu_recent.addAction(tr("无最近文件"))
            action.setEnabled(False)
            return
        
        for path in recent_files:
            action = self.menu_recent.addAction(Path(path).name)
            action.setData(path)
            action.triggered.connect(lambda checked, p=path: self.load_image(p))
    
    def _apply_config(self) -> None:
        """应用配置（启动时调用 + 设置对话框"应用"按钮触发）"""
        # 1. 主题
        theme = self.config.get('ui.theme', 'light')
        if theme in ('light', 'dark'):
            apply_theme(QApplication.instance(), theme)
        else:
            try:
                QApplication.instance().setStyleSheet("")
            except Exception:
                pass
            theme_map = {'fusion': 'Fusion', 'windows': 'Windows', 'windowsvista': 'WindowsVista'}
            style_name = theme_map.get(theme, 'Fusion')
            from PyQt5.QtWidgets import QStyleFactory
            available = QStyleFactory.keys()
            if style_name.lower() in [s.lower() for s in available]:
                QApplication.instance().setStyle(style_name)

        # 2. 语言
        lang = self.config.get('ui.language', 'zh_CN')
        if get_language() != lang:
            set_language(lang)
            self.retranslate_ui()

        # 3. 工具栏可见性
        show_toolbar = self.config.get('ui.show_toolbar', True)
        self.toolbar.setVisible(show_toolbar)

        # 4. 状态栏可见性
        show_statusbar = self.config.get('ui.show_statusbar', True)
        self.statusBar().setVisible(show_statusbar)

        # 5. 窗口几何/状态
        geometry = self.config.get('ui.window_geometry')
        if geometry:
            try:
                self.restoreGeometry(QByteArray.fromBase64(geometry.encode()))
            except Exception:
                pass

        state = self.config.get('ui.window_state')
        if state:
            try:
                self.restoreState(QByteArray.fromBase64(state.encode()))
            except Exception:
                pass

        # 6. 确保主要面板可见（滤镜 / 属性）
        if not self.filter_dock.isVisible():
            self.filter_dock.show()
        if not self.property_dock.isVisible():
            self.property_dock.show()

    def retranslate_ui(self) -> None:
        """重新翻译所有UI字符串（语言切换时调用）"""
        # 窗口标题
        self.setWindowTitle(tr("水下图像增强软件 - Underwater Image Enhancement"))

        # 菜单栏
        self.menu_file.setTitle(tr("文件(&F)"))
        self.menu_edit.setTitle(tr("编辑(&E)"))
        self.menu_view.setTitle(tr("视图(&V)"))
        self.menu_enhance.setTitle(tr("增强(&A)"))
        self.menu_tools.setTitle(tr("工具(&T)"))
        self.menu_help.setTitle(tr("帮助(&H)"))
        self.menu_recent.setTitle(tr("最近文件"))
        self.menu_enhance_mode.setTitle(tr("增强模式"))

        # 文件菜单 actions
        self.action_open.setText(tr("打开图像..."))
        self.action_open.setStatusTip(tr("打开图像文件"))
        self.action_open_folder.setText(tr("打开文件夹..."))
        self.action_open_folder.setStatusTip(tr("打开文件夹进行批量处理"))
        self.action_save.setText(tr("保存"))
        self.action_save_as.setText(tr("另存为..."))
        self.action_save_session.setText(tr("保存会话"))
        self.action_restore_session.setText(tr("恢复上次会话"))
        self.action_exit.setText(tr("退出"))

        # 编辑菜单 actions
        self.action_undo.setText(tr("撤销"))
        self.action_redo.setText(tr("重做"))
        self.action_reset.setText(tr("重置图像"))
        self.action_reset_all.setText(tr("重置所有"))

        # 视图菜单 actions
        self.action_zoom_in.setText(tr("放大"))
        self.action_zoom_out.setText(tr("缩小"))
        self.action_zoom_fit.setText(tr("适应窗口"))
        self.action_zoom_100.setText(tr("实际大小 (100%)"))
        self.action_show_comparison.setText(tr("对比模式"))
        self.action_toggle_filter.setText(tr("滤镜调节面板"))
        self.action_toggle_property.setText(tr("图像属性面板"))
        self.action_toggle_thumbnail.setText(tr("缩略图浏览器"))

        # 增强菜单 actions
        self.action_enhance.setText(tr("自动增强"))
        self.action_mode_auto.setText(tr("自动"))
        self.action_mode_standard.setText(tr("标准"))
        self.action_mode_light.setText(tr("轻度"))
        self.action_mode_strong.setText(tr("强力"))
        self.action_deblur.setText(tr("去模糊"))

        # 工具菜单 actions
        self.action_batch.setText(tr("批量处理..."))
        self.action_video.setText(tr("视频处理..."))
        self.action_preset_manager.setText(tr("预设管理..."))
        self.action_export_comparison.setText(tr("导出对比图..."))
        self.action_settings.setText(tr("设置..."))

        # 帮助菜单 actions
        self.action_about.setText(tr("关于"))
        self.action_metrics.setText(tr("质量指标"))

        # 工具栏
        self.toolbar.setWindowTitle(tr("主工具栏"))
        self.label_enhance_mode.setText(tr("增强模式:"))
        self.label_model_selector.setText(tr("模型:"))

        # 增强模式下拉框（保留 itemData 键）
        self.combo_mode.clear()
        self.combo_mode.addItem(tr("自动"), "auto")
        self.combo_mode.addItem(tr("标准"), "standard")
        self.combo_mode.addItem(tr("轻度"), "light")
        self.combo_mode.addItem(tr("强力"), "strong")

        # 状态栏
        self.label_status.setText(tr("就绪"))
        self.status_label_zoom.setText(tr("  缩放:"))
        self.status_label_size.setText(tr("  尺寸:"))
        self.status_label_model.setText(tr("  模型:"))

        # 更新模型状态文本
        if self.enhancer and self.enhancer.is_model_loaded:
            self.label_model.setText(tr("已加载"))
        else:
            self.label_model.setText(tr("未加载"))

        # 停靠窗口
        self.filter_dock.setWindowTitle(tr("滤镜调节"))
        self.property_dock.setWindowTitle(tr("图像属性"))
        self.thumbnail_dock.setWindowTitle(tr("缩略图"))

        # 滤镜面板 & 属性面板
        self.filter_panel.retranslate_ui()
        self.property_panel.retranslate_ui()
        # 最近文件菜单
        self._update_recent_files_menu()

        # 属性面板内容
        self._update_property_panel()
        if self.original_image is not None and self.enhanced_image is not None:
            self._update_metrics_panel(self.original_image, self.enhanced_image)

        # 级联到子组件
        self.compare_panel.retranslate_ui()
        self.image_viewer.retranslate_ui()
    
    # -- model management --
    def _refresh_model_combo(self) -> None:
        """刷新模型下拉框"""
        self.combo_model.blockSignals(True)
        self.combo_model.clear()
        models = self.model_manager.models
        for m in models:
            self.combo_model.addItem(m.name, m.path)
        if models:
            self.combo_model.setCurrentIndex(self.model_manager.current_index)
        self.combo_model.blockSignals(False)

    def _on_model_switched(self, model_path: str) -> None:
        """模型切换回调"""
        if self.enhancer:
            self.model_manager.reload_enhancer(self.enhancer, model_path)
            self.label_model.setText(tr("已加载: ") + Path(model_path).stem)
            if self.original_image is not None and self.is_enhanced:
                self._on_enhance()

    def _on_model_combo_changed(self, index: int) -> None:
        if index >= 0:
            self.model_manager.switch_to(index)

    # -- undo / redo --
    def _push_undo(self, image: np.ndarray) -> None:
        """保存撤销状态"""
        self._undo_stack.append(image.copy())
        if len(self._undo_stack) > self._max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.action_undo.setEnabled(True)
        self.action_redo.setEnabled(False)

    def _on_undo(self) -> None:
        """撤销"""
        if not self._undo_stack:
            return
        current = self.image_viewer.current_image
        if current is not None:
            self._redo_stack.append(current.copy())
        prev = self._undo_stack.pop()
        self.image_viewer.set_image(prev)
        self.action_redo.setEnabled(True)
        self.action_undo.setEnabled(len(self._undo_stack) > 0)

    def _on_redo(self) -> None:
        """重做"""
        if not self._redo_stack:
            return
        current = self.image_viewer.current_image
        if current is not None:
            self._undo_stack.append(current.copy())
        next_img = self._redo_stack.pop()
        self.image_viewer.set_image(next_img)
        self.action_undo.setEnabled(True)
        self.action_redo.setEnabled(len(self._redo_stack) > 0)

    # -- deblur --
    def _on_deblur(self) -> None:
        """去模糊处理"""
        image = self.image_viewer.current_image
        if image is None:
            return
        try:
            self._push_undo(image)
            result = deblur(image)
            self.image_viewer.set_image(result)
            self.label_status.setText(tr("去模糊完成"))
        except Exception as e:
            logger.error(f"去模糊失败: {e}")
            self.label_status.setText(tr("去模糊失败"))

    # -- preset manager --
    def _on_preset_manager(self) -> None:
        """打开预设管理对话框"""
        fp = self.filter_panel
        params = {
            "brightness": fp.slider_brightness.value() / 100.0,
            "contrast": fp.slider_contrast.value() / 100.0 + 1.0,
            "saturation": fp.slider_saturation.value() / 100.0 + 1.0,
            "sharpness": fp.slider_sharpness.value() / 100.0 + 1.0,
            "wb": fp.slider_wb.value() / 100.0,
        }
        dialog = PresetDialog(self, current_params=params)
        dialog.preset_applied.connect(self._on_preset_applied)
        dialog.exec_()

    def _on_preset_applied(self, preset: dict) -> None:
        """应用自定义预设"""
        fp = self.filter_panel
        fp.slider_brightness.setValue(int(preset.get("brightness", 0) * 100))
        fp.slider_contrast.setValue(int((preset.get("contrast", 1.0) - 1) * 100))
        fp.slider_saturation.setValue(int((preset.get("saturation", 1.0) - 1) * 100))
        fp.slider_sharpness.setValue(int((preset.get("sharpness", 1.0) - 1) * 100))
        fp.slider_wb.setValue(int(preset.get("wb", 0) * 100))
        self._on_filter_changed()

    # -- export comparison --
    def _on_export_comparison(self) -> None:
        """导出对比图"""
        dialog = ExportDialog(self, self.original_image, self.enhanced_image)
        dialog.exec_()

    # -- session --
    def _on_save_session(self) -> None:
        """保存会话"""
        fp = self.filter_panel
        data = {
            "image_path": str(self.current_file_path) if self.current_file_path else "",
            "filters": {
                "brightness": fp.slider_brightness.value(),
                "contrast": fp.slider_contrast.value(),
                "saturation": fp.slider_saturation.value(),
                "sharpness": fp.slider_sharpness.value(),
                "wb": fp.slider_wb.value(),
            },
            "mode": self.combo_mode.currentData(),
            "model": self.combo_model.currentData() if self.combo_model.count() > 0 else "",
            "language": get_language(),
            "theme": current_theme(),
        }
        self.session_manager.save(data)
        self.label_status.setText(tr("会话已保存"))

    def _on_restore_session(self) -> None:
        """恢复上次会话"""
        data = self.session_manager.restore()
        if not data:
            QMessageBox.information(self, tr("提示"), tr("没有可恢复的会话"))
            return
        image_path = data.get("image_path", "")
        if image_path and Path(image_path).exists():
            self.load_image(image_path)
        filters = data.get("filters", {})
        if filters:
            fp = self.filter_panel
            fp.slider_brightness.setValue(filters.get("brightness", 0))
            fp.slider_contrast.setValue(filters.get("contrast", 0))
            fp.slider_saturation.setValue(filters.get("saturation", 0))
            fp.slider_sharpness.setValue(filters.get("sharpness", 0))
            fp.slider_wb.setValue(filters.get("wb", 0))
            self._on_filter_changed()
        self.label_status.setText(tr("会话已恢复"))

    def _save_config(self) -> None:
        """保存配置"""
        self.config.set('ui.window_geometry', self.saveGeometry().toBase64().data().decode())
        self.config.set('ui.window_state', self.saveState().toBase64().data().decode())
        self.config.save()
    
    # ==================== 事件处理 ====================
    
    def _on_open_image(self) -> None:
        """打开图像"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            tr("打开图像"),
            "",
            tr("图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)")
        )
        
        if path:
            self.load_image(path)
    
    def _on_open_folder(self) -> None:
        """打开文件夹"""
        folder = QFileDialog.getExistingDirectory(
            self,
            tr("选择文件夹"),
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder:
            self.thumbnail_strip.load_folder(folder)
            self.thumbnail_dock.show()
            self.action_toggle_thumbnail.setChecked(True)
            self._on_batch_processing(folder)
    
    def _on_save(self) -> None:
        """保存图像"""
        image = self.image_viewer.current_image
        if image is None:
            return

        if self.current_file_path:
            output_path = self.output_dir / f"{self.current_file_path.stem}_enhanced{self.current_file_path.suffix}"
        else:
            output_path = self.output_dir / "enhanced.png"

        self._save_image(output_path)
    
    def _on_save_as(self) -> None:
        """另存为"""
        image = self.image_viewer.current_image
        if image is None:
            return

        default_name = f"{self.current_file_path.stem}_enhanced.png" if self.current_file_path else "enhanced.png"
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("保存图像"),
            str(self.output_dir / default_name),
            tr("PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;所有文件 (*)")
        )

        if path:
            self._save_image(Path(path))
    
    def _save_image(self, path: Path) -> bool:
        """保存图像到文件"""
        try:
            image = self.image_viewer.current_image
            if image is None:
                return False
            
            write_image(image, path)
            
            self.statusBar().showMessage(tr("已保存: ") + str(path), 3000)
            logger.info(f"图像已保存: {path}")
            return True

        except Exception as e:
            QMessageBox.critical(self, tr("错误"), tr("保存失败:\n") + str(e))
            return False
    
    def _on_reset(self) -> None:
        """重置图像"""
        if self.original_image is None:
            return
        
        self.enhanced_image = None
        self.is_enhanced = False
        self.image_viewer.set_image(self.original_image)
        self.label_status.setText(tr("已重置为原始图像"))
        self._enable_image_actions(True)
    
    def _on_enhance(self) -> None:
        """增强图像"""
        if self.original_image is None or self.enhancer is None:
            return

        try:
            self.label_status.setText(tr("正在增强..."))
            self.setCursor(Qt.WaitCursor)

            # 保存撤销状态
            current_display = self.image_viewer.current_image
            if current_display is not None:
                self._push_undo(current_display)

            mode_key = self.combo_mode.currentData()
            mode_map = {
                "auto": EnhancementMode.AUTO,
                "standard": EnhancementMode.STANDARD,
                "light": EnhancementMode.LIGHT,
                "strong": EnhancementMode.STRONG
            }
            mode = mode_map.get(mode_key, EnhancementMode.AUTO)

            self.enhanced_image = self.enhancer.enhance(
                self.original_image,
                mode=mode
            )
            self.is_enhanced = True

            self.image_viewer.set_image(self.enhanced_image)
            self._update_metrics_panel(self.original_image, self.enhanced_image)
            self.property_panel.set_histograms(self.original_image, self.enhanced_image)

            self.label_status.setText(tr("增强完成"))
            self.action_show_comparison.setEnabled(True)
            self.action_export_comparison.setEnabled(True)

            self.unsetCursor()
            logger.info("图像增强完成")

        except Exception as e:
            self.unsetCursor()
            self.label_status.setText(tr("增强失败"))
            logger.error(f"增强失败: {e}")
            QMessageBox.critical(self, tr("错误"), tr("增强失败:\n") + str(e))
    
    def _on_mode_changed(self, mode_key: str) -> None:
        """增强模式改变 (mode_key is itemData: auto/standard/light/strong)"""
        mode_action_map = {
            "auto": self.action_mode_auto,
            "standard": self.action_mode_standard,
            "light": self.action_mode_light,
            "strong": self.action_mode_strong
        }

        for key, action in mode_action_map.items():
            action.setChecked(key == mode_key)
    
    def _on_toggle_comparison(self, checked: bool) -> None:
        """切换对比模式"""
        self.show_comparison = checked
        
        if checked and self.original_image is not None and self.enhanced_image is not None:
            # 切换到对比面板
            self.image_viewer.hide()
            self.compare_panel.show()
            
            # 设置对比图像
            if self.original_image is not None:
                self.compare_panel.set_images(self.original_image, self.enhanced_image)
        else:
            # 切换到普通视图
            self.compare_panel.hide()
            self.image_viewer.show()
            
            if self.enhanced_image is not None:
                self.image_viewer.set_image(
                    self.enhanced_image if self.is_enhanced else self.original_image
                )
    
    def _on_filter_changed(self, value: int = None) -> None:
        """滤镜参数改变（实时预览）"""
        if self.original_image is None:
            return

        fp = self.filter_panel
        brightness = fp.slider_brightness.value() / 100.0 + 1.0
        contrast = fp.slider_contrast.value() / 100.0 + 1.0
        saturation = fp.slider_saturation.value() / 100.0 + 1.0
        sharpness = fp.slider_sharpness.value() / 100.0 + 1.0
        wb = fp.slider_wb.value() / 100.0 + 1.0
        
        # 获取基础图像（增强或原始）
        base_image = self.enhanced_image if self.is_enhanced else self.original_image
        
        # 应用滤镜
        try:
            filtered = self.adjust_filters.apply(
                base_image,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                sharpness=sharpness,
                white_balance=wb
            )
            
            # 更新显示
            self.image_viewer.set_image(filtered)
            
        except Exception as e:
            logger.error(f"滤镜应用失败: {e}")
    
    def _apply_preset(self, preset: str) -> None:
        """应用预设"""
        presets = {
            "original": (0, 1.0, 1.0, 1.0, 0, 1.0),
            "auto": (0.1, 1.1, 1.1, 1.0, 0, 1.0),
            "vivid": (0.1, 1.3, 1.4, 1.2, 0, 1.1),
            "soft": (0.05, 0.9, 0.9, 0.8, 0, 0.95)
        }

        if preset not in presets:
            return

        brightness, contrast, saturation, sharpness, wb, gamma = presets[preset]
        fp = self.filter_panel

        fp.slider_brightness.setValue(int(brightness * 100))
        fp.slider_contrast.setValue(int((contrast - 1) * 100))
        fp.slider_saturation.setValue(int((saturation - 1) * 100))
        fp.slider_sharpness.setValue(int((sharpness - 1) * 100))
        fp.slider_wb.setValue(int(wb * 100))

        self._on_filter_changed()
    
    def _reset_filters(self) -> None:
        """重置所有滤镜"""
        fp = self.filter_panel
        fp.slider_brightness.setValue(0)
        fp.slider_contrast.setValue(0)
        fp.slider_saturation.setValue(0)
        fp.slider_sharpness.setValue(0)
        fp.slider_wb.setValue(0)
        
        # 恢复显示
        if self.original_image is not None:
            base = self.enhanced_image if self.is_enhanced else self.original_image
            self.image_viewer.set_image(base)
    
    def _on_zoom_in(self) -> None:
        """放大"""
        self.image_viewer.zoom_in()
    
    def _on_zoom_out(self) -> None:
        """缩小"""
        self.image_viewer.zoom_out()
    
    def _on_zoom_fit(self) -> None:
        """适应窗口"""
        self.image_viewer.fit_to_window()
    
    def _on_zoom_100(self) -> None:
        """实际大小"""
        self.image_viewer.set_zoom(1.0)
    
    def _on_zoom_changed(self, zoom: float) -> None:
        """缩放比例改变"""
        self.label_zoom.setText(f"{zoom * 100:.0f}%")
    
    def _on_progress(self, current: int, total: int, message: str = "") -> None:
        """处理进度更新"""
        if message:
            self.label_status.setText(message)
    
    def _on_batch_processing(self, folder: str = None) -> None:
        """批量处理"""
        if folder is None or isinstance(folder, bool):
            # Allow selecting individual image files or a folder
            files, _ = QFileDialog.getOpenFileNames(
                self,
                tr("选择要处理的图像文件"),
                "",
                tr("图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)")
            )

            if files:
                self._process_file_list(files)
                return

            folder = QFileDialog.getExistingDirectory(
                self,
                tr("或选择包含图像的文件夹"),
                "",
                QFileDialog.ShowDirsOnly
            )

            if not folder:
                return

        # BatchProcessDialog processes and closes itself in __init__
        BatchProcessDialog(self, folder, self.enhancer, self.output_dir)

    def _process_file_list(self, files: list) -> None:
        """Process a list of selected image files"""
        from ..utils.file_io import write_image, read_image
        from datetime import datetime

        total = len(files)
        progress = QProgressDialog(tr("批量处理中..."), tr("取消"), 0, total, self)
        progress.setWindowTitle(tr("批量处理 (") + str(total) + tr(" 个文件)"))
        progress.setMinimumDuration(0)

        batch_dir = self.output_dir / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for i, path_str in enumerate(files):
            if progress.wasCanceled():
                break

            path = Path(path_str)
            progress.setLabelText(tr("处理: ") + path.name)
            progress.setValue(i)

            try:
                image = read_image(path)
                enhanced = self.enhancer.enhance(image)
                output_path = batch_dir / f"{path.stem}_enhanced{path.suffix}"
                write_image(enhanced, output_path)
                success_count += 1
            except Exception as e:
                logger.error(f"处理失败 {path}: {e}")

        progress.setValue(total)
        progress.close()
        QMessageBox.information(self, tr("完成"),
            tr("批量处理完成\n成功: ") + f"{success_count}/{total}" + tr("\n保存至: ") + str(batch_dir))
    
    def _on_video_processing(self, video_path: str = None) -> None:
        """视频处理"""
        if video_path is None or isinstance(video_path, bool):
            video_path, _ = QFileDialog.getOpenFileName(
                self,
                tr("选择视频文件"),
                "",
                tr("视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)")
            )
            
            if not video_path:
                return
        
        # 显示视频处理对话框
        dialog = VideoProcessDialog(self, video_path, self.enhancer, self.output_dir)
        dialog.exec_()
    
    def _on_settings(self) -> None:
        """打开设置对话框"""
        dialog = SettingsDialog(self, self.config)
        dialog.settings_applied.connect(self._apply_config)
        if dialog.exec_():
            self._apply_config()
    
    def _on_about(self) -> None:
        """显示关于对话框"""
        QMessageBox.about(
            self,
            tr("关于"),
            tr("<h3>水下图像增强软件 v1.0.0</h3>"
            "<p>基于FUnIE-GAN的深度学习图像增强工具</p>"
            "<p>提供多种增强模式和滤镜调节功能</p>"
            "<hr>"
            "<p>© 2024 Underwater Image Enhancement Team</p>")
        )
    
    def _on_show_metrics(self) -> None:
        """显示质量指标"""
        if self.original_image is None or self.enhanced_image is None:
            return
        
        metrics = self.metrics.evaluate(self.original_image, self.enhanced_image)
        
        QMessageBox.information(
            self,
            tr("质量指标"),
            metrics.summary()
        )
    
    # ==================== 覆盖方法 ====================
    
    def closeEvent(self, event) -> None:
        """窗口关闭事件"""
        self._save_config()
        self._on_save_session()
        event.accept()
    
    def dragEnterEvent(self, event) -> None:
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event) -> None:
        """拖拽放下事件"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_file():
                self.load_image(path)



