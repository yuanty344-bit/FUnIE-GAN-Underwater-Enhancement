"""
滑动对比面板模块

提供增强前后的滑动对比功能，
通过拖动滑块查看图像变化。
"""

from typing import Optional, Tuple

import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent

from PyQt5.QtGui import QPalette

from ..i18n import tr

class ComparePanel(QWidget):
    """
    滑动对比面板
    
    在同一视图中显示增强前后的图像，
    通过可拖动的垂直滑块进行对比。
    
    Signals:
        position_changed: 滑块位置改变信号 (0.0 - 1.0)
    """
    
    position_changed = pyqtSignal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化对比面板
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 图像数据
        self._original: Optional[np.ndarray] = None
        self._enhanced: Optional[np.ndarray] = None
        
        # 显示状态
        self._split_position: float = 0.5  # 滑块位置 (0.0 - 1.0)
        self._is_dragging: bool = False
        
        # 缩放
        self._zoom: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        
        # 初始化UI
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化UI"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建对比视图
        self.compare_view = CompareView()
        layout.addWidget(self.compare_view)
        
        # 创建滑块
        slider_layout = QHBoxLayout()
        self.label_original = QLabel(tr("原图"))
        slider_layout.addWidget(self.label_original)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider)

        self.label_enhanced = QLabel(tr("增强"))
        slider_layout.addWidget(self.label_enhanced)

        layout.addLayout(slider_layout)
        
        # 连接信号
        self.compare_view.position_changed.connect(
            lambda pos: self.slider.setValue(int(pos * 100))
        )
    
    def set_images(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> None:
        """
        设置对比图像
        
        Args:
            original: 原始图像
            enhanced: 增强后的图像
        """
        self._original = original
        self._enhanced = enhanced
        
        # 转换图像
        self.compare_view.set_images(original, enhanced)
        
        # 适应窗口
        self.compare_view.fit_to_window()
    
    def set_split_position(self, position: float) -> None:
        """
        设置分割位置
        
        Args:
            position: 位置比例 (0.0 - 1.0)
        """
        self._split_position = max(0.0, min(1.0, position))
        self.slider.setValue(int(self._split_position * 100))
        self.compare_view.set_split_position(self._split_position)
        self.position_changed.emit(self._split_position)
    
    def _on_slider_changed(self, value: int) -> None:
        """滑块值改变"""
        self._split_position = value / 100.0
        self.compare_view.set_split_position(self._split_position)
        self.position_changed.emit(self._split_position)
    
    def fit_to_window(self) -> None:
        """适应窗口"""
        self.compare_view.fit_to_window()

    def set_zoom(self, zoom: float) -> None:
        """设置缩放"""
        self._zoom = zoom
        self.compare_view.set_zoom(zoom)

    def retranslate_ui(self) -> None:
        """重新翻译UI字符串"""
        self.label_original.setText(tr("原图"))
        self.label_enhanced.setText(tr("增强"))
        self.compare_view.update()


class CompareView(QWidget):
    """
    对比视图组件
    
    显示两个图像的对比效果，
    左侧显示原图，右侧显示增强图。
    """
    
    position_changed = pyqtSignal(float)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # 图像数据
        self._original_pixmap: Optional[QPixmap] = None
        self._enhanced_pixmap: Optional[QPixmap] = None
        
        # 显示状态
        self._split_position: float = 0.5
        self._is_dragging: bool = False
        
        # 缩放和平移
        self._zoom: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        
        # 鼠标位置
        self._mouse_x: int = 0
        
        # 设置
        self.setMouseTracking(True)
        self.setBackgroundRole(QPalette.Dark)
        self.setAutoFillBackground(True)
    
    def set_images(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> None:
        """
        设置对比图像
        
        Args:
            original: 原始图像
            enhanced: 增强后的图像
        """
        self._original_pixmap = self._numpy_to_pixmap(original)
        self._enhanced_pixmap = self._numpy_to_pixmap(enhanced)
        self.update()
    
    def _numpy_to_pixmap(self, image: np.ndarray) -> Optional[QPixmap]:
        """将numpy数组转换为QPixmap"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # 转换为QImage
        if len(image.shape) == 2:
            fmt = QImage.Format_Grayscale8
        elif image.shape[2] == 3:
            try:
                import cv2
                rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = np.ascontiguousarray(rgb)
            except ImportError:
                rgb = image.copy()
                rgb[:, :, 0], rgb[:, :, 2] = image[:, :, 2].copy(), image[:, :, 0].copy()
                image = np.ascontiguousarray(rgb)
            fmt = QImage.Format_RGB888
        elif image.shape[2] == 4:
            fmt = QImage.Format_RGBA8888
        else:
            return None
        
        bytes_per_line = image.strides[0]
        qimage = QImage(
            image.data,
            w, h,
            bytes_per_line,
            fmt
        )
        
        return QPixmap.fromImage(qimage)
    
    def set_split_position(self, position: float) -> None:
        """设置分割位置"""
        self._split_position = max(0.0, min(1.0, position))
        self.update()
    
    def set_zoom(self, zoom: float) -> None:
        """设置缩放"""
        self._zoom = max(0.01, min(10.0, zoom))
        self.update()
    
    def fit_to_window(self) -> None:
        """适应窗口"""
        if self._original_pixmap is None:
            return
        
        available = self.size()
        image_size = self._original_pixmap.size()
        
        scale_w = available.width() / image_size.width()
        scale_h = available.height() / image_size.height()
        self._zoom = min(scale_w, scale_h)
        
        # 居中
        self._offset = QPoint(
            (available.width() - image_size.width() * self._zoom) / 2,
            (available.height() - image_size.height() * self._zoom) / 2
        )
        
        self.update()
    
    def paintEvent(self, event) -> None:
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 填充背景
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        
        if self._original_pixmap is None or self._enhanced_pixmap is None:
            self._draw_placeholder(painter)
            return
        
        # 计算绘制尺寸
        image_size = self._original_pixmap.size()
        scaled_size = image_size * self._zoom
        
        # 计算分割线位置
        split_x = int(scaled_size.width() * self._split_position)
        
        # 缩放图像
        original_scaled = self._original_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        enhanced_scaled = self._enhanced_pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 绘制增强图（右侧）
        enhanced_rect = QRect(
            self._offset.x(),
            self._offset.y(),
            enhanced_scaled.width(),
            enhanced_scaled.height()
        )
        painter.drawPixmap(enhanced_rect, enhanced_scaled)
        
        # 绘制原图（左侧，使用裁剪）
        # 创建裁剪后的图像
        if self._split_position < 1.0:
            original_scaled_copy = original_scaled.copy()
            painter.save()
            
            # 创建裁剪区域
            clip_rect = QRect(
                self._offset.x(),
                self._offset.y(),
                int(original_scaled.width() * self._split_position),
                original_scaled.height()
            )
            
            # 绘制被裁剪的部分（原图）
            if clip_rect.width() > 0:
                painter.setClipRect(clip_rect)
                painter.drawPixmap(enhanced_rect, original_scaled_copy)
            
            painter.restore()
        
        # 绘制分割线
        self._draw_split_line(painter, split_x)
        
        # 绘制标签
        self._draw_labels(painter, split_x)
    
    def _draw_split_line(self, painter: QPainter, x: int) -> None:
        """绘制分割线"""
        painter.save()
        
        # 绘制白色线条
        pen = QPen(QColor(255, 255, 255))
        pen.setWidth(3)
        painter.setPen(pen)
        
        painter.drawLine(
            self._offset.x() + x,
            self._offset.y(),
            self._offset.x() + x,
            self._offset.y() + int(self._original_pixmap.height() * self._zoom)
        )
        
        # 绘制黑色描边
        pen.setColor(QColor(0, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        
        painter.drawLine(
            self._offset.x() + x - 1,
            self._offset.y(),
            self._offset.x() + x - 1,
            self._offset.y() + int(self._original_pixmap.height() * self._zoom)
        )
        painter.drawLine(
            self._offset.x() + x + 1,
            self._offset.y(),
            self._offset.x() + x + 1,
            self._offset.y() + int(self._original_pixmap.height() * self._zoom)
        )
        
        painter.restore()
    
    def _draw_labels(self, painter: QPainter, split_x: int) -> None:
        """绘制标签"""
        painter.save()
        
        from PyQt5.QtGui import QFont
        
        # 设置字体
        font = QFont()
        font.setPixelSize(14)
        font.setBold(True)
        painter.setFont(font)
        
        # 标签背景
        label_width = 60
        label_height = 24
        
        # 原图标签
        original_label_rect = QRect(
            self._offset.x() + 10,
            self._offset.y() + 10,
            label_width,
            label_height
        )
        
        # 增强标签
        enhanced_label_rect = QRect(
            self._offset.x() + int(self._original_pixmap.width() * self._zoom) - label_width - 10,
            self._offset.y() + 10,
            label_width,
            label_height
        )
        
        # 绘制原图标签
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(original_label_rect, 4, 4)
        
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            original_label_rect,
            Qt.AlignCenter,
            tr("原图")
        )

        # 绘制增强标签
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 150))
        painter.drawRoundedRect(enhanced_label_rect, 4, 4)

        painter.setPen(QColor(255, 255, 255))
        painter.drawText(
            enhanced_label_rect,
            Qt.AlignCenter,
            tr("增强")
        )
        
        painter.restore()
    
    def _draw_placeholder(self, painter: QPainter) -> None:
        """绘制占位符"""
        painter.save()
        
        from PyQt5.QtGui import QFont
        font = QFont()
        font.setPixelSize(16)
        painter.setFont(font)
        
        text = tr("请先加载图像")
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        
        center = self.rect().center()
        text_pos = QPoint(
            center.x() - text_rect.width() / 2,
            center.y() - text_rect.height() / 2
        )
        
        painter.drawText(text_pos, text)
        
        painter.restore()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标按下"""
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._update_split_position(event.x())
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """鼠标移动"""
        self._mouse_x = event.x()
        
        if self._is_dragging:
            self._update_split_position(event.x())
        else:
            # 更新鼠标样式
            if self._original_pixmap:
                self.setCursor(Qt.CrossCursor)
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """鼠标释放"""
        self._is_dragging = False
    
    def _update_split_position(self, x: int) -> None:
        """更新分割位置"""
        if self._original_pixmap is None:
            return
        
        # 计算相对于图像的位置
        relative_x = x - self._offset.x()
        total_width = self._original_pixmap.width() * self._zoom
        
        if total_width > 0:
            position = relative_x / total_width
            position = max(0.0, min(1.0, position))
            
            self._split_position = position
            self.position_changed.emit(position)
            self.update()
