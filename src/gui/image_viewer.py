"""
图像显示组件模块

提供图像查看和缩放功能，
支持鼠标滚轮缩放、拖拽平移等交互。
"""
from PyQt5.QtGui import QPalette

from ..i18n import tr

import math
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea,
    QSizePolicy, QLabel
)
from PyQt5.QtCore import (
    Qt, QPoint, QSize, pyqtSignal, pyqtProperty
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QWheelEvent,
    QMouseEvent, QResizeEvent, QCursor
)


class ImageViewer(QWidget):
    """
    图像显示查看器组件
    
    支持：
    - 图像显示（支持多种格式）
    - 鼠标滚轮缩放
    - 拖拽平移
    - 适应窗口
    - 实际大小显示
    
    Signals:
        zoom_changed: 缩放比例改变信号
        image_double_clicked: 图像双击信号
        position_clicked: 点击图像位置信号
    """
    
    # 信号定义
    zoom_changed = pyqtSignal(float)  # 缩放比例
    image_double_clicked = pyqtSignal(int, int)  # x, y 坐标
    position_clicked = pyqtSignal(int, int)  # x, y 坐标
    file_dropped = pyqtSignal(str)  # 拖放文件路径
    # 缩放限制
    MIN_ZOOM = 0.01  # 最小缩放 (1%)
    MAX_ZOOM = 10.0  # 最大缩放 (1000%)
    ZOOM_STEP = 0.1  # 每次缩放变化量
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        初始化图像查看器
        
        Args:
            parent: 父窗口
        """
        super().__init__(parent)
        
        # 图像数据
        self._image: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._image_size: QSize = QSize(0, 0)
        
        # 缩放和平移状态
        self._zoom: float = 1.0
        self._offset: QPoint = QPoint(0, 0)
        self._is_dragging: bool = False
        self._drag_start: QPoint = QPoint(0, 0)
        
        # 平滑缩放
        self._smooth_zoom_enabled: bool = True
        self._magnifier = None
        
        # 初始化UI
        self._init_ui()
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
    
    def _init_ui(self) -> None:
        """初始化UI"""
        # 设置背景色
        self.setBackgroundRole(QPalette.Dark)
        self.setAutoFillBackground(True)
        
        # 设置大小策略
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 300)
        
        # 鼠标样式
        self.setCursor(QCursor(Qt.OpenHandCursor))
    
    # ==================== 属性 ====================

    @property
    def current_image(self):
        """获取当前显示的图像（numpy数组）"""
        return self._image

    @pyqtProperty(float)
    def zoom(self) -> float:
        """获取当前缩放比例"""
        return self._zoom
    
    @zoom.setter
    def zoom(self, value: float) -> None:
        """设置缩放比例"""
        self.set_zoom(value)
    
    # ==================== 公共方法 ====================
    
    def set_image(self, image: np.ndarray) -> None:
        """
        设置显示的图像
        
        Args:
            image: numpy数组，shape为(H, W, 3)或(H, W)，dtype为uint8
        """
        if image is None or image.size == 0:
            self._image = None
            self._pixmap = None
            self._image_size = QSize(0, 0)
            self.update()
            return
        
        # 保存原始图像
        self._image = image.copy()
        
        # 转换为QPixmap
        self._pixmap = self._numpy_to_pixmap(image)
        
        if self._pixmap:
            self._image_size = self._pixmap.size()
        else:
            self._image_size = QSize(0, 0)
        
        # 适应窗口显示
        self.fit_to_window()
    
    def _numpy_to_pixmap(self, image: np.ndarray) -> Optional[QPixmap]:
        """
        将numpy数组转换为QPixmap
        
        Args:
            image: 图像数组
            
        Returns:
            QPixmap对象
        """
        h, w = image.shape[:2]
        
        # 确定格式
        if len(image.shape) == 2:
            # 灰度图
            fmt = QImage.Format_Grayscale8
        elif image.shape[2] == 3:
            # RGB图，需要转换为QImage格式
            # QImage使用BGR顺序
            try:
                import cv2
                rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image = np.ascontiguousarray(rgb)
                fmt = QImage.Format_RGB888
            except ImportError:
                # 无OpenCV时手动转换
                rgb = image.copy()
                rgb[:, :, 0], rgb[:, :, 2] = image[:, :, 2].copy(), image[:, :, 0].copy()
                image = np.ascontiguousarray(rgb)
                fmt = QImage.Format_RGB888
        elif image.shape[2] == 4:
            # RGBA图
            fmt = QImage.Format_RGBA8888
        else:
            return None
        
        # 创建QImage
        bytes_per_line = image.strides[0]
        qimage = QImage(
            image.data,
            w, h,
            bytes_per_line,
            fmt
        )
        
        # 转换为QPixmap
        return QPixmap.fromImage(qimage)
    
    def set_zoom(self, zoom: float) -> None:
        """
        设置缩放比例
        
        Args:
            zoom: 缩放比例
        """
        # 限制缩放范围
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        
        self._zoom = new_zoom
        self.update()
        self.zoom_changed.emit(self._zoom)
    
    def zoom_in(self, factor: float = None) -> None:
        """
        放大
        
        Args:
            factor: 放大倍数，None则使用默认步长
        """
        if factor is None:
            factor = 1.0 + self.ZOOM_STEP
        
        self.set_zoom(self._zoom * factor)
    
    def zoom_out(self, factor: float = None) -> None:
        """
        缩小

        Args:
            factor: 缩小倍数，None则使用默认步长
        """
        if factor is None:
            factor = 1.0 - self.ZOOM_STEP

        self.set_zoom(self._zoom * factor)
    
    def zoom_at_point(self, point: QPoint, factor: float = 1.1) -> None:
        """
        在指定点进行缩放
        
        Args:
            point: 缩放中心点（widget坐标）
            factor: 缩放因子，>1放大，<1缩小
        """
        if self._pixmap is None:
            return
        
        # 计算点在图像上的位置
        image_pos = self._widget_to_image(point)
        
        # 计算新的偏移量以保持缩放中心
        new_zoom = self._zoom * factor
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        
        # 新偏移量
        new_offset = point - image_pos * new_zoom
        
        # 更新状态
        self._zoom = new_zoom
        self._offset = new_offset
        
        self.update()
        self.zoom_changed.emit(self._zoom)
    
    def fit_to_window(self) -> None:
        """适应窗口显示"""
        if self._pixmap is None:
            return
        
        # 获取可用空间
        available = self.size()
        
        # 计算适应窗口的缩放比例
        scale_w = available.width() / self._image_size.width()
        scale_h = available.height() / self._image_size.height()
        scale = min(scale_w, scale_h)
        
        # 居中偏移
        self._zoom = scale
        self._offset = QPoint(
            (available.width() - self._image_size.width() * scale) / 2,
            (available.height() - self._image_size.height() * scale) / 2
        )
        
        self.update()
        self.zoom_changed.emit(self._zoom)
    
    def reset_view(self) -> None:
        """重置视图到初始状态"""
        self.set_zoom(1.0)
        self._offset = QPoint(0, 0)
        self.update()
        self.zoom_changed.emit(self._zoom)
    
    def center_on_image(self) -> None:
        """将视图居中到图像中心"""
        if self._pixmap is None:
            return
        
        available = self.size()
        center_offset = QPoint(
            (available.width() - self._pixmap.width() * self._zoom) / 2,
            (available.height() - self._pixmap.height() * self._zoom) / 2
        )
        
        self._offset = center_offset
        self.update()

    def set_magnifier(self, magnifier) -> None:
        """设置放大镜组件"""
        from .widgets.magnifier import Magnifier
        self._magnifier = magnifier

    def _show_context_menu(self, pos) -> None:
        from PyQt5.QtWidgets import QMenu, QAction, QApplication
        menu = QMenu(self)

        action_copy = menu.addAction(tr("复制图像"))
        action_reset_zoom = menu.addAction(tr("重置缩放"))
        action_fit = menu.addAction(tr("适应窗口"))
        menu.addSeparator()
        action_copy_data = menu.addAction(tr("复制像素信息"))

        action = menu.exec_(self.mapToGlobal(pos))

        if action == action_copy:
            if self._pixmap:
                QApplication.clipboard().setPixmap(self._pixmap)
        elif action == action_reset_zoom:
            self.set_zoom(1.0)
            self.center_on_image()
        elif action == action_fit:
            self.fit_to_window()
        elif action == action_copy_data:
            img_info = self.get_image_at_point(pos)
            if img_info:
                x, y, r, g, b = img_info
                text = f"({x}, {y}) R:{r} G:{g} B:{b}"
                QApplication.clipboard().setText(text)

    def _update_magnifier(self, widget_pos) -> None:
        if not hasattr(self, '_magnifier') or not self._magnifier:
            return
        if self._pixmap:
            self._magnifier.set_source(self._pixmap, self._offset,
                                       self._zoom, (self._image_size.width(), self._image_size.height()))
            self._magnifier.show_at(self.mapToGlobal(widget_pos), widget_pos)

    def leaveEvent(self, event) -> None:
        if hasattr(self, '_magnifier') and self._magnifier:
            self._magnifier.hide()
        super().leaveEvent(event)

    def retranslate_ui(self) -> None:
        """重新翻译UI字符串（占位文本在 paintEvent 中通过 tr() 动态获取）"""
        self.update()

    def get_image_at_point(self, point: QPoint) -> Optional[tuple]:
        """
        获取图像上指定点的坐标和像素值
        
        Args:
            point: widget坐标
            
        Returns:
            (x, y, r, g, b) 或 None
        """
        if self._image is None:
            return None
        
        img_point = self._widget_to_image(point)
        
        if 0 <= img_point.x() < self._image.shape[1] and 0 <= img_point.y() < self._image.shape[0]:
            x, y = img_point.x(), img_point.y()
            
            if len(self._image.shape) == 3:
                r, g, b = self._image[y, x, :3]
            else:
                r = g = b = self._image[y, x]
            
            return (x, y, int(r), int(g), int(b))
        
        return None
    
    def _widget_to_image(self, point: QPoint) -> QPoint:
        """
        将widget坐标转换为图像坐标
        
        Args:
            point: widget坐标
            
        Returns:
            图像坐标
        """
        if self._zoom == 0:
            return QPoint(0, 0)
        
        return QPoint(
            int((point.x() - self._offset.x()) / self._zoom),
            int((point.y() - self._offset.y()) / self._zoom)
        )
    
    # ==================== 绘制 ====================
    
    def paintEvent(self, event) -> None:
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # 填充背景
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        
        if self._pixmap is None:
            # 显示占位符
            self._draw_placeholder(painter)
            return
        
        # 计算绘制区域
        scaled_size = QSize(
            int(self._image_size.width() * self._zoom),
            int(self._image_size.height() * self._zoom)
        )
        
        dest_rect = self._offset
        source_rect = self._image_size
        
        # 绘制图像
        scaled_pixmap = self._pixmap.scaled(
            scaled_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # 计算实际偏移（用于居中）
        actual_offset = QPoint(
            self._offset.x() + (scaled_size.width() - scaled_pixmap.width()) / 2,
            self._offset.y() + (scaled_size.height() - scaled_pixmap.height()) / 2
        )
        
        painter.drawPixmap(actual_offset, scaled_pixmap)
        
        # 绘制网格线（仅在高倍放大时）
        if self._zoom >= 2.0:
            self._draw_grid(painter, actual_offset, scaled_pixmap.size())
    
    def _draw_placeholder(self, painter: QPainter) -> None:
        """绘制占位符"""
        painter.save()
        
        # 设置字体
        from PyQt5.QtGui import QFont
        font = QFont()
        font.setPixelSize(16)
        painter.setFont(font)
        
        # 绘制文字
        text = tr("拖放图像到此处")
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        
        center = self.rect().center()
        text_pos = QPoint(
            center.x() - text_rect.width() / 2,
            center.y() - text_rect.height() / 2
        )
        
        painter.drawText(text_pos, text)
        
        painter.restore()
    
    def _draw_grid(
        self,
        painter: QPainter,
        offset: QPoint,
        size: QSize
    ) -> None:
        """绘制网格线"""
        painter.save()
        painter.setPen(Qt.gray)
        painter.setOpacity(0.3)
        
        grid_size = 100 * self._zoom
        
        # 垂直线
        x = offset.x() % grid_size
        while x < size.width():
            painter.drawLine(x + offset.x(), 0, x + offset.x(), self.height())
            x += grid_size
        
        # 水平线
        y = offset.y() % grid_size
        while y < size.height():
            painter.drawLine(0, y + offset.y(), self.width(), y + offset.y())
            y += grid_size
        
        painter.restore()
    
    # ==================== 鼠标事件 ====================
    
    def wheelEvent(self, event: QWheelEvent) -> None:
        """鼠标滚轮事件（缩放）"""
        if self._pixmap is None:
            return
        
        # 获取滚轮角度
        angle = event.angleDelta().y()
        
        if angle > 0:
            # 放大
            self.zoom_at_point(event.pos(), 1.1)
        elif angle < 0:
            # 缩小
            self.zoom_at_point(event.pos(), 0.9)
        
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._drag_start = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            
            # 发射点击信号
            img_info = self.get_image_at_point(event.pos())
            if img_info:
                self.position_clicked.emit(img_info[0], img_info[1])
        
        elif event.button() == Qt.RightButton:
            self._show_context_menu(event.pos())
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """鼠标移动事件"""
        if self._is_dragging:
            delta = event.pos() - self._drag_start
            self._offset += delta
            self._drag_start = event.pos()
            self.update()
        else:
            if self._pixmap:
                self.setCursor(QCursor(Qt.OpenHandCursor))
                if hasattr(self, '_magnifier') and self._magnifier:
                    self._update_magnifier(event.pos())
    
    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
            self.setCursor(QCursor(Qt.OpenHandCursor))
    
    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """鼠标双击事件"""
        if event.button() == Qt.LeftButton and self._pixmap:
            if abs(self._zoom - 1.0) < 0.01:
                self.fit_to_window()
            else:
                self.set_zoom(1.0)
                self.center_on_image()

    def dragEnterEvent(self, event) -> None:
        """拖拽进入事件"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile()
                from pathlib import Path
                if Path(path).suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        """拖拽放下事件"""
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """窗口大小改变事件"""
        super().resizeEvent(event)
        
        # 保持图像在可视区域内
        if self._pixmap and self._zoom > 0:
            self._constrain_offset()
    
    def _constrain_offset(self) -> None:
        """限制偏移量，确保图像不会完全移出视图"""
        if self._pixmap is None:
            return
        
        scaled_w = self._image_size.width() * self._zoom
        scaled_h = self._image_size.height() * self._zoom
        
        # 水平限制
        if scaled_w <= self.width():
            self._offset.setX((self.width() - scaled_w) / 2)
        else:
            self._offset.setX(min(0, max(self.width() - scaled_w, self._offset.x())))
        
        # 垂直限制
        if scaled_h <= self.height():
            self._offset.setY((self.height() - scaled_h) / 2)
        else:
            self._offset.setY(min(0, max(self.height() - scaled_h, self._offset.y())))
