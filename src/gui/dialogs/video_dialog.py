"""
视频处理对话框

处理前可预览视频片段，选择起止时间范围处理。
"""
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QProgressDialog, QMessageBox, QPushButton, QLabel,
    QDoubleSpinBox, QCheckBox, QSlider, QGroupBox
)
from PyQt5.QtCore import Qt

from ...i18n import tr
from ...processors.frame_extractor import get_video_info, extract_frame
import cv2
import numpy as np


class VideoProcessDialog(QDialog):
    """视频处理对话框（带预览）"""

    def __init__(
        self,
        parent,
        video_path: str,
        enhancer,
        output_dir: Path
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.enhancer = enhancer
        self.output_dir = output_dir
        self._info = get_video_info(video_path)

        self.setWindowTitle(tr("视频处理"))
        self.setMinimumWidth(480)
        self._init_ui()
        self._load_preview_frame()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # 视频信息
        if self._info:
            info_text = (
                f"{tr('分辨率')}: {self._info['width']}x{self._info['height']} | "
                f"{tr('帧率')}: {self._info['fps']:.1f} fps | "
                f"{tr('总帧数')}: {self._info['frame_count']} | "
                f"{tr('时长')}: {self._info['duration']:.1f}s"
            )
        else:
            info_text = tr("无法读取视频信息")
        self.label_info = QLabel(info_text)
        layout.addWidget(self.label_info)

        # 时间范围
        group_range = QGroupBox(tr("处理范围"))
        range_layout = QFormLayout()

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0, 999999)
        self.spin_start.setDecimals(1)
        self.spin_start.setValue(0)
        self.spin_start.setSuffix(" s")
        range_layout.addRow(tr("起始时间:"), self.spin_start)

        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0, 999999)
        self.spin_end.setDecimals(1)
        if self._info:
            self.spin_end.setValue(self._info['duration'])
        self.spin_end.setSuffix(" s")
        range_layout.addRow(tr("结束时间:"), self.spin_end)

        group_range.setLayout(range_layout)
        layout.addWidget(group_range)

        # 预览按钮
        btn_layout = QHBoxLayout()
        self.btn_preview_start = QPushButton(tr("预览起始帧"))
        self.btn_preview_start.clicked.connect(lambda: self._preview_at(self.spin_start.value()))
        btn_layout.addWidget(self.btn_preview_start)

        self.btn_preview_end = QPushButton(tr("预览结束帧"))
        self.btn_preview_end.clicked.connect(lambda: self._preview_at(self.spin_end.value()))
        btn_layout.addWidget(self.btn_preview_end)
        layout.addLayout(btn_layout)

        self.label_preview = QLabel()
        self.label_preview.setFixedHeight(200)
        self.label_preview.setAlignment(Qt.AlignCenter)
        self.label_preview.setStyleSheet("background-color: #222; border: 1px solid #555;")
        self.label_preview.setText(tr("点击预览按钮查看帧"))
        layout.addWidget(self.label_preview)

        # 处理按钮
        self.btn_process = QPushButton(tr("开始处理"))
        self.btn_process.clicked.connect(self._on_process)
        self.btn_process.setMinimumHeight(36)
        layout.addWidget(self.btn_process)

    def _load_preview_frame(self):
        frame = extract_frame(self.video_path, 0)
        if frame is not None:
            self._set_preview_pixmap(frame)

    def _preview_at(self, time_sec: float):
        if not self._info:
            return
        frame_idx = int(time_sec * self._info['fps'])
        frame = extract_frame(self.video_path, frame_idx)
        if frame is not None:
            self._set_preview_pixmap(frame)
        else:
            self.label_preview.setText(tr("无法提取该帧"))

    def _set_preview_pixmap(self, frame: np.ndarray):
        from PyQt5.QtGui import QPixmap, QImage
        h, w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(self.label_preview.width(), self.label_preview.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_preview.setPixmap(scaled)

    def _on_process(self):
        from ...processors.video_processor import VideoProcessor

        start_time = self.spin_start.value()
        end_time = self.spin_end.value()

        progress = QProgressDialog(tr("视频处理中..."), tr("取消"), 0, 100, self)
        progress.setWindowTitle(tr("视频处理"))
        progress.setMinimumDuration(0)
        progress.show()

        try:
            processor = VideoProcessor(
                self.video_path,
                self.enhancer,
                str(self.output_dir / f"{Path(self.video_path).stem}_enhanced.mp4"),
                codec='avc1',
                start_sec=start_time,
                end_sec=end_time,
            )

            success = processor.process(
                progress_callback=lambda current, total, msg: (
                    progress.setValue(int(current / total * 100)),
                    progress.setLabelText(msg),
                    None
                )
            )

            progress.setValue(100)
            progress.close()

            if success:
                QMessageBox.information(self, tr("完成"),
                    tr("视频处理完成\n输出: ") + str(processor.output_path))
            else:
                QMessageBox.critical(self, tr("错误"),
                    tr("视频处理失败：无法打开视频文件或创建输出文件。"))

        except Exception as e:
            progress.close()
            QMessageBox.critical(self, tr("错误"), tr("视频处理失败:\n") + str(e))
