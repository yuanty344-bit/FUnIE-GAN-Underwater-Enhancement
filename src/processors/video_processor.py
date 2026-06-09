"""
视频处理器模块

提供视频文件的逐帧增强处理功能，
使用OpenCV进行视频读取和写入。
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Tuple, Union, Dict
from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..utils.file_io import ensure_directories_exist

logger = logging.getLogger(__name__)


class VideoCodec(Enum):
    """视频编码器"""
    AVC1 = 'avc1'
    MP4V = 'mp4v'
    XVID = 'xvid'
    DIVX = 'divx'
    H264 = 'avc1'
    MJPG = 'MJPG'
    WMV1 = 'WMV1'
    
    @classmethod
    def from_string(cls, codec_str: str) -> 'VideoCodec':
        """从字符串创建编码器"""
        codec_map = {
            'avc1': cls.AVC1,
            'mp4v': cls.MP4V,
            'xvid': cls.XVID,
            'divx': cls.DIVX,
            'h264': cls.H264,
            'H264': cls.H264,
            'mjpg': cls.MJPG,
            'MJPG': cls.MJPG,
            'wmv1': cls.WMV1,
            'WMV1': cls.WMV1
        }
        return codec_map.get(codec_str.lower(), cls.MP4V)


@dataclass
class VideoInfo:
    """视频信息"""
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str


@dataclass
class FrameResult:
    """帧处理结果"""
    frame_index: int
    success: bool
    error: Optional[str]
    processing_time: float


class VideoProcessor:
    """
    视频处理器
    
    提供视频文件的逐帧增强功能，
    支持进度跟踪和参数配置。
    
    Attributes:
        input_path: 输入视频路径
        output_path: 输出视频路径
        enhancer: 图像增强器
        frame_interval: 帧处理间隔（每n帧处理1帧）
        
    Example:
        >>> processor = VideoProcessor('input.mp4', enhancer, 'output.mp4')
        >>> processor.process(progress_callback=my_callback)
    """
    
    def __init__(
        self,
        input_path: Union[str, Path],
        enhancer,  # ImageEnhancer
        output_path: Union[str, Path],
        frame_interval: int = 1,
        codec: str = 'mp4v',
        target_fps: Optional[float] = None
    ):
        """
        初始化视频处理器
        
        Args:
            input_path: 输入视频路径
            enhancer: 图像增强器实例
            output_path: 输出视频路径
            frame_interval: 帧处理间隔（每n帧处理1帧）
            codec: 视频编码器
            target_fps: 目标帧率，None则保持原帧率
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.enhancer = enhancer
        self.frame_interval = max(1, frame_interval)
        self.codec = VideoCodec.from_string(codec)
        self.target_fps = target_fps
        
        # 确保输出目录存在
        ensure_directories_exist([self.output_path.parent])
        
        # 运行时状态
        self._is_running = False
        self._is_paused = False
        self._is_cancelled = False
        
        # 视频信息
        self._video_info: Optional[VideoInfo] = None
        
        # OpenCV对象
        self._capture = None
        self._writer = None
        
        # 统计
        self._processed_frames = 0
        self._failed_frames = 0
        self._total_frames = 0
        
        # 检查OpenCV可用性
        self._cv2_available = False
        try:
            import cv2
            self._cv2 = cv2
            self._cv2_available = True
        except ImportError:
            logger.warning("OpenCV未安装，视频处理功能不可用")
    
    @property
    def is_available(self) -> bool:
        """检查视频处理是否可用"""
        return self._cv2_available
    
    def get_video_info(self) -> Optional[VideoInfo]:
        """
        获取视频信息
        
        Returns:
            VideoInfo对象，失败返回None
        """
        if not self._cv2_available:
            return None
        
        try:
            capture = self._cv2.VideoCapture(str(self.input_path))
            
            if not capture.isOpened():
                logger.error(f"无法打开视频: {self.input_path}")
                return None
            
            width = int(capture.get(self._cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(self._cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(self._cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(self._cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            fourcc = capture.get(self._cv2.CAP_PROP_FOURCC)
            codec_str = self._fourcc_to_str(int(fourcc))
            
            capture.release()
            
            self._video_info = VideoInfo(
                path=self.input_path,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
                duration=duration,
                codec=codec_str
            )
            
            return self._video_info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return None
    
    def _fourcc_to_str(self, fourcc: int) -> str:
        """将FourCC码转换为字符串"""
        try:
            return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        except:
            return "Unknown"
    
    def _create_video_writer(self, fps: float, width: int, height: int):
        """创建视频写入器"""
        # Try primary codec first
        fourcc = self._cv2.VideoWriter_fourcc(*self.codec.value)
        writer = self._cv2.VideoWriter(
            str(self.output_path), fourcc, fps, (width, height)
        )

        if not writer.isOpened():
            # Try fallback codecs
            fallback_codecs = ['avc1', 'mp4v', 'H264', 'X264', 'MJPG', 'XVID', 'DIVX']
            for codec in fallback_codecs:
                if codec.upper() == self.codec.value.upper():
                    continue
                try:
                    fourcc = self._cv2.VideoWriter_fourcc(*codec)
                    writer = self._cv2.VideoWriter(
                        str(self.output_path), fourcc, fps, (width, height)
                    )
                    if writer.isOpened():
                        logger.info(f"Using fallback codec: {codec}")
                        break
                except Exception:
                    continue

        if not writer.isOpened():
            # Last resort: try AVI container with MJPG
            avi_path = str(self.output_path.with_suffix('.avi'))
            logger.info(f"MP4 failed, trying AVI: {avi_path}")
            try:
                fourcc = self._cv2.VideoWriter_fourcc(*'MJPG')
                writer = self._cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
                if writer.isOpened():
                    self.output_path = Path(avi_path)
            except Exception:
                pass

        return writer
    
    def process(
        self,
        progress_callback: Optional[Callable] = None,
        skip_frames: int = 0
    ) -> bool:
        """
        处理视频
        
        Args:
            progress_callback: 进度回调，签名为 callback(current, total, message)
            skip_frames: 开始时跳过的帧数
            
        Returns:
            处理是否成功
        """
        if not self._cv2_available:
            logger.error("OpenCV不可用，无法处理视频")
            return False
        
        # 获取视频信息
        video_info = self.get_video_info()
        if video_info is None:
            return False
        
        # 打开视频
        self._capture = self._cv2.VideoCapture(str(self.input_path))
        
        if not self._capture.isOpened():
            logger.error(f"无法打开视频: {self.input_path}")
            return False
        
        # 跳过开头的帧
        if skip_frames > 0:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, skip_frames)
        
        # 创建写入器
        output_fps = self.target_fps or video_info.fps
        self._writer = self._create_video_writer(
            output_fps,
            video_info.width,
            video_info.height
        )
        
        if not self._writer.isOpened():
            logger.error("无法创建视频写入器")
            self._capture.release()
            return False
        
        # 初始化状态
        self._is_running = True
        self._is_paused = False
        self._is_cancelled = False
        self._processed_frames = 0
        self._failed_frames = 0
        self._total_frames = video_info.frame_count - skip_frames
        
        start_time = time.time()
        
        frame_index = skip_frames
        processed_index = 0
        
        logger.info(f"开始处理视频: {video_info.frame_count} 帧")
        
        while self._is_running:
            # 检查取消
            if self._is_cancelled:
                break
            
            # 读取帧
            ret, frame = self._capture.read()
            
            if not ret:
                # 视频结束
                break
            
            frame_index += 1
            
            # 是否处理当前帧
            should_process = (frame_index - skip_frames) % self.frame_interval == 0
            
            if should_process:
                # 转换为RGB
                frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
                
                try:
                    # 增强
                    enhanced = self.enhancer.enhance(frame_rgb)
                    
                    # 转换回BGR
                    enhanced_bgr = self._cv2.cvtColor(enhanced, self._cv2.COLOR_RGB2BGR)
                    
                    # 写入
                    self._writer.write(enhanced_bgr)
                    self._processed_frames += 1
                    
                except Exception as e:
                    logger.error(f"帧 {frame_index} 处理失败: {e}")
                    self._failed_frames += 1
                    # 写入原帧
                    self._writer.write(frame)
            else:
                # 写入原帧
                self._writer.write(frame)
            
            processed_index += 1
            
            # 报告进度
            if progress_callback:
                message = f"处理帧: {frame_index}/{video_info.frame_count}"
                progress_callback(frame_index, video_info.frame_count, message)
        
        # 清理
        self._capture.release()
        self._writer.release()
        
        self._is_running = False
        
        elapsed = time.time() - start_time
        logger.info(f"视频处理完成: {self._processed_frames} 帧成功, "
                   f"{self._failed_frames} 帧失败, 耗时 {elapsed:.2f}秒")
        
        return self._processed_frames > 0
    
    def process_with_preview(
        self,
        frame_callback: Callable[[np.ndarray, np.ndarray, int], None],
        progress_callback: Optional[Callable] = None
    ) -> bool:
        """
        处理视频并提供预览回调
        
        Args:
            frame_callback: 帧回调，签名为 callback(original, enhanced, frame_index)
            progress_callback: 进度回调
            
        Returns:
            处理是否成功
        """
        if not self._cv2_available:
            return False
        
        video_info = self.get_video_info()
        if video_info is None:
            return False
        
        self._capture = self._cv2.VideoCapture(str(self.input_path))
        
        if not self._capture.isOpened():
            return False
        
        self._is_running = True
        self._is_cancelled = False
        self._processed_frames = 0
        
        frame_index = 0
        
        while self._is_running:
            if self._is_cancelled:
                break
            
            ret, frame = self._capture.read()
            
            if not ret:
                break
            
            frame_index += 1
            
            # 转换颜色空间
            frame_rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            
            try:
                # 增强
                enhanced = self.enhancer.enhance(frame_rgb)
                
                # 回调
                frame_callback(frame_rgb, enhanced, frame_index)
                
                self._processed_frames += 1
                
            except Exception as e:
                logger.error(f"帧 {frame_index} 处理失败: {e}")
            
            if progress_callback:
                progress_callback(frame_index, video_info.frame_count,
                                f"预览帧: {frame_index}/{video_info.frame_count}")
        
        self._capture.release()
        self._is_running = False
        
        return True
    
    def pause(self) -> None:
        """暂停处理"""
        self._is_paused = True
        logger.info("视频处理已暂停")
    
    def resume(self) -> None:
        """继续处理"""
        self._is_paused = False
        logger.info("视频处理已继续")
    
    def stop(self) -> None:
        """
        停止处理
        
        用于回调中检查是否需要停止
        """
        self._is_running = False
        self._is_cancelled = True
        logger.info("视频处理已停止")
    
    def cancel(self) -> None:
        """取消处理"""
        self._is_running = False
        self._is_cancelled = True
        
        # 清理资源
        if self._capture:
            self._capture.release()
        if self._writer:
            self._writer.release()
        
        # 删除不完整的输出文件
        if self.output_path.exists():
            try:
                self.output_path.unlink()
            except:
                pass
        
        logger.info("视频处理已取消")
    
    @property
    def progress(self) -> float:
        """获取处理进度 (0.0 - 1.0)"""
        if self._total_frames == 0:
            return 0.0
        return self._processed_frames / self._total_frames
    
    @property
    def statistics(self) -> Dict:
        """获取处理统计"""
        return {
            'processed_frames': self._processed_frames,
            'failed_frames': self._failed_frames,
            'total_frames': self._total_frames,
            'progress': self.progress,
            'is_running': self._is_running,
            'is_cancelled': self._is_cancelled
        }
    
    def __del__(self):
        """析构时清理资源"""
        if self._capture:
            self._capture.release()
        if self._writer:
            self._writer.release()
