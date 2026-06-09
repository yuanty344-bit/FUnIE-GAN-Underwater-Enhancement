"""
图像增强逻辑模块

该模块负责调用FUnIE-GAN模型进行图像增强，
提供高级的图像处理接口和质量评估功能。
"""

import os
import logging
import time
from pathlib import Path
from typing import Optional, Union, List, Tuple, Callable
from enum import Enum
import json

import numpy as np

from .funie_wrapper import FunieWrapper
from .metrics import ImageMetrics

logger = logging.getLogger(__name__)


class EnhancementMode(Enum):
    """增强模式枚举"""
    AUTO = 'auto'  # 自动选择最优模式
    STANDARD = 'standard'  # 标准增强
    LIGHT = 'light'  # 轻度增强
    STRONG = 'strong'  # 强力增强
    CUSTOM = 'custom'  # 自定义参数


class ProcessingStatus(Enum):
    """处理状态枚举"""
    IDLE = 'idle'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    ERROR = 'error'
    CANCELLED = 'cancelled'


class ImageEnhancer:
    """
    图像增强器主类
    
    整合FUnIE-GAN模型和图像处理工具，
    提供统一的图像增强接口。
    
    Attributes:
        model_wrapper: FUnIE-GAN模型封装器实例
        metrics: 图像质量评估器实例
        mode: 当前增强模式
        status: 当前处理状态
        
    Example:
        >>> enhancer = ImageEnhancer('./models/funie_gan_generator.h5')
        >>> enhancer.load_model()
        >>> enhanced = enhancer.enhance(input_image)
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        framework: str = 'tensorflow',
        device: Optional[str] = None,
        auto_load: bool = True
    ):
        """
        初始化图像增强器
        
        Args:
            model_path: 模型文件路径
            framework: 深度学习框架
            device: 计算设备
            auto_load: 是否自动加载模型
        """
        # 初始化模型封装器
        self.model_wrapper = FunieWrapper(
            model_path=model_path,
            framework=framework,
            device=device
        )
        
        # 初始化质量评估器
        self.metrics = ImageMetrics()
        
        # 状态管理
        self.mode = EnhancementMode.AUTO
        self.status = ProcessingStatus.IDLE
        self._last_result = None
        self._last_original = None
        
        # 处理统计
        self._stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
        
        # 进度回调
        self._progress_callback = None
        
        # 自动加载模型
        if auto_load:
            self.load_model()
        
        logger.info(f"ImageEnhancer初始化完成: model_path={model_path}")
    
    def load_model(self) -> bool:
        """
        加载增强模型
        
        Returns:
            加载是否成功
        """
        return self.model_wrapper.load()
    
    def unload_model(self) -> None:
        """
        卸载模型，释放资源
        """
        self.model_wrapper.unload()
    
    @property
    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model_wrapper.is_loaded
    
    def set_mode(self, mode: Union[str, EnhancementMode]) -> None:
        """
        设置增强模式
        
        Args:
            mode: 增强模式 (字符串或枚举)
        """
        if isinstance(mode, str):
            mode = EnhancementMode(mode.lower())
        
        if not isinstance(mode, EnhancementMode):
            raise ValueError(f"无效的增强模式: {mode}")
        
        self.mode = mode
        logger.info(f"增强模式已设置为: {mode.value}")
    
    def set_progress_callback(self, callback: Optional[Callable]) -> None:
        """
        设置进度回调函数
        
        Args:
            callback: 回调函数，签名为 callback(current, total, message)
        """
        self._progress_callback = callback
    
    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        """
        报告处理进度
        
        Args:
            current: 当前进度
            total: 总数
            message: 状态消息
        """
        if self._progress_callback:
            self._progress_callback(current, total, message)
    
    def enhance(
        self,
        image: np.ndarray,
        mode: Optional[Union[str, EnhancementMode]] = None,
        return_metrics: bool = False,
        return_comparison: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict], Tuple[np.ndarray, np.ndarray]]:
        """
        对图像进行增强处理
        
        Args:
            image: 输入图像，numpy数组，shape为(H, W, 3)或(H, W)，范围[0, 255]
            mode: 增强模式，为None时使用当前设置
            return_metrics: 是否返回质量评估指标
            return_comparison: 是否返回(增强图, 原图)元组
            
        Returns:
            增强后的图像
            或 (增强图像, 指标字典) 当 return_metrics=True
            或 (增强图像, 原始图像) 当 return_comparison=True
            或 三者元组当两者都为True
            
        Raises:
            RuntimeError: 模型未加载时抛出
            ValueError: 输入图像格式错误时抛出
        """
        # 验证输入
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
        
        # 更新状态
        self.status = ProcessingStatus.PROCESSING
        start_time = time.time()
        
        try:
            # 确保图像格式正确
            image = self._validate_image(image)
            
            # 记录原始图像
            self._last_original = image.copy()
            
            # 记录进度
            self._report_progress(1, 4, "预处理图像...")
            
            # 应用预处理（根据增强模式）
            if mode:
                self.set_mode(mode)
            
            processed = self._preprocess_by_mode(image)
            
            # 记录进度
            self._report_progress(2, 4, "正在增强...")
            
            # 执行增强
            enhanced = self.model_wrapper.predict(processed)
            
            # 记录进度
            self._report_progress(3, 4, "后处理...")
            
            # 应用后处理
            enhanced = self._postprocess_by_mode(enhanced)
            
            # 保存结果
            self._last_result = enhanced.copy()
            
            # 记录进度
            self._report_progress(4, 4, "完成")
            
            # 更新统计
            elapsed = time.time() - start_time
            self._stats['total_processed'] += 1
            self._stats['total_time'] += elapsed
            self._stats['success_count'] += 1
            
            # 构建返回结果
            result = enhanced
            
            if return_metrics and return_comparison:
                metrics = self.compute_metrics(image, enhanced)
                return result, metrics, image
            elif return_metrics:
                metrics = self.compute_metrics(image, enhanced)
                return result, metrics
            elif return_comparison:
                return result, image
            else:
                return result
                
        except Exception as e:
            self._stats['error_count'] += 1
            self.status = ProcessingStatus.ERROR
            logger.error(f"图像增强失败: {e}")
            raise
        
        finally:
            if self.status != ProcessingStatus.ERROR:
                self.status = ProcessingStatus.IDLE
    
    def _validate_image(self, image: np.ndarray) -> np.ndarray:
        """
        验证并规范化输入图像
        
        Args:
            image: 输入图像
            
        Returns:
            规范化的图像
        """
        # 复制图像避免修改原图
        image = image.copy()
        
        # 转换为3通道
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA转RGB
            image = image[:, :, :3]
        
        # 确保数据类型为uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
    
    def _preprocess_by_mode(self, image: np.ndarray) -> np.ndarray:
        """
        根据增强模式进行预处理
        
        Args:
            image: 输入图像
            
        Returns:
            预处理后的图像
        """
        # 大多数模式下，预处理由模型封装器处理
        # 这里可以添加特定模式的预处理逻辑
        return image
    
    def _postprocess_by_mode(self, image: np.ndarray) -> np.ndarray:
        """
        根据增强模式进行后处理
        
        Args:
            image: 输入图像
            
        Returns:
            后处理后的图像
        """
        if self.mode == EnhancementMode.LIGHT:
            # 轻度增强：混合原图和增强图
            alpha = 0.5
            image = (alpha * image + (1 - alpha) * self._last_original).astype(np.uint8)
            
        elif self.mode == EnhancementMode.STRONG:
            # 强力增强：增加对比度
            import cv2
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        elif self.mode == EnhancementMode.CUSTOM:
            # 自定义模式：应用用户配置的参数
            # 具体参数由GUI层传入
            pass
        
        return image
    
    def compute_metrics(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> dict:
        """
        计算增强图像的质量指标
        
        Args:
            original: 原始图像
            enhanced: 增强后的图像
            
        Returns:
            指标字典
        """
        return self.metrics.evaluate(original, enhanced)
    
    def batch_enhance(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        preserve_structure: bool = True,
        overwrite: bool = False
    ) -> List[dict]:
        """
        批量增强图像
        
        Args:
            image_paths: 图像文件路径列表
            output_dir: 输出目录，None则在原目录添加后缀
            preserve_structure: 是否保持目录结构
            overwrite: 是否覆盖已存在的文件
            
        Returns:
            处理结果列表，每项包含路径和状态信息
        """
        from ..utils.file_io import read_image, write_image
        from ..utils.image_utils import get_image_files
        
        results = []
        total = len(image_paths)
        
        logger.info(f"开始批量处理: {total} 张图像")
        
        for i, path in enumerate(image_paths):
            path = Path(path)
            
            # 报告进度
            self._report_progress(i + 1, total, f"处理中: {path.name}")
            
            try:
                # 读取图像
                image = read_image(path)
                
                # 增强处理
                enhanced = self.enhance(image)
                
                # 确定输出路径
                if output_dir:
                    output_path = Path(output_dir)
                    if preserve_structure:
                        # 保持相对目录结构
                        rel_path = path.name
                    else:
                        rel_path = path.name
                    output_path = output_path / rel_path
                else:
                    # 在原目录添加后缀
                    output_path = path.parent / f"{path.stem}_enhanced{path.suffix}"
                
                # 确保输出目录存在
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 保存图像
                if overwrite or not output_path.exists():
                    write_image(enhanced, output_path)
                    status = 'saved'
                else:
                    status = 'skipped_existing'
                
                results.append({
                    'input': str(path),
                    'output': str(output_path),
                    'status': status,
                    'error': None
                })
                
                logger.info(f"[{i+1}/{total}] {path.name} -> {output_path.name}")
                
            except Exception as e:
                logger.error(f"[{i+1}/{total}] 处理失败: {path.name}, 错误: {e}")
                results.append({
                    'input': str(path),
                    'output': None,
                    'status': 'error',
                    'error': str(e)
                })
        
        # 统计结果
        success = sum(1 for r in results if r['status'] == 'saved')
        skipped = sum(1 for r in results if r['status'] == 'skipped_existing')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        logger.info(f"批量处理完成: 成功 {success}, 跳过 {skipped}, 失败 {failed}")
        
        return results
    
    @property
    def statistics(self) -> dict:
        """获取处理统计信息"""
        stats = self._stats.copy()
        if stats['total_processed'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = stats['success_count'] / stats['total_processed']
        else:
            stats['average_time'] = 0.0
            stats['success_rate'] = 0.0
        return stats
    
    def reset_statistics(self) -> None:
        """重置统计信息"""
        self._stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
    
    def get_last_result(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        获取上次处理的结果
        
        Returns:
            (增强图像, 原始图像) 元组，可能为None
        """
        return self._last_result, self._last_original
    
    def save_result(
        self,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 95
    ) -> bool:
        """
        保存上次增强结果
        
        Args:
            output_path: 输出路径
            format: 图像格式，None则根据扩展名自动判断
            quality: JPEG质量 (1-100)
            
        Returns:
            保存是否成功
        """
        from ..utils.file_io import write_image
        
        if self._last_result is None:
            logger.warning("没有可保存的增强结果")
            return False
        
        try:
            write_image(
                self._last_result,
                output_path,
                format=format,
                quality=quality
            )
            logger.info(f"结果已保存: {output_path}")
            return True
        except Exception as e:
            logger.error(f"保存失败: {e}")
            return False
    
    def export_results_json(self, results: List[dict], output_path: str) -> None:
        """
        导出一批处理结果为JSON
        
        Args:
            results: 处理结果列表
            output_path: 输出JSON文件路径
        """
        export_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': self.statistics,
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已导出: {output_path}")
    
    def __repr__(self) -> str:
        return f"ImageEnhancer(mode={self.mode.value}, status={self.status.value}, loaded={self.is_model_loaded})"
    
    def __del__(self):
        """析构时清理资源"""
        self.unload_model()
