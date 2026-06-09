"""
手动滤镜调整模块

提供实时滤镜调节功能，
支持亮度、对比度、饱和度、锐度、白平衡等调整。
"""

import logging
from typing import Optional, Tuple, Union, Callable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterParams:
    """
    滤镜参数数据类
    
    所有参数范围为 0.0-2.0，其中 1.0 表示原始值（无变化）
    """
    brightness: float = 1.0  # 亮度
    contrast: float = 1.0  # 对比度
    saturation: float = 1.0  # 饱和度
    sharpness: float = 1.0  # 锐度
    white_balance: float = 1.0  # 白平衡
    gamma: float = 1.0  # gamma校正
    hue: float = 1.0  # 色相
    
    def is_default(self) -> bool:
        """检查是否为默认参数（无调整）"""
        return (
            self.brightness == 1.0 and
            self.contrast == 1.0 and
            self.saturation == 1.0 and
            self.sharpness == 1.0 and
            self.white_balance == 1.0 and
            self.gamma == 1.0 and
            self.hue == 1.0
        )
    
    def reset(self) -> None:
        """重置为默认参数"""
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.sharpness = 1.0
        self.white_balance = 1.0
        self.gamma = 1.0
        self.hue = 1.0


class AdjustFilters:
    """
    滤镜调整器
    
    提供多种图像滤镜的实时调整功能，
    支持链式应用和组合调整。
    
    Example:
        >>> filters = AdjustFilters()
        >>> result = filters.apply(
        ...     image,
        ...     brightness=1.2,
        ...     contrast=1.1,
        ...     saturation=1.0
        ... )
    """
    
    def __init__(self):
        """初始化滤镜调整器"""
        self._cv2_available = False
        self._pil_available = False
        
        try:
            import cv2
            self._cv2 = cv2
            self._cv2_available = True
        except ImportError:
            logger.debug("OpenCV不可用，将使用PIL")
        
        try:
            from PIL import Image, ImageEnhance, ImageFilter
            self._PIL = Image
            self._ImageEnhance = ImageEnhance
            self._ImageFilter = ImageFilter
            self._pil_available = True
        except ImportError:
            logger.debug("PIL不可用")
    
    def apply(
        self,
        image: np.ndarray,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
        white_balance: float = 1.0,
        gamma: float = 1.0,
        hue: float = 1.0,
        use_gpu: bool = False
    ) -> np.ndarray:
        """
        应用滤镜调整
        
        按顺序应用所有滤镜：亮度 → 对比度 → 白平衡 → 饱和度 → 锐度 → Gamma
        
        Args:
            image: 输入图像，uint8数组，范围[0, 255]
            brightness: 亮度调整 (0.0-2.0, 1.0=不变)
            contrast: 对比度调整 (0.0-2.0, 1.0=不变)
            saturation: 饱和度调整 (0.0-2.0, 1.0=不变)
            sharpness: 锐度调整 (0.0-2.0, 1.0=不变)
            white_balance: 白平衡调整 (0.0-2.0, 1.0=不变)
            gamma: Gamma校正 (0.0-2.0, 1.0=不变)
            hue: 色相调整 (0.0-2.0, 1.0=不变，暂不支持)
            use_gpu: 是否使用GPU加速
            
        Returns:
            调整后的图像
        """
        if image is None or image.size == 0:
            return image
        
        # 确保是RGB图像
        result = self._ensure_rgb(image)
        
        # 按顺序应用滤镜
        # 1. 亮度
        if abs(brightness - 1.0) > 0.01:
            result = self._adjust_brightness(result, brightness)
        
        # 2. 对比度
        if abs(contrast - 1.0) > 0.01:
            result = self._adjust_contrast(result, contrast)
        
        # 3. 白平衡
        if abs(white_balance - 1.0) > 0.01:
            result = self._adjust_white_balance_auto(result, white_balance)
        
        # 4. 饱和度
        if abs(saturation - 1.0) > 0.01:
            result = self._adjust_saturation(result, saturation)
        
        # 5. 锐度
        if abs(sharpness - 1.0) > 0.01:
            result = self._adjust_sharpness(result, sharpness)
        
        # 6. Gamma校正
        if abs(gamma - 1.0) > 0.01:
            result = self._adjust_gamma(result, gamma)
        
        # 确保输出是uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _ensure_rgb(self, image: np.ndarray) -> np.ndarray:
        """确保图像是RGB格式"""
        if len(image.shape) == 2:
            # 灰度图转RGB
            return np.stack([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            # RGBA转RGB
            return image[:, :, :3]
        return image
    
    def _adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整亮度"""
        if self._cv2_available:
            # 使用OpenCV
            if factor > 1.0:
                # 增加亮度
                hsv = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2HSV)
                h, s, v = self._cv2.split(hsv)
                v = np.clip(v * factor, 0, 255).astype(np.uint8)
                hsv = self._cv2.merge([h, s, v])
                result = self._cv2.cvtColor(hsv, self._cv2.COLOR_HSV2RGB)
            else:
                # 降低亮度
                result = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        else:
            # 使用PIL
            pil_img = self._PIL.fromarray(image)
            enhancer = self._ImageEnhance.Brightness(pil_img)
            result = np.array(enhancer.enhance(factor))
        
        return result
    
    def _adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整对比度"""
        if self._cv2_available:
            # 计算均值
            mean = np.mean(image)
            
            # 应用对比度调整
            result = np.clip(
                (image - mean) * factor + mean,
                0, 255
            ).astype(np.uint8)
        else:
            pil_img = self._PIL.fromarray(image)
            enhancer = self._ImageEnhance.Contrast(pil_img)
            result = np.array(enhancer.enhance(factor))
        
        return result
    
    def _adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整饱和度"""
        if self._cv2_available:
            hsv = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2HSV)
            h, s, v = self._cv2.split(hsv)
            s = np.clip(s * factor, 0, 255).astype(np.uint8)
            hsv = self._cv2.merge([h, s, v])
            result = self._cv2.cvtColor(hsv, self._cv2.COLOR_HSV2RGB)
        else:
            pil_img = self._PIL.fromarray(image)
            enhancer = self._ImageEnhance.Color(pil_img)
            result = np.array(enhancer.enhance(factor))
        
        return result
    
    def _adjust_sharpness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """调整锐度"""
        if self._cv2_available:
            if factor > 1.0:
                # 锐化
                kernel = np.array([
                    [-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]
                ]) * ((factor - 1.0) / 9.0)
                
                center = 1.0 - (factor - 1.0) * 8.0 / 9.0
                kernel[1, 1] = center
                
                result = self._cv2.filter2D(image, -1, kernel)
            elif factor < 1.0:
                # 模糊
                blur_amount = int((1.0 - factor) * 5)
                blur_amount = max(1, blur_amount)
                if blur_amount % 2 == 0:
                    blur_amount += 1
                result = self._cv2.GaussianBlur(image, (blur_amount, blur_amount), 0)
            else:
                result = image
        else:
            pil_img = self._PIL.fromarray(image)
            enhancer = self._ImageEnhance.Sharpness(pil_img)
            result = np.array(enhancer.enhance(factor))
        
        return result
    
    def _adjust_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Gamma校正"""
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype(np.uint8)
        
        if self._cv2_available:
            result = self._cv2.LUT(image, table)
        else:
            result = table[image]
        
        return result
    
    def _adjust_white_balance_auto(
        self,
        image: np.ndarray,
        strength: float = 1.0
    ) -> np.ndarray:
        """
        自动白平衡校正（基于灰度世界假设）
        
        Args:
            image: 输入RGB图像
            strength: 调整强度 (0.0-2.0)
            
        Returns:
            白平衡校正后的图像
        """
        # 计算各通道平均值
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        # 计算灰度平均值
        gray_mean = (r_mean + g_mean + b_mean) / 3
        
        # 计算增益
        r_gain = gray_mean / (r_mean + 1e-8)
        g_gain = gray_mean / (g_mean + 1e-8)
        b_gain = gray_mean / (b_mean + 1e-8)
        
        # 混合原始增益和均衡增益
        # 当strength=1时使用完整增益，strength<1时减弱效果
        r_gain = 1.0 + (r_gain - 1.0) * strength
        g_gain = 1.0 + (g_gain - 1.0) * strength
        b_gain = 1.0 + (b_gain - 1.0) * strength
        
        # 应用增益
        result = image.astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] * r_gain, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * g_gain, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] * b_gain, 0, 255)
        
        return result.astype(np.uint8)
    
    def apply_with_params(
        self,
        image: np.ndarray,
        params: FilterParams
    ) -> np.ndarray:
        """
        使用FilterParams应用滤镜
        
        Args:
            image: 输入图像
            params: 滤镜参数
            
        Returns:
            调整后的图像
        """
        return self.apply(
            image,
            brightness=params.brightness,
            contrast=params.contrast,
            saturation=params.saturation,
            sharpness=params.sharpness,
            white_balance=params.white_balance,
            gamma=params.gamma,
            hue=params.hue
        )
    
    def create_preview(
        self,
        image: np.ndarray,
        params: FilterParams,
        thumbnail_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        创建预览缩略图
        
        用于在调整时快速预览效果
        
        Args:
            image: 输入图像
            params: 滤镜参数
            thumbnail_size: 缩略图尺寸
            
        Returns:
            预览图像
        """
        # 缩小图像
        h, w = image.shape[:2]
        scale = min(thumbnail_size[0] / w, thumbnail_size[1] / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if self._cv2_available:
                thumbnail = self._cv2.resize(image, (new_w, new_h))
            else:
                pil_img = self._PIL.fromarray(image)
                pil_img = pil_img.resize((new_w, new_h), self._PIL.Image.LANCZOS)
                thumbnail = np.array(pil_img)
        else:
            thumbnail = image
        
        # 应用滤镜
        return self.apply_with_params(thumbnail, params)
    
    def get_preview_params(
        self,
        brightness_delta: float = 0.0,
        contrast_delta: float = 0.0,
        saturation_delta: float = 0.0,
        sharpness_delta: float = 0.0,
        wb_delta: float = 0.0,
        gamma_delta: float = 0.0
    ) -> FilterParams:
        """
        从增量值创建滤镜参数
        
        用于滑块调整
        
        Args:
            brightness_delta: 亮度增量 (-1.0 to 1.0)
            contrast_delta: 对比度增量
            saturation_delta: 饱和度增量
            sharpness_delta: 锐度增量
            wb_delta: 白平衡增量
            gamma_delta: gamma增量
            
        Returns:
            FilterParams对象
        """
        return FilterParams(
            brightness=1.0 + brightness_delta,
            contrast=1.0 + contrast_delta,
            saturation=1.0 + saturation_delta,
            sharpness=1.0 + sharpness_delta,
            white_balance=1.0 + wb_delta,
            gamma=1.0 + gamma_delta
        )


# 预设滤镜配置
class FilterPresets:
    """预设滤镜"""
    
    # 原始（无调整）
    ORIGINAL = FilterParams()
    
    # 自动增强
    AUTO = FilterParams(
        brightness=1.1,
        contrast=1.1,
        saturation=1.1,
        sharpness=1.0,
        white_balance=1.1,
        gamma=1.0
    )
    
    # 鲜艳
    VIVID = FilterParams(
        brightness=1.1,
        contrast=1.3,
        saturation=1.4,
        sharpness=1.2,
        white_balance=1.0,
        gamma=1.1
    )
    
    # 柔和
    SOFT = FilterParams(
        brightness=1.05,
        contrast=0.9,
        saturation=0.9,
        sharpness=0.8,
        white_balance=1.0,
        gamma=0.95
    )
    
    # 高对比度
    HIGH_CONTRAST = FilterParams(
        brightness=1.0,
        contrast=1.5,
        saturation=1.1,
        sharpness=1.2,
        white_balance=1.0,
        gamma=1.0
    )
    
    # 复古
    VINTAGE = FilterParams(
        brightness=0.95,
        contrast=0.9,
        saturation=0.8,
        sharpness=0.9,
        white_balance=0.9,
        gamma=1.1
    )
    
    # 水下优化
    UNDERWATER = FilterParams(
        brightness=1.15,
        contrast=1.1,
        saturation=1.2,
        sharpness=1.1,
        white_balance=1.3,  # 增强白平衡校正
        gamma=0.95
    )
