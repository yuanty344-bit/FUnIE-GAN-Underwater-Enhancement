"""
图像质量评估指标模块

提供多种图像质量评估指标的计算：
- UIQM (Underwater Image Quality Measure) - 水下图像质量指标
- SSIM (Structural Similarity Index) - 结构相似性指数
- PSNR (Peak Signal-to-Noise Ratio) - 峰值信噪比
- MAE (Mean Absolute Error) - 平均绝对误差
- UCIQE (Underwater Color Image Quality Evaluation) - 水下色图图像质量评估
"""

import logging
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass, asdict
import warnings

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """
    图像质量指标数据类
    
    包含所有评估指标的计算结果
    """
    # UIQM 指标 (Underwater Image Quality Measure)
    uiqm: float = 0.0  # 总体UIQM分数
    uicm: float = 0.0  # 色彩度量
    uism: float = 0.0  # 清晰度/锐度度量
    uiconm: float = 0.0  # 对比度度量
    
    # UCIQE 指标 (Underwater Color Image Quality Evaluation)
    uciqe: float = 0.0
    
    # 传统图像质量指标
    ssim: float = 0.0  # 结构相似性指数
    psnr: float = 0.0  # 峰值信噪比 (dB)
    mae: float = 0.0  # 平均绝对误差
    mse: float = 0.0  # 均方误差
    
    # 额外指标
    entropy: float = 0.0  # 信息熵
    brightness: float = 0.0  # 平均亮度
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return asdict(self)
    
    def summary(self) -> str:
        """生成指标摘要字符串"""
        lines = [
            "=" * 40,
            "图像质量评估结果",
            "=" * 40,
            f"\n【UIQM 水下图像质量指标】",
            f"  总体分数 (UIQM): {self.uiqm:.4f}",
            f"  色彩度量 (UICM): {self.uicm:.4f}",
            f"  清晰度量 (UISM): {self.uism:.4f}",
            f"  对比度量 (UICONM): {self.uiconm:.4f}",
            f"\n【UCIQE 水下色图质量评估】",
            f"  UCIQE分数: {self.uciqe:.4f}",
            f"\n【传统质量指标】",
            f"  SSIM: {self.ssim:.4f}",
            f"  PSNR: {self.psnr:.2f} dB",
            f"  MAE: {self.mae:.4f}",
            f"  MSE: {self.mse:.4f}",
            f"\n【统计信息】",
            f"  信息熵: {self.entropy:.4f}",
            f"  平均亮度: {self.brightness:.2f}",
            "=" * 40
        ]
        return "\n".join(lines)


class ImageMetrics:
    """
    图像质量评估器
    
    提供多种图像质量评估指标的计算方法，
    特别针对水下图像进行了优化。
    
    Attributes:
        use_scikit_image: 是否使用scikit-image计算SSIM
        bit_depth: 图像位深，默认为8
        
    Example:
        >>> metrics = ImageMetrics()
        >>> result = metrics.evaluate(original_image, enhanced_image)
        >>> print(result.uiqm, result.ssim, result.psnr)
    """
    
    def __init__(
        self,
        use_scikit_image: bool = True,
        bit_depth: int = 8
    ):
        """
        初始化图像质量评估器
        
        Args:
            use_scikit_image: 是否优先使用scikit-image（更准确）
            bit_depth: 图像位深，默认为8位
        """
        self.use_scikit_image = use_scikit_image
        self.bit_depth = bit_depth
        self.max_value = 2 ** bit_depth - 1
        
        # 尝试导入可选依赖
        self._skimage_available = False
        self._cv2_available = False
        self._scipy_available = False

        try:
            import cv2
            self._cv2 = cv2
            self._cv2_available = True
        except ImportError:
            logger.warning("OpenCV 未安装，部分功能可能受限")

        try:
            from scipy.ndimage import uniform_filter
            self._scipy_available = True
        except ImportError:
            logger.warning("scipy 未安装，SSIM计算将使用替代方法")

        try:
            from skimage.metrics import structural_similarity
            from skimage.metrics import peak_signal_noise_ratio
            from skimage.metrics import mean_squared_error
            self._skimage = None  # 不需要存储
            self._skimage_available = True
        except ImportError:
            logger.warning("scikit-image 未安装，将使用替代方法计算SSIM")
    
    def evaluate(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        include_all: bool = True
    ) -> QualityMetrics:
        """
        评估增强图像的质量
        
        Args:
            original: 原始图像
            enhanced: 增强后的图像
            include_all: 是否计算所有指标
            
        Returns:
            QualityMetrics 对象
        """
        # 验证输入
        original = self._prepare_image(original)
        enhanced = self._prepare_image(enhanced)
        
        # 确保尺寸一致
        if original.shape != enhanced.shape:
            enhanced = self._resize_match(original, enhanced)
        
        # 创建结果对象
        metrics = QualityMetrics()
        
        # 计算各项指标
        try:
            # UIQM 指标（水下图像专用）
            uiqm_result = self.compute_uiqm(enhanced)
            metrics.uiqm = uiqm_result['uiqm']
            metrics.uicm = uiqm_result['uicm']
            metrics.uism = uiqm_result['uism']
            metrics.uiconm = uiqm_result['uiconm']
            
            # UCIQE 指标
            metrics.uciqe = self.compute_uciqe(enhanced)
            
            # 传统指标
            if include_all or True:
                metrics.ssim = self.compute_ssim(original, enhanced)
                metrics.psnr = self.compute_psnr(original, enhanced)
                metrics.mae = self.compute_mae(original, enhanced)
                metrics.mse = self.compute_mse(original, enhanced)
                
            # 统计指标
            metrics.entropy = self.compute_entropy(enhanced)
            metrics.brightness = self.compute_brightness(enhanced)
            
        except Exception as e:
            logger.error(f"指标计算出错: {e}")
        
        return metrics
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """
        准备图像数据
        
        Args:
            image: 输入图像
            
        Returns:
            标准化后的图像
        """
        image = image.astype(np.float64)
        
        # 如果图像范围是[0, 255]，归一化到[0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    
    def _resize_match(self, target: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        调整源图像尺寸以匹配目标
        
        Args:
            target: 目标图像
            source: 源图像
            
        Returns:
            调整后的源图像
        """
        if self._cv2_available:
            import cv2
            return cv2.resize(
                source,
                (target.shape[1], target.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            from PIL import Image
            pil_img = Image.fromarray((source * 255).astype(np.uint8))
            pil_img = pil_img.resize(
                (target.shape[1], target.shape[0]),
                Image.BILINEAR
            )
            return np.array(pil_img).astype(np.float64) / 255.0
    
    def compute_uiqm(self, image: np.ndarray) -> dict:
        """
        计算 UIQM (Underwater Image Quality Measure)
        
        UIQM = c1 * UICM + c2 * UISM + c3 * UICONM
        
        其中:
        - UICM: 色彩度量 (Underwater Image Colorfulness Measure)
        - UISM: 锐度/清晰度度量 (Underwater Image Sharpness Measure)
        - UICONM: 对比度度量 (Underwater Image Contrast Measure)
        
        Args:
            image: 输入图像，范围[0, 1]
            
        Returns:
            包含UIQM及各分量的字典
            
        Reference:
            M. Yang and J. Hu, "An Objective Quality Assessment
            Method for Underwater Images", ICMEW 2019.
        """
        # UIQM 权重参数
        c1, c2, c3 = 0.0282, 0.2953, 3.5753
        
        # 计算各分量
        uicm = self._compute_uicm(image)
        uism = self._compute_uism(image)
        uiconm = self._compute_uiconm(image)
        
        # 计算总体UIQM
        uiqm = c1 * uicm + c2 * uism + c3 * uiconm
        
        return {
            'uiqm': float(uiqm),
            'uicm': float(uicm),
            'uism': float(uism),
            'uiconm': float(uiconm)
        }
    
    def _compute_uicm(self, image: np.ndarray) -> float:
        """
        计算 UICM - 色彩度量
        
        基于图像色彩丰富度和色偏程度
        
        Args:
            image: 归一化RGB图像
            
        Returns:
            UICM值
        """
        # 转换到LAB色彩空间（近似计算）
        # 使用更简化的色彩度量方法
        
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        
        # 计算色彩饱和度
        max_rgb = np.maximum(np.maximum(R, G), B)
        min_rgb = np.minimum(np.minimum(R, G), B)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        
        # 计算色彩丰富度（饱和度的均值）
        colorfulness = np.mean(saturation)
        
        # 计算色偏（偏离灰色的程度）
        gray = (R + G + B) / 3
        color_bias = np.mean(np.abs(R - gray) + np.abs(G - gray) + np.abs(B - gray))
        
        # UICM: 色彩丰富度越高越好，色偏越低越好
        uicm = colorfulness * 2 - color_bias * 0.5
        
        return max(0, uicm)
    
    def _compute_uism(self, image: np.ndarray) -> float:
        """
        计算 UISM - 锐度/清晰度度量
        
        基于图像的局部梯度/边缘强度
        
        Args:
            image: 归一化RGB图像
            
        Returns:
            UISM值
        """
        # 转换为灰度图
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # 计算Sobel梯度
        if self._cv2_available:
            import cv2
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        else:
            # 简化的Sobel算子
            grad_x = np.zeros_like(gray)
            grad_y = np.zeros_like(gray)
            
            # X方向梯度
            grad_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
            # Y方向梯度
            grad_y[1:-1, :] = gray[2:, :] - gray[:-2, :]
        
        # 计算梯度幅度
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # UISM: 使用PCSM方法，计算边缘感知度
        # 这里使用简化的计算
        uism = np.mean(gradient_magnitude)
        
        return float(uism)
    
    def _compute_uiconm(self, image: np.ndarray) -> float:
        """
        计算 UICONM - 对比度度量
        
        基于图像的局部对比度
        
        Args:
            image: 归一化RGB图像
            
        Returns:
            UICONM值
        """
        # 转换到灰度
        gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        
        # 使用局部标准差作为对比度度量
        # 简化的滑动窗口计算
        window_size = 5
        pad = window_size // 2
        
        # 填充
        padded = np.pad(gray, pad, mode='edge')
        
        # 计算局部均值和方差
        uiconm = 0
        count = 0
        
        for i in range(pad, padded.shape[0] - pad):
            for j in range(pad, padded.shape[1] - pad):
                window = padded[i-pad:i+pad+1, j-pad:j+pad+1]
                local_std = np.std(window)
                uiconm += local_std
                count += 1
        
        uiconm = uiconm / count if count > 0 else 0
        
        return float(uiconm)
    
    def compute_uciqe(self, image: np.ndarray) -> float:
        """
        计算 UCIQE (Underwater Color Image Quality Evaluation)
        
        UCIQE = c1 * sigma_c + c2 * con_l + c3 * avg_saturation
        
        Args:
            image: 归一化RGB图像
            
        Returns:
            UCIQE值
            
        Reference:
            M. Yang et al., "UCIQE: An Universal Image Quality
            Index for Underwater Imaging", SRL 2015.
        """
        # UCIQE 权重
        c1, c2, c3 = 0.4680, 0.2745, 0.2576
        
        # 色度偏差 (sigma_c)
        R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        chroma = np.sqrt(((R - 0.5) ** 2 + (G - 0.5) ** 2 + (B - 0.5) ** 2))
        sigma_c = np.std(chroma)
        
        # 对比度 (con_l)
        gray = 0.299 * R + 0.587 * G + 0.114 * B
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        cumsum = np.cumsum(hist)
        min_intensity = np.searchsorted(cumsum, cumsum[-1] * 0.05)
        max_intensity = np.searchsorted(cumsum, cumsum[-1] * 0.95, side='right')
        con_l = (max_intensity - min_intensity) / 256.0
        
        # 平均饱和度 (avg_saturation)
        max_rgb = np.maximum(np.maximum(R, G), B)
        min_rgb = np.minimum(np.minimum(R, G), B)
        saturation = (max_rgb - min_rgb) / (max_rgb + 1e-8)
        avg_saturation = np.mean(saturation)
        
        # UCIQE
        uciqe = c1 * sigma_c + c2 * con_l + c3 * avg_saturation
        
        return float(uciqe)
    
    def compute_ssim(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        window_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
        L: float = 1.0
    ) -> float:
        """
        计算 SSIM (Structural Similarity Index)
        
        SSIM 衡量两幅图像的结构相似性，范围[-1, 1]，越接近1越好
        
        Args:
            original: 原始图像
            enhanced: 增强图像
            window_size: 滑动窗口大小
            k1, k2: 稳定常数
            L: 像素值范围
            
        Returns:
            SSIM值
            
        Reference:
            Z. Wang et al., "Image Quality Assessment: From Error
            Visibility to Structural Similarity", IEEE TIP 2004.
        """
        if self._skimage_available and self.use_scikit_image:
            from skimage.metrics import structural_similarity
            # 转换为灰度图进行SSIM计算
            if len(original.shape) == 3:
                gray1 = np.mean(original, axis=2)
                gray2 = np.mean(enhanced, axis=2)
            else:
                gray1 = original
                gray2 = enhanced
            return structural_similarity(gray1, gray2, data_range=1.0)
        
        # 手动实现 SSIM
        # 转换为灰度图
        if len(original.shape) == 3:
            gray1 = 0.299 * original[:, :, 0] + 0.587 * original[:, :, 1] + 0.114 * original[:, :, 2]
            gray2 = 0.299 * enhanced[:, :, 0] + 0.587 * enhanced[:, :, 1] + 0.114 * enhanced[:, :, 2]
        else:
            gray1 = original
            gray2 = enhanced
        
        # 常数
        C1 = (k1 * L) ** 2
        C2 = (k2 * L) ** 2
        
        # 计算局部统计量
        mu1 = self._local_mean(gray1, window_size)
        mu2 = self._local_mean(gray2, window_size)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = self._local_variance(gray1, mu1, window_size)
        sigma2_sq = self._local_variance(gray2, mu2, window_size)
        sigma12 = self._local_covariance(gray1, gray2, mu1, mu2, window_size)
        
        # SSIM 公式
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / denominator
        
        return float(np.mean(ssim_map))
    
    def _box_filter(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """局部均值滤波器（优先scipy，回退cv2，再回退numpy）"""
        if self._scipy_available:
            from scipy.ndimage import uniform_filter
            return uniform_filter(image, size=window_size)
        if self._cv2_available:
            return self._cv2.blur(image.astype(np.float32), (window_size, window_size))
        # Pure numpy fallback: separable box filter via convolve1d
        kernel = np.ones(window_size) / window_size
        result = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode='same'), axis=0, arr=image)
        result = np.apply_along_axis(lambda r: np.convolve(r, kernel, mode='same'), axis=1, arr=result)
        return result

    def _local_mean(self, image: np.ndarray, window_size: int) -> np.ndarray:
        """计算局部均值"""
        return self._box_filter(image, window_size)

    def _local_variance(
        self,
        image: np.ndarray,
        mean: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """计算局部方差"""
        return self._box_filter(image ** 2, window_size) - mean ** 2

    def _local_covariance(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        mean1: np.ndarray,
        mean2: np.ndarray,
        window_size: int
    ) -> np.ndarray:
        """计算局部协方差"""
        return self._box_filter(image1 * image2, window_size) - mean1 * mean2
    
    def compute_psnr(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        max_value: float = 1.0
    ) -> float:
        """
        计算 PSNR (Peak Signal-to-Noise Ratio)
        
        PSNR 衡量图像质量，单位dB，越高越好
        
        Args:
            original: 原始图像
            enhanced: 增强图像
            max_value: 最大像素值
            
        Returns:
            PSNR值 (dB)
        """
        if self._skimage_available and self.use_scikit_image:
            from skimage.metrics import peak_signal_noise_ratio
            return peak_signal_noise_ratio(original, enhanced, data_range=max_value)
        
        # 手动计算
        mse = self.compute_mse(original, enhanced)
        
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(max_value / np.sqrt(mse))
        return float(psnr)
    
    def compute_mse(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        计算 MSE (Mean Squared Error)
        
        Args:
            original: 原始图像
            enhanced: 增强图像
            
        Returns:
            MSE值
        """
        if self._skimage_available and self.use_scikit_image:
            from skimage.metrics import mean_squared_error
            return mean_squared_error(original, enhanced)
        
        # 手动计算
        return float(np.mean((original - enhanced) ** 2))
    
    def compute_mae(self, original: np.ndarray, enhanced: np.ndarray) -> float:
        """
        计算 MAE (Mean Absolute Error)
        
        Args:
            original: 原始图像
            enhanced: 增强图像
            
        Returns:
            MAE值
        """
        return float(np.mean(np.abs(original - enhanced)))
    
    def compute_entropy(self, image: np.ndarray) -> float:
        """
        计算图像信息熵
        
        Args:
            image: 输入图像
            
        Returns:
            熵值
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image
        
        # 计算直方图
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        
        # 计算概率分布
        prob = hist / hist.sum()
        
        # 计算熵
        entropy = -np.sum(prob * np.log2(prob + 1e-10))
        
        return float(entropy)
    
    def compute_brightness(self, image: np.ndarray) -> float:
        """
        计算图像平均亮度
        
        Args:
            image: 输入图像，范围[0, 1]
            
        Returns:
            平均亮度值
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
        else:
            gray = image
        
        return float(np.mean(gray))
    
    def compare_metrics(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> Tuple[QualityMetrics, QualityMetrics]:
        """
        分别计算原始和增强图像的指标
        
        Args:
            original: 原始图像
            enhanced: 增强图像
            
        Returns:
            (原始图像指标, 增强图像指标)
        """
        return (
            self.evaluate(original, original),
            self.evaluate(original, enhanced)
        )
    
    def print_comparison(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> None:
        """
        打印原始和增强图像的指标对比
        
        Args:
            original: 原始图像
            enhanced: 增强图像
        """
        _, enhanced_metrics = self.compare_metrics(original, enhanced)
        print(enhanced_metrics.summary())
