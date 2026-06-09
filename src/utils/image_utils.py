"""
图像处理工具模块

提供常用的图像处理函数，
包括格式转换、尺寸调整、滤镜处理等。
"""

import math
import logging
from typing import Optional, Tuple, Union, List

import numpy as np

logger = logging.getLogger(__name__)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """
    RGB图像转灰度图
    
    Args:
        image: RGB图像数组
        
    Returns:
        灰度图像数组
    """
    if len(image.shape) == 2:
        return image
    
    # 使用标准加权
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return gray.astype(image.dtype)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    灰度图转RGB图像
    
    Args:
        image: 灰度图像数组
        
    Returns:
        RGB图像数组
    """
    if len(image.shape) == 3:
        return image
    
    return np.stack([image, image, image], axis=-1)


def normalize_image(
    image: np.ndarray,
    dtype: type = np.uint8
) -> np.ndarray:
    """
    归一化图像到指定范围
    
    Args:
        image: 输入图像
        dtype: 输出数据类型
        
    Returns:
        归一化后的图像
    """
    image = image.astype(np.float32)
    
    # 获取当前范围
    min_val = image.min()
    max_val = image.max()
    
    # 避免除零
    if max_val - min_val < 1e-8:
        return np.full_like(image, 0 if dtype == np.uint8 else 0.5).astype(dtype)
    
    # 归一化到[0, 1]
    normalized = (image - min_val) / (max_val - min_val)
    
    # 转换到目标类型
    if dtype == np.uint8:
        normalized = (normalized * 255).astype(np.uint8)
    
    return normalized.astype(dtype)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    interpolation: str = 'bilinear'
) -> np.ndarray:
    """
    调整图像尺寸
    
    Args:
        image: 输入图像
        size: 目标尺寸 (width, height)
        interpolation: 插值方法
            - 'nearest': 最近邻插值
            - 'bilinear': 双线性插值
            - 'bicubic': 双三次插值
            - 'lanczos': Lanczos插值
            
    Returns:
        调整后的图像
    """
    try:
        import cv2
        
        inter_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        inter_code = inter_map.get(interpolation, cv2.INTER_LINEAR)
        
        resized = cv2.resize(
            image,
            (size[0], size[1]),
            interpolation=inter_code
        )
        
    except ImportError:
        from PIL import Image
        
        pil_image = Image.fromarray(image)
        pil_resized = pil_image.resize(
            (size[0], size[1]),
            Image.BILINEAR if interpolation == 'bilinear' else Image.NEAREST
        )
        resized = np.array(pil_resized)
    
    return resized


def resize_keep_aspect(
    image: np.ndarray,
    target_size: Tuple[int, int],
    padding: bool = True
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    保持宽高比调整图像尺寸
    
    Args:
        image: 输入图像
        target_size: 目标最大尺寸 (width, height)
        padding: 是否用黑色填充以保持比例
        
    Returns:
        (调整后的图像, (original_width, original_height))
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 计算缩放比例
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # 调整尺寸
    resized = resize_image(image, (new_w, new_h))
    
    if padding and (new_w != target_w or new_h != target_h):
        # 创建画布
        canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
        
        # 计算居中偏移
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # 放置图像
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        resized = canvas
    
    return resized, (w, h)


def crop_image(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    裁剪图像
    
    Args:
        image: 输入图像
        x, y: 裁剪区域左上角坐标
        width, height: 裁剪区域尺寸
        
    Returns:
        裁剪后的图像
    """
    h, w = image.shape[:2]
    
    # 边界检查
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    width = min(width, w - x)
    height = min(height, h - y)
    
    return image[y:y+height, x:x+width]


def center_crop(
    image: np.ndarray,
    crop_size: Tuple[int, int]
) -> np.ndarray:
    """
    中心裁剪图像
    
    Args:
        image: 输入图像
        crop_size: 裁剪尺寸 (width, height)
        
    Returns:
        裁剪后的图像
    """
    h, w = image.shape[:2]
    crop_w, crop_h = crop_size
    
    # 计算裁剪起点
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    
    return crop_image(image, x, y, crop_w, crop_h)


def adjust_brightness(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    调整图像亮度
    
    Args:
        image: 输入图像 (uint8)
        factor: 亮度因子
            - 1.0: 保持不变
            - >1.0: 增加亮度
            - <1.0: 降低亮度
            
    Returns:
        调整后的图像
    """
    image = image.astype(np.float32)
    adjusted = image * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_contrast(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    调整图像对比度
    
    Args:
        image: 输入图像 (uint8)
        factor: 对比度因子
            - 1.0: 保持不变
            - >1.0: 增加对比度
            - <1.0: 降低对比度
            
    Returns:
        调整后的图像
    """
    image = image.astype(np.float32)
    mean = np.mean(image)
    adjusted = mean + factor * (image - mean)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_saturation(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    调整图像饱和度
    
    Args:
        image: 输入图像 (uint8)
        factor: 饱和度因子
            - 1.0: 保持不变
            - >1.0: 增加饱和度
            - <1.0: 降低饱和度
            
    Returns:
        调整后的图像
    """
    try:
        import cv2
        
        # 转换到HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # 调整饱和度
        hsv[:, :, 1] = hsv[:, :, 1] * factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # 转回RGB
        hsv = hsv.astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
    except ImportError:
        # 无OpenCV时使用PIL
        from PIL import Image, ImageEnhance
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_image)
        adjusted = np.array(enhancer.enhance(factor))
    
    return adjusted


def adjust_sharpness(
    image: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    调整图像锐度
    
    Args:
        image: 输入图像 (uint8)
        factor: 锐度因子
            - 1.0: 保持不变
            - >1.0: 增加锐度
            - <1.0: 降低锐度
            
    Returns:
        调整后的图像
    """
    try:
        import cv2
        
        # 定义锐化核
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) * (factor - 1) / 9
        
        # 添加原图核
        center = (factor - 1) * 8 / 9
        kernel[1, 1] += center
        
        # 应用卷积
        adjusted = cv2.filter2D(image, -1, kernel)
        
    except ImportError:
        from PIL import Image, ImageEnhance
        
        pil_image = Image.fromarray(image)
        enhancer = ImageEnhance.Sharpness(pil_image)
        adjusted = np.array(enhancer.enhance(factor))
    
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def auto_white_balance(image: np.ndarray) -> np.ndarray:
    """
    自动白平衡校正（基于灰度世界假设）
    
    Args:
        image: 输入RGB图像 (uint8)
        
    Returns:
        白平衡校正后的图像
    """
    # 计算各通道平均值
    r_mean = np.mean(image[:, :, 0])
    g_mean = np.mean(image[:, :, 1])
    b_mean = np.mean(image[:, :, 2])
    
    # 计算增益
    gray_mean = (r_mean + g_mean + b_mean) / 3
    
    r_gain = gray_mean / (r_mean + 1e-8)
    g_gain = gray_mean / (g_mean + 1e-8)
    b_gain = gray_mean / (b_mean + 1e-8)
    
    # 应用增益
    balanced = image.astype(np.float32)
    balanced[:, :, 0] = np.clip(balanced[:, :, 0] * r_gain, 0, 255)
    balanced[:, :, 1] = np.clip(balanced[:, :, 1] * g_gain, 0, 255)
    balanced[:, :, 2] = np.clip(balanced[:, :, 2] * b_gain, 0, 255)
    
    return balanced.astype(np.uint8)


def clahe_enhancement(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    对比度受限自适应直方图均衡化 (CLAHE)
    
    Args:
        image: 输入图像 (uint8)
        clip_limit: 对比度裁剪限制
        tile_size: 网格大小
        
    Returns:
        增强后的图像
    """
    try:
        import cv2
        
        # 转换为灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 创建CLAHE对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        
        # 应用CLAHE
        enhanced = clahe.apply(gray)
        
        # 如果是彩色图像，处理各通道
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                result[:, :, i] = clahe.apply(image[:, :, i])
            enhanced = result
        
    except ImportError:
        # 无OpenCV时的简化实现
        from PIL import Image
        
        pil_image = Image.fromarray(image)
        if pil_image.mode != 'L':
            pil_image = pil_image.convert('L')
        
        from PIL import ImageOps
        enhanced = np.array(ImageOps.equalize(pil_image))
        
        if len(image.shape) == 3:
            enhanced = np.stack([enhanced] * 3, axis=-1)
    
    return enhanced


def blend_images(
    image1: np.ndarray,
    image2: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    混合两张图像
    
    Args:
        image1: 第一张图像
        image2: 第二张图像
        alpha: 混合因子 (image1的权重)
        
    Returns:
        混合后的图像
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    
    blended = alpha * image1 + (1 - alpha) * image2
    return np.clip(blended, 0, 255).astype(np.uint8)


def compute_histogram(
    image: np.ndarray,
    bins: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算图像直方图
    
    Args:
        image: 输入图像
        bins: 直方图 bins 数量
        
    Returns:
        (直方图数组, bin边界数组)
    """
    if len(image.shape) == 3:
        gray = rgb_to_gray(image)
    else:
        gray = image
    
    hist, bin_edges = np.histogram(gray.flatten(), bins=bins, range=(0, 256))
    
    return hist, bin_edges


def create_thumbnail(
    image: np.ndarray,
    max_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    创建图像缩略图
    
    Args:
        image: 输入图像
        max_size: 最大尺寸
        
    Returns:
        缩略图
    """
    h, w = image.shape[:2]
    max_w, max_h = max_size
    
    # 计算缩放比例
    scale = min(max_w / w, max_h / h)
    
    if scale >= 1:
        return image
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return resize_image(image, (new_w, new_h))


def stack_images_horizontal(
    images: List[np.ndarray],
    spacing: int = 5,
    background: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    水平拼接多张图像
    
    Args:
        images: 图像列表
        spacing: 图像间距
        background: 背景颜色
        
    Returns:
        拼接后的图像
    """
    if not images:
        return np.array([])
    
    # 确保高度一致
    heights = [img.shape[0] for img in images]
    max_height = max(heights)
    
    # 调整每张图像高度并垂直居中
    processed = []
    for img in images:
        if img.shape[0] < max_height:
            # 创建相同尺寸的画布
            canvas = np.full((max_height, img.shape[1], 3), background, dtype=np.uint8)
            y_offset = (max_height - img.shape[0]) // 2
            canvas[y_offset:y_offset+img.shape[0]] = img
            processed.append(canvas)
        else:
            processed.append(img)
    
    # 水平拼接
    total_width = sum(img.shape[1] for img in processed) + spacing * (len(processed) - 1)
    result = np.full((max_height, total_width, 3), background, dtype=np.uint8)
    
    x_offset = 0
    for img in processed:
        result[:, x_offset:x_offset+img.shape[1]] = img
        x_offset += img.shape[1] + spacing
    
    return result


def stack_images_vertical(
    images: List[np.ndarray],
    spacing: int = 5,
    background: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    垂直拼接多张图像
    
    Args:
        images: 图像列表
        spacing: 图像间距
        background: 背景颜色
        
    Returns:
        拼接后的图像
    """
    if not images:
        return np.array([])
    
    # 确保宽度一致
    widths = [img.shape[1] for img in images]
    max_width = max(widths)
    
    # 调整每张图像宽度并水平居中
    processed = []
    for img in images:
        if img.shape[1] < max_width:
            canvas = np.full((img.shape[0], max_width, 3), background, dtype=np.uint8)
            x_offset = (max_width - img.shape[1]) // 2
            canvas[:, x_offset:x_offset+img.shape[1]] = img
            processed.append(canvas)
        else:
            processed.append(img)
    
    # 垂直拼接
    total_height = sum(img.shape[0] for img in processed) + spacing * (len(processed) - 1)
    result = np.full((total_height, max_width, 3), background, dtype=np.uint8)
    
    y_offset = 0
    for img in processed:
        result[y_offset:y_offset+img.shape[0]] = img
        y_offset += img.shape[0] + spacing
    
    return result
