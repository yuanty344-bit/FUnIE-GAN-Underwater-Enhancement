"""
去雾/去模糊处理模块

提供暗通道先验去雾和简易去模糊功能。
"""
import logging
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """计算暗通道"""
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def estimate_atmosphere(image: np.ndarray, dark: np.ndarray,
                        top_percent: float = 0.001) -> np.ndarray:
    """估计大气光"""
    h, w = dark.shape
    num_pixels = int(h * w * top_percent)
    flat_dark = dark.ravel()
    flat_img = image.reshape(-1, 3)
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    atmosphere = np.max(flat_img[indices], axis=0)
    return atmosphere.astype(np.float32)


def dehaze(image: np.ndarray, omega: float = 0.95, patch_size: int = 15,
           t0: float = 0.1) -> np.ndarray:
    """
    暗通道先验去雾

    Args:
        image: RGB图像 uint8
        omega: 去雾强度 (0~1), 越大效果越强
        patch_size: 暗通道窗口大小
        t0: 透射率下限
    """
    img = image.astype(np.float32) / 255.0
    dark = dark_channel(img, patch_size)
    atmosphere = estimate_atmosphere(img, dark)

    # 透射率估计
    transmission = 1.0 - omega * dark_channel(img / atmosphere.max(), patch_size)
    transmission = np.maximum(transmission, t0)
    transmission = np.expand_dims(transmission, axis=2)

    # 恢复
    result = (img - atmosphere) / transmission + atmosphere
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def deblur(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    简易去模糊（锐化增强）

    Args:
        image: RGB图像 uint8
        strength: 锐化强度 (0~3)
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # 调整核以控制强度
    kernel[1, 1] = 4.0 + strength

    result = cv2.filter2D(image, -1, kernel)
    return np.clip(result, 0, 255).astype(np.uint8)


def remove_haze_pipeline(image: np.ndarray,
                         dehaze_strength: float = 0.95,
                         deblur_strength: float = 0.5) -> np.ndarray:
    """完整去雾+去模糊管线"""
    result = dehaze(image, omega=dehaze_strength)
    if deblur_strength > 0:
        result = deblur(result, strength=deblur_strength)
    return result
