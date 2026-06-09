"""
文件读写工具模块

提供图像和视频文件的读写功能，
支持多种格式的自动检测和转换。
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
import json

import numpy as np

logger = logging.getLogger(__name__)


# 支持的图像格式
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.jpe', '.jfif',
    '.png', '.bmp', '.dib',
    '.tif', '.tiff', '.webp',
    '.gif', '.tga', '.ras'
}

# 支持的视频格式
VIDEO_EXTENSIONS = {
    '.avi', '.mp4', '.m4v', '.mkv',
    '.mov', '.wmv', '.flv', '.webm'
}


def is_image_file(path: Union[str, Path]) -> bool:
    """
    检查文件是否为支持的图像格式
    
    Args:
        path: 文件路径
        
    Returns:
        是否为图像文件
    """
    path = Path(path)
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Union[str, Path]) -> bool:
    """
    检查文件是否为支持的视频格式
    
    Args:
        path: 文件路径
        
    Returns:
        是否为视频文件
    """
    path = Path(path)
    return path.suffix.lower() in VIDEO_EXTENSIONS


def get_image_files(
    directory: Union[str, Path],
    recursive: bool = False,
    extensions: Optional[set] = None
) -> List[Path]:
    """
    获取目录中的所有图像文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归搜索子目录
        extensions: 指定的扩展名集合，None则使用默认
        
    Returns:
        图像文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"目录不存在: {directory}")
        return []
    
    if not directory.is_dir():
        if is_image_file(directory):
            return [directory]
        return []
    
    if extensions is None:
        extensions = IMAGE_EXTENSIONS
    
    image_files = []
    
    if recursive:
        # 递归搜索
        for ext in extensions:
            image_files.extend(directory.rglob(f"*{ext}"))
    else:
        # 仅搜索当前目录
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
    
    # 排序保证一致性
    image_files.sort()
    
    return image_files


def get_video_files(
    directory: Union[str, Path],
    recursive: bool = False
) -> List[Path]:
    """
    获取目录中的所有视频文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归搜索子目录
        
    Returns:
        视频文件路径列表
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    video_files = []
    
    if recursive:
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in VIDEO_EXTENSIONS:
            video_files.extend(directory.glob(f"*{ext}"))
    
    video_files.sort()
    return video_files


def read_image(
    path: Union[str, Path],
    mode: str = 'RGB'
) -> np.ndarray:
    """
    读取图像文件
    
    Args:
        path: 图像文件路径
        mode: 读取模式
            - 'RGB': 3通道RGB
            - 'RGBA': 4通道RGBA
            - 'L': 灰度图
            - '1': 8位灰度
            - 'BGR': OpenCV BGR格式
            
    Returns:
        numpy数组，shape为(H, W, C)
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的图像格式
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"图像文件不存在: {path}")
    
    if not is_image_file(path):
        raise ValueError(f"不支持的图像格式: {path.suffix}")
    
    try:
        import cv2
        
        # 使用OpenCV读取
        if mode == 'BGR':
            image = cv2.imread(str(path))
        elif mode == 'RGB':
            image = cv2.imread(str(path))
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif mode == 'RGBA':
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None and len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            elif image is not None and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        elif mode == 'L' or mode == '1':
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(str(path))
            
    except ImportError:
        # 无OpenCV时使用PIL
        from PIL import Image
        pil_image = Image.open(str(path))
        
        if mode == 'RGB':
            pil_image = pil_image.convert('RGB')
        elif mode == 'RGBA':
            pil_image = pil_image.convert('RGBA')
        elif mode == 'L' or mode == '1':
            pil_image = pil_image.convert('L')
        
        image = np.array(pil_image)
    
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    
    logger.debug(f"读取图像: {path}, shape={image.shape}")
    return image


def write_image(
    image: np.ndarray,
    path: Union[str, Path],
    format: Optional[str] = None,
    quality: int = 95
) -> bool:
    """
    保存图像到文件
    
    Args:
        image: 图像数组，范围[0, 255]，dtype为uint8
        path: 输出文件路径
        format: 图像格式，None则根据扩展名自动判断
        quality: JPEG质量 (1-100)
        
    Returns:
        保存是否成功
    """
    path = Path(path)
    
    # 确保输出目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 规范化图像格式
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # 确定格式
    if format is None:
        format = path.suffix.lstrip('.').upper()
        if format == 'JPG':
            format = 'JPEG'
    
    success = False
    
    try:
        import cv2
        
        # 使用OpenCV保存
        if format == 'JPEG':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_params)
        elif format == 'PNG':
            compress = int((100 - quality) * 9 / 100)  # 转换为PNG压缩级别
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, compress]
            success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_params)
        elif format == 'WEBP':
            encode_params = [cv2.IMWRITE_WEBP_QUALITY, quality]
            success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), encode_params)
        else:
            success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
    except ImportError:
        # 无OpenCV时使用PIL
        from PIL import Image
        
        pil_image = Image.fromarray(image)
        
        if format == 'JPEG':
            # JPEG不支持透明通道
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                # 白色背景
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                background.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = background
            pil_image.save(str(path), format='JPEG', quality=quality)
            success = True
        elif format == 'PNG':
            pil_image.save(str(path), format='PNG')
            success = True
        else:
            pil_image.save(str(path))
            success = True
    
    if success:
        logger.debug(f"保存图像: {path}")
    else:
        logger.error(f"保存图像失败: {path}")
    
    return success


def read_json(
    path: Union[str, Path]
) -> dict:
    """
    读取JSON文件
    
    Args:
        path: JSON文件路径
        
    Returns:
        解析后的字典
    """
    path = Path(path)
    
    if not path.exists():
        logger.warning(f"JSON文件不存在: {path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def write_json(
    data: dict,
    path: Union[str, Path],
    indent: int = 4
) -> bool:
    """
    写入JSON文件
    
    Args:
        data: 要保存的字典
        path: 输出文件路径
        indent: 缩进空格数
        
    Returns:
        保存是否成功
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.debug(f"保存JSON: {path}")
        return True
    except Exception as e:
        logger.error(f"保存JSON失败: {e}")
        return False


def ensure_directories_exist(directories: List[Union[str, Path]]) -> None:
    """
    确保指定的目录都存在，不存在则创建
    
    Args:
        directories: 目录路径列表
    """
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"创建目录: {directory}")


def get_file_info(path: Union[str, Path]) -> Dict[str, any]:
    """
    获取文件信息
    
    Args:
        path: 文件路径
        
    Returns:
        包含文件信息的字典
    """
    path = Path(path)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    info = {
        'name': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size': stat.st_size,
        'size_mb': stat.st_size / (1024 * 1024),
        'modified': stat.st_mtime,
        'is_image': is_image_file(path),
        'is_video': is_video_file(path)
    }
    
    return info


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    复制文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        
    Returns:
        复制是否成功
    """
    import shutil
    
    src = Path(src)
    dst = Path(dst)
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.debug(f"复制文件: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"复制文件失败: {e}")
        return False


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    移动文件
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        
    Returns:
        移动是否成功
    """
    import shutil
    
    src = Path(src)
    dst = Path(dst)
    
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        logger.debug(f"移动文件: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"移动文件失败: {e}")
        return False


def get_available_formats() -> Dict[str, List[str]]:
    """
    获取当前环境支持的图像和视频格式
    
    Returns:
        格式信息字典
    """
    formats = {
        'images': [],
        'videos': []
    }
    
    # 尝试导入各库检查支持格式
    try:
        import cv2
        formats['images'] = [fmt.decode() if isinstance(fmt, bytes) else fmt 
                           for fmt in cv2.imread_readDB2() if fmt]
        formats['videos'] = [fmt.decode() if isinstance(fmt, bytes) else fmt 
                           for fmt in cv2.videoio_registry.getBackends() if fmt]
    except ImportError:
        pass
    
    # PIL支持格式
    try:
        from PIL import Image
        pil_formats = list(Image.registered_extensions().keys())
        formats['images'] = list(set(formats['images'] + pil_formats))
    except ImportError:
        pass
    
    return formats
