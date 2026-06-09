"""
配置管理模块

提供JSON格式的配置文件读写和管理功能，
支持配置的加载、保存和热更新。
"""

import os
import logging
import json
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import copy

from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """模型配置数据类"""
    path: str = './models'
    device: str = 'auto'  # auto, cpu, gpu
    batch_size: int = 1
    framework: str = 'pytorch'  # 改成 pytorch 默认
    precision: str = 'float32'  # float32, float16
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessingConfig:
    """处理配置数据类"""
    output_format: str = 'png'
    quality: int = 95
    preserve_metadata: bool = True
    auto_enhance: bool = False
    default_mode: str = 'auto'
    
    def to_dict(self) -> dict:  # ✅ 加上这个方法
        return asdict(self)


@dataclass
class UIConfig:
    """界面配置数据类"""
    theme: str = 'fusion'  # fusion, windows, windowsvista
    language: str = 'zh_CN'
    show_toolbar: bool = True
    show_statusbar: bool = True
    recent_files: List[str] = field(default_factory=list)
    max_recent_files: int = 10
    window_geometry: Optional[str] = None
    window_state: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FilterConfig:
    """滤镜配置数据类"""
    brightness: float = 0.0  # -100 to 100
    contrast: float = 0.0
    saturation: float = 0.0
    sharpness: float = 0.0
    white_balance: float = 0.0
    gamma: float = 1.0
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchConfig:
    """批量处理配置数据类"""
    parallel: bool = True
    max_workers: int = 4
    skip_existing: bool = True
    output_subdir: str = 'enhanced'
    save_metrics: bool = True
    
    def to_dict(self) -> dict:  # ✅ 加上这个方法
        return asdict(self)


@dataclass
class VideoConfig:
    """视频处理配置数据类"""
    output_fps: Optional[float] = None  # None表示保持原帧率
    output_codec: str = 'mp4v'
    frame_interval: int = 1  # 每隔n帧处理一次
    output_format: str = 'mp4'
    
    def to_dict(self) -> dict:  # ✅ 加上这个方法
        return asdict(self)


@dataclass
class MetricsConfig:
    """评估指标配置数据类"""
    compute_uiqm: bool = True
    compute_uciqe: bool = True
    compute_ssim: bool = True
    compute_psnr: bool = True
    save_metrics: bool = False
    
    def to_dict(self) -> dict:  # ✅ 加上这个方法
        return asdict(self)


class ConfigManager(QObject):
    """
    配置管理器

    提供配置的加载、保存、合并和访问功能。

    Attributes:
        config_path: 配置文件路径
        config: 当前配置字典

    Example:
        >>> config = ConfigManager('./config.json')
        >>> config.load()
        >>> config.set('model.device', 'gpu')
        >>> config.save()
    """

    config_changed = pyqtSignal(str, object)

    # 默认配置
    DEFAULT_CONFIG = {
        'version': '1.0.0',
        'last_modified': None,
        'model': ModelConfig().to_dict(),
        'processing': ProcessingConfig().to_dict(),
        'ui': UIConfig().to_dict(),
        'filters': FilterConfig().to_dict(),
        'batch': BatchConfig().to_dict(),
        'video': VideoConfig().to_dict(),
        'metrics': MetricsConfig().to_dict()
    }
    
    def __init__(
        self,
        config_path: Union[str, Path] = './config.json',
        auto_load: bool = False
    ):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
            auto_load: 是否自动加载配置
        """
        QObject.__init__(self)
        self.config_path = Path(config_path) if config_path else None
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._backup = None

        if auto_load and self.config_path and self.config_path.exists():
            self.load()
    
    def load(self) -> bool:
        """
        从文件加载配置
        
        Returns:
            加载是否成功
        """
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"配置文件不存在，使用默认配置: {self.config_path}")
            return False
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # 深度合并配置
            self.config = self._deep_merge(self.DEFAULT_CONFIG, loaded_config)
            
            # 验证版本兼容性
            self._migrate_config()
            
            logger.info(f"已加载配置: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            return False
    
    def save(self, path: Optional[Union[str, Path]] = None) -> bool:
        """
        保存配置到文件
        
        Args:
            path: 输出路径，None则使用config_path
            
        Returns:
            保存是否成功
        """
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            logger.error("未指定保存路径")
            return False
        
        # 更新时间戳
        self.config['last_modified'] = datetime.now().isoformat()
        
        try:
            # 确保目录存在
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"已保存配置: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"配置保存失败: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔，如 'model.device'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, auto_save: bool = False) -> bool:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔，如 'model.device'
            value: 配置值
            auto_save: 是否自动保存
            
        Returns:
            设置是否成功
        """
        keys = key.split('.')
        config = self.config
        
        try:
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value

            if auto_save:
                self.save()

            logger.debug(f"设置配置: {key} = {value}")
            self.config_changed.emit(key, value)
            return True
            
        except Exception as e:
            logger.error(f"设置配置失败: {e}")
            return False
    
    def reset(self) -> None:
        """重置为默认配置"""
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        logger.info("已重置为默认配置")
    
    def clear_recent_files(self) -> None:
        """清空最近文件列表"""
        self.set('ui.recent_files', [])
        self.save()
        logger.info("已清空最近文件列表")

    def add_recent_file(self, file_path: str) -> None:
        """
        添加文件到最近文件列表

        Args:
            file_path: 文件路径
        """
        recent_files = self.get('ui.recent_files', [])
        if not isinstance(recent_files, list):
            recent_files = []
        # 去重：如果已在列表中则移除
        recent_files = [f for f in recent_files if f != file_path]
        # 插入到最前面
        recent_files.insert(0, file_path)
        # 限制数量
        max_files = self.get('ui.max_recent_files', 10)
        recent_files = recent_files[:max_files]
        self.set('ui.recent_files', recent_files)

    def backup(self) -> None:
        """创建当前配置备份"""
        self._backup = copy.deepcopy(self.config)
        logger.info("已创建配置备份")
    
    def restore(self) -> bool:
        """恢复到上一个备份"""
        if self._backup is None:
            logger.warning("没有可用的备份")
            return False
        
        self.config = copy.deepcopy(self._backup)
        logger.info("已恢复配置备份")
        return True
    
    def _deep_merge(self, base: dict, update: dict) -> dict:
        """
        深度合并两个字典
        
        Args:
            base: 基础字典
            update: 更新字典
            
        Returns:
            合并后的字典
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _migrate_config(self) -> None:
        """配置版本迁移，处理兼容性"""
        version = self.config.get('version', '0.0.0')
        
        # 这里可以添加版本迁移逻辑
        if version < '1.0.0':
            logger.info(f"迁移配置从版本 {version} 到 1.0.0")
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return f"ConfigManager(path='{self.config_path}', version={self.config.get('version')})"


# 全局配置实例
_global_config = None

def get_config(config_path: Optional[str] = None, auto_load: bool = True) -> ConfigManager:
    """
    获取全局配置管理器单例
    
    Args:
        config_path: 配置文件路径
        auto_load: 是否自动加载
        
    Returns:
        ConfigManager 实例
    """
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager(config_path, auto_load=auto_load)
    
    return _global_config