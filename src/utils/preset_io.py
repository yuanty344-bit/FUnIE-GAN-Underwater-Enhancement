"""
滤镜预设导入/导出工具

JSON 格式读写滤镜预设。
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


def get_builtin_presets_path() -> str:
    """获取内置预设文件路径"""
    base = Path(__file__).resolve().parents[2] / "resources" / "presets"
    return str(base / "defaults.json")


def load_presets(preset_path: str) -> List[Dict]:
    """从 JSON 文件加载预设"""
    try:
        with open(preset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "presets" in data:
            return data["presets"]
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"预设加载失败: {e}")
        return []


def save_presets(presets: List[Dict], preset_path: str) -> bool:
    """保存预设到 JSON 文件"""
    try:
        Path(preset_path).parent.mkdir(parents=True, exist_ok=True)
        with open(preset_path, "w", encoding="utf-8") as f:
            json.dump({"presets": presets}, f, indent=2, ensure_ascii=False)
        logger.info(f"预设已保存: {preset_path}")
        return True
    except Exception as e:
        logger.error(f"预设保存失败: {e}")
        return False


def load_builtin_presets() -> List[Dict]:
    """加载内置预设（优先从 JSON 文件加载，回退到硬编码）"""
    path = get_builtin_presets_path()
    presets = load_presets(path)
    if presets:
        return presets
    return [
        {"name": "original", "brightness": 0.0, "contrast": 1.0, "saturation": 1.0,
         "sharpness": 1.0, "wb": 0.0, "gamma": 1.0},
        {"name": "auto", "brightness": 0.1, "contrast": 1.1, "saturation": 1.1,
         "sharpness": 1.0, "wb": 0.0, "gamma": 1.0},
        {"name": "vivid", "brightness": 0.1, "contrast": 1.3, "saturation": 1.4,
         "sharpness": 1.2, "wb": 0.0, "gamma": 1.1},
        {"name": "soft", "brightness": 0.05, "contrast": 0.9, "saturation": 0.9,
         "sharpness": 0.8, "wb": 0.0, "gamma": 0.95},
    ]


def get_user_presets_path() -> Path:
    """获取用户预设文件路径"""
    return Path.home() / ".funie_gan" / "user_presets.json"


def load_user_presets() -> List[Dict]:
    """加载用户自定义预设"""
    path = get_user_presets_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return load_presets(str(path))


def save_user_presets(presets: List[Dict]) -> bool:
    """保存用户自定义预设"""
    path = get_user_presets_path()
    return save_presets(presets, str(path))


def export_preset(preset: Dict, output_path: str) -> bool:
    """导出单个预设为 JSON 文件"""
    return save_presets([preset], output_path)


def import_preset(input_path: str) -> Optional[Dict]:
    """从 JSON 文件导入单个预设"""
    presets = load_presets(input_path)
    return presets[0] if presets else None
