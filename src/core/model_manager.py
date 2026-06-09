"""
模型管理器

管理多个模型的扫描、加载、切换和信息查询。
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable, Dict

logger = logging.getLogger(__name__)


class ModelInfo:
    """模型信息"""
    def __init__(self, name: str, path: str, size_mb: float = 0):
        self.name = name
        self.path = path
        self.size_mb = size_mb


class ModelManager:
    """多模型管理器"""

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self._models: List[ModelInfo] = []
        self._current_index: int = -1
        self._on_switch: Optional[Callable] = None
        self.scan()

    def scan(self) -> List[ModelInfo]:
        """扫描模型目录，返回模型信息列表"""
        self._models = []
        if not self.models_dir.exists():
            return self._models
        for p in sorted(self.models_dir.glob("*.pth")):
            info = ModelInfo(
                name=p.stem,
                path=str(p),
                size_mb=p.stat().st_size / (1024 * 1024),
            )
            self._models.append(info)
        if self._models and self._current_index < 0:
            self._current_index = 0
        return self._models

    @property
    def models(self) -> List[ModelInfo]:
        if not self._models:
            self.scan()
        return self._models

    @property
    def current(self) -> Optional[ModelInfo]:
        if 0 <= self._current_index < len(self._models):
            return self._models[self._current_index]
        return None

    @property
    def current_index(self) -> int:
        return self._current_index

    def get_path(self) -> Optional[str]:
        """获取当前模型路径"""
        c = self.current
        return c.path if c else None

    def get_name(self) -> str:
        """获取当前模型名称"""
        c = self.current
        return c.name if c else ""

    def get_model_names(self) -> List[str]:
        """获取所有模型名称列表"""
        return [m.name for m in self._models]

    def set_on_switch(self, callback: Callable) -> None:
        """设置模型切换回调"""
        self._on_switch = callback

    def switch_to(self, index: int) -> bool:
        """切换到指定索引的模型"""
        if 0 <= index < len(self._models):
            self._current_index = index
            logger.info(f"模型切换至: {self._models[index].name}")
            if self._on_switch:
                self._on_switch(self._models[index].path)
            return True
        return False

    def switch_to_name(self, name: str) -> bool:
        """按名称切换模型"""
        for i, m in enumerate(self._models):
            if m.name == name:
                return self.switch_to(i)
        return False

    def reload_enhancer(self, enhancer, model_path: str) -> bool:
        """重新加载增强器模型"""
        try:
            enhancer.model_wrapper.unload()
            enhancer.model_wrapper.model_path = Path(model_path)
            enhancer.load_model()
            logger.info(f"增强器已重新加载模型: {model_path}")
            return True
        except Exception as e:
            logger.error(f"重新加载模型失败: {e}")
            return False

    def to_dict(self) -> List[Dict]:
        """导出为字典列表"""
        return [{"name": m.name, "path": m.path, "size_mb": m.size_mb}
                for m in self._models]
