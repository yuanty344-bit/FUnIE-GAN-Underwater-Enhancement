"""
FUnIE-GAN 模型封装接口 - 已对接真实代码
类名: GeneratorFunieGAN
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, List
from enum import Enum
import json

import numpy as np

logger = logging.getLogger(__name__)


class ModelFramework(Enum):
    TENSORFLOW = 'tensorflow'
    PYTORCH = 'pytorch'
    UNKNOWN = 'unknown'


class ModelStatus(Enum):
    UNLOADED = 'unloaded'
    LOADING = 'loading'
    LOADED = 'loaded'
    ERROR = 'error'


class FunieWrapper:
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        framework: str = 'pytorch',
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.model_path = Path(model_path) if model_path else None
        self.framework = ModelFramework(framework.lower())
        self.device = self._setup_device(device)
        self.status = ModelStatus.UNLOADED
        self.config = {
            'input_shape': (256, 256, 3),
            'normalization': {'mode': 'tanh', 'input_scale': 127.5},
            'preprocessing': {'resize': True, 'normalize': True},
            'postprocessing': {'clip': True, 'denormalize': True}
        }
        self._model = None
        
        logger.info(f"FunieWrapper初始化完成: framework={self.framework.value}, device={self.device}")
    
    def _setup_device(self, device: Optional[str] = None) -> str:
        if device:
            return device
        try:
            import torch
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            return 'cpu'
    
    def load(self) -> bool:
        if self.status == ModelStatus.LOADED:
            return True
        
        self.status = ModelStatus.LOADING
        logger.info("正在加载模型...")
        
        try:
            self._load_pytorch_model()
            self.status = ModelStatus.LOADED
            logger.info("模型加载成功！")
            return True
        except Exception as e:
            self.status = ModelStatus.ERROR
            logger.error(f"模型加载失败: {e}")
            print(f"错误详情: {e}")
            return False
    
    def _load_pytorch_model(self) -> None:
        import torch
        import sys
        
        # 1. 添加路径
        project_root = Path(__file__).parent.parent.parent
        funie_pytorch_path = project_root / '3rdparty' / 'FUnIE-GAN' / 'PyTorch'
        
        if not funie_pytorch_path.exists():
            raise FileNotFoundError(f"找不到目录: {funie_pytorch_path}")
        
        sys.path.insert(0, str(funie_pytorch_path))
        print(f"已添加路径: {funie_pytorch_path}")
        
        # 2. 导入正确的类名：GeneratorFunieGAN
        from nets.funiegan import GeneratorFunieGAN
        print("✓ 成功导入 GeneratorFunieGAN")
        
        # 3. 设备
        device = torch.device(self.device)
        
        # 4. 初始化模型（看原始代码的构造函数）
        self._model = GeneratorFunieGAN()
        print("✓ 模型初始化成功")
        
        # 5. 加载权重
        default_model_path = funie_pytorch_path / 'models' / 'funie_generator.pth'
        if self.model_path and self.model_path.exists():
            state_dict = torch.load(str(self.model_path), map_location=device)
            self._model.load_state_dict(state_dict)
            print(f"✓ 加载权重: {self.model_path}")
        elif default_model_path.exists():
            state_dict = torch.load(str(default_model_path), map_location=device)
            self._model.load_state_dict(state_dict)
            self.model_path = default_model_path
            print(f"✓ 加载权重: {default_model_path}")
        else:
            print(f"⚠ 未找到权重文件: {default_model_path}")
            print("使用随机初始化（也能运行，只是效果不好）")
        
        # 6. 移到设备，评估模式
        self._model = self._model.to(device)
        self._model.eval()
        print(f"✓ 模型已移到 {device} 并设为评估模式")
    
    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self.status = ModelStatus.UNLOADED
    
    def predict(self, image: np.ndarray, return_original: bool = False):
        if self.status != ModelStatus.LOADED:
            raise RuntimeError("模型未加载！")
        
        original_shape = image.shape
        
        # 预处理：resize + 归一化到 [-1, 1]
        processed = self._preprocess(image)
        
        # 推理
        output = self._predict_pytorch(processed)
        
        # 后处理：反归一化 + resize回去
        enhanced = self._postprocess(output, original_shape)
        
        if return_original:
            return enhanced, image
        return enhanced
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        import cv2
        
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        
        # pad to multiple of 32, preserve aspect ratio up to max_size
        if self.config['preprocessing']['resize']:
            h, w = image.shape[:2]
            max_size = 1024
            scale = min(max_size / max(h, w), 1.0)
            if scale < 1.0:
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                h, w = new_h, new_w

            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32
            if pad_h or pad_w:
                image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                           cv2.BORDER_REFLECT)

            self._preprocess_pad = (h, w)
        
        # 归一化到 [-1, 1]
        if self.config['preprocessing']['normalize']:
            image = image.astype(np.float32)
            image = (image - 127.5) / 127.5
        
        return image
    
    def _postprocess(self, output: np.ndarray, target_shape: Tuple) -> np.ndarray:
        import cv2
        
        # 反归一化到 [0, 255]
        output = output * 127.5 + 127.5
        
        # crop back to original size, then resize if needed
        h, w = target_shape[:2]
        oh, ow = output.shape[:2]
        pad_h, pad_w = getattr(self, '_preprocess_pad', (oh, ow))
        if oh > pad_h or ow > pad_w:
            output = output[:pad_h, :pad_w]
        if output.shape[:2] != (h, w):
            output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)
        
        # 裁剪范围
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output
    
    def _predict_pytorch(self, image: np.ndarray) -> np.ndarray:
        import torch
        
        device = torch.device(self.device)
        
        # (H,W,C) -> (B,C,H,W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        tensor = tensor.to(device)
        
        # 推理
        with torch.no_grad():
            output_tensor = self._model(tensor)
        
        # (B,C,H,W) -> (H,W,C)
        output = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return output
    
    @property
    def is_loaded(self) -> bool:
        return self.status == ModelStatus.LOADED


def create_wrapper(framework='pytorch', auto_load=True):
    wrapper = FunieWrapper(framework=framework)
    if auto_load:
        wrapper.load()
    return wrapper