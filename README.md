# 水下图像增强软件 v1.0.0

基于 FUnIE-GAN (PyTorch) 深度学习模型的水下图像增强桌面工具。

---

## 功能特性

**核心功能**
- 单图像增强：打开水下图片，一键 AI 增强，支持四种增强强度（自动/标准/轻度/强力）
- 批量处理：选择多张图片或整个文件夹，自动存入 `output/batch_YYYYMMDD_HHMMSS/` 子文件夹
- 视频增强：逐帧处理水下视频，输出增强后的视频文件
- 实时对比：左右分屏滑动对比增强前后效果

**手动调节**
- 亮度、对比度、饱和度、锐度、白平衡滑块实时调节
- 四种预设一键切换：原始 / 自动增强 / 鲜艳 / 柔和
- 所有调节在增强结果基础上叠加，所见即所得

**质量评估**
- UIQM（水下图像质量指标）
- SSIM（结构相似性）+ PSNR（峰值信噪比）
- MAE（平均绝对误差）+ UCIQE

**界面特性**
- PyQt5 现代化图形界面，中文菜单
- 可拖放图片到窗口直接打开
- 鼠标滚轮缩放、拖拽平移
- 滤镜面板 / 属性面板可显示/隐藏
- 最近文件列表，保存窗口布局记忆

---

## 安装说明

### 系统要求

- Python 3.8+
- Windows / macOS / Linux
- 推荐 8GB+ RAM，NVIDIA GPU 可选

### 安装步骤

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

# 2. 安装依赖
pip install -r requirements.txt
```

预训练模型已内置在 `3rdparty/FUnIE-GAN/PyTorch/models/funie_generator.pth`，无需额外下载。

```bash
# 3. 启动
python main.py
```

---

## 使用方法

### 图形界面

| 操作 | 方式 |
|------|------|
| 打开图片 | 文件 → 打开图像 / Ctrl+O / 拖拽图片到窗口 |
| AI 增强 | 工具栏「增强」按钮 / Ctrl+E |
| 切换增强强度 | 工具栏下拉框 或 增强菜单 → 增强模式 |
| 手动微调 | 左侧滤镜面板拖动滑块 |
| 对比原图 | 视图 → 对比模式 |
| 保存结果 | Ctrl+S（存到 output/） 或 另存为 |
| 批量处理 | 工具 → 批量处理 → 选多张图片或文件夹 |
| 视频处理 | 工具 → 视频处理 → 选择视频文件 |

### 命令行

```bash
python main.py -i image.jpg                    # 打开指定图片
python main.py --batch ./input_images/         # 批量处理文件夹
python main.py --video underwater.mp4          # 处理视频
python main.py -o ./my_output/                 # 指定输出目录
```

---

## 项目结构

```
FUnIE-GAN-master/
├── main.py                         # 程序入口
├── requirements.txt                # Python 依赖
├── config.json                     # 应用配置文件
│
├── src/                            # 核心源码
│   ├── core/
│   │   ├── funie_wrapper.py        # FUnIE-GAN PyTorch 模型封装
│   │   ├── enhancer.py             # 图像增强管线（预处理→推理→后处理）
│   │   └── metrics.py              # 质量评估（UIQM/SSIM/PSNR/MAE/UCIQE）
│   │
│   ├── gui/
│   │   ├── main_window.py          # 主窗口（菜单、工具栏、信号连接）
│   │   ├── image_viewer.py         # 图像显示组件（缩放、拖拽、拖放）
│   │   ├── compare_panel.py        # 增强前后滑动对比面板
│   │   ├── settings_dialog.py      # 设置对话框（模型/处理/界面/批量/视频）
│   │   └── icons.py                # 应用图标（QPainter 矢量绘制）
│   │
│   ├── processors/
│   │   ├── batch_processor.py      # 批量处理
│   │   ├── video_processor.py      # 视频逐帧处理
│   │   └── adjust_filters.py       # 滤镜调节（亮度/对比度/饱和度/锐度/白平衡/Gamma）
│   │
│   └── utils/
│       ├── config.py               # JSON 配置管理器
│       ├── file_io.py              # 图像/视频文件读写
│       └── image_utils.py          # 图像处理工具函数
│
├── 3rdparty/FUnIE-GAN/             # 原始 FUnIE-GAN 代码（仅保留 PyTorch 部分）
│   └── PyTorch/
│       ├── nets/funiegan.py        # GeneratorFunieGAN 模型定义
│       ├── nets/commons.py         # 网络公共组件
│       └── models/
│           └── funie_generator.pth # 预训练生成器权重（~28 MB）
│
├── models/                         # 模型目录（可放置额外模型文件）
├── output/                         # 增强结果输出目录
├── logs/                           # 运行日志
└── temp/                           # 临时文件
```

---

## 配置文件

`config.json` 结构：

```json
{
  "model": {
    "path": "./models",
    "device": "auto",
    "batch_size": 1,
    "framework": "pytorch",
    "precision": "float32"
  },
  "processing": {
    "output_format": "png",
    "quality": 95,
    "preserve_metadata": true,
    "auto_enhance": false,
    "default_mode": "auto"
  },
  "ui": {
    "theme": "fusion",
    "language": "zh_CN",
    "show_toolbar": true,
    "show_statusbar": true,
    "recent_files": [],
    "max_recent_files": 10
  },
  "filters": {
    "brightness": 0.0,
    "contrast": 0.0,
    "saturation": 0.0,
    "sharpness": 0.0,
    "white_balance": 0.0,
    "gamma": 1.0
  },
  "batch": {
    "parallel": true,
    "max_workers": 4,
    "skip_existing": true,
    "output_subdir": "enhanced"
  },
  "video": {
    "output_fps": null,
    "output_codec": "mp4v",
    "frame_interval": 1,
    "output_format": "mp4"
  },
  "metrics": {
    "compute_uiqm": true,
    "compute_uciqe": true,
    "compute_ssim": true,
    "compute_psnr": true
  }
}
```

---

## FUnIE-GAN 模型说明

本软件使用 **GeneratorFunieGAN**：一个 5 层全卷积 UNet 生成器，32× 下采样，输入自动补齐到 32 的倍数。

- 预处理：保持宽高比缩放到 ≤1024px，反射填充到 32 倍数，归一化到 [-1, 1]
- 后处理：裁剪填充区域，Lanczos 缩放回原始尺寸，裁剪到 [0, 255]
- 推理框架：PyTorch
- 模型来源：[xahidbuffon/FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN)

---

## 常见问题

**程序启动失败？**
- 检查 Python ≥ 3.8
- `pip install -r requirements.txt` 确保依赖完整
- PyQt5 需单独确认安装成功

**模型加载失败？**
- 确认 `3rdparty/FUnIE-GAN/PyTorch/models/funie_generator.pth` 存在
- 检查 PyTorch 是否正确安装

**增强效果不好？**
- 尝试切换增强模式（自动/标准/轻度/强力）
- 使用左侧滤镜面板手动微调白平衡和对比度
- 水下图片颜色偏差大时，适当调整白平衡滑块

**GPU 不可用？**
- 程序自动回退到 CPU，无需手动干预

---

## 许可证

本项目基于 MIT 许可证开源。

FUnIE-GAN 模型版权归原作者 [xahidbuffon](https://github.com/xahidbuffon) 所有。

## 致谢

- [FUnIE-GAN](https://github.com/xahidbuffon/FUnIE-GAN) — 核心增强算法
- PyQt5 — GUI 框架
- OpenCV — 图像与视频处理
- PyTorch — 深度学习推理
