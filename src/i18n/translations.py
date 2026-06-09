"""
翻译字典

中文UI字符串 -> 英文翻译映射表。
tr() 函数优先查此字典，找不到则降级返回中文原文。
"""

TRANSLATIONS = {
    # ========== main_window.py ==========
    # 窗口标题
    "水下图像增强软件 - Underwater Image Enhancement":
        "Underwater Image Enhancement v1.0.0",

    # 菜单栏
    "文件(&F)": "&File",
    "打开图像...": "Open Image...",
    "打开图像文件": "Open an image file",
    "打开文件夹...": "Open Folder...",
    "打开文件夹进行批量处理": "Open a folder for batch processing",
    "保存": "Save",
    "另存为...": "Save As...",
    "最近文件": "Recent Files",
    "退出": "Exit",
    "编辑(&E)": "&Edit",
    "重置图像": "Reset Image",
    "重置所有": "Reset All",
    "视图(&V)": "&View",
    "放大": "Zoom In",
    "缩小": "Zoom Out",
    "适应窗口": "Fit to Window",
    "实际大小 (100%)": "Actual Size (100%)",
    "对比模式": "Compare Mode",
    "滤镜调节面板": "Filter Panel",
    "图像属性面板": "Properties Panel",
    "增强(&A)": "En&hance",
    "自动增强": "Auto Enhance",
    "增强模式": "Enhance Mode",
    "自动": "Auto",
    "标准": "Standard",
    "轻度": "Mild",
    "强力": "Strong",
    "工具(&T)": "&Tools",
    "批量处理...": "Batch Process...",
    "视频处理...": "Video Process...",
    "设置...": "Settings...",
    "帮助(&H)": "&Help",
    "关于": "About",
    "质量指标": "Quality Metrics",
    "主工具栏": "Main Toolbar",
    "增强模式:": "Enhance Mode:",

    # 状态栏
    "就绪": "Ready",
    "  缩放:": "  Zoom:",
    "  尺寸:": "  Size:",
    "  模型:": "  Model:",
    "未加载": "Not Loaded",
    "已加载": "Loaded",
    "错误": "Error",
    "正在加载...": "Loading...",
    "加载失败": "Load Failed",
    "已加载: ": "Loaded: ",
    "已保存: ": "Saved: ",
    "已重置为原始图像": "Reset to Original",
    "正在增强...": "Enhancing...",
    "增强完成": "Enhancement Complete",
    "增强失败": "Enhancement Failed",
    "文件不存在: ": "File not found: ",

    # 停靠窗口
    "滤镜调节": "Filter Adjustments",
    "图像属性": "Image Properties",

    # 滤镜面板
    "预设": "Presets",
    "原始": "Original",
    "鲜艳": "Vivid",
    "柔和": "Soft",
    "亮度": "Brightness",
    "对比度": "Contrast",
    "饱和度": "Saturation",
    "锐度": "Sharpness",
    "白平衡": "White Balance",

    # 属性面板
    "文件信息:": "File Info:",
    "质量指标:": "Quality Metrics:",
    "文件名: ": "Filename: ",
    "路径: ": "Path: ",
    "大小: ": "Size: ",
    "尺寸: ": "Dimensions: ",
    "通道: ": "Channels: ",
    "无": "None",
    "【增强后质量指标】": "[Post-Enhancement Metrics]",

    # 最近文件
    "无最近文件": "No Recent Files",

    # 文件对话框
    "打开图像": "Open Image",
    "选择文件夹": "Select Folder",
    "保存图像": "Save Image",
    "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;所有文件 (*)":
        "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)",
    "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;所有文件 (*)":
        "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;All Files (*)",

    # 消息框
    "图像加载失败:\n": "Image load failed:\n",
    "保存失败:\n": "Save failed:\n",
    "增强失败:\n": "Enhancement failed:\n",
    "文件不存在": "File Not Found",

    # 批量处理
    "批量处理中...": "Batch Processing...",
    "批量处理": "Batch Processing",
    "完成": "Complete",
    "处理: ": "Processing: ",
    "批量处理完成\n成功: ": "Batch complete\nSuccess: ",
    "\n保存至: ": "\nSaved to: ",
    "取消": "Cancel",
    "提示": "Info",
    "文件夹中没有找到图像文件": "No image files found in the folder",
    "选择要处理的图像文件": "Select Image Files to Process",
    "或选择包含图像的文件夹": "Or Select a Folder Containing Images",
    "批量处理 (": "Batch Process (",
    " 个文件)": " files)",

    # 视频处理
    "视频处理中...": "Video Processing...",
    "视频处理": "Video Processing",
    "视频处理完成\n输出: ": "Video processing complete\nOutput: ",
    "视频处理失败：无法打开视频文件或创建输出文件。\n请检查文件格式和输出目录权限。":
        "Video processing failed: cannot open input video or create output.\n"
        "Check the file format and output directory permissions.",
    "视频处理失败:\n": "Video processing failed:\n",
    "选择视频文件": "Select Video File",
    "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)":
        "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)",

    # 关于对话框
    "<h3>水下图像增强软件 v1.0.0</h3>"
    "<p>基于FUnIE-GAN的深度学习图像增强工具</p>"
    "<p>提供多种增强模式和滤镜调节功能</p>"
    "<hr>"
    "<p>© 2024 Underwater Image Enhancement Team</p>":
        "<h3>Underwater Image Enhancement v1.0.0</h3>"
        "<p>Deep learning image enhancement based on FUnIE-GAN</p>"
        "<p>Multiple enhancement modes and adjustment filters</p>"
        "<hr>"
        "<p>© 2024 Underwater Image Enhancement Team</p>",

    # ========== settings_dialog.py ==========
    "设置": "Settings",
    "模型": "Model",
    "处理": "Processing",
    "界面": "UI",
    "批量处理": "Batch",
    "视频处理": "Video",
    "模型路径:": "Model Path:",
    "浏览...": "Browse...",
    "计算设备:": "Compute Device:",
    "深度学习框架:": "DL Framework:",
    "批处理大小:": "Batch Size:",
    "计算精度:": "Precision:",
    "检查模型": "Check Model",
    "模型状态: 未检查": "Model Status: not checked",
    "模型状态: 未指定路径": "Model Status: no path specified",
    "模型状态: 已找到 ": "Model Status: found ",
    " 个模型文件": " model file(s)",
    "模型状态: 未找到模型文件": "Model Status: no model files found",
    "选择模型目录": "Select Model Directory",
    "确认": "Confirm",
    "确定要清空最近文件列表吗？": "Clear recent file list?",
    "最近文件列表已清空": "Recent file list cleared",
    "输出格式:": "Output Format:",
    "JPEG质量:": "JPEG Quality:",
    "保留元数据:": "Preserve Metadata:",
    "加载时自动增强:": "Auto-enhance on Load:",
    "默认增强模式:": "Default Mode:",
    "界面主题:": "UI Theme:",
    "界面语言:": "UI Language:",
    "显示工具栏:": "Show Toolbar:",
    "显示状态栏:": "Show Statusbar:",
    "最近文件数量:": "Max Recent Files:",
    "清空最近文件": "Clear Recent Files",
    "启用并行处理:": "Parallel Processing:",
    "最大工作线程:": "Max Workers:",
    "跳过已处理文件:": "Skip Existing:",
    "输出子目录:": "Output Subdir:",
    "命名模板:": "Naming Template:",
    "可用变量: {name} {date} {time} {ext}": "Variables: {name} {date} {time} {ext}",
    "保存处理指标:": "Save Metrics:",
    " 张": " imgs",
    "最高质量": "Best",
    " 个": "",
    " 帧": " frames",
    "浅色 (Light)": "Light",
    "深色 (Dark)": "Dark",
    "简体中文": "Simplified Chinese",
    "输出帧率:": "Output FPS:",
    "视频编码器:": "Video Codec:",
    "处理帧间隔:": "Frame Interval:",
    "留空则保持原帧率": "Leave empty to keep original FPS",

    # ========== compare_panel.py ==========
    "原图": "Original",
    "增强": "Enhanced",
    "请先加载图像": "Please load an image first",

    # ========== image_viewer.py ==========
    "拖放图像到此处": "Drag & Drop Image Here",
    "复制图像": "Copy Image",
    "重置缩放": "Reset Zoom",
    "导出对比图": "Export Comparison",
    "复制像素信息": "Copy Pixel Info",

    # ========== main_window.py additional ==========
    "撤销": "Undo",
    "重做": "Redo",
    "去模糊": "Deblur",
    "去模糊完成": "Deblur Complete",
    "去模糊失败": "Deblur Failed",
    "预设管理...": "Presets...",
    "导出对比图...": "Export Comparison...",
    "缩略图浏览器": "Thumbnail Browser",
    "缩略图": "Thumbnails",
    "模型:": "Model:",
    "保存会话": "Save Session",
    "恢复上次会话": "Restore Session",
    "会话已保存": "Session Saved",
    "会话已恢复": "Session Restored",
    "没有可恢复的会话": "No session to restore",
    "变换已应用": "Transform Applied",
    " 个模型文件": " model file(s)",

    # ========== preset_dialog.py ==========
    "预设管理": "Preset Manager",
    "管理您的自定义滤镜预设": "Manage your custom filter presets",
    "保存当前": "Save Current",
    "应用": "Apply",
    "导出...": "Export...",
    "导入...": "Import...",
    "删除": "Delete",
    "未命名": "Unnamed",
    "没有可保存的滤镜参数": "No filter parameters to save",
    "保存预设": "Save Preset",
    "预设名称:": "Preset Name:",
    "请先选择一个预设": "Please select a preset first",
    "导出预设": "Export Preset",
    "导入预设": "Import Preset",
    "无法导入预设，文件格式不正确": "Cannot import preset: invalid file format",
    "确认删除": "Confirm Delete",
    "确定要删除预设 \"": "Delete preset \"",
    "\" 吗？": "\"?",
    "提示": "Info",

    # ========== export_dialog.py ==========
    "导出对比图": "Export Comparison",
    "拼接布局": "Layout",
    "左右对比 (推荐)": "Side by Side (Recommended)",
    "上下对比": "Stacked",
    "左右分割线": "Split View",
    "布局:": "Layout:",
    "导出格式": "Format",
    "格式:": "Format:",
    "导出...": "Export...",
    "导出": "Export",
    "请先加载并增强图像": "Please load and enhance an image first",
    "保存对比图": "Save Comparison Image",
    "对比图已导出:\n": "Comparison exported:\n",
    "导出失败": "Export Failed",
    "完成": "Complete",

    # ========== video_dialog.py ==========
    "分辨率": "Resolution",
    "帧率": "FPS",
    "总帧数": "Total Frames",
    "时长": "Duration",
    "无法读取视频信息": "Cannot read video info",
    "处理范围": "Range",
    "起始时间:": "Start:",
    "结束时间:": "End:",
    "预览起始帧": "Preview Start",
    "预览结束帧": "Preview End",
    "点击预览按钮查看帧": "Click preview to view frame",
    "开始处理": "Start Processing",
    "无法提取该帧": "Cannot extract frame",
    "处理": "Process",
    "预览": "Preview",

    # ========== histogram_widget.py ==========
    "增强前": "Before",
    "增强后": "After",
    "无直方图数据": "No Histogram Data",
    "RGB直方图:": "RGB Histogram:",


    # ========== main.py ==========
    "水下图像增强软件": "Underwater Image Enhancement",
    "程序启动失败": "Program Start Failed",
    "发生严重错误，程序无法启动:\n\n": "A critical error occurred:\n\n",
    "\n\n详情请查看控制台输出。": "\n\nCheck console output for details.",

    # ========== batch_dialog.py ==========
    "输出命名模板:": "Naming Template:",
    "{name}_enhanced": "{name}_enhanced",
}
