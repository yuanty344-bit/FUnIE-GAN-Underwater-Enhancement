# 水下图像增强软件 (Underwater Image Enhancement Software)
"""
主程序入口
提供命令行参数解析和应用程序初始化
"""

import sys
import os
from pathlib import Path

# ✅ 先设置高DPI属性，再创建QApplication！这个顺序不能变
from PyQt5.QtWidgets import QApplication, QMessageBox, QStyleFactory
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtGui import QIcon

# 先设置属性！必须在创建QApplication之前
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

from src.gui.main_window import MainWindow
from src.gui.icons import set_app_icon
from src.utils.config import ConfigManager
from src.utils.file_io import ensure_directories_exist
from src.i18n import tr, set_language


def setup_application_style(app: QApplication, theme: str = 'light') -> None:
    """
    设置应用程序样式和主题

    Args:
        app: QApplication实例
        theme: 主题名称 (light/dark/fusion/windows/windowsvista)
    """
    if theme in ('light', 'dark'):
        from src.gui.theme import apply_theme
        apply_theme(app, theme)
        return
    theme_map = {'fusion': 'Fusion', 'windows': 'Windows', 'windowsvista': 'WindowsVista'}
    style_name = theme_map.get(theme, 'Fusion')
    available_styles = QStyleFactory.keys()
    if style_name in available_styles:
        app.setStyle(style_name)


def setup_application_data(app: QApplication) -> None:
    """
    设置应用程序数据存储路径

    Args:
        app: QApplication实例
    """
    app.setOrganizationName('UnderwaterImageEnhancement')
    app.setOrganizationDomain('github.com')
    app.setApplicationName(tr('水下图像增强软件'))
    app.setApplicationVersion('1.0.0')


def parse_arguments() -> dict:
    """
    解析命令行参数
    
    Returns:
        参数字典
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='水下图像增强软件 - 基于FUnIE-GAN的深度学习图像增强工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法:
  python main.py                          # 启动GUI界面
  python main.py --input image.jpg        # 打开指定图像
  python main.py --batch ./images/        # 批量处理目录中的图像
  python main.py --video underwater.mp4   # 处理视频文件
        '''
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='输入图像文件路径'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=str,
        help='批量处理模式，指定图像目录路径'
    )
    
    parser.add_argument(
        '-v', '--video',
        type=str,
        help='视频文件路径，进行视频增强处理'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./output',
        help='输出目录路径（默认: ./output）'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='./models',
        help='模型文件目录（默认: ./models）'
    )
    
    args = parser.parse_args()
    return {
        'input_image': args.input,
        'batch_dir': args.batch,
        'video_file': args.video,
        'output_dir': args.output,
        'model_dir': args.model
    }


def main():
    """
    主函数
    """
    try:
        # 创建应用程序实例
        app = QApplication(sys.argv)

        # 尽早加载配置以应用主题和语言
        config = ConfigManager('./config.json', auto_load=True)

        # 从配置设置语言（在创建任何含文本的widget之前）
        set_language(config.get('ui.language', 'zh_CN'))

        # 从配置设置主题
        theme = config.get('ui.theme', 'light')
        setup_application_style(app, theme)
        setup_application_data(app)
        set_app_icon(app)

        # 打印欢迎信息
        print("=" * 50)
        print("水下图像增强软件 v1.0.0")
        print("=" * 50)
        print("基于FUnIE-GAN的深度学习图像增强工具")

        # 解析命令行参数
        params = parse_arguments()

        # 确保必要的目录存在
        ensure_directories_exist([
            params['output_dir'],
            params['model_dir'],
            './temp',
            './logs'
        ])

        # 创建并显示主窗口
        main_window = MainWindow(
            config=config,
            model_dir=params['model_dir'],
            output_dir=params['output_dir'],
            initial_image=params['input_image'],
            batch_mode=params['batch_dir'],
            video_file=params['video_file']
        )
        main_window.show()

        print("程序启动成功！")
        print("=" * 50)
        
        # 运行事件循环
        return app.exec_()
        
    except Exception as e:
        error_msg = f"程序启动失败: {str(e)}\n\n{traceback.format_exc()}"
        print(f"错误: {error_msg}")

        # 显示错误对话框
        try:
            QMessageBox.critical(
                None,
                tr("程序启动失败"),
                tr("发生严重错误，程序无法启动:\n\n") + str(e) + tr("\n\n详情请查看控制台输出。")
            )
        except:
            pass

        return 1


if __name__ == '__main__':
    import traceback
    sys.exit(main())