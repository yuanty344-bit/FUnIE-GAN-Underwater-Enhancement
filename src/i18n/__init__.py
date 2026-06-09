"""
i18n 国际化模块

提供轻量级字符串翻译功能，无外部依赖。
中文模式直接返回原文，英文模式查字典翻译，未找到则降级返回中文。
"""

from .translations import TRANSLATIONS

_current_language = 'zh_CN'


def set_language(lang: str) -> None:
    """设置当前界面语言 ('zh_CN' 或 'en')"""
    global _current_language
    _current_language = lang


def get_language() -> str:
    """获取当前界面语言"""
    return _current_language


def tr(text: str) -> str:
    """
    翻译字符串。

    中文模式下原文返回；其他语言下查字典，
    找不到对应翻译时降级返回原文（中文）。
    """
    if _current_language == 'zh_CN':
        return text
    return TRANSLATIONS.get(text, text)
