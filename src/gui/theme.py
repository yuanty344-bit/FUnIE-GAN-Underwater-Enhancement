"""
明暗主题管理

加载 resources/themes/ 下的 .qss 文件并应用到 QApplication。
"""
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QStyleFactory


_theme_name = "light"


def apply_theme(app: QApplication, name: str = "light") -> bool:
    """应用主题样式表，返回是否成功"""
    global _theme_name
    _theme_name = name

    if name in ("light", "dark"):
        qss_path = Path(__file__).resolve().parents[2] / "resources" / "themes" / f"{name}.qss"
        if not qss_path.exists():
            return False
        try:
            with open(qss_path, "r", encoding="utf-8") as f:
                qss = f.read()
            # Set Fusion as base for consistent cross-platform appearance
            if "Fusion" in QStyleFactory.keys():
                app.setStyle("Fusion")
            app.setStyleSheet(qss)
            return True
        except Exception:
            return False
    else:
        # Clear QSS for native styles
        app.setStyleSheet("")
        if name in QStyleFactory.keys():
            app.setStyle(name)
        return True


def current_theme() -> str:
    return _theme_name


def toggle_theme(app: QApplication) -> str:
    """切换明暗主题，返回新主题名"""
    name = "dark" if _theme_name == "light" else "light"
    apply_theme(app, name)
    return name


def get_available_themes() -> list:
    """获取可用主题列表"""
    themes_dir = Path(__file__).resolve().parents[2] / "resources" / "themes"
    themes = []
    for qss in themes_dir.glob("*.qss"):
        themes.append(qss.stem)
    return themes
