"""
应用图标模块

为工具栏按钮和应用程序生成可辨识的矢量风格图标。
所有图标通过 QPainter 绘制，无需外部资源文件。
"""

from PyQt5.QtCore import Qt, QSize, QRectF, QPointF
from PyQt5.QtGui import (
    QIcon, QPixmap, QPainter, QPen, QBrush, QColor,
    QPainterPath, QPolygonF, QFont, QLinearGradient
)

# 主题色系
OCEAN_BLUE = QColor(25, 118, 210)        # 主色 - 深海蓝
TEAL = QColor(0, 188, 212)               # 强调 - 青绿
CORAL = QColor(255, 112, 67)             # 珊瑚橙
SEAFOAM = QColor(38, 166, 154)           # 海沫绿
SAND = QColor(255, 183, 77)              # 沙金色
DEEP = QColor(55, 71, 133)               # 深海紫
WHITE = QColor(255, 255, 255)
DARK = QColor(48, 48, 48)
GRAY = QColor(158, 158, 158)


def _paint_app_icon(painter: QPainter, size: int):
    """应用图标：海洋波浪 + 星光增强"""
    s = size
    m = s / 64.0  # 基于 64x64 的缩放系数
    painter.setRenderHint(QPainter.Antialiasing)

    # 背景圆 - 海洋渐变
    gradient = QLinearGradient(0, 0, 0, s)
    gradient.setColorAt(0, QColor(3, 78, 162))
    gradient.setColorAt(0.5, QColor(2, 119, 189))
    gradient.setColorAt(1, QColor(0, 150, 199))
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(QRectF(0, 0, s, s), 12 * m, 12 * m)

    # 波浪曲线
    wave = QPainterPath()
    wave.moveTo(0, 38 * m)
    wave.cubicTo(12 * m, 30 * m, 20 * m, 46 * m, 32 * m, 38 * m)
    wave.cubicTo(44 * m, 30 * m, 52 * m, 46 * m, 64 * m, 38 * m)
    wave.lineTo(64 * m, 64 * m)
    wave.lineTo(0, 64 * m)
    wave.closeSubpath()
    painter.setBrush(QColor(0, 131, 179, 140))
    painter.drawPath(wave)

    # 鱼形剪影（简化为弧线 + 三角尾）
    fish = QPainterPath()
    fish.moveTo(26 * m, 28 * m)
    fish.cubicTo(18 * m, 20 * m, 18 * m, 36 * m, 26 * m, 28 * m)
    fish.lineTo(32 * m, 18 * m)
    fish.lineTo(30 * m, 28 * m)
    fish.lineTo(32 * m, 38 * m)
    fish.closeSubpath()
    painter.setBrush(QColor(255, 255, 255, 200))
    painter.drawPath(fish)

    # 星光 / 增强标记
    star = QPolygonF()
    cx, cy = 46 * m, 14 * m
    r_outer = 8 * m
    r_inner = 3 * m
    for i in range(8):
        angle = i * 3.14159 / 4 - 3.14159 / 2
        r = r_outer if i % 2 == 0 else r_inner
        star.append(QPointF(cx + r * __import__('math').cos(angle),
                           cy + r * __import__('math').sin(angle)))
    painter.setBrush(QColor(255, 235, 59))
    painter.drawPolygon(star)


def _paint_open_icon(painter: QPainter, size: int, color: QColor = SAND):
    """打开图标：文件夹 + 向上箭头"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    # 文件夹形状
    folder = QPainterPath()
    folder.moveTo(6 * m, 22 * m)
    folder.lineTo(6 * m, 16 * m)
    folder.cubicTo(8 * m, 10 * m, 14 * m, 10 * m, 18 * m, 10 * m)
    folder.lineTo(26 * m, 10 * m)
    folder.lineTo(24 * m, 16 * m)
    folder.lineTo(24 * m, 22 * m)
    folder.closeSubpath()
    painter.setBrush(QBrush(color.lighter(120)))
    painter.setPen(QPen(color.darker(110), 2.2 * m))
    painter.drawPath(folder)

    # 向上箭头
    arrow = QPainterPath()
    arrow.moveTo(32 * m, 44 * m)
    arrow.lineTo(32 * m, 24 * m)
    arrow.moveTo(24 * m, 32 * m)
    arrow.lineTo(32 * m, 22 * m)
    arrow.lineTo(40 * m, 32 * m)
    painter.setPen(QPen(color, 3.2 * m, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(arrow)

    # 横线底座
    painter.setPen(QPen(color, 3 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(20 * m, 48 * m), QPointF(44 * m, 48 * m))


def _paint_save_icon(painter: QPainter, size: int, color: QColor = SEAFOAM):
    """保存图标：向下箭头进入托盘"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    # 托盘
    tray = QPainterPath()
    tray.moveTo(8 * m, 46 * m)
    tray.lineTo(8 * m, 56 * m)
    tray.lineTo(56 * m, 56 * m)
    tray.lineTo(56 * m, 46 * m)
    tray.moveTo(14 * m, 46 * m)
    tray.lineTo(14 * m, 52 * m)
    tray.lineTo(50 * m, 52 * m)
    tray.lineTo(50 * m, 46 * m)
    painter.setPen(QPen(color, 2.2 * m))
    painter.setBrush(QBrush(color.lighter(130)))
    painter.drawPath(tray)

    # 向下箭头
    painter.setPen(QPen(color.darker(110), 3.2 * m, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    painter.setBrush(Qt.NoBrush)
    painter.drawLine(QPointF(32 * m, 10 * m), QPointF(32 * m, 38 * m))
    arrow_tip = QPainterPath()
    arrow_tip.moveTo(22 * m, 28 * m)
    arrow_tip.lineTo(32 * m, 40 * m)
    arrow_tip.lineTo(42 * m, 28 * m)
    painter.drawPath(arrow_tip)


def _paint_enhance_icon(painter: QPainter, size: int, color: QColor = CORAL):
    """增强图标：魔法棒 + 星光"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)
    import math

    # 魔法棒
    wand = QPainterPath()
    wand.moveTo(44 * m, 10 * m)
    wand.lineTo(18 * m, 48 * m)
    painter.setPen(QPen(color.darker(120), 3 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawPath(wand)

    # 棒头小星
    star = QPolygonF()
    cx, cy = 44 * m, 8 * m
    for i in range(8):
        r = 6 * m if i % 2 == 0 else 2.5 * m
        a = i * math.pi / 4 - math.pi / 2
        star.append(QPointF(cx + r * math.cos(a), cy + r * math.sin(a)))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(color))
    painter.drawPolygon(star)

    # 散布星光点
    for px, py, r in [(26, 28, 3.5), (14, 36, 2.5), (36, 18, 2), (20, 20, 1.8)]:
        painter.setBrush(QBrush(QColor(255, 215, 0)))
        painter.drawEllipse(QPointF(px * m, py * m), r * m, r * m)


def _paint_zoom_in_icon(painter: QPainter, size: int, color: QColor = OCEAN_BLUE):
    """放大图标：放大镜 + 加号"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    # 镜框
    cx, cy, r = 24 * m, 24 * m, 16 * m
    painter.setPen(QPen(color, 3.5 * m))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(cx, cy), r, r)

    # 手柄
    handle = QPainterPath()
    handle.moveTo(36 * m, 36 * m)
    handle.lineTo(52 * m, 52 * m)
    painter.setPen(QPen(color, 3.8 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawPath(handle)

    # 加号
    painter.setPen(QPen(color, 3 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(24 * m, 16 * m), QPointF(24 * m, 32 * m))
    painter.drawLine(QPointF(16 * m, 24 * m), QPointF(32 * m, 24 * m))


def _paint_zoom_out_icon(painter: QPainter, size: int, color: QColor = OCEAN_BLUE):
    """缩小图标：放大镜 + 减号"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    cx, cy, r = 24 * m, 24 * m, 16 * m
    painter.setPen(QPen(color, 3.5 * m))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(cx, cy), r, r)

    handle = QPainterPath()
    handle.moveTo(36 * m, 36 * m)
    handle.lineTo(52 * m, 52 * m)
    painter.setPen(QPen(color, 3.8 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawPath(handle)

    # 减号
    painter.setPen(QPen(color, 3 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(16 * m, 24 * m), QPointF(32 * m, 24 * m))


def _paint_zoom_fit_icon(painter: QPainter, size: int, color: QColor = OCEAN_BLUE):
    """适应窗口图标：向内收缩的角标"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(color, 3 * m, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    # 四个角的 L 形
    l = 16 * m  # L臂长
    d = 10 * m  # 离边距
    # 左上
    painter.drawPolyline(QPointF(d + l, d), QPointF(d, d), QPointF(d, d + l))
    # 右上
    painter.drawPolyline(QPointF(s - d - l, d), QPointF(s - d, d), QPointF(s - d, d + l))
    # 左下
    painter.drawPolyline(QPointF(d, s - d - l), QPointF(d, s - d), QPointF(d + l, s - d))
    # 右下
    painter.drawPolyline(QPointF(s - d, s - d - l), QPointF(s - d, s - d), QPointF(s - d - l, s - d))


def _paint_reset_icon(painter: QPainter, size: int, color: QColor = GRAY):
    """重置图标：逆时针箭头"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(color, 3 * m, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    import math

    # 圆弧箭头
    arc = QPainterPath()
    arc.arcMoveTo(QRectF(14 * m, 10 * m, 36 * m, 36 * m), 90)
    arc.arcTo(QRectF(14 * m, 10 * m, 36 * m, 36 * m), 90, 300)
    painter.drawPath(arc)

    # 箭头尖
    tip = QPainterPath()
    tip.moveTo(22 * m, 12 * m)
    tip.lineTo(50 * m, 10 * m)
    tip.lineTo(46 * m, 24 * m)
    painter.setBrush(QBrush(WHITE))
    painter.drawPath(tip)
    painter.setBrush(Qt.NoBrush)
    painter.drawPath(tip)


def _paint_batch_icon(painter: QPainter, size: int, color: QColor = TEAL):
    """批量处理图标：堆叠的图片"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    # 后面一张（偏移）
    painter.setPen(QPen(color.darker(120), 2 * m))
    painter.setBrush(QBrush(color.lighter(130)))
    painter.drawRoundedRect(QRectF(6 * m, 6 * m, 36 * m, 28 * m), 4 * m, 4 * m)

    # 前面一张
    painter.setPen(QPen(color, 2.2 * m))
    painter.setBrush(QBrush(WHITE))
    painter.drawRoundedRect(QRectF(20 * m, 20 * m, 38 * m, 30 * m), 4 * m, 4 * m)

    # 小画面图标
    painter.setPen(QPen(color.lighter(80), 1.5 * m))
    painter.drawEllipse(QPointF(34 * m, 32 * m), 6 * m, 5 * m)
    painter.drawLine(QPointF(28 * m, 42 * m), QPointF(40 * m, 42 * m))
    # 小三角（山）
    painter.drawPolyline(QPointF(28 * m, 42 * m), QPointF(34 * m, 36 * m), QPointF(40 * m, 42 * m))


def _paint_video_icon(painter: QPainter, size: int, color: QColor = DEEP):
    """视频图标：播放按钮 / 胶片"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    # 胶片边框
    painter.setPen(QPen(color, 2.2 * m))
    painter.setBrush(Qt.NoBrush)
    painter.drawRoundedRect(QRectF(8 * m, 14 * m, 48 * m, 36 * m), 6 * m, 6 * m)

    # 播放三角
    play = QPainterPath()
    play.moveTo(26 * m, 22 * m)
    play.lineTo(26 * m, 42 * m)
    play.lineTo(42 * m, 32 * m)
    play.closeSubpath()
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.NoPen)
    painter.drawPath(play)

    # 胶片齿孔
    painter.setBrush(QBrush(color.lighter(150)))
    for i in range(4):
        py = 18 * m + i * 9 * m
        painter.drawEllipse(QPointF(11 * m, py), 2.5 * m, 2.5 * m)
        painter.drawEllipse(QPointF(53 * m, py), 2.5 * m, 2.5 * m)


def _paint_settings_icon(painter: QPainter, size: int, color: QColor = GRAY):
    """设置图标：齿轮"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)
    import math

    # 齿轮外圈齿
    cx, cy = 32 * m, 32 * m
    r_outer = 20 * m
    r_inner = 14 * m
    teeth = 8
    wheel = QPolygonF()
    for i in range(teeth * 2):
        angle = i * math.pi / teeth - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        wheel.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))
    painter.setPen(QPen(color.darker(110), 2.2 * m))
    painter.setBrush(QBrush(color.lighter(120)))
    painter.drawPolygon(wheel)

    # 中心圆
    painter.setBrush(QBrush(WHITE))
    painter.setPen(QPen(color, 2 * m))
    painter.drawEllipse(QPointF(cx, cy), 8 * m, 8 * m)


def _paint_about_icon(painter: QPainter, size: int, color: QColor = OCEAN_BLUE):
    """关于图标：信息圆圈"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(color, 3 * m))
    painter.setBrush(Qt.NoBrush)
    painter.drawEllipse(QPointF(32 * m, 32 * m), 22 * m, 22 * m)

    # i 的竖线
    painter.setPen(QPen(color, 3.5 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(32 * m, 22 * m), QPointF(32 * m, 26 * m))
    painter.drawLine(QPointF(32 * m, 34 * m), QPointF(32 * m, 44 * m))
    # i 的点
    painter.setBrush(QBrush(color))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(QPointF(32 * m, 18 * m), 3 * m, 3 * m)


def _paint_compare_icon(painter: QPainter, size: int, color: QColor = OCEAN_BLUE):
    """对比图标：左右分屏"""
    s = size
    m = s / 64.0
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setPen(QPen(color, 2.2 * m))
    painter.setBrush(QBrush(color.lighter(160)))
    painter.drawRoundedRect(QRectF(4 * m, 8 * m, 24 * m, 30 * m), 3 * m, 3 * m)
    painter.drawRoundedRect(QRectF(36 * m, 8 * m, 24 * m, 30 * m), 3 * m, 3 * m)

    # 中间分割线和箭头
    painter.setPen(QPen(color, 2 * m, Qt.SolidLine, Qt.RoundCap))
    painter.drawLine(QPointF(32 * m, 14 * m), QPointF(32 * m, 32 * m))
    # 左右箭头
    painter.drawLine(QPointF(26 * m, 23 * m), QPointF(30 * m, 18 * m))
    painter.drawLine(QPointF(30 * m, 18 * m), QPointF(30 * m, 28 * m))
    painter.drawLine(QPointF(38 * m, 23 * m), QPointF(34 * m, 18 * m))
    painter.drawLine(QPointF(34 * m, 18 * m), QPointF(34 * m, 28 * m))


# 图标注册表
_ICON_PAINTERS = {
    'app':        _paint_app_icon,
    'open':       _paint_open_icon,
    'save':       _paint_save_icon,
    'enhance':    _paint_enhance_icon,
    'zoom_in':    _paint_zoom_in_icon,
    'zoom_out':   _paint_zoom_out_icon,
    'zoom_fit':   _paint_zoom_fit_icon,
    'reset':      _paint_reset_icon,
    'batch':      _paint_batch_icon,
    'video':      _paint_video_icon,
    'settings':   _paint_settings_icon,
    'about':      _paint_about_icon,
    'compare':    _paint_compare_icon,
}

_icon_cache = {}


def get_icon(name: str, size: int = 64) -> QIcon:
    """获取指定名称的图标，自动缓存"""
    cache_key = (name, size)
    if cache_key in _icon_cache:
        return _icon_cache[cache_key]

    painter = _ICON_PAINTERS.get(name)
    if painter is None:
        return QIcon()

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    p = QPainter(pixmap)
    try:
        painter(p, size)
    finally:
        p.end()

    icon = QIcon(pixmap)
    _icon_cache[cache_key] = icon
    return icon


def set_app_icon(app):
    """设置应用程序图标"""
    icon = get_icon('app', 256)
    app.setWindowIcon(icon)
