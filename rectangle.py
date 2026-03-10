"""
工具栏按钮组件。

这个文件负责两件事：
1. 定义按钮的命中区域和停留点击逻辑。
2. 用 OpenCV 图元直接绘制按钮外观与图标。

由于项目运行在摄像头交互场景中，这里的“点击”不是鼠标按下，
而是指针在按钮上停留一段时间后自动触发。
"""

import time

import cv2
import numpy as np


class UIButton:
    """
    支持停留点击的工具栏按钮。

    当前按钮既承担“状态判断”，也承担“自我绘制”，
    这样主程序可以只负责布局和动作分发，不需要关心具体绘制细节。
    """

    def __init__(self, x, y, size, color, label=None, icon_type=None, is_circle=False):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.label = label
        self.icon_type = icon_type
        self.is_circle = is_circle
        self.is_hovered = False
        self.hover_start_time = 0.0
        self.click_duration = 0.6
        self.id = None

    def _contains_point(self, mx, my) -> bool:
        """
        判断一个点是否落在按钮命中区域内。

        - 圆形按钮使用半径范围判断。
        - 方形按钮使用包围盒判断。
        """
        half_size = self.size // 2
        if self.is_circle:
            dx = mx - self.x
            dy = my - self.y
            return dx * dx + dy * dy < half_size * half_size
        return abs(mx - self.x) < half_size and abs(my - self.y) < half_size

    def _reset_hover(self) -> None:
        """
        清空悬停状态。

        这个动作在两种情况下都会发生：
        - 指针离开按钮区域
        - 停留点击已经触发完成
        """
        self.is_hovered = False
        self.hover_start_time = 0.0

    def update(self, mx, my):
        """
        更新按钮悬停状态，并返回点击结果与进度。

        返回值格式：
        - 第 1 项：是否已触发停留点击
        - 第 2 项：当前停留进度，范围在 0 到 1 之间

        这里的进度主要用于主程序绘制按钮周围的进度环。
        """
        if not self._contains_point(mx, my):
            self._reset_hover()
            return False, 0.0

        current_time = time.time()
        if not self.is_hovered:
            self.is_hovered = True
            self.hover_start_time = current_time

        elapsed = current_time - self.hover_start_time
        if elapsed >= self.click_duration:
            self._reset_hover()
            return True, 1.0

        return False, elapsed / self.click_duration

    def draw(self, img) -> None:
        """
        绘制按钮主体、图标和文字。

        绘制顺序固定为：
        1. 按钮底形
        2. 按钮图标
        3. 文字标签
        """
        self._draw_shape(img)
        self._draw_icon(img)
        self._draw_label(img)

    def _draw_shape(self, img) -> None:
        """
        绘制按钮底形和外边框。

        当前实现使用纯 OpenCV 图元绘制，避免依赖额外图像资源。
        """
        half_size = self.size // 2
        if self.is_circle:
            cv2.circle(img, (self.x, self.y), half_size, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (self.x, self.y), half_size + 2, (255, 255, 255), 1, cv2.LINE_AA)
            return

        top_left = (self.x - half_size, self.y - half_size)
        bottom_right = (self.x + half_size, self.y + half_size)
        cv2.rectangle(img, top_left, bottom_right, self.color, -1)
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_icon(self, img) -> None:
        """
        根据按钮配置分发对应图标绘制函数。

        这里没有使用字典映射，而是保留简单分支，
        主要是为了让当前支持的图标类型一眼可见。
        """
        if self.icon_type == "clear":
            self._draw_clear_icon(img)
        elif self.icon_type == "eraser":
            self._draw_eraser_icon(img)
        elif self.icon_type == "brush":
            self._draw_brush_icon(img)

    def _draw_label(self, img) -> None:
        """
        在按钮中央绘制文字标签。

        主要用于显示笔刷尺寸等简单文本信息。
        """
        if not self.label:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (text_width, text_height), _ = cv2.getTextSize(self.label, font, font_scale, 1)
        text_origin = (self.x - text_width // 2, self.y + text_height // 2)
        cv2.putText(img, self.label, text_origin, font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_clear_icon(self, img) -> None:
        """绘制清空按钮使用的垃圾桶图标。"""
        cx, cy, radius = self.x, self.y, self.size // 4
        cv2.rectangle(img, (cx - radius, cy - radius + 4), (cx + radius, cy + radius), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx - radius - 4, cy - radius + 4), (cx + radius + 4, cy - radius + 4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx - 4, cy - radius + 4), (cx - 4, cy - radius), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx + 4, cy - radius + 4), (cx + 4, cy - radius), (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_eraser_icon(self, img) -> None:
        """
        绘制橡皮擦图标。

        当前采用菱形轮廓加局部填充的方式，
        目的是在很小的按钮尺寸下仍然保持辨识度。
        """
        cx, cy, radius = self.x, self.y, self.size // 3
        outline = np.array([[cx - radius, cy], [cx, cy - radius], [cx + radius, cy], [cx, cy + radius]], np.int32)
        fill = np.array([[cx - radius, cy], [cx, cy - radius], [cx, cy], [cx - radius // 2, cy + radius // 2]], np.int32)
        cv2.polylines(img, [outline], True, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.fillPoly(img, [fill], (255, 255, 255))

    def _draw_brush_icon(self, img) -> None:
        """
        绘制笔刷图标。

        中间实心圆代表笔尖，横竖辅助线用来强化“发光刷头”的视觉识别。
        """
        cx, cy, radius = self.x, self.y, self.size // 4
        cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.line(img, (cx - radius - 6, cy), (cx + radius + 6, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - radius - 6), (cx, cy + radius + 6), (255, 255, 255), 1, cv2.LINE_AA)
