"""
QuantumPaint AR - UI 基础组件库
提供具备物理碰撞检测与程序化矢量图标渲染能力的交互控件。
"""

import cv2
import numpy as np
import time

class UIButton:
    """
    具备交互状态管理与程序化几何图标绘制能力的工业级按钮。
    
    支持功能：
    - 悬停 dwell 点击逻辑（确认机制防止误触）。
    - 圆形/矩形两种碰撞体判定。
    - 程序化矢量图标渲染（零文件依赖）。
    """
    def __init__(self, x, y, size, color, label=None, icon_type=None, is_circle=False):
        """
        初始化 UI 按钮。
        
        Args:
            x, y: 按钮中心坐标。
            size: 按钮尺寸。
            color: 按钮底色 (BGR)。
            label: 按钮显示的文字标签。
            icon_type: 矢量图标类型 ('clear', 'eraser', 'brush')。
            is_circle: 是否采用圆形碰撞体。
        """
        self.x, self.y = x, y
        self.size = size
        self.color = color
        self.label = label
        self.icon_type = icon_type  
        self.is_circle = is_circle
        
        # 交互状态机
        self.is_hovered = False
        self.hover_start_time = 0
        self.click_duration = 0.6  # 悬停触发时长阈值 (秒)
        self.id = None             # 业务逻辑绑定 ID

    def update(self, mx, my):
        """
        更新按钮交互状态。
        
        Args:
            mx, my: 当前交互点（如食指指尖）的像素坐标。
            
        Returns:
            tuple: (is_clicked, progress)
                - is_clicked: bool, 是否在当前帧触发了点击事件。
                - progress: float, 当前点击确认的加载进度 (0.0 - 1.0)。
        """
        # 1. 物理碰撞检测
        if self.is_circle:
            # 欧式距离判定 (圆形)
            dist = ((mx - self.x)**2 + (my - self.y)**2)**0.5
            in_range = dist < self.size // 2
        else:
            # 边界判定 (矩形)
            in_range = abs(mx - self.x) < self.size // 2 and abs(my - self.y) < self.size // 2

        # 2. 状态机逻辑
        if in_range:
            if not self.is_hovered:
                self.is_hovered = True
                self.hover_start_time = time.time()
            
            elapsed = time.time() - self.hover_start_time
            if elapsed >= self.click_duration:
                self.is_hovered = False # 触发后重置，防止连续触发
                return True, 1.0
            return False, elapsed / self.click_duration
        else:
            self.is_hovered = False
            return False, 0.0

    def draw(self, img):
        """
        在图像上渲染按钮外观。
        
        Args:
            img: OpenCV 目标图像矩阵。
        """
        s = self.size
        
        # 1. 绘制底色与高亮边框
        if self.is_circle:
            cv2.circle(img, (self.x, self.y), s // 2, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (self.x, self.y), s // 2 + 2, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (self.x - s // 2, self.y - s // 2), 
                          (self.x + s // 2, self.y + s // 2), self.color, -1)
            cv2.rectangle(img, (self.x - s // 2, self.y - s // 2), 
                          (self.x + s // 2, self.y + s // 2), (255, 255, 255), 1, cv2.LINE_AA)

        # 2. 调度程序化矢量图标渲染
        if self.icon_type == 'clear':
            self._draw_clear_icon(img)
        elif self.icon_type == 'eraser':
            self._draw_eraser_icon(img)
        elif self.icon_type == 'brush':
            self._draw_brush_icon(img)
        
        # 3. 绘制文字标签
        if self.label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.5
            (tw, th), _ = cv2.getTextSize(self.label, font, fs, 1)
            cv2.putText(img, self.label, (self.x - tw // 2, self.y + th // 2), 
                        font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_clear_icon(self, img):
        """绘制矢量垃圾桶图标：体现『清空』语义。"""
        cx, cy, r = self.x, self.y, self.size // 4
        # 桶身轮廓
        cv2.rectangle(img, (cx-r, cy-r+4), (cx+r, cy+r), (255, 255, 255), 2, cv2.LINE_AA)
        # 桶盖与提手
        cv2.line(img, (cx-r-4, cy-r+4), (cx+r+4, cy-r+4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx-4, cy-r+4), (cx-4, cy-r), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx+4, cy-r+4), (cx+4, cy-r), (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_eraser_icon(self, img):
        """绘制斜切几何体图标：体现『橡皮擦』语义。"""
        cx, cy, r = self.x, self.y, self.size // 3
        pts = np.array([[cx-r, cy], [cx, cy-r], [cx+r, cy], [cx, cy+r]], np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        # 填充前端，模拟摩擦消耗端视觉
        fill_pts = np.array([[cx-r, cy], [cx, cy-r], [cx, cy], [cx-r//2, cy+r//2]], np.int32)
        cv2.fillPoly(img, [fill_pts], (255, 255, 255))

    def _draw_brush_icon(self, img):
        """绘制发光圆点图标：体现『笔刷』语义。"""
        cx, cy, r = self.x, self.y, self.size // 4
        cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        # 绘制交互十字光芒，增强 AR 科技感
        cv2.line(img, (cx-r-6, cy), (cx+r+6, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy-r-6), (cx, cy+r+6), (255, 255, 255), 1, cv2.LINE_AA)
