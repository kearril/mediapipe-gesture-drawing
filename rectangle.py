import cv2
import numpy as np
import time

class UIButton:
    """具备几何图标绘制能力的工业级 UI 按钮"""
    def __init__(self, x, y, size, color, label=None, icon_type=None, is_circle=False):
        self.x, self.y = x, y
        self.size = size
        self.color = color
        self.label = label
        self.icon_type = icon_type  # 'clear', 'eraser', 'brush'
        self.is_circle = is_circle
        
        self.is_hovered = False
        self.hover_start_time = 0
        self.click_duration = 0.6  # 悬停 0.6 秒触发点击
        self.id = None

    def update(self, mx, my):
        """更新悬停状态并返回是否触发点击"""
        dist = ((mx - self.x)**2 + (my - self.y)**2)**0.5
        if self.is_circle:
            in_range = dist < self.size // 2
        else:
            in_range = abs(mx - self.x) < self.size // 2 and abs(my - self.y) < self.size // 2

        if in_range:
            if not self.is_hovered:
                self.is_hovered = True
                self.hover_start_time = time.time()
            
            elapsed = time.time() - self.hover_start_time
            if elapsed >= self.click_duration:
                self.is_hovered = False # 触发后重置
                return True, 1.0
            return False, elapsed / self.click_duration
        else:
            self.is_hovered = False
            return False, 0.0

    def draw(self, img):
        """根据类型绘制按钮外观"""
        s = self.size
        # 1. 绘制底色/外圈
        if self.is_circle:
            cv2.circle(img, (self.x, self.y), s // 2, self.color, -1, cv2.LINE_AA)
            cv2.circle(img, (self.x, self.y), s // 2 + 2, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (self.x - s // 2, self.y - s // 2), 
                          (self.x + s // 2, self.y + s // 2), self.color, -1)
            cv2.rectangle(img, (self.x - s // 2, self.y - s // 2), 
                          (self.x + s // 2, self.y + s // 2), (255, 255, 255), 1, cv2.LINE_AA)

        # 2. 绘制程序化几何图标
        if self.icon_type == 'clear':
            self._draw_clear_icon(img)
        elif self.icon_type == 'eraser':
            self._draw_eraser_icon(img)
        elif self.icon_type == 'brush':
            self._draw_brush_icon(img)
        
        # 3. 绘制文字标签 (如果有)
        if self.label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.5
            (tw, th), _ = cv2.getTextSize(self.label, font, fs, 1)
            cv2.putText(img, self.label, (self.x - tw // 2, self.y + th // 2), 
                        font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_clear_icon(self, img):
        """绘制几何垃圾桶图标"""
        cx, cy, r = self.x, self.y, self.size // 4
        # 桶身
        cv2.rectangle(img, (cx-r, cy-r+4), (cx+r, cy+r), (255, 255, 255), 2, cv2.LINE_AA)
        # 桶盖
        cv2.line(img, (cx-r-4, cy-r+4), (cx+r+4, cy-r+4), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx-4, cy-r+4), (cx-4, cy-r), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(img, (cx+4, cy-r+4), (cx+4, cy-r), (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_eraser_icon(self, img):
        """绘制斜切橡皮图标"""
        cx, cy, r = self.x, self.y, self.size // 3
        pts = np.array([[cx-r, cy], [cx, cy-r], [cx+r, cy], [cx, cy+r]], np.int32)
        cv2.polylines(img, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        # 填充一半表示擦除端
        fill_pts = np.array([[cx-r, cy], [cx, cy-r], [cx, cy], [cx-r//2, cy+r//2]], np.int32)
        cv2.fillPoly(img, [fill_pts], (255, 255, 255))

    def _draw_brush_icon(self, img):
        """绘制带光芒的画笔图标"""
        cx, cy, r = self.x, self.y, self.size // 4
        cv2.circle(img, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        # 绘制十字光芒
        cv2.line(img, (cx-r-6, cy), (cx+r+6, cy), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy-r-6), (cx, cy+r+6), (255, 255, 255), 1, cv2.LINE_AA)
