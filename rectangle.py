import cv2
import numpy as np

# 这个类用于在屏幕上定义和操作矩形区域（可以看作是“虚拟按钮”）
class Rect():
    def __init__(self, x, y, width, height, color, text='', alpha=0.5):
        # 矩形的左上角坐标 (x, y)
        self.x = x
        self.y = y
        # 矩形的宽度和高度
        self.width = width
        self.height = height
        # 矩形的颜色 (BGR格式)
        self.color = color
        # 矩形中心显示的文本
        self.text = text
        # 矩形的透明度 (0.0 完全透明, 1.0 完全不透明)
        self.alpha = alpha

    def draw_rect(self, img, text_color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2):
        # 在传入的图像 (img) 上绘制带透明效果的矩形
        rec_roi = img[self.y: self.y + self.height, self.x: self.x + self.width]
        w_rect = np.ones(rec_roi.shape, dtype=np.uint8)
        w_rect[:] = self.color
        # 使用权重混合实现透明效果
        res = cv2.addWeighted(rec_roi, self.alpha, w_rect, 1 - self.alpha, 1.0)
        img[self.y: self.y + self.height, self.x: self.x + self.width] = res

        # 计算文字位置，使其在矩形正中心显示
        text_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.width / 2 - text_size[0][0] / 2),
                    int(self.y + self.height / 2 + text_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, fontFace, fontScale, text_color, thickness)

    def add_image(self, img, img2):
        # 在矩形区域内叠加一张带Alpha通道（透明背景）的图片（例如图标）
        rec_roi = img[self.y: self.y + self.height, self.x: self.x + self.width]
        # 分离Alpha通道
        alpha = img2[:, :, -1]
        img_bgr = img2[:, :, :-1]
        # 只在Alpha通道为不透明（255）的地方进行像素替换
        rec_roi[alpha == 255] = img_bgr[alpha == 255]
        img[self.y: self.y + self.height, self.x: self.x + self.width] = rec_roi
        return img

    def is_over(self, x, y):
        # 判断传入的坐标点 (x, y) 是否在这个矩形范围内
        # 常用于检测手指是否“点击”或“悬停”在按钮上
        if (self.x + self.width > x > self.x) and (self.y + self.height > y > self.y):
            return True
        return False
