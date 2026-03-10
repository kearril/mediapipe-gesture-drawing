"""
QuantumPaint AR - 核心控制系统
集成多线程视频流、AI 手势引擎、模块化 HUD 以及高性能绘图逻辑。
"""

import cv2
import numpy as np
import time
import threading
import math
from PIL import Image, ImageDraw, ImageFont
from landmarker import HandLandmarker, draw_marker, analyze_hand_data
from rectangle import UIButton

class CONFIG:
    """系统全局配置中心：集中管理 UI、算法、渲染及资源路径。"""
    WIN_NAME = "Quantum Paint AR"
    FRAME_W, FRAME_H = 1280, 720
    UI_DEADZONE_Y = 135        # 顶部操作条防误触避让区高度
    GESTURE_CONFIRM_SEC = 2.0  # 核心手势指令触发所需时长
    GESTURE_GRACE_SEC = 0.3    # 手势检测瞬间丢失的容错缓冲时长
    ALPHA_BASE = 0.3           # 滤波平滑基数（手部静止或慢速移动时）
    ALPHA_MAX = 0.9            # 滤波响应上限（手部快速移动时）
    FONT_PATH = "C:/Windows/Fonts/STKAITI.TTF"

class WebcamStream:
    """
    高性能多线程摄像头驱动类。
    通过后台线程持续采集帧，消除 OpenCV 默认阻塞读取导致的 I/O 延迟。
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_H)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        """启动后台采集线程。"""
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """线程循环：实时获取硬件图像。"""
        while self.started:
            (grabbed, frame) = self.stream.read()
            with self.read_lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        """获取当前最新帧的副本。"""
        with self.read_lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """关闭设备并终止线程。"""
        self.started = False
        self.stream.release()

class HUDManager:
    """
    负责增强现实界面的状态面板绘制。
    采用 ROI（局部感兴趣区域）渲染技术，最小化跨库（OpenCV/PIL）处理开销。
    """
    def __init__(self):
        self.px, self.py, self.pw, self.ph = 20, 20, 280, 115
        self._init_fonts()

    def _init_fonts(self):
        """加载矢量字体，失败时自动降级。"""
        try:
            self.font_main = ImageFont.truetype(CONFIG.FONT_PATH, 18)
            self.font_sub = ImageFont.truetype(CONFIG.FONT_PATH, 16)
        except Exception:
            self.font_main = self.font_sub = ImageFont.load_default()

    def render(self, frame, pen_type, is_drawing, fps, alpha, color):
        """
        绘制复合 HUD 面板。
        
        Args:
            frame: 目标画布。
            pen_type: 笔刷模式 ('brush' 或 'eraser')。
            is_drawing: 当前是否处于绘图激活态。
            fps: 系统运行实时帧率。
            alpha: 绘图滤波系数。
            color: 当前笔刷选定的 BGR 颜色。
        """
        # 1. 绘制半透明背景面板
        roi = frame[self.py:self.py+self.ph, self.px:self.px+self.pw]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0,0), (self.pw, self.ph), (30,30,30), -1)
        cv2.addWeighted(overlay, 0.75, roi, 0.25, 0, roi)
        cv2.rectangle(frame, (self.px, self.py), (self.px+self.pw, self.py+self.ph), (120,120,120), 1, cv2.LINE_AA)
        
        # 2. 绘制颜色状态球
        indicator_center = (self.px + self.pw - 45, self.py + self.ph // 2)
        cv2.circle(frame, indicator_center, 22, (255,255,255), 2, cv2.LINE_AA)
        cv2.circle(frame, indicator_center, 18, color, -1)

        # 3. 渲染文字遥测数据 (PIL 局部合成)
        text_area = frame[self.py:self.py+self.ph, self.px:self.px+self.pw-80]
        pil_img = Image.fromarray(cv2.cvtColor(text_area, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        mode_text = "橡皮擦" if pen_type == 'eraser' else "发光笔刷"
        status_text = "正在绘画" if is_drawing else "待机中"
        status_color = (50, 255, 50) if is_drawing else (180, 180, 180)
        
        draw.text((15, 10), f"绘画模式: {mode_text}", font=self.font_main, fill=(50, 180, 255))
        draw.text((15, 35), f"交互状态: {status_text}", font=self.font_main, fill=status_color[::-1])
        draw.text((15, 60), f"系统帧率: {int(fps)} FPS", font=self.font_sub, fill=(255, 255, 0))
        draw.text((15, 85), f"滤波系数: {alpha:.2f}", font=self.font_sub, fill=(255, 150, 255))
        
        frame[self.py:self.py+self.ph, self.px:self.px+self.pw-80] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return frame

class QuantumPaintApp:
    """
    QuantumPaint AR 系统主控制器。
    集成数据流管线、业务逻辑分发、手势指令状态机及绘画引擎。
    """
    def __init__(self):
        # 初始化核心引擎
        self.vs = WebcamStream().start()
        self.hands = HandLandmarker()
        self.hud = HUDManager()
        
        # 应用交互状态
        self.current_color = (0, 255, 0)
        self.current_size = 5
        self.pen_type = 'brush'
        self.is_drawing_active = False
        
        # 信号处理与计时器
        self.canvas = None
        self.prev_draw_pt = self.smooth_pt = None
        self.current_alpha = self.display_alpha = 0.5
        
        # 手势指令状态机 (具备 Hysteresis 容错)
        self.active_gesture = None
        self.gesture_start_time = 0
        self.last_seen_time = 0
        
        # UI 与 遥测数据
        self.dock_buttons = self._init_dock()
        self.prev_time = time.time()
        self.display_fps = 0
        self.fps_list = [] 

    def _init_dock(self):
        """初始化顶部交互操作栏的所有按钮。"""
        btns = []
        base_y = 62 
        # 颜色板初始化
        colors = [(255, 255, 255), (91, 183, 230), (22, 45, 93), (35, 51, 235), 
                  (51, 134, 240), (85, 254, 255), (76, 250, 117), (61, 205, 105), 
                  (253, 252, 116), (245, 30, 0), (247, 61, 234), (192, 49, 111)]
        for i, col in enumerate(colors):
            b = UIButton(350 + i * 45, base_y, 30, col, is_circle=True)
            b.id = ("color", col); btns.append(b)
        
        # 功能工具组初始化
        tools = [('clear', 'clear'), ('eraser', 'eraser'), ('brush', 'brush')]
        for i, (icon, name) in enumerate(tools):
            b = UIButton(930 + i * 65, base_y, 55, (200, 200, 200), icon_type=icon)
            b.id = ("tool", name); btns.append(b)
            
        # 笔刷尺寸组初始化
        for i, s in enumerate([5, 15, 30]):
            b = UIButton(1135 + i * 45, base_y, 40, (100, 100, 100), label=str(s), is_circle=False)
            b.id = ("size", s); btns.append(b)
        return btns

    def run(self):
        """系统主运行循环。"""
        cv2.namedWindow(CONFIG.WIN_NAME, cv2.WINDOW_NORMAL)
        while True:
            # 1. 采集与基础预处理
            frame = self.vs.read()
            if frame is None: continue
            frame = cv2.flip(frame, 1)
            if self.canvas is None: self.canvas = np.zeros_like(frame)
            
            # 2. 遥测计算 (FPS)
            curr = time.time()
            dt = curr - self.prev_time
            self.fps_list.append(dt if dt > 0 else 0.033)
            if len(self.fps_list) > 30: self.fps_list.pop(0)
            self.display_fps = 1 / (sum(self.fps_list) / len(self.fps_list))
            self.prev_time = curr

            # 3. 业务逻辑管线
            self._logic_step(frame)
            
            # 4. 视觉合成与 HUD
            combined = cv2.add(frame, self.canvas)
            combined = self.hud.render(combined, self.pen_type, self.is_drawing_active, 
                                       self.display_fps, self.display_alpha, self.current_color)
            
            # 5. 显示与按键监听
            cv2.imshow(CONFIG.WIN_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break 
            elif key == ord(' '): self.is_drawing_active = not self.is_drawing_active

        self._cleanup()

    def _logic_step(self, frame):
        """分发并处理一帧内的所有数据子流。"""
        self.hands.detect_async(frame)
        res = self.hands.result
        
        # AI 解析
        l_pos, r_pos, fingers, gesture = analyze_hand_data(frame, res)
        
        # 指令、交互与绘图
        frame[:] = draw_marker(frame, res)
        self._handle_gesture_commands(frame, gesture)
        
        ptr = r_pos[8] if r_pos else (l_pos[8] if l_pos else None)
        self._handle_ui_interaction(frame, ptr)
        self._handle_drawing(fingers, l_pos, r_pos)
        self._draw_cursor(frame, ptr)

    def _handle_gesture_commands(self, frame, gesture):
        """执行全局手势指令逻辑（带断连缓冲）。"""
        g_name, g_center = gesture["name"], gesture["center"]
        now = time.time()

        if g_name == "FIST":
            if self.active_gesture != "FIST":
                self.active_gesture = "FIST"
                self.gesture_start_time = now
            self.last_seen_time = now 
            
            # 计算并绘制进度环
            elapsed = now - self.gesture_start_time
            progress = min(1.0, elapsed / CONFIG.GESTURE_CONFIRM_SEC)
            cv2.ellipse(frame, g_center, (40, 40), 0, -90, int(progress*360)-90, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, "CLEARING", (g_center[0]-40, g_center[1]+60), 0, 0.5, (0,0,255), 1, cv2.LINE_AA)

            if elapsed >= CONFIG.GESTURE_CONFIRM_SEC:
                self.canvas = np.zeros_like(self.canvas)
                self.active_gesture = None
        else:
            # 缓冲期判定：防止瞬间丢失导致计时重置
            if self.active_gesture == "FIST" and (now - self.last_seen_time) > CONFIG.GESTURE_GRACE_SEC:
                self.active_gesture = None

    def _handle_drawing(self, fingers, l_pos, r_pos):
        """自适应滤波绘画引擎。"""
        if not self.is_drawing_active:
            self.prev_draw_pt = self.smooth_pt = None
            return

        # 确定交互点 (右手优先)
        raw = r_pos[8] if (r_pos and fingers.get('RIGHT_INDEX')) else \
              (l_pos[8] if (l_pos and fingers.get('LEFT_INDEX')) else None)
        
        if raw and raw[1] > CONFIG.UI_DEADZONE_Y:
            if self.smooth_pt is None:
                self.smooth_pt = raw
            else:
                # 动态 Alpha 计算：根据移动距离自适应调整滤波强度
                dist = math.sqrt((raw[0]-self.smooth_pt[0])**2 + (raw[1]-self.smooth_pt[1])**2)
                self.current_alpha = np.interp(dist, [2, 50], [CONFIG.ALPHA_BASE, CONFIG.ALPHA_MAX])
                self.display_alpha = 0.1 * self.current_alpha + 0.9 * self.display_alpha
                
                sx = int(self.current_alpha * raw[0] + (1 - self.current_alpha) * self.smooth_pt[0])
                sy = int(self.current_alpha * raw[1] + (1 - self.current_alpha) * self.smooth_pt[1])
                self.smooth_pt = (sx, sy)
            
            # 在画布上绘制线条
            if self.prev_draw_pt:
                is_eraser = self.pen_type == 'eraser'
                color = (0,0,0) if is_eraser else self.current_color
                thickness = self.current_size * 5 if is_eraser else self.current_size * 4
                cv2.line(self.canvas, self.prev_draw_pt, self.smooth_pt, color, thickness, cv2.LINE_AA)
                if not is_eraser:
                    cv2.line(self.canvas, self.prev_draw_pt, self.smooth_pt, color, self.current_size * 2, cv2.LINE_AA)
            self.prev_draw_pt = self.smooth_pt
        else:
            self.prev_draw_pt = self.smooth_pt = None

    def _handle_ui_interaction(self, frame, pt):
        """处理顶部操作条的悬停交互与确认逻辑。"""
        # 1. 渲染操作条背景 (ROI 加权)
        roi = frame[20:105, 310:1270]
        mask = roi.copy(); cv2.rectangle(mask, (0,0), (960, 85), (35,35,35), -1)
        cv2.addWeighted(mask, 0.5, roi, 0.5, 0, roi)
        cv2.rectangle(frame, (310, 20), (1270, 105), (80, 80, 80), 1, cv2.LINE_AA)

        # 2. 更新并绘制每个按钮
        for btn in self.dock_buttons:
            if pt:
                is_clk, prog = btn.update(pt[0], pt[1])
                if btn.is_hovered:
                    cv2.drawMarker(frame, pt, (255,255,255), cv2.MARKER_CROSS, 15, 1)
                    if prog > 0: 
                        cv2.ellipse(frame, (btn.x, btn.y), (btn.size//2 + 8, btn.size//2 + 8), 
                                    0, -90, int(prog*360)-90, (255, 255, 255), 3, cv2.LINE_AA)
                if is_clk: self._execute_action(btn)
            else:
                btn.update(-1, -1)
            btn.draw(frame)

    def _execute_action(self, btn):
        """执行按钮点击对应的系统指令。"""
        bt, bv = btn.id
        if bt == "color": self.current_color = bv
        elif bt == "size": self.current_size = bv
        elif bt == "tool":
            if bv == "clear": 
                self.canvas = np.zeros_like(self.canvas)
                self.is_drawing_active = False
            else: self.pen_type = bv

    def _draw_cursor(self, frame, pt):
        """绘制交互点随动光标。"""
        if pt and pt[1] > CONFIG.UI_DEADZONE_Y:
            cv2.circle(frame, pt, self.current_size + 10, self.current_color, 2, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)
            if self.is_drawing_active:
                pulse = int(abs(np.sin(time.time() * 5)) * 5)
                cv2.circle(frame, pt, self.current_size + 10 + pulse, self.current_color, 1, cv2.LINE_AA)

    def _cleanup(self):
        """安全释放系统资源。"""
        self.vs.stop(); self.hands.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    QuantumPaintApp().run()
