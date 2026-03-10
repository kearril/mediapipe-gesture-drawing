import cv2
import numpy as np
import time
import threading
import math
from PIL import Image, ImageDraw, ImageFont
from landmarker import HandLandmarker, draw_marker, analyze_hand_data
from rectangle import UIButton

class CONFIG:
    """系统全局配置中心"""
    WIN_NAME = "Quantum Paint AR"
    FRAME_W, FRAME_H = 1280, 720
    UI_DEADZONE_Y = 135  
    GESTURE_CONFIRM_SEC = 2.0  
    GESTURE_GRACE_SEC = 0.3    # 手势丢失容错时间 (防闪烁)
    ALPHA_BASE = 0.3     
    ALPHA_MAX = 0.9      
    FONT_PATH = "C:/Windows/Fonts/STKAITI.TTF"

class WebcamStream:
    """多线程摄像头驱动引擎"""
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_H)
        (self.grabbed, self.frame) = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started: return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            with self.read_lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.read_lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.started = False
        self.stream.release()

class QuantumPaintApp:
    def __init__(self):
        self.vs = WebcamStream().start()
        self.hands = HandLandmarker()
        
        self.current_color = (0, 255, 0)
        self.current_size = 5
        self.pen_type = 'brush'
        self.is_drawing_active = False
        
        self.canvas = None
        self.prev_draw_pt = None
        self.smooth_pt = None
        self.current_alpha = 0.5
        self.display_alpha = 0.5 
        
        # --- 增强型手势状态机 ---
        self.active_gesture = None      # 当前锁定的手势名
        self.gesture_start_time = 0     # 首次检测到时间
        self.last_seen_time = 0         # 最后一次看到该手势的时间
        
        self.dock_buttons = self._init_dock()
        self.prev_time = time.time()
        self.display_fps = 0
        self.fps_list = [] 
        self._init_fonts()

    def _init_fonts(self):
        try:
            self.font_main = ImageFont.truetype(CONFIG.FONT_PATH, 18)
            self.font_sub = ImageFont.truetype(CONFIG.FONT_PATH, 16)
        except:
            self.font_main = self.font_sub = ImageFont.load_default()

    def _init_dock(self):
        btns = []
        base_y = 62 
        colors = [(255, 255, 255), (91, 183, 230), (22, 45, 93), (35, 51, 235), 
                  (51, 134, 240), (85, 254, 255), (76, 250, 117), (61, 205, 105), 
                  (253, 252, 116), (245, 30, 0), (247, 61, 234), (192, 49, 111)]
        for i, col in enumerate(colors):
            b = UIButton(350 + i * 45, base_y, 30, col, is_circle=True)
            b.id = ("color", col); btns.append(b)
        tools = [('clear', 'clear'), ('eraser', 'eraser'), ('brush', 'brush')]
        for i, (icon, name) in enumerate(tools):
            b = UIButton(930 + i * 65, base_y, 55, (200, 200, 200), icon_type=icon)
            b.id = ("tool", name); btns.append(b)
        for i, s in enumerate([5, 15, 30]):
            b = UIButton(1135 + i * 45, base_y, 40, (100, 100, 100), label=str(s), is_circle=False)
            b.id = ("size", s); btns.append(b)
        return btns

    def run(self):
        cv2.namedWindow(CONFIG.WIN_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = self.vs.read()
            if frame is None: continue
            frame = cv2.flip(frame, 1)
            if self.canvas is None: self.canvas = np.zeros_like(frame)
            
            curr = time.time()
            dt = curr - self.prev_time
            self.fps_list.append(dt if dt > 0 else 0.033)
            if len(self.fps_list) > 30: self.fps_list.pop(0)
            avg_dt = sum(self.fps_list) / len(self.fps_list)
            self.display_fps = 1 / avg_dt if avg_dt > 0 else 0
            self.prev_time = curr

            self._logic_step(frame)
            combined = cv2.add(frame, self.canvas)
            combined = self._draw_overlay(combined)
            cv2.imshow(CONFIG.WIN_NAME, combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break 
            elif key == ord(' '): 
                self.is_drawing_active = not self.is_drawing_active

        self._cleanup()

    def _logic_step(self, frame):
        self.hands.detect_async(frame)
        res = self.hands.result
        l_pos, r_pos, fingers, gesture = analyze_hand_data(frame, res)
        frame[:] = draw_marker(frame, res)
        
        # 核心改进：传入当前检测手势及掌心
        self._handle_gesture_commands(frame, gesture)
        
        ptr = r_pos[8] if r_pos else (l_pos[8] if l_pos else None)
        self._handle_ui_interaction(frame, ptr)
        self._handle_drawing(fingers, l_pos, r_pos)
        self._draw_cursor(frame, ptr)

    def _handle_gesture_commands(self, frame, gesture):
        """带有容错机制的手势指令状态机"""
        g_name, g_center = gesture["name"], gesture["center"]
        now = time.time()

        if g_name == "FIST":
            # 1. 状态初始化或恢复
            if self.active_gesture != "FIST":
                self.active_gesture = "FIST"
                self.gesture_start_time = now
            
            self.last_seen_time = now # 刷新“最后看见”时间
            
            # 2. 计算进度
            elapsed = now - self.gesture_start_time
            progress = min(1.0, elapsed / CONFIG.GESTURE_CONFIRM_SEC)
            
            # 3. 视觉反馈
            cv2.ellipse(frame, g_center, (40, 40), 0, -90, int(progress*360)-90, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.putText(frame, "CLEARING", (g_center[0]-40, g_center[1]+60), 0, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # 4. 触发判定
            if elapsed >= CONFIG.GESTURE_CONFIRM_SEC:
                self.canvas = np.zeros_like(self.canvas)
                self.active_gesture = None # 触发后重置
        else:
            # 核心容错逻辑：如果当前没看到 FIST，但距离上次看到还没超过缓冲期
            if self.active_gesture == "FIST":
                if (now - self.last_seen_time) > CONFIG.GESTURE_GRACE_SEC:
                    # 真正丢失，重置状态机
                    self.active_gesture = None
                else:
                    # 在缓冲期内，保持进度条显示，但进度不增加（或缓慢增加）
                    # 为了视觉平滑，我们不画进度条，但也不重置计时器，让用户有接回手势的机会
                    pass

    def _handle_drawing(self, fingers, l_pos, r_pos):
        if not self.is_drawing_active:
            self.prev_draw_pt = self.smooth_pt = None
            return
        raw = r_pos[8] if (r_pos and fingers.get('RIGHT_INDEX')) else \
              (l_pos[8] if (l_pos and fingers.get('LEFT_INDEX')) else None)
        if raw and raw[1] > CONFIG.UI_DEADZONE_Y:
            if self.smooth_pt is None:
                self.smooth_pt = raw
            else:
                dist = math.sqrt((raw[0]-self.smooth_pt[0])**2 + (raw[1]-self.smooth_pt[1])**2)
                self.current_alpha = np.interp(dist, [2, 50], [CONFIG.ALPHA_BASE, CONFIG.ALPHA_MAX])
                self.display_alpha = 0.1 * self.current_alpha + 0.9 * self.display_alpha
                sx = int(self.current_alpha * raw[0] + (1 - self.current_alpha) * self.smooth_pt[0])
                sy = int(self.current_alpha * raw[1] + (1 - self.current_alpha) * self.smooth_pt[1])
                self.smooth_pt = (sx, sy)
            if self.prev_draw_pt:
                color = (0,0,0) if self.pen_type == 'eraser' else self.current_color
                thickness = self.current_size * 5 if self.pen_type == 'eraser' else self.current_size * 4
                cv2.line(self.canvas, self.prev_draw_pt, self.smooth_pt, color, thickness, cv2.LINE_AA)
                if self.pen_type != 'eraser':
                    cv2.line(self.canvas, self.prev_draw_pt, self.smooth_pt, color, self.current_size * 2, cv2.LINE_AA)
            self.prev_draw_pt = self.smooth_pt
        else:
            self.prev_draw_pt = self.smooth_pt = None

    def _handle_ui_interaction(self, frame, pt):
        roi = frame[20:105, 310:1270]
        mask = roi.copy(); cv2.rectangle(mask, (0,0), (960, 85), (35,35,35), -1)
        cv2.addWeighted(mask, 0.5, roi, 0.5, 0, roi)
        cv2.rectangle(frame, (310, 20), (1270, 105), (80, 80, 80), 1, cv2.LINE_AA)
        for btn in self.dock_buttons:
            if pt:
                is_clk, prog = btn.update(pt[0], pt[1])
                if btn.is_hovered:
                    cv2.drawMarker(frame, pt, (255,255,255), cv2.MARKER_CROSS, 15, 1)
                    if prog > 0: 
                        cv2.ellipse(frame, (btn.x, btn.y), (btn.size//2 + 8, btn.size//2 + 8), 
                                    0, -90, int(prog*360)-90, (255, 255, 255), 3, cv2.LINE_AA)
                if is_clk: self._execute_action(btn)
            else: btn.update(-1, -1)
            btn.draw(frame)

    def _execute_action(self, btn):
        bt, bv = btn.id
        if bt == "color": self.current_color = bv
        elif bt == "size": self.current_size = bv
        elif bt == "tool":
            if bv == "clear": self.canvas = np.zeros_like(self.canvas); self.is_drawing_active = False
            else: self.pen_type = bv

    def _draw_cursor(self, frame, pt):
        if pt and pt[1] > CONFIG.UI_DEADZONE_Y:
            cv2.circle(frame, pt, self.current_size + 10, self.current_color, 2, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)
            if self.is_drawing_active:
                pulse = int(abs(np.sin(time.time() * 5)) * 5)
                cv2.circle(frame, pt, self.current_size + 10 + pulse, self.current_color, 1, cv2.LINE_AA)

    def _draw_overlay(self, frame):
        px, py, pw, ph = 20, 20, 280, 115
        p_roi = frame[py:py+ph, px:px+pw]
        ov = p_roi.copy(); cv2.rectangle(ov, (0,0), (pw, ph), (30,30,30), -1)
        cv2.addWeighted(ov, 0.75, p_roi, 0.25, 0, p_roi)
        cv2.rectangle(frame, (px, py), (px+pw, py+ph), (120,120,120), 1, cv2.LINE_AA)
        c_ctr = (px + pw - 45, py + ph // 2)
        cv2.circle(frame, c_ctr, 22, (255,255,255), 2, cv2.LINE_AA)
        cv2.circle(frame, c_ctr, 18, self.current_color, -1)
        txt_roi = frame[py:py+ph, px:px+pw-80]
        pil = Image.fromarray(cv2.cvtColor(txt_roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        mode_str = "橡皮擦" if self.pen_type == 'eraser' else "发光笔刷"
        status_str = "正在绘画" if self.is_drawing_active else "待机中"
        status_col = (50, 255, 50) if self.is_drawing_active else (180, 180, 180)
        draw.text((15, 10), f"绘画模式: {mode_str}", font=self.font_main, fill=(50, 180, 255))
        draw.text((15, 35), f"交互状态: {status_str}", font=self.font_main, fill=status_col[::-1])
        draw.text((15, 60), f"系统帧率: {int(self.display_fps)} FPS", font=self.font_sub, fill=(255, 255, 0))
        draw.text((15, 85), f"滤波系数: {self.display_alpha:.2f}", font=self.font_sub, fill=(255, 150, 255))
        frame[py:py+ph, px:px+pw-80] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return frame

    def _cleanup(self):
        self.vs.stop(); self.hands.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    QuantumPaintApp().run()
