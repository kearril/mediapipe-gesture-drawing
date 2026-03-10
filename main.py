"""
AR 手势绘画演示主程序。

这个文件负责把项目里的几个核心能力串起来：
- 摄像头采集
- 手部关键点检测
- 工具栏与状态面板绘制
- 手势指令处理
- 笔迹滤波与落笔渲染

整体设计思路是“一个主循环 + 若干小辅助函数”：
主循环负责按帧推进，辅助函数分别处理 FPS、手势、绘图、工具栏和光标，
这样既能保持实时渲染流程清晰，也便于后续继续扩展交互逻辑。
"""

import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from landmarker import HandLandmarker, analyze_hand_data, draw_marker
from rectangle import UIButton
from stroke_filter import StrokeSmoother


class CONFIG:
    """
    项目运行参数集中配置。

    这里统一收纳窗口尺寸、工具栏布局、手势阈值、滤波参数等内容，
    这样当交互手感或 UI 布局需要微调时，可以优先在这里调整，
    不必在主逻辑中到处查找硬编码常量。
    """

    WIN_NAME = "Quantum Paint AR"  # OpenCV 窗口标题，会显示在程序窗口顶部。
    FRAME_W, FRAME_H = 1280, 720  # 摄像头采集与显示使用的目标分辨率，直接影响画布尺寸和按钮布局基准。

    UI_DEADZONE_Y = 135  # 顶部工具栏下方的绘图死区高度，指针落在这条线以上时不会在画布上落笔。
    FPS_SAMPLES = 30  # HUD 计算滚动帧率时保留的历史帧数，值越大显示越稳，但响应越慢。

    GESTURE_CONFIRM_SEC = 2.0  # 触发全局手势指令前需要持续保持的时间，当前主要用于握拳清屏。
    GESTURE_GRACE_SEC = 0.3  # 手势短暂丢失时允许继续沿用上次识别结果的宽限时间，避免轻微抖动导致中断。

    STROKE_MIN_CUTOFF = 1.2  # 一欧元滤波的最小截止频率，值越小越稳，值越大越跟手。
    STROKE_BETA = 0.04  # 一欧元滤波的速度响应系数，移动越快时它越决定滤波要放松到什么程度。
    STROKE_D_CUTOFF = 1.0  # 对速度本身做平滑时使用的截止频率，用来避免速度估计过于抖动。
    STROKE_DROPOUT_SEC = 0.12  # 允许短时丢点仍保留笔迹上下文的时间窗口，超过后会彻底重置当前一笔。

    TOOLBAR_TOP = 20  # 工具栏上边界的 y 坐标。
    TOOLBAR_BOTTOM = 105  # 工具栏下边界的 y 坐标。
    TOOLBAR_LEFT = 310  # 工具栏左边界的 x 坐标。
    TOOLBAR_RIGHT = 1270  # 工具栏右边界的 x 坐标，决定整条工具栏的总宽度。
    TOOLBAR_BG = (35, 35, 35)  # 工具栏半透明底板的主色，使用 BGR 顺序。
    TOOLBAR_BORDER = (80, 80, 80)  # 工具栏边框颜色，用于提升按钮区域与背景的分离感。

    HUD_X = 20  # HUD 面板左上角的 x 坐标。
    HUD_Y = 20  # HUD 面板左上角的 y 坐标。
    HUD_W = 280  # HUD 面板宽度，影响文字区与颜色指示区的布局。
    HUD_H = 115  # HUD 面板高度，决定状态信息的垂直排布空间。

    FONT_PATH = "C:/Windows/Fonts/STKAITI.TTF"  # HUD 中文字体路径；当前使用华文楷体，不存在时会自动回退到默认字体。


class WebcamStream:
    """
    摄像头异步读取器。

    由于摄像头读取本身可能阻塞，主线程如果直接 `read()`，
    会影响界面刷新、手部检测提交和笔迹绘制的连贯性。
    因此这里使用后台线程持续拉取最新帧，主循环只取副本。
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_W)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_H)
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.thread = None
        self.read_lock = threading.Lock()

    def start(self):
        """
        启动后台采集线程。

        如果已经启动，则直接返回自身，避免重复创建线程。
        """
        if self.started:
            return self

        self.started = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        """
        持续读取摄像头最新画面，并在锁保护下更新共享状态。

        主线程只负责读取 `self.frame` 的副本，
        实际采集动作统一在这里完成。
        """
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """
        读取当前缓存的最新帧副本。

        返回副本而不是原对象，目的是避免主线程在渲染时
        意外修改后台线程仍可能替换的同一块内存。
        """
        with self.read_lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        """停止采集并释放摄像头资源。"""
        self.started = False
        self.stream.release()


class HUDManager:
    """
    负责左上角 HUD 状态面板的绘制。

    HUD 显示的是用户当前最关心的状态：
    - 正在用什么工具
    - 当前是否处于绘图状态
    - 实时帧率
    - 当前滤波响应程度
    - 当前颜色
    """

    def __init__(self):
        self.panel_rect = (CONFIG.HUD_X, CONFIG.HUD_Y, CONFIG.HUD_W, CONFIG.HUD_H)
        self._init_fonts()

    def _init_fonts(self):
        """
        加载 HUD 用字体。

        优先尝试项目当前指定的中文字体；
        如果本机没有这套字体，则退回到 PIL 默认字体，
        保证界面至少还能正常显示。
        """
        try:
            self.font_main = ImageFont.truetype(CONFIG.FONT_PATH, 18)
            self.font_sub = ImageFont.truetype(CONFIG.FONT_PATH, 16)
        except OSError:
            self.font_main = ImageFont.load_default()
            self.font_sub = self.font_main

    def render(self, frame, pen_type, is_drawing, fps, response_ratio, color):
        """
        在画面左上角绘制状态面板。

        这里先用 OpenCV 画半透明底板和颜色指示圆，
        再借助 PIL 处理中文文本绘制，最后写回到原图像。
        这样可以同时兼顾 OpenCV 图元效率和中文显示效果。
        """
        panel_x, panel_y, panel_w, panel_h = self.panel_rect
        roi = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        overlay = roi.copy()

        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, roi, 0.25, 0, roi)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )

        indicator_center = (panel_x + panel_w - 45, panel_y + panel_h // 2)
        cv2.circle(frame, indicator_center, 22, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, indicator_center, 18, color, -1)

        text_area = frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w - 80]
        pil_img = Image.fromarray(cv2.cvtColor(text_area, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        mode_text = "橡皮擦" if pen_type == "eraser" else "发光笔刷"
        status_text = "正在绘画" if is_drawing else "待机中"
        status_color = (50, 255, 50) if is_drawing else (180, 180, 180)

        draw.text((15, 10), f"绘画模式: {mode_text}", font=self.font_main, fill=(50, 180, 255))
        draw.text((15, 35), f"交互状态: {status_text}", font=self.font_main, fill=status_color[::-1])
        draw.text((15, 60), f"系统帧率: {int(fps)} FPS", font=self.font_sub, fill=(255, 255, 0))
        draw.text((15, 85), f"滤波响应: {response_ratio:.2f}", font=self.font_sub, fill=(255, 150, 255))

        frame[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w - 80] = cv2.cvtColor(
            np.array(pil_img),
            cv2.COLOR_RGB2BGR,
        )
        return frame


class QuantumPaintApp:
    """
    应用主控制器。

    这个类本身不去实现底层检测或滤波算法，
    它的职责是协调各个模块，让每一帧按固定顺序流转：
    1. 读取摄像头画面
    2. 提交手部检测
    3. 分析检测结果
    4. 处理手势、工具栏、绘图和光标
    5. 组合画布与 HUD 后显示
    """

    def __init__(self):
        self.vs = WebcamStream().start()
        self.hands = HandLandmarker()
        self.hud = HUDManager()

        self.current_color = (0, 255, 0)
        self.current_size = 5
        self.pen_type = "brush"
        self.is_drawing_active = False

        self.canvas = None
        self.prev_draw_pt = None
        self.current_alpha = 0.0
        self.display_alpha = 0.0
        self.stroke_smoother = StrokeSmoother(
            min_cutoff=CONFIG.STROKE_MIN_CUTOFF,
            beta=CONFIG.STROKE_BETA,
            d_cutoff=CONFIG.STROKE_D_CUTOFF,
            dropout_tolerance=CONFIG.STROKE_DROPOUT_SEC,
        )

        self.active_gesture = None
        self.gesture_start_time = 0.0
        self.last_seen_time = 0.0

        self.dock_buttons = self._init_dock()
        self.prev_time = time.time()
        self.display_fps = 0.0
        self.fps_list = []

    def _init_dock(self):
        """
        创建顶部工具栏按钮。

        当前工具栏由三组按钮组成：
        - 颜色选择
        - 工具切换（清空 / 橡皮 / 画笔）
        - 笔刷尺寸

        每个按钮都通过 `id` 保存语义信息，后续点击时统一分发处理。
        """
        buttons = []
        base_y = 62

        colors = [
            (255, 255, 255),
            (91, 183, 230),
            (22, 45, 93),
            (35, 51, 235),
            (51, 134, 240),
            (85, 254, 255),
            (76, 250, 117),
            (61, 205, 105),
            (253, 252, 116),
            (245, 30, 0),
            (247, 61, 234),
            (192, 49, 111),
        ]
        for index, color in enumerate(colors):
            button = UIButton(350 + index * 45, base_y, 30, color, is_circle=True)
            button.id = ("color", color)
            buttons.append(button)

        tools = [("clear", "clear"), ("eraser", "eraser"), ("brush", "brush")]
        for index, (icon, name) in enumerate(tools):
            button = UIButton(930 + index * 65, base_y, 55, (200, 200, 200), icon_type=icon)
            button.id = ("tool", name)
            buttons.append(button)

        for index, size in enumerate([5, 15, 30]):
            button = UIButton(1135 + index * 45, base_y, 40, (100, 100, 100), label=str(size))
            button.id = ("size", size)
            buttons.append(button)

        return buttons

    def _update_fps(self, timestamp: float) -> None:
        """
        更新滚动平均帧率。

        这里不是直接用单帧耗时算 FPS，
        而是维护一个短窗口平均值，目的是让 HUD 数字更稳定、
        不至于每一帧都剧烈抖动。
        """
        frame_time = timestamp - self.prev_time
        self.fps_list.append(frame_time if frame_time > 0 else 0.033)
        if len(self.fps_list) > CONFIG.FPS_SAMPLES:
            self.fps_list.pop(0)
        self.display_fps = 1 / (sum(self.fps_list) / len(self.fps_list))
        self.prev_time = timestamp

    def _reset_drawing_state(self) -> None:
        """
        重置一笔绘制过程中的临时状态。

        这里不会删除画布内容，
        只会清除“当前这条笔迹”的上下文信息，
        例如上一绘制点、滤波器内部状态和界面展示的响应值。
        """
        self.prev_draw_pt = None
        self.stroke_smoother.reset()
        self.current_alpha = 0.0
        self.display_alpha = 0.0

    def _get_pointer(self, left_pos, right_pos):
        """
        选出用于界面交互的当前指针位置。

        规则是：
        - 优先使用右手食指
        - 若右手不存在，再退回左手食指

        这个指针用于工具栏悬停和光标显示，
        不等同于一定会参与绘图。
        """
        if right_pos:
            return right_pos[8]
        if left_pos:
            return left_pos[8]
        return None

    def _get_draw_input(self, fingers, left_pos, right_pos):
        """
        选出当前真正用于绘图的输入点。

        与 `_get_pointer()` 的区别在于：
        - 这里只接受“对应食指处于伸出状态”的点
        - 必须位于工具栏死区下方，才能落笔

        这样做可以避免用户在操作工具栏时误画到画布上。
        """
        if right_pos and fingers.get("RIGHT_INDEX"):
            point = right_pos[8]
        elif left_pos and fingers.get("LEFT_INDEX"):
            point = left_pos[8]
        else:
            point = None

        if point and point[1] > CONFIG.UI_DEADZONE_Y:
            return point
        return None

    def _get_stroke_style(self):
        """
        根据当前工具状态解析笔触样式。

        - 橡皮擦本质上是用黑色粗线覆盖画布。
        - 普通画笔会绘制两层线条，形成更亮、更有体积感的笔触效果。
        """
        is_eraser = self.pen_type == "eraser"
        color = (0, 0, 0) if is_eraser else self.current_color
        outer_thickness = self.current_size * 5 if is_eraser else self.current_size * 4
        inner_thickness = None if is_eraser else self.current_size * 2
        return color, outer_thickness, inner_thickness

    def _draw_stroke_segment(self, draw_pt) -> None:
        """
        在画布上绘制一段笔迹。

        第一帧只有当前位置，没有前一个点，因此只记录起点；
        从第二帧开始，才会把上一点和当前点连接成线段。
        """
        if not self.prev_draw_pt:
            self.prev_draw_pt = draw_pt
            return

        color, outer_thickness, inner_thickness = self._get_stroke_style()
        cv2.line(self.canvas, self.prev_draw_pt, draw_pt, color, outer_thickness, cv2.LINE_AA)
        if inner_thickness is not None:
            cv2.line(self.canvas, self.prev_draw_pt, draw_pt, color, inner_thickness, cv2.LINE_AA)
        self.prev_draw_pt = draw_pt

    def _draw_toolbar(self, frame) -> None:
        """
        绘制顶部半透明工具栏背景。

        按钮自身内容由每个 `UIButton` 负责绘制，
        这里仅负责统一绘制底板和边框，形成一个完整的工具区。
        """
        roi = frame[
            CONFIG.TOOLBAR_TOP:CONFIG.TOOLBAR_BOTTOM,
            CONFIG.TOOLBAR_LEFT:CONFIG.TOOLBAR_RIGHT,
        ]
        overlay = roi.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (CONFIG.TOOLBAR_RIGHT - CONFIG.TOOLBAR_LEFT, CONFIG.TOOLBAR_BOTTOM - CONFIG.TOOLBAR_TOP),
            CONFIG.TOOLBAR_BG,
            -1,
        )
        cv2.addWeighted(overlay, 0.5, roi, 0.5, 0, roi)
        cv2.rectangle(
            frame,
            (CONFIG.TOOLBAR_LEFT, CONFIG.TOOLBAR_TOP),
            (CONFIG.TOOLBAR_RIGHT, CONFIG.TOOLBAR_BOTTOM),
            CONFIG.TOOLBAR_BORDER,
            1,
            cv2.LINE_AA,
        )

    def _draw_hover_feedback(self, frame, pointer, button, progress) -> None:
        """
        为当前悬停按钮绘制视觉反馈。

        反馈由两部分组成：
        - 准星：表示当前指针中心位置
        - 进度环：表示停留点击已经累计到什么程度
        """
        cv2.drawMarker(frame, pointer, (255, 255, 255), cv2.MARKER_CROSS, 15, 1)
        if progress <= 0:
            return

        radius = button.size // 2 + 8
        cv2.ellipse(
            frame,
            (button.x, button.y),
            (radius, radius),
            0,
            -90,
            int(progress * 360) - 90,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

    def run(self):
        """
        启动主循环，直到用户主动退出程序。

        每一轮循环都会完成以下工作：
        1. 获取最新摄像头画面
        2. 更新时间与 FPS
        3. 执行检测、手势、绘图和界面逻辑
        4. 合成画面并显示
        5. 处理键盘输入
        """
        cv2.namedWindow(CONFIG.WIN_NAME, cv2.WINDOW_NORMAL)

        while True:
            frame = self.vs.read()
            if frame is None:
                continue

            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                # 画布与视频帧尺寸保持一致，便于直接相加合成。
                self.canvas = np.zeros_like(frame)

            timestamp = time.time()
            self._update_fps(timestamp)
            self._logic_step(frame, timestamp)

            combined = cv2.add(frame, self.canvas)
            combined = self.hud.render(
                combined,
                self.pen_type,
                self.is_drawing_active,
                self.display_fps,
                self.display_alpha,
                self.current_color,
            )

            cv2.imshow(CONFIG.WIN_NAME, combined)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            if key == ord(" "):
                # 空格键仅切换“是否允许绘图”，不清空已有画布。
                self.is_drawing_active = not self.is_drawing_active

        self._cleanup()

    def _logic_step(self, frame, timestamp):
        """
        处理一帧中的主要业务逻辑。

        这个方法是主循环里的核心分发点，
        用来把检测结果依次送到各个子流程中：
        骨架绘制、手势命令、工具栏交互、笔迹绘制和光标渲染。
        """
        self.hands.detect_async(frame)
        result = self.hands.result
        left_pos, right_pos, fingers, gesture = analyze_hand_data(frame, result)

        frame[:] = draw_marker(frame, result)
        self._handle_gesture_commands(frame, gesture)

        pointer = self._get_pointer(left_pos, right_pos)
        self._handle_ui_interaction(frame, pointer)
        self._handle_drawing(fingers, left_pos, right_pos, timestamp)
        self._draw_cursor(frame, pointer)

    def _handle_gesture_commands(self, frame, gesture):
        """
        处理全局手势指令。

        当前只支持一个系统级手势：
        - `FIST`：持续保持一段时间后清空画布

        这里专门加入了确认时长与宽限窗口，
        是为了避免轻微识别抖动导致清空动作误触发或频繁中断。
        """
        gesture_name = gesture["name"]
        gesture_center = gesture["center"]
        now = time.time()

        if gesture_name == "FIST":
            if self.active_gesture != "FIST":
                self.active_gesture = "FIST"
                self.gesture_start_time = now
            self.last_seen_time = now

            elapsed = now - self.gesture_start_time
            progress = min(1.0, elapsed / CONFIG.GESTURE_CONFIRM_SEC)
            cv2.ellipse(
                frame,
                gesture_center,
                (40, 40),
                0,
                -90,
                int(progress * 360) - 90,
                (0, 0, 255),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "CLEARING",
                (gesture_center[0] - 40, gesture_center[1] + 60),
                0,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            if elapsed >= CONFIG.GESTURE_CONFIRM_SEC:
                self._clear_canvas(reset_drawing_active=False)
                self.active_gesture = None
            return

        if self.active_gesture == "FIST" and (now - self.last_seen_time) > CONFIG.GESTURE_GRACE_SEC:
            self.active_gesture = None

    def _handle_drawing(self, fingers, left_pos, right_pos, timestamp):
        """
        处理当前帧的落笔逻辑。

        主要流程如下：
        1. 判断当前是否允许绘图
        2. 取得有效指尖点
        3. 把原始点送入平滑器
        4. 将平滑后的结果转换成画布线段

        这里专门保留了短时丢点连续性：
        如果只是很短暂地失去检测，不会立刻把上一笔上下文全清掉。
        """
        if not self.is_drawing_active:
            self._reset_drawing_state()
            return

        filtered_point = self.stroke_smoother.process(
            self._get_draw_input(fingers, left_pos, right_pos),
            timestamp,
        )
        self.current_alpha = self.stroke_smoother.response_ratio
        self.display_alpha = 0.2 * self.current_alpha + 0.8 * self.display_alpha

        if filtered_point is None:
            # 只有在平滑器自己确认“这不是短时丢点，而是完整重置”时，
            # 才清空上一笔的连接点，避免短暂漏检造成明显断笔。
            if self.stroke_smoother.last_output is None:
                self.prev_draw_pt = None
            return

        draw_pt = (int(round(filtered_point[0])), int(round(filtered_point[1])))
        self._draw_stroke_segment(draw_pt)

    def _handle_ui_interaction(self, frame, pointer):
        """
        处理顶部工具栏交互。

        如果当前有指针，就更新每个按钮的悬停进度；
        如果没有指针，则主动喂一个无效坐标给按钮，
        让按钮自行清空悬停状态。
        """
        self._draw_toolbar(frame)

        for button in self.dock_buttons:
            if pointer:
                is_clicked, progress = button.update(pointer[0], pointer[1])
                if button.is_hovered:
                    self._draw_hover_feedback(frame, pointer, button, progress)
                if is_clicked:
                    self._execute_action(button)
            else:
                button.update(-1, -1)
            button.draw(frame)

    def _execute_action(self, button):
        """
        执行工具栏按钮对应动作。

        这里不区分按钮具体来源，
        统一通过 `button.id` 里的语义信息做分发，
        便于后续继续扩充按钮类型。
        """
        action_type, value = button.id
        if action_type == "color":
            self.current_color = value
        elif action_type == "size":
            self.current_size = value
        elif action_type == "tool":
            if value == "clear":
                self._clear_canvas(reset_drawing_active=True)
            else:
                self.pen_type = value

    def _clear_canvas(self, reset_drawing_active):
        """
        清空画布，并同步重置临时笔迹状态。

        这样做可以保证：
        - 画布消失时，不会残留上一笔的滤波上下文
        - 下一次重新开始绘制时，不会从旧状态突然接出一条线
        """
        if self.canvas is not None:
            self.canvas = np.zeros_like(self.canvas)
        self._reset_drawing_state()
        if reset_drawing_active:
            self.is_drawing_active = False

    def _draw_cursor(self, frame, pointer):
        """
        在画面中绘制当前画笔光标。

        只有当指针位于工具栏死区下方时才显示，
        这样用户在操作工具栏按钮时不会被画布光标干扰。
        """
        if not pointer or pointer[1] <= CONFIG.UI_DEADZONE_Y:
            return

        cv2.circle(frame, pointer, self.current_size + 10, self.current_color, 2, cv2.LINE_AA)
        cv2.circle(frame, pointer, 3, (255, 255, 255), -1)
        if self.is_drawing_active:
            # 呼吸感脉冲只在允许绘图时显示，用于增强“当前可落笔”的反馈。
            pulse = int(abs(np.sin(time.time() * 5)) * 5)
            cv2.circle(frame, pointer, self.current_size + 10 + pulse, self.current_color, 1, cv2.LINE_AA)

    def _cleanup(self):
        """释放摄像头、检测器和窗口资源。"""
        self.vs.stop()
        self.hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    QuantumPaintApp().run()
