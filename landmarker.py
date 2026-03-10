"""
手部关键点检测与手势分析模块。

这个文件承担两类职责：
1. 包装 MediaPipe 的异步手部检测器，持续接收最新检测结果。
2. 把检测结果整理成主程序更容易使用的数据结构，例如：
   - 左右手关键点像素坐标
   - 食指是否伸出
   - 当前是否命中语义手势（如 FIST）
"""
import time
import cv2
import mediapipe as mp
import numpy as np


def _to_pixel(value: float, limit: int) -> int:
    """
    将 0 到 1 之间的归一化坐标转换成图像坐标。

    MediaPipe 输出的关键点坐标通常是归一化值，
    这里统一把它限制在图像边界内，避免后续访问越界。
    """
    return max(0, min(int(value * limit), limit - 1))


def _landmarks_to_pixels(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    """
    把整组关键点从归一化坐标映射成像素坐标。

    返回值是按原关键点索引顺序排列的 `(x, y)` 列表，
    这样主程序可以继续通过固定索引访问指定关节点。
    """
    return [(_to_pixel(landmark.x, width), _to_pixel(landmark.y, height)) for landmark in landmarks]


def _is_finger_up(landmarks, tip_index: int, joint_index: int) -> bool:
    """
    根据指尖与下方关节的相对高度判断手指是否伸出。

    这里使用的是非常轻量的几何规则：
    当指尖 y 坐标小于下方关节 y 坐标时，视为该手指向上伸出。
    这个规则简单，但足够满足当前绘画交互需要。
    """
    return landmarks[tip_index].y < landmarks[joint_index].y


def _detect_gesture(landmarks, pixel_positions, index_up: bool):
    """
    识别当前这只手是否命中已支持的手势。

    目前项目只支持一个全局手势：
    - `FIST`：食指、中指、无名指、小拇指都未伸出

    返回结构保持统一：
    - `name`：手势名，没有命中时为 `None`
    - `center`：手势中心点，这里使用 9 号关键点作为参考位置
    """
    middle_up = _is_finger_up(landmarks, 12, 10)
    ring_up = _is_finger_up(landmarks, 16, 14)
    pinky_up = _is_finger_up(landmarks, 20, 18)

    if not index_up and not middle_up and not ring_up and not pinky_up:
        return {"name": "FIST", "center": pixel_positions[9]}

    return {"name": None, "center": None}


class HandLandmarker:
    """
    MediaPipe 异步手部检测器的轻量包装类。

    主程序每帧调用 `detect_async()` 提交最新画面，
    检测结果会通过回调异步写入 `self.result`，
    主循环再读取这份最近结果参与后续处理。
    """

    def __init__(self) -> None:
        self.result = None
        self.last_timestamp_ms = 0
        self.detector = self._create_detector()

    def _create_detector(self):
        """
        创建直播流模式下的手部检测器。

        当前配置固定为：
        - 最多检测两只手
        - 使用 live stream 模式
        - 三项置信度阈值都设为 0.5
        """

        def update_result(result: mp.tasks.vision.HandLandmarkerResult, _output_img: mp.Image, _timestamp_ms: int):
            # 主程序只关心“当前最新结果”，因此这里直接覆盖保存。
            self.result = result

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=update_result,
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _next_timestamp_ms(self) -> int:
        """
        生成严格递增的毫秒级时间戳。

        MediaPipe 的异步接口要求时间戳单调递增，
        所以即使系统时钟精度不足，也要手动保证不会回退。
        """
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def detect_async(self, frame) -> None:
        """
        提交一帧 BGR 图像给异步手部检测器。

        这里会先把 OpenCV 的 BGR 图像转成 RGB，
        再包装成 MediaPipe 需要的 `mp.Image` 结构。
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb_frame),
        )
        self.detector.detect_async(mp_image, self._next_timestamp_ms())

    def close(self) -> None:
        """释放底层检测器资源。"""
        if self.detector:
            self.detector.close()


def draw_marker(rgb_img, result: mp.tasks.vision.HandLandmarkerResult):
    """
    在图像副本上绘制检测到的手部骨架。

    这里返回的是新的图像副本，而不是原地修改原输入引用，
    这样主流程在组合不同渲染层时更安全直观。
    """
    if result is None or not result.hand_landmarks:
        return rgb_img

    output = np.copy(rgb_img)
    for landmarks in result.hand_landmarks:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            output,
            landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
            mp.tasks.vision.drawing_styles.get_default_hand_connections_style(),
        )
    return output


def analyze_hand_data(img, result: mp.tasks.vision.HandLandmarkerResult):
    """
    把原始检测结果整理成主程序直接可用的数据。

    返回四项内容：
    - `left_pos`：左手关键点像素坐标列表；没有左手时为空列表
    - `right_pos`：右手关键点像素坐标列表；没有右手时为空列表
    - `finger_states`：当前只维护左右食指是否伸出
    - `gesture_info`：当前识别到的语义手势与中心点

    当前策略偏轻量，优先满足绘画交互，不追求完整手势系统。
    """
    left_pos, right_pos = [], []
    finger_states = {"LEFT_INDEX": False, "RIGHT_INDEX": False}
    gesture_info = {"name": None, "center": None}

    if result is None or not result.hand_landmarks:
        return left_pos, right_pos, finger_states, gesture_info

    height, width = img.shape[:2]
    for index, handedness in enumerate(result.handedness):
        label = handedness[0].category_name
        landmarks = result.hand_landmarks[index]
        pixel_positions = _landmarks_to_pixels(landmarks, width, height)
        index_up = _is_finger_up(landmarks, 8, 6)

        if label == "Right":
            right_pos = pixel_positions
        else:
            left_pos = pixel_positions

        # 主程序当前只用食指状态控制是否落笔，因此这里先维护食指即可。
        finger_states[f"{label.upper()}_INDEX"] = index_up

        # 手势只保留第一条命中结果，避免同一帧被多个结果反复覆盖。
        if gesture_info["name"] is None:
            gesture_info = _detect_gesture(landmarks, pixel_positions, index_up)

    return left_pos, right_pos, finger_states, gesture_info
