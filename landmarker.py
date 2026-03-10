"""
QuantumPaint AR - 手部关键点检测与语义分析模块
基于 MediaPipe Tasks API 实现高性能的手势追踪与语义识别。
"""

import time
import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker
import numpy as np
import math
import cv2

class HandLandmarker():
    """
    手部关键点检测引擎封装类。
    采用异步流模式 (Live Stream) 以确保主线程不被阻塞。
    """
    def __init__(self):
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.result = None
        self.last_timestamp_ms = 0
        self._create_landmarker()

    def _create_landmarker(self):
        """初始化 MediaPipe 检测器，设置回调函数。"""
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, 
                          output_img: mp.Image, timestamp_ms: int):
            self.result = result

        # 配置检测选项
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=update_result)
        
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        """
        向检测器发送异步检测请求。
        
        Args:
            frame: OpenCV 格式的 BGR 图像矩阵。
        """
        # 1. 颜色空间转换与内存连续化
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        # 2. 构造 MediaPipe 图像对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. 时间戳校验（必须严格递增）
        ts = int(time.time() * 1000)
        if ts <= self.last_timestamp_ms:
            ts = self.last_timestamp_ms + 1
        self.last_timestamp_ms = ts
        
        # 4. 执行异步推理
        self.landmarker.detect_async(mp_image, ts)

    def close(self):
        """释放底层计算资源。"""
        if self.landmarker:
            self.landmarker.close()

def draw_marker(rgb_img, result: mp.tasks.vision.HandLandmarkerResult):
    """
    在图像上绘制手部骨架关键点。
    
    Args:
        rgb_img: 原始图像。
        result: MediaPipe 推理结果。
        
    Returns:
        绘制了骨架的图像副本。
    """
    if result is None or not result.hand_landmarks:
        return rgb_img
        
    out = np.copy(rgb_img)
    for landmarks in result.hand_landmarks:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            out, landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
            mp.tasks.vision.drawing_styles.get_default_hand_connections_style())
    return out

def analyze_hand_data(img, result: mp.tasks.vision.HandLandmarkerResult):
    """
    核心手部数据解析器：集成坐标映射、指尖状态判定与语义手势识别。
    
    Args:
        img: 用于获取分辨率参考的输入图像。
        result: 推理结果对象。
        
    Returns:
        tuple: (left_pos, right_pos, finger_states, gesture_info)
            - left_pos/right_pos: 像素坐标列表。
            - finger_states: 食指伸缩状态字典。
            - gesture_info: 包含识别到的手势名及中心点的字典。
    """
    left_pos, right_pos = [], []
    finger_states = {'LEFT_INDEX': False, 'RIGHT_INDEX': False}
    gesture_info = {"name": None, "center": None}
    
    if result is None or not result.hand_landmarks:
        return left_pos, right_pos, finger_states, gesture_info
        
    h, w, _ = img.shape
    
    # 遍历检测到的所有手部（通常为 1-2 只）
    for index, hand in enumerate(result.handedness):
        label = hand[0].category_name # 'Left' 或 'Right'
        landmarks = result.hand_landmarks[index]
        
        # 1. 坐标像素解算：从归一化 [0,1] 映射到 [0, W/H]
        pixel_positions = []
        for lm in landmarks:
            px = min(math.floor(lm.x * w), w - 1)
            py = min(math.floor(lm.y * h), h - 1)
            pixel_positions.append((px, py))
        
        if label == 'Right':
            right_pos = pixel_positions
        else:
            left_pos = pixel_positions
        
        # 2. 食指伸缩检测 (食指指尖 8 是否在第 2 关节 6 上方)
        is_index_up = landmarks[8].y < landmarks[6].y
        finger_states[f'{label.upper()}_INDEX'] = is_index_up
        
        # 3. 语义手势分析引擎 (目前仅激活首只手的识别)
        if gesture_info["name"] is None:
            # 判定其余三指状态
            is_middle_up = landmarks[12].y < landmarks[10].y
            is_ring_up   = landmarks[16].y < landmarks[14].y
            is_pinky_up  = landmarks[20].y < landmarks[18].y
            
            # 语义：握拳 (FIST) -> 四指全部收起
            if not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
                gesture_info = {"name": "FIST", "center": pixel_positions[9]}
                
    return left_pos, right_pos, finger_states, gesture_info
