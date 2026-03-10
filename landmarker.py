import time
import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker
import numpy as np
import math
import cv2

class HandLandmarker():
    """手部关键点检测与语义分析类"""
    def __init__(self):
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.result = None
        self.last_timestamp_ms = 0
        self.createLandmarker()

    def createLandmarker(self):
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_img: mp.Image, timestamp_ms: int):
            self.result = result

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
        """发送异步检测请求"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        ts = int(time.time() * 1000)
        if ts <= self.last_timestamp_ms: ts = self.last_timestamp_ms + 1
        self.last_timestamp_ms = ts
        self.landmarker.detect_async(mp_image, ts)

    def close(self):
        if self.landmarker: self.landmarker.close()

def draw_marker(rgb_img, result: mp.tasks.vision.HandLandmarkerResult):
    """绘制手部骨架"""
    if result is None or not result.hand_landmarks: return rgb_img
    out = np.copy(rgb_img)
    for landmarks in result.hand_landmarks:
        mp.tasks.vision.drawing_utils.draw_landmarks(
            out, landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
            mp.tasks.vision.drawing_styles.get_default_hand_connections_style())
    return out

def analyze_hand_data(img, result: mp.tasks.vision.HandLandmarkerResult):
    """
    高度集成的核心数据分析函数
    一次性处理：坐标转换、指线状态判定、语义手势识别
    返回: (left_pos, right_pos, finger_states, gesture_info)
    """
    left_pos, right_pos = [], []
    finger_states = {'LEFT_INDEX': False, 'RIGHT_INDEX': False}
    gesture_info = {"name": None, "center": None}
    
    if result is None or not result.hand_landmarks:
        return left_pos, right_pos, finger_states, gesture_info
        
    h, w, _ = img.shape
    for index, hand in enumerate(result.handedness):
        label = hand[0].category_name
        lms = result.hand_landmarks[index]
        
        # 1. 坐标像素映射
        pos_list = []
        for lm in lms:
            pos_list.append((min(math.floor(lm.x * w), w - 1), min(math.floor(lm.y * h), h - 1)))
        
        if label == 'Right': right_pos = pos_list
        else: left_pos = pos_list
        
        # 2. 状态判定 (食指)
        is_index_up = lms[8].y < lms[6].y
        finger_states[f'{label.upper()}_INDEX'] = is_index_up
        
        # 3. 语义手势分析 (仅识别第一个检测到的有效手势)
        if gesture_info["name"] is None:
            is_middle_up = lms[12].y < lms[10].y
            is_ring_up = lms[16].y < lms[14].y
            is_pinky_up = lms[20].y < lms[18].y
            
            palm_center = pos_list[9] # 掌心点坐标
            
            # 握拳检测: 四指全收
            if not is_index_up and not is_middle_up and not is_ring_up and not is_pinky_up:
                gesture_info = {"name": "FIST", "center": palm_center}
                
    return left_pos, right_pos, finger_states, gesture_info
