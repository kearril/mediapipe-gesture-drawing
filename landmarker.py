import time
import mediapipe as mp
from mediapipe.tasks.python.vision import hand_landmarker
import numpy as np
import math
import cv2

# 手部关键点检测类，基于 MediaPipe Tasks API 封装
class HandLandmarker():
    def __init__(self):
        # 初始化模型对象和结果容器
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.result = None
        self.last_timestamp_ms = 0
        self.createLandmarker()

    def createLandmarker(self):
        # 回调函数：当 MediaPipe 完成一帧检测后，会调用此函数更新结果
        def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_img: mp.Image, timestamp_ms: int):
            self.result = result

        # 配置检测器选项
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM, # 使用直播流模式
            num_hands=2, # 最多检测两只手
            min_hand_detection_confidence=0.8,
            min_hand_presence_confidence=0.8,
            min_tracking_confidence=0.8,
            result_callback=update_result)
        # 从选项创建检测器实例
        self.landmarker = self.landmarker.create_from_options(options)

    def detect_async(self, frame):
        # 1. 将 OpenCV 的 BGR 格式转换为 MediaPipe 要求的 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. 确保内存数组是连续的，以避免底层报错
        rgb_frame = np.ascontiguousarray(rgb_frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 3. 生成单调递增的时间戳（直播流模式必需）
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        
        # 4. 发送异步检测请求
        self.landmarker.detect_async(mp_image, timestamp_ms)

    def close(self):
        # 释放检测器资源
        if self.landmarker:
            self.landmarker.close()

# 辅助函数：在图像上绘制手部骨架和关键点
def draw_marker(rgb_img, result: mp.tasks.vision.HandLandmarkerResult):
    if result is None or not hasattr(result, 'hand_landmarks') or not result.hand_landmarks:
        return rgb_img
    
    output_img = np.copy(rgb_img)
    for hand_landmarks in result.hand_landmarks:
        # 使用 MediaPipe 自带的绘图工具
        mp.tasks.vision.drawing_utils.draw_landmarks(
            output_img,
            hand_landmarks,
            mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS,
            mp.tasks.vision.drawing_styles.get_default_hand_landmarks_style(),
            mp.tasks.vision.drawing_styles.get_default_hand_connections_style())
    return output_img

# 辅助函数：获取所有关键点的像素坐标
def positions(img, result: mp.tasks.vision.HandLandmarkerResult):
    left_pos_list = []
    right_pos_list = []
    if result is None or not hasattr(result, 'hand_landmarks'):
        return left_pos_list, right_pos_list
        
    height, width, _ = img.shape
    for index, hand in enumerate(result.handedness):
        hand_label = hand[0].category_name # 获取是左手还是右手
        hand_landmarks = result.hand_landmarks[index]
        for id in hand_landmarks:
            # 将归一化坐标 (0-1) 转换为像素坐标 (0-width/height)
            px = min(math.floor(id.x * width), width - 1)
            py = min(math.floor(id.y * height), height - 1)
            if hand_label == 'Right':
                right_pos_list.append((px, py))
            elif hand_label == 'Left':
                left_pos_list.append((px, py))
    return left_pos_list, right_pos_list

# 辅助函数：判断食指和小指是否伸出
def up_fingers(result: mp.tasks.vision.HandLandmarkerResult):
    finger_statue = {'LEFT_INDEX': False, 'LEFT_PINKY': False,
                     'RIGHT_INDEX': False, 'RIGHT_PINKY': False}
    if result is None or not hasattr(result, 'hand_landmarks'):
        return finger_statue
        
    finger_tip_id = [hand_landmarker.HandLandmark.INDEX_FINGER_TIP,
                     hand_landmarker.HandLandmark.PINKY_TIP]
    
    for index, hand in enumerate(result.handedness):
        hand_label = hand[0].category_name
        hand_landmarks = result.hand_landmarks[index]
        for tip_id in finger_tip_id:
            finger_name = tip_id.name.split("_")[0] # 获取手指名称
            # 逻辑：如果指尖的 Y 坐标小于其下方第二个关节的 Y 坐标，则判定为伸出
            if hand_landmarks[tip_id].y < hand_landmarks[tip_id - 2].y:
                finger_statue[hand_label.upper() + "_" + finger_name] = True
    return finger_statue

