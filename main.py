
from landmarker import *
from rectangle import *

# 创建颜色选择面板的矩形对象列表
def color_panel_rect():
    cpanel = list()
    # 定义 12 种颜色及其 BGR 值
    color_bgr = {"0": (255, 255, 255), "1": (91, 183, 230), "2": (22, 45, 93), "3": (35, 51, 235), "4": (51, 134, 240),
                 "5": (85, 254, 255), "6": (76, 250, 117), "7": (61, 205, 105), "8": (253, 252, 116), "9": (245, 30, 0),
                 "10": (247, 61, 234), "11": (192, 49, 111)}
    for key, value in color_bgr.items():
        # 在屏幕左侧纵向排列按钮
        cpanel.append(Rect(0, int(key)*60, 60, 60, value, alpha=0.1))
    return cpanel

# 创建功能面板（清屏、橡皮擦、画笔）
def pen_panel_rect():
    ppanel = list()
    for index in range(3):
        # 在屏幕右侧顶部排列按钮
        ppanel.append(Rect(1215, 20 + index * 60, 60, 60, (255, 255, 255), alpha=0.5))
    return ppanel

# 创建画笔粗细选择面板
def pen_size_rect():
    psize = list()
    for index, size in enumerate(range(5, 35, 5)):
        # 在功能面板下方排列粗细按钮
        psize.append(Rect(1215, 210+index*60, 60, 60, (0, 0, 0), str(size), alpha=0.3))
    return psize

def main_func():
    # 1. 初始化所有 UI 元素
    color_panel = color_panel_rect()
    pen_panel = pen_panel_rect()
    pen_size = pen_size_rect()

    # 按钮对应的图片路径和功能名称
    pen_images = {'0': 'images/clear.png', '1': 'images/eraser.png', '2': 'images/drawing.png'}
    pen_types = {'0': 'clear', '1': 'eraser', '2': 'brush'}

    # 2. 初始化手势检测器和摄像头
    hands = HandLandmarker()
    camera_live = cv2.VideoCapture(0)
    camera_live.set(3, 1280) # 设置宽度
    camera_live.set(4, 720)  # 设置高度
    cv2.namedWindow('Virtual Paint', cv2.WINDOW_NORMAL)

    # 3. 设置初始状态
    color = (0, 255, 0) # 默认颜色：绿色
    pensize = 5         # 默认粗细
    drawing = False     # 默认绘画状态：关闭
    canvas = None       # 画布对象（用于保存笔迹）
    pen_type = 'brush'  # 默认工具：画笔

    # 4. 进入实时处理循环
    while camera_live.isOpened():
        read, frame = camera_live.read()
        if not read:
            continue
        # 画面镜像处理，符合直觉
        frame = cv2.flip(frame, 1)
        # 异步检测当前帧的手势
        hands.detect_async(frame)
        
        # 捕获当前检测结果快照，防止处理过程中被异步更新修改
        current_result = hands.result
        
        # 在实时画面上绘制手势骨架
        frame = draw_marker(frame, current_result)

        # 获取手部关键点坐标和手指伸缩状态
        left_position, right_position = positions(frame, current_result)
        finger_statue = up_fingers(current_result)

        # 检查是否检测到了足够的手部关键点
        has_left = len(left_position) > 20
        has_right = len(right_position) > 20

        # 5. UI 交互逻辑：使用小指点击按钮
        # 遍历颜色面板
        for index in range(len(color_panel)):
            color_panel[index].draw_rect(frame)
            if isinstance(finger_statue, dict):
                # 如果左手或右手的小指伸出并悬停在颜色按钮上，则切换颜色
                if has_left and finger_statue.get('LEFT_PINKY') and color_panel[index].is_over(left_position[20][0], left_position[20][1]):
                    color = color_panel[index].color
                elif has_right and finger_statue.get('RIGHT_PINKY') and color_panel[index].is_over(right_position[20][0], right_position[20][1]):
                    color = color_panel[index].color

        # 遍历粗细面板
        for index in range(len(pen_size)):
            pen_size[index].draw_rect(frame)
            if isinstance(finger_statue, dict):
                if has_left and finger_statue.get('LEFT_PINKY') and pen_size[index].is_over(left_position[20][0], left_position[20][1]):
                    pensize = int(pen_size[index].text)
                elif has_right and finger_statue.get('RIGHT_PINKY') and pen_size[index].is_over(right_position[20][0], right_position[20][1]):
                    pensize = int(pen_size[index].text)

        # 绘制功能按钮的图标
        for key, value in pen_images.items():
            image = cv2.imread(value, cv2.IMREAD_UNCHANGED)
            frame = pen_panel[int(key)].add_image(frame, image)

        # 遍历功能按钮逻辑（清屏、切换工具）
        for key, value in pen_types.items():
            if isinstance(finger_statue, dict):
                if int(key) == 0: # 清屏按钮
                    if has_left and finger_statue.get('LEFT_PINKY') and pen_panel[int(key)].is_over(left_position[20][0], left_position[20][1]):
                        canvas = np.zeros_like(frame) # 重置画布
                        drawing = False
                    elif has_right and finger_statue.get('RIGHT_PINKY') and pen_panel[int(key)].is_over(right_position[20][0], right_position[20][1]):
                        canvas = np.zeros_like(frame)
                        drawing = False
                elif int(key) > 0: # 切换橡皮擦/画笔
                    if has_left and finger_statue.get('LEFT_PINKY') and pen_panel[int(key)].is_over(left_position[20][0], left_position[20][1]):
                        pen_type = value
                    elif has_right and finger_statue.get('RIGHT_PINKY') and pen_panel[int(key)].is_over(right_position[20][0], right_position[20][1]):
                        pen_type = value

        # 6. 绘图逻辑：使用食指绘图
        if drawing:
            if canvas is None:
                canvas = np.zeros_like(frame)
            if isinstance(finger_statue, dict):
                # 如果食指伸出，则根据当前工具（画笔或橡皮擦）在画布上画圆
                if has_left and finger_statue.get('LEFT_INDEX') and pen_type == 'brush':
                    cv2.circle(canvas, left_position[8], pensize, color, cv2.FILLED)
                elif has_right and finger_statue.get('RIGHT_INDEX') and pen_type == 'brush':
                    cv2.circle(canvas, right_position[8], pensize, color, cv2.FILLED)
                elif has_left and finger_statue.get('LEFT_INDEX') and pen_type == 'eraser':
                    cv2.circle(canvas, left_position[8], pensize, (0, 0, 0), cv2.FILLED)
                elif has_right and finger_statue.get('RIGHT_INDEX') and pen_type == 'eraser':
                    cv2.circle(canvas, right_position[8], pensize, (0, 0, 0), cv2.FILLED)

        # 将笔迹画布叠加到原始画面上
        if canvas is not None:
            frame = cv2.add(frame, canvas)

        # 7. 绘制顶部的状态栏 overlay
        status_bg = frame.copy()
        cv2.rectangle(status_bg, (300, 0), (980, 60), (50, 50, 50), -1)
        cv2.addWeighted(status_bg, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (320, 15), (360, 45), color, -1)
        cv2.rectangle(frame, (320, 15), (360, 45), (255, 255, 255), 2)
        status_msg = f"Tool: {pen_type.upper()} | Drawing: {'ON' if drawing else 'OFF'} | Size: {pensize}"
        cv2.putText(frame, status_msg, (380, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'D' to Toggle Drawing | 'ESC' to Exit", (420, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Virtual Paint', frame)

        # 监听按键
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # 按 ESC 退出
            break
        elif k == 100: # 按 'D' 键手动切换绘画开关
            drawing = not drawing

    # 释放资源
    camera_live.release()
    hands.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_func()
