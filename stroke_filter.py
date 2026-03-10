"""
笔迹滤波模块。

这个文件专门负责把手部关键点产生的原始轨迹做时间域平滑，
目标是在“静止时更稳”和“快速移动时不明显拖尾”之间取得平衡。

当前实现采用两层结构：
1. `LowPassFilter`：最基础的一阶低通滤波器，只处理单个标量。
2. `OneEuroFilter`：在低通滤波基础上，根据速度动态调整截止频率。
3. `StrokeSmoother`：把一欧元滤波器扩展到二维坐标，并处理短时丢点容错。
"""

import math
from typing import Optional

Point = tuple[float, float]


class LowPassFilter:
    """
    用于单个标量信号的一阶低通滤波器。

    这个类本身非常简单，只维护一个内部状态值：
    - 第一次输入时直接采用当前值，避免初始阶段出现突兀跳变。
    - 之后按 `alpha * 当前值 + (1 - alpha) * 上次结果` 做平滑。

    它不关心时间戳、速度或业务语义，只负责“给定 alpha 后怎么平滑”。
    """

    def __init__(self) -> None:
        self.initialized = False
        self.value = 0.0

    def reset(self, value: float) -> None:
        """
        重置滤波器内部状态。

        这个方法通常在以下场景使用：
        - 第一次接收到有效输入时，用当前值作为初始状态。
        - 发生长时间丢点后，重新用新的原始点重新起步。
        """
        self.value = value
        self.initialized = True

    def filter(self, value: float, alpha: float) -> float:
        """
        对一个新的标量值执行低通滤波。

        参数：
        - `value`：本次输入值。
        - `alpha`：平滑系数，越接近 1 越跟手，越接近 0 越平滑。

        返回：
        - 滤波后的结果，并同步写回内部状态。
        """
        if not self.initialized:
            self.reset(value)
            return value

        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


class OneEuroFilter:
    """
    一欧元滤波器的单轴实现。

    一欧元滤波器的核心思想是：
    - 速度慢时，使用更低的截止频率，让结果更稳，压掉细碎抖动。
    - 速度快时，自动提高截止频率，让结果更跟手，减少明显拖尾。

    这里它只处理一个标量轴（如 x 或 y），因此二维轨迹会由两个实例分别处理。
    """

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_raw: Optional[float] = None
        self.last_time: Optional[float] = None
        self.current_cutoff = min_cutoff

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """
        根据截止频率和采样时间间隔计算低通滤波 alpha。

        这个公式是一欧元滤波器的标准换算方式：
        截止频率越高，alpha 越大，滤波越“跟手”。
        """
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _clamp_dt(dt: float) -> float:
        """
        限制相邻采样间隔，避免极端帧时间把滤波行为拉坏。

        这样做有两个目的：
        - 防止时间间隔过小导致速度估计异常放大。
        - 防止时间间隔过大导致一次更新把状态拉得过猛。
        """
        return max(1e-3, min(dt, 0.1))

    def _seed(self, value: float, timestamp: float) -> None:
        """
        用当前值为滤波器建立初始状态。

        当滤波器还没有历史信息时，不做平滑，直接以当前值起步。
        这样可以避免第一帧因为缺少上下文而产生不必要的延迟。
        """
        self.x_filter.reset(value)
        self.dx_filter.reset(0.0)
        self.last_raw = value
        self.last_time = timestamp
        self.current_cutoff = self.min_cutoff

    def reset(self, value: Optional[float] = None, timestamp: Optional[float] = None) -> None:
        """
        清空内部状态，并可选地立刻用一个新值重新起步。

        如果同时提供 `value` 与 `timestamp`，重置后会马上完成一次种子初始化；
        如果不提供，则滤波器回到“尚未接收有效输入”的状态。
        """
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_raw = None
        self.last_time = timestamp
        self.current_cutoff = self.min_cutoff

        if value is not None and timestamp is not None:
            self._seed(value, timestamp)

    def filter(self, value: float, timestamp: float) -> float:
        """
        对当前轴的新输入执行一欧元滤波。

        流程分三步：
        1. 根据时间戳计算相邻输入间隔，并估计当前速度。
        2. 先对速度做一次低通滤波，避免速度本身太抖。
        3. 根据平滑后的速度动态更新截止频率，再对原始值做主滤波。

        返回值就是当前轴经过平滑后的结果。
        """
        if self.last_time is None or self.last_raw is None:
            self._seed(value, timestamp)
            return value

        dt = self._clamp_dt(timestamp - self.last_time)
        derivative = (value - self.last_raw) / dt
        derivative_hat = self.dx_filter.filter(derivative, self._alpha(self.d_cutoff, dt))

        self.current_cutoff = self.min_cutoff + self.beta * abs(derivative_hat)
        filtered = self.x_filter.filter(value, self._alpha(self.current_cutoff, dt))

        self.last_raw = value
        self.last_time = timestamp
        return filtered


class StrokeSmoother:
    """
    面向二维笔迹坐标的平滑器。

    这个类是业务层真正使用的入口，它做了两件事：
    - 分别对 x/y 轴应用一欧元滤波。
    - 处理短时丢点容错，避免检测偶发丢失时笔迹立刻断掉。

    额外维护的 `response_ratio` 用来给 HUD 显示“当前滤波响应程度”，
    它不是严格数学意义上的 alpha，而是一个便于观察的归一化指标。
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.04,
        d_cutoff: float = 1.0,
        dropout_tolerance: float = 0.12,
        responsiveness_cutoff: float = 12.0,
    ) -> None:
        self.filter_x = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.filter_y = OneEuroFilter(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff)
        self.dropout_tolerance = dropout_tolerance
        self.responsiveness_cutoff = responsiveness_cutoff
        self.last_seen_time: Optional[float] = None
        self.last_output: Optional[Point] = None
        self.response_ratio = 0.0

    def reset(self) -> None:
        """
        清空整个平滑器状态。

        一般在以下情况调用：
        - 用户主动停止绘图。
        - 画布被清空，需要重新开始一笔。
        - 丢点时间过长，需要彻底放弃上一段轨迹上下文。
        """
        self.filter_x.reset()
        self.filter_y.reset()
        self.last_seen_time = None
        self.last_output = None
        self.response_ratio = 0.0

    def _timed_out(self, timestamp: float) -> bool:
        """判断当前时间点是否已经超过短时丢点容忍窗口。"""
        return self.last_seen_time is not None and (timestamp - self.last_seen_time) > self.dropout_tolerance

    def _seed_filters(self, point: Point, timestamp: float) -> Point:
        """
        在长时间丢点后，用新的原始点重新建立滤波状态。

        这里直接返回原始点，是为了让恢复后的第一帧不要再额外产生拖尾。
        """
        x, y = point
        self.filter_x.reset(x, timestamp)
        self.filter_y.reset(y, timestamp)
        self.last_seen_time = timestamp
        self.last_output = point
        self.response_ratio = 0.0
        return point

    def _update_response_ratio(self) -> None:
        """
        更新给界面显示用的响应比例。

        当前实现取 x/y 两轴中更激进的截止频率，
        再按经验上限做归一化，范围控制在 0 到 1 之间。
        """
        cutoff = max(self.filter_x.current_cutoff, self.filter_y.current_cutoff)
        self.response_ratio = min(1.0, max(0.0, cutoff / self.responsiveness_cutoff))

    def process(self, point: Optional[Point], timestamp: float) -> Optional[Point]:
        """
        处理一个新的二维输入点。

        参数：
        - `point`：当前原始坐标；如果为 `None`，表示本帧没有拿到有效点。
        - `timestamp`：当前帧对应的时间戳，用于计算时间间隔。

        返回：
        - 平滑后的二维坐标；
        - 如果当前没有有效点，则返回 `None`。

        需要特别注意的行为：
        - 短时丢点时不会立刻清空滤波状态，这样恢复后更平滑。
        - 长时丢点时会彻底重置，避免把旧轨迹硬接到新轨迹上。
        """
        if point is None:
            if self._timed_out(timestamp):
                self.reset()
            return None

        if self._timed_out(timestamp):
            return self._seed_filters(point, timestamp)

        x, y = point
        filtered_point = (
            self.filter_x.filter(x, timestamp),
            self.filter_y.filter(y, timestamp),
        )
        self.last_seen_time = timestamp
        self.last_output = filtered_point
        self._update_response_ratio()
        return filtered_point
