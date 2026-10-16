"""图像处理：数码变倍、多帧降噪、夜间配色、自动对比度、十字线。

输入输出都是 RGB 的 numpy uint8 数组（高 × 宽 × 3）。按树莓派 Zero 2 W 的算力设计：
只用简单的数组运算，统计量隔点采样。
"""

from __future__ import annotations

import numpy as np

from .controller import NIGHT, State

# 绿色夜视：模仿传统夜视仪的荧光屏颜色，人眼对绿色最敏感
GREEN = np.array([0.35, 1.0, 0.45], dtype=np.float32)


def luma(frame: np.ndarray) -> np.ndarray:
    """RGB → 亮度（float32，0–255）。"""
    f = frame.astype(np.float32)
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def mean_luma(frame: np.ndarray, step: int = 8) -> float:
    """画面平均亮度 0–1，隔 step 个像素取一个点。"""
    return float(luma(frame[::step, ::step]).mean() / 255.0)


def digital_zoom(frame: np.ndarray, factor: float, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """从画面中间裁出 1/factor，再放大到 out_size（宽, 高）。用最近邻，快。"""
    h, w = frame.shape[:2]
    out_w, out_h = out_size or (w, h)
    if factor <= 1 and (out_w, out_h) == (w, h):
        return frame
    cw, ch = w / max(factor, 1.0), h / max(factor, 1.0)
    x0, y0 = (w - cw) / 2, (h - ch) / 2
    xs = (x0 + (np.arange(out_w) + 0.5) * cw / out_w).astype(np.intp).clip(0, w - 1)
    ys = (y0 + (np.arange(out_h) + 0.5) * ch / out_h).astype(np.intp).clip(0, h - 1)
    return frame[ys[:, None], xs[None, :]]


class TemporalDenoiser:
    """多帧平均降噪：静止的地方越叠越干净，动的地方直接用新画面，不拖影。

    strength 是新画面占的比例（越小越干净、越容易拖影）；
    某个像素和平均值差超过 motion_threshold，就认为它在动。
    """

    def __init__(self, strength: float = 0.3, motion_threshold: float = 40.0):
        self.strength = strength
        self.motion_threshold = motion_threshold
        self._avg: np.ndarray | None = None

    def reset(self) -> None:
        self._avg = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32)
        if self._avg is None or self._avg.shape != f.shape:
            self._avg = f
            return frame
        diff = np.abs(f - self._avg).max(axis=-1, keepdims=True)
        alpha = np.where(diff > self.motion_threshold, 1.0, self.strength).astype(np.float32)
        self._avg += alpha * (f - self._avg)
        return self._avg.round().astype(np.uint8)


def auto_contrast(gray: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5, step: int = 4) -> np.ndarray:
    """把最暗的 1% 拉到黑、最亮的 0.5% 拉到白（float32，0–1）。"""
    sample = gray[::step, ::step]
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    span = max(float(hi - lo), 8.0)  # 太暗的画面别把噪点放大成满屏雪花
    return np.clip((gray - lo) / span, 0.0, 1.0)


def night_palette(frame: np.ndarray, palette: str) -> np.ndarray:
    """夜间配色：green 绿色、white 黑白、color 保留彩色只拉对比度。"""
    if palette == "color":
        out = frame.astype(np.float32)
        lo, hi = np.percentile(out[::4, ::4], [1.0, 99.5])
        out = np.clip((out - lo) / max(float(hi - lo), 8.0), 0.0, 1.0)
        return (out * 255).astype(np.uint8)
    y = auto_contrast(luma(frame))
    y = np.sqrt(y)  # 提亮暗部，夜里多看清一点
    if palette == "green":
        return (y[..., None] * GREEN * 255).astype(np.uint8)
    return np.repeat((y * 255).astype(np.uint8)[..., None], 3, axis=-1)


def draw_reticle(frame: np.ndarray, color: tuple[int, int, int] = (255, 60, 60), gap: int = 6, arm: int = 18) -> np.ndarray:
    """画面中心的十字线，中间留空不挡目标。"""
    out = frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    out[cy, cx - gap - arm:cx - gap] = color
    out[cy, cx + gap:cx + gap + arm] = color
    out[cy - gap - arm:cy - gap, cx] = color
    out[cy + gap:cy + gap + arm, cx] = color
    return out


class Pipeline:
    """按当前状态把相机画面处理成显示和推流用的画面。"""

    def __init__(self, out_size: tuple[int, int] | None = None):
        self.out_size = out_size
        self.denoiser = TemporalDenoiser()
        self._last_key: tuple | None = None

    def process(self, frame: np.ndarray, state: State) -> np.ndarray:
        key = (state.mode, state.zoom)
        if key != self._last_key:  # 换模式或变倍后，旧的平均画面作废
            self.denoiser.reset()
            self._last_key = key
        out = digital_zoom(frame, state.zoom, self.out_size)
        if state.mode == NIGHT:
            # 倍率越高手越抖，降噪越要轻
            self.denoiser.strength = {1: 0.3, 2: 0.45, 4: 0.6}[state.zoom]
            out = night_palette(self.denoiser.process(out), state.palette)
        if state.reticle:
            out = draw_reticle(out)
        return out
