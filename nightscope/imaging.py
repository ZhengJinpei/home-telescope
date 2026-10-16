"""图像处理：多帧降噪、夜间配色、自动对比度、十字线。

输入输出都是 RGB 的 numpy uint8 数组（高 × 宽 × 3）。按树莓派 Zero 2 W 的算力设计：
- 数码变倍交给相机的图像处理器（ISP）做，不占 CPU，见 hardware.PiCamera.set_zoom
- 夜间先转成单色再处理，数据量少三分之二
- 降噪用整数定点运算，配色和对比度用 256 项查找表
"""

from __future__ import annotations

import numpy as np

from .controller import NIGHT, State

# 绿色夜视：模仿传统夜视仪的荧光屏颜色，人眼对绿色最敏感
GREEN = np.array([0.35, 1.0, 0.45], dtype=np.float32)
WHITE = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def gray(frame: np.ndarray) -> np.ndarray:
    """RGB → 亮度，uint8。用 (R + 2G + B) / 4 近似，只要加法和移位。"""
    f = frame.astype(np.uint16)
    return ((f[..., 0] + 2 * f[..., 1] + f[..., 2]) >> 2).astype(np.uint8)


def mean_luma(frame: np.ndarray, step: int = 8) -> float:
    """画面平均亮度 0–1，隔 step 个像素取一个点。"""
    return float(gray(frame[::step, ::step]).mean() / 255.0)


def digital_zoom(frame: np.ndarray, factor: float, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """软件变倍：从画面中间裁出 1/factor，再放大到 out_size（宽, 高）。用最近邻。

    真机上变倍由相机 ISP 完成，这里给模拟相机和缩小推流画面用。
    """
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

    用 int16 定点数（4 位小数）。每帧新画面占 1/2^shift：shift 越大越干净，也越容易拖影；
    shift = 0 相当于不降噪。

    判断“在动”按 block × block 的小块算：小块里新画面和平均画面的平均差超过 motion_threshold 级
    就算动。噪点在小块里正负抵消（8×8 块噪声降到 1/8），真正移动的东西不会，所以夜里噪点再大也不会误判。
    只处理单色画面（高 × 宽）。
    """

    FRAC = 4

    def __init__(self, shift: int = 2, motion_threshold: int = 12, block: int = 8):
        self.shift = shift
        self.motion_threshold = motion_threshold
        self.block = block
        self._acc: np.ndarray | None = None

    def reset(self) -> None:
        self._acc = None

    def _moving(self, d: np.ndarray) -> np.ndarray:
        b = self.block
        h, w = d.shape[:2]
        hb, wb = h // b, w // b
        sums = d[:hb * b, :wb * b].reshape(hb, b, wb, b).sum(axis=(1, 3), dtype=np.int32)
        blocks = np.abs(sums) > (self.motion_threshold << self.FRAC) * b * b
        mask = np.ones((h, w), dtype=bool)  # 边上凑不满一块的像素不降噪
        mask[:hb * b, :wb * b].reshape(hb, b, wb, b)[:] = blocks[:, None, :, None]
        return mask

    def process(self, img: np.ndarray) -> np.ndarray:
        x = img.astype(np.int16) << self.FRAC
        if self._acc is None or self._acc.shape != x.shape or self.shift == 0:
            self._acc = x
            return img
        d = x - self._acc
        moving = self._moving(d)
        self._acc += d >> self.shift
        np.copyto(self._acc, x, where=moving)
        return (self._acc >> self.FRAC).astype(np.uint8)


def stretch_lut(gray_img: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5,
                step: int = 4, lift: bool = True) -> np.ndarray:
    """自动对比度的查找表：最暗的 1% 变黑、最亮的 0.5% 变白，返回 256 个 0–1 的值。

    用直方图找百分位，比排序快得多。lift=True 时再开平方，提亮暗部。
    """
    hist = np.bincount(gray_img[::step, ::step].ravel(), minlength=256)
    cdf = np.cumsum(hist) / max(hist.sum(), 1)
    lo = int(np.searchsorted(cdf, low_pct / 100))
    hi = int(np.searchsorted(cdf, high_pct / 100))
    span = max(hi - lo, 8)  # 太暗的画面别把噪点放大成满屏雪花
    v = np.clip((np.arange(256, dtype=np.float32) - lo) / span, 0.0, 1.0)
    return np.sqrt(v) if lift else v


def night_palette(frame: np.ndarray, palette: str, denoiser: TemporalDenoiser | None = None) -> np.ndarray:
    """夜间画面：green 绿色、white 黑白（先转单色），color 保留彩色只拉对比度。"""
    y = gray(frame)
    if palette == "color":
        img = frame
        if denoiser:
            # 只给亮度降噪，颜色用新画面：人眼对亮度噪点最敏感，计算量省三分之二
            yd = denoiser.process(y)
            corr = yd.astype(np.int16) - y
            img = np.clip(frame + corr[..., None], 0, 255).astype(np.uint8)
            y = yd
        lut = (stretch_lut(y, lift=False) * 255).astype(np.uint8)
        return np.take(lut, img)  # 比 lut[img] 快好几倍
    if denoiser:
        y = denoiser.process(y)
    tint = GREEN if palette == "green" else WHITE
    lut = (stretch_lut(y)[:, None] * tint * 255).astype(np.uint8)  # 256 × 3
    return np.take(lut, y, axis=0)


def draw_reticle(frame: np.ndarray, color: tuple[int, int, int] = (255, 60, 60), gap: int = 6, arm: int = 18,
                 inplace: bool = False) -> np.ndarray:
    """画面中心的十字线，中间留空不挡目标。"""
    out = frame if inplace else frame.copy()
    h, w = out.shape[:2]
    cx, cy = w // 2, h // 2
    out[cy, cx - gap - arm:cx - gap] = color
    out[cy, cx + gap:cx + gap + arm] = color
    out[cy - gap - arm:cy - gap, cx] = color
    out[cy + gap:cy + gap + arm, cx] = color
    return out


# 倍率越高手越抖，降噪要越轻，否则一动就拖影
DENOISE_SHIFT = {1: 2, 2: 1, 4: 1}


class Pipeline:
    """按当前状态把相机画面处理成显示和推流用的画面。"""

    def __init__(self):
        self.denoiser = TemporalDenoiser()
        self._last_key: tuple | None = None

    def process(self, frame: np.ndarray, state: State) -> np.ndarray:
        key = (state.mode, state.zoom, state.palette)
        if key != self._last_key:  # 换模式、变倍或换颜色后，旧的平均画面作废
            self.denoiser.reset()
            self._last_key = key
        if state.mode == NIGHT:
            self.denoiser.shift = DENOISE_SHIFT.get(state.zoom, 1)
            out = night_palette(frame, state.palette, self.denoiser)
        else:
            out = frame.copy() if state.reticle else frame
        if state.reticle:
            draw_reticle(out, inplace=True)
        return out
