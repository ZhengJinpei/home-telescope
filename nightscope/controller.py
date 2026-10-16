"""夜视仪的“大脑”：按键、昼夜模式切换、红外灯、变倍、拍照录像。

这里只做决定，不碰硬件：每次调用返回一串命令，由 app.py 交给硬件执行。
这样在电脑上就能把所有逻辑测清楚。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from .design import IR_LEVELS

DAY, NIGHT = "day", "night"
ZOOMS = (1, 2, 4)
PALETTES = ("color", "green", "white")  # 夜里的显示方式：彩色、绿色、黑白

# 自动切换昼夜：场景亮度 = 画面平均亮度 / (曝光秒数 × 增益)，只和外面有多亮有关
DARK_BELOW = 2.0  # 低于这个值连续 DARK_SECONDS 秒 → 切到夜间
BRIGHT_ABOVE = 8.0  # 夜间关着红外灯时，高于这个值连续 BRIGHT_SECONDS 秒 → 切回白天
DARK_SECONDS = 3.0
BRIGHT_SECONDS = 5.0

# 手持防抖：倍率越高，单帧曝光越要短，否则画面糊
MAX_EXPOSURE_US = {1: 33_333, 2: 25_000, 4: 16_667}


@dataclass
class State:
    mode: str = DAY
    auto_mode: bool = True  # 自动切换昼夜
    zoom: int = 1
    ir_level: int = 0  # 0–3，见 design.IR_LEVELS
    palette: str = "green"
    recording: bool = False
    photos: int = 0
    reticle: bool = True
    scene_light: float | None = None  # 最近一次估计的场景亮度
    battery: float | None = None  # 电量 0–1，没有电量计时为 None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 命令


@dataclass(frozen=True)
class SetIRCut:
    """红外截止滤镜：白天放进去（颜色正），晚上移开（让红外光进来）。"""

    engaged: bool


@dataclass(frozen=True)
class SetIR:
    power: float  # 0–1


@dataclass(frozen=True)
class SetZoom:
    """数码变倍：真机上由相机的图像处理器裁剪，不占 CPU。"""

    factor: int


@dataclass(frozen=True)
class SetExposureLimit:
    max_us: int


@dataclass(frozen=True)
class TakePhoto:
    pass


@dataclass(frozen=True)
class StartRecording:
    pass


@dataclass(frozen=True)
class StopRecording:
    pass


@dataclass(frozen=True)
class Shutdown:
    pass


Command = SetIRCut | SetIR | SetZoom | SetExposureLimit | TakePhoto | StartRecording | StopRecording | Shutdown


# ------------------------------------------------------------------ 控制器

BUTTONS = ("ir", "zoom", "mode", "shutter", "power")


@dataclass
class Controller:
    clock: Callable[[], float] = time.monotonic
    state: State = field(default_factory=State)
    _dark_since: float | None = None
    _bright_since: float | None = None

    def startup(self) -> list[Command]:
        """开机时把硬件设到和状态一致。"""
        return self._apply_mode() + self._apply_zoom()

    # -------------------------------------------------------------- 按键

    def press(self, button: str, long: bool = False) -> list[Command]:
        """短按 / 长按（按住 1 秒以上）。

        红外：短按换档（关→弱→中→强→关），长按直接关
        变倍：短按 1×→2×→4×→1×，长按回到 1×
        模式：短按白天/夜间互换（同时关掉自动切换）；长按时，白天恢复自动切换，夜间换显示颜色
        拍照：短按拍照，长按开始/停止录像
        电源：长按关机
        """
        s = self.state
        if button not in BUTTONS:
            raise ValueError(f"没有“{button}”这个按键")
        if button == "ir":
            if s.mode == DAY:
                s.message = "白天不需要红外灯"
                return []
            s.ir_level = 0 if long else (s.ir_level + 1) % len(IR_LEVELS)
            s.message = f"红外灯 {['关', '弱', '中', '强'][s.ir_level]}"
            return [SetIR(IR_LEVELS[s.ir_level])]
        if button == "zoom":
            s.zoom = 1 if long else ZOOMS[(ZOOMS.index(s.zoom) + 1) % len(ZOOMS)]
            s.message = f"{s.zoom}× 变倍"
            return self._apply_zoom()
        if button == "mode":
            if long:
                if s.mode == NIGHT:
                    s.palette = PALETTES[(PALETTES.index(s.palette) + 1) % len(PALETTES)]
                    s.message = f"夜间显示：{_PALETTE_NAMES[s.palette]}"
                else:
                    s.auto_mode = True
                    s.message = "自动切换白天/夜间"
                return []
            s.auto_mode = False
            s.mode = NIGHT if s.mode == DAY else DAY
            s.message = "夜间模式" if s.mode == NIGHT else "白天模式"
            self._dark_since = self._bright_since = None
            return self._apply_mode()
        if button == "shutter":
            if long:
                s.recording = not s.recording
                s.message = "开始录像" if s.recording else "录像已保存"
                return [StartRecording() if s.recording else StopRecording()]
            s.photos += 1
            s.message = "已拍照"
            return [TakePhoto()]
        # 电源键：只认长按，防止碰到就关机
        if not long:
            return []
        cmds: list[Command] = [StopRecording()] if s.recording else []
        s.recording = False
        s.message = "正在关机"
        return cmds + [SetIR(0.0), Shutdown()]

    def set_ir_level(self, level: int) -> list[Command]:
        """手机上直接选档位。"""
        if self.state.mode == DAY:
            return []
        self.state.ir_level = max(0, min(level, len(IR_LEVELS) - 1))
        return [SetIR(IR_LEVELS[self.state.ir_level])]

    # -------------------------------------------------------------- 自动昼夜

    def on_frame(self, mean_luma: float, exposure_us: float, gain: float) -> list[Command]:
        """每帧调用一次。mean_luma 是画面平均亮度 0–1。"""
        s = self.state
        light = mean_luma / max(exposure_us * 1e-6 * max(gain, 1.0), 1e-6)
        s.scene_light = light
        if not s.auto_mode:
            return []
        now = self.clock()
        if s.mode == DAY:
            self._bright_since = None
            if light < DARK_BELOW:
                self._dark_since = self._dark_since if self._dark_since is not None else now
                if now - self._dark_since >= DARK_SECONDS:
                    s.mode = NIGHT
                    s.message = "天黑了，切到夜间模式"
                    self._dark_since = None
                    return self._apply_mode()
            else:
                self._dark_since = None
            return []
        # 夜间：红外灯开着时画面亮是灯照的，判断不了天亮没亮，不自动切回
        self._dark_since = None
        if s.ir_level == 0 and light > BRIGHT_ABOVE:
            self._bright_since = self._bright_since if self._bright_since is not None else now
            if now - self._bright_since >= BRIGHT_SECONDS:
                s.mode = DAY
                s.message = "天亮了，切到白天模式"
                self._bright_since = None
                return self._apply_mode()
        else:
            self._bright_since = None
        return []

    def _apply_zoom(self) -> list[Command]:
        return [SetZoom(self.state.zoom), SetExposureLimit(MAX_EXPOSURE_US[self.state.zoom])]

    def _apply_mode(self) -> list[Command]:
        s = self.state
        if s.mode == DAY:
            s.ir_level = 0
            return [SetIR(0.0), SetIRCut(True)]
        return [SetIRCut(False), SetIR(IR_LEVELS[s.ir_level])]


_PALETTE_NAMES = {"color": "彩色", "green": "绿色", "white": "黑白"}
