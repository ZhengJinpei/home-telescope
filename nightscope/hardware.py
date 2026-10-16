"""硬件：相机、红外灯、红外截止滤镜、按键、屏幕。

每样都有真机版和模拟版。真机版只在树莓派上导入 picamera2 / gpiozero，
尚未在实物上验证，第一次接线后请按 docs/NIGHTVISION.md 逐项测试。
"""

from __future__ import annotations

import io
import math
import os
import queue
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

# ------------------------------------------------------------------ 相机


class Camera(Protocol):
    size: tuple[int, int]  # (宽, 高)

    def capture(self) -> tuple[np.ndarray, dict]:
        """返回 (RGB 画面, 元数据)。元数据里有 ExposureTime（微秒）和 AnalogueGain。"""

    def set_exposure_limit(self, max_us: int) -> None: ...

    def start_recording(self, path: Path) -> None: ...

    def stop_recording(self) -> None: ...

    def close(self) -> None: ...


class PiCamera:
    """树莓派相机（IMX462 等），用 picamera2。"""

    def __init__(self, size: tuple[int, int] = (1280, 720)):
        from picamera2 import Picamera2  # 只在树莓派上有

        self.size = size
        self.cam = Picamera2()
        # picamera2 的 "BGR888" 在内存里是 R、G、B 顺序，正好是 numpy 的 RGB
        config = self.cam.create_video_configuration(
            main={"size": size, "format": "BGR888"},
            controls={"FrameDurationLimits": (33_333, 33_333)},
        )
        self.cam.configure(config)
        self.cam.start()
        self._recording = False

    def capture(self) -> tuple[np.ndarray, dict]:
        request = self.cam.capture_request()
        try:
            return request.make_array("main"), request.get_metadata()
        finally:
            request.release()

    def set_exposure_limit(self, max_us: int) -> None:
        # 自动曝光的最长曝光受帧间隔限制：帧间隔就是手持防抖要的最长曝光
        self.cam.set_controls({"FrameDurationLimits": (max_us, max_us)})

    def start_recording(self, path: Path) -> None:
        from picamera2.encoders import H264Encoder
        from picamera2.outputs import FfmpegOutput

        self.cam.start_encoder(H264Encoder(bitrate=8_000_000), FfmpegOutput(str(path)))
        self._recording = True

    def stop_recording(self) -> None:
        if self._recording:
            self.cam.stop_encoder()
            self._recording = False

    def close(self) -> None:
        self.stop_recording()
        self.cam.stop()
        self.cam.close()


class SimulatedCamera:
    """电脑上用的假相机：一片夜里的草地和树，一只“鹿”慢慢走过。

    ambient：环境光 0–1（1 是白天，0.001 左右是有月光的夜里）。
    红外灯只有在红外截止滤镜移开时才照得进来。
    """

    TARGET = 0.4  # 自动曝光想把平均亮度调到这里
    MAX_GAIN = 64.0

    def __init__(self, size: tuple[int, int] = (640, 360), ambient: float = 1.0, seed: int = 0):
        self.size = size
        self.ambient = ambient
        self.ir_power = 0.0
        self.ircut_engaged = True
        self.max_exposure_us = 33_333
        self.recording_path: Path | None = None
        self._rng = np.random.default_rng(seed)
        self._t = 0
        w, h = size
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sky = yy < h * 0.45
        scene = np.where(sky[..., None], [0.35, 0.45, 0.6], [0.25, 0.4, 0.2]).astype(np.float32)
        for k in range(6):  # 几棵树
            x = w * (0.08 + 0.17 * k)
            trunk = (np.abs(xx - x) < w * 0.01) & (yy > h * 0.3)
            crown = (xx - x) ** 2 + (yy - h * 0.3) ** 2 < (h * 0.12) ** 2
            scene[trunk | crown] = [0.1, 0.2, 0.08]
        self._scene = scene
        self._yy, self._xx = yy, xx

    def _frame_reflectance(self) -> np.ndarray:
        w, h = self.size
        scene = self._scene.copy()
        x = w * (0.2 + 0.6 * (0.5 + 0.5 * math.sin(self._t / 60)))
        y = h * 0.65
        body = ((self._xx - x) / (w * 0.06)) ** 2 + ((self._yy - y) / (h * 0.06)) ** 2 < 1
        scene[body] = [0.55, 0.4, 0.3]
        eyes = ((self._xx - x - w * 0.05) ** 2 + (self._yy - y + h * 0.04) ** 2) < (h * 0.008) ** 2
        scene[eyes] = [1.0, 1.0, 1.0]  # 动物眼睛反射红外灯，亮晶晶的
        return scene

    def capture(self) -> tuple[np.ndarray, dict]:
        self._t += 1
        reflect = self._frame_reflectance()
        light = self.ambient
        if not self.ircut_engaged:
            light += 0.02 * self.ir_power  # 红外灯在近处的照度，和月光差不多到十倍
        signal = reflect * light  # 每秒、单位增益下传感器收到的“亮度”
        # 自动曝光：先加长曝光（不超过手持防抖的上限），不够再加增益
        needed = self.TARGET / (float(signal.mean()) + 1e-9)
        exposure_s = min(needed, self.max_exposure_us * 1e-6)
        gain = float(np.clip(needed / exposure_s, 1.0, self.MAX_GAIN))
        img = signal * exposure_s * gain
        # 增益越高噪点越多
        img = img + self._rng.normal(0.0, 0.004 * math.sqrt(gain), img.shape).astype(np.float32)
        if not self.ircut_engaged:  # 红外光让颜色偏粉
            img = img * np.array([1.15, 0.9, 1.0], dtype=np.float32)
        frame = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if self.recording_path is not None:
            with open(self.recording_path, "ab") as f:
                f.write(encode_jpeg(frame, 60))
        return frame, {"ExposureTime": exposure_s * 1e6, "AnalogueGain": gain}

    def set_exposure_limit(self, max_us: int) -> None:
        self.max_exposure_us = max_us

    def start_recording(self, path: Path) -> None:
        self.recording_path = path.with_suffix(".mjpeg")

    def stop_recording(self) -> None:
        self.recording_path = None

    def close(self) -> None:
        self.stop_recording()


# ------------------------------------------------------------------ 红外灯、滤镜、按键


class IO(Protocol):
    def set_ir(self, power: float) -> None: ...

    def set_ircut(self, engaged: bool) -> None: ...

    def button_events(self) -> list[tuple[str, bool]]:
        """自上次调用以来的按键：[(按键名, 是否长按)]。"""

    def shutdown(self) -> None: ...

    def close(self) -> None: ...


# 默认接线（BCM 编号），见 docs/NIGHTVISION.md
DEFAULT_PINS = {
    "ir_pwm": 18,  # 红外灯恒流驱动板的 PWM / DIM 脚（硬件 PWM）
    "ircut": 23,  # 红外截止滤镜控制脚：高电平 = 滤镜放进去（白天）
    "buttons": {"ir": 5, "zoom": 6, "mode": 13, "shutter": 19, "power": 26},
}


class GpioIO:
    """树莓派 GPIO，用 gpiozero。按键接在 GPIO 和 GND 之间，用内部上拉。"""

    def __init__(self, pins: dict = DEFAULT_PINS, hold_s: float = 1.0):
        from gpiozero import Button, DigitalOutputDevice, PWMOutputDevice

        self._events: queue.Queue[tuple[str, bool]] = queue.Queue()
        self._ir = PWMOutputDevice(pins["ir_pwm"], frequency=1000)
        self._ircut = DigitalOutputDevice(pins["ircut"])
        self._held: dict[str, bool] = {}
        self._buttons = []
        for name, pin in pins["buttons"].items():
            b = Button(pin, pull_up=True, hold_time=hold_s, bounce_time=0.03)
            b.when_held, b.when_released = self._callbacks(name)
            self._held[name] = False
            self._buttons.append(b)

    def _callbacks(self, name: str):
        """按住超过 hold_s 算长按（按住时就触发），否则松手时算短按。"""
        def held():
            self._held[name] = True
            self._events.put((name, True))

        def released():
            if not self._held[name]:
                self._events.put((name, False))
            self._held[name] = False

        return held, released

    def set_ir(self, power: float) -> None:
        self._ir.value = max(0.0, min(power, 1.0))

    def set_ircut(self, engaged: bool) -> None:
        self._ircut.value = engaged

    def button_events(self) -> list[tuple[str, bool]]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def shutdown(self) -> None:
        subprocess.run(["sudo", "shutdown", "-h", "now"], check=False)

    def close(self) -> None:
        self._ir.value = 0
        for dev in [self._ir, self._ircut, *self._buttons]:
            dev.close()


class SimulatedIO:
    """模拟版：把红外灯和滤镜的状态告诉模拟相机；按键从 press() 或手机网页进来。"""

    def __init__(self, camera: SimulatedCamera | None = None):
        self.camera = camera
        self.ir_power = 0.0
        self.ircut_engaged = True
        self.shut_down = False
        self._events: queue.Queue[tuple[str, bool]] = queue.Queue()

    def press(self, name: str, long: bool = False) -> None:
        self._events.put((name, long))

    def set_ir(self, power: float) -> None:
        self.ir_power = power
        if self.camera:
            self.camera.ir_power = power

    def set_ircut(self, engaged: bool) -> None:
        self.ircut_engaged = engaged
        if self.camera:
            self.camera.ircut_engaged = engaged

    def button_events(self) -> list[tuple[str, bool]]:
        out = []
        while not self._events.empty():
            out.append(self._events.get_nowait())
        return out

    def shutdown(self) -> None:
        self.shut_down = True

    def close(self) -> None:
        pass


# ------------------------------------------------------------------ 屏幕


class Display(Protocol):
    def show(self, frame: np.ndarray) -> None: ...

    def close(self) -> None: ...


class NullDisplay:
    def show(self, frame: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比例缩放到屏幕里，上下或左右留黑边。"""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    xs = ((np.arange(nw) + 0.5) / scale).astype(np.intp).clip(0, w - 1)
    ys = ((np.arange(nh) + 0.5) / scale).astype(np.intp).clip(0, h - 1)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = frame[ys[:, None], xs[None, :]]
    return out


def to_rgb565(frame: np.ndarray) -> bytes:
    r = frame[..., 0].astype(np.uint16) >> 3
    g = frame[..., 1].astype(np.uint16) >> 2
    b = frame[..., 2].astype(np.uint16) >> 3
    return ((r << 11) | (g << 5) | b).astype("<u2").tobytes()


def to_bgrx(frame: np.ndarray) -> bytes:
    out = np.empty(frame.shape[:2] + (4,), dtype=np.uint8)
    out[..., 0], out[..., 1], out[..., 2], out[..., 3] = frame[..., 2], frame[..., 1], frame[..., 0], 255
    return out.tobytes()


class FramebufferDisplay:
    """直接写 Linux 帧缓冲（/dev/fb0）：目镜里的微型屏幕接在树莓派 HDMI 上。

    不需要桌面环境，也不需要 pygame。支持 16 位（RGB565）和 32 位（BGRX）。
    """

    def __init__(self, device: str = "/dev/fb0", sysfs: str = "/sys/class/graphics/fb0"):
        sys_dir = Path(sysfs)
        self.width, self.height = (int(v) for v in (sys_dir / "virtual_size").read_text().strip().split(","))
        self.bpp = int((sys_dir / "bits_per_pixel").read_text().strip())
        stride_file = sys_dir / "stride"
        self.stride = int(stride_file.read_text().strip()) if stride_file.exists() else self.width * self.bpp // 8
        if self.bpp not in (16, 32):
            raise ValueError(f"不支持 {self.bpp} 位帧缓冲")
        self._fd = os.open(device, os.O_RDWR)

    def show(self, frame: np.ndarray) -> None:
        img = letterbox(frame, self.width, self.height)
        data = to_rgb565(img) if self.bpp == 16 else to_bgrx(img)
        row = self.width * self.bpp // 8
        if row == self.stride:
            os.pwrite(self._fd, data, 0)
        else:
            for y in range(self.height):
                os.pwrite(self._fd, data[y * row:(y + 1) * row], y * self.stride)

    def close(self) -> None:
        os.close(self._fd)


# ------------------------------------------------------------------ 照片和录像


def encode_jpeg(frame: np.ndarray, quality: int = 75) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def media_name(prefix: str, suffix: str, when: float | None = None) -> str:
    stamp = datetime.fromtimestamp(when if when is not None else time.time()).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}{suffix}"
