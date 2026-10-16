"""主循环：取一帧 → 按键和自动昼夜 → 处理画面 → 送到目镜屏幕和手机。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import controller as ctl
from .hardware import IO, Camera, Display, encode_jpeg, media_name
from .imaging import Pipeline, digital_zoom, mean_luma
from .server import Hub


@dataclass
class Stats:
    frames: int = 0
    fps: float = 0.0
    _t0: float = 0.0
    _n0: int = 0

    def tick(self, now: float) -> None:
        self.frames += 1
        if now - self._t0 >= 2.0:
            self.fps = (self.frames - self._n0) / (now - self._t0) if self._t0 else 0.0
            self._t0, self._n0 = now, self.frames


class App:
    def __init__(
        self,
        camera: Camera,
        io: IO,
        display: Display,
        hub: Hub,
        media_dir: Path,
        *,
        controller: ctl.Controller | None = None,
        pipeline: Pipeline | None = None,
        stream_size: tuple[int, int] | None = None,  # None：和处理后的画面一样大
        clock: Callable[[], float] = time.monotonic,
    ):
        self.camera, self.io, self.display, self.hub = camera, io, display, hub
        self.media_dir = media_dir
        media_dir.mkdir(parents=True, exist_ok=True)
        self.controller = controller or ctl.Controller(clock=clock)
        self.pipeline = pipeline or Pipeline()
        self.stream_size = stream_size
        self.clock = clock
        self.stats = Stats()
        self.running = True
        self.last_photo: Path | None = None
        self._recording_path: Path | None = None
        self._shutdown_requested = False
        self.execute(self.controller.startup())

    # -------------------------------------------------------------- 命令

    def execute(self, commands: list) -> bool:
        """执行控制器的命令。返回 True 表示这一帧要拍照。"""
        photo = False
        for cmd in commands:
            if isinstance(cmd, ctl.SetIRCut):
                self.io.set_ircut(cmd.engaged)
            elif isinstance(cmd, ctl.SetIR):
                self.io.set_ir(cmd.power)
            elif isinstance(cmd, ctl.SetZoom):
                self.camera.set_zoom(cmd.factor)
            elif isinstance(cmd, ctl.SetExposureLimit):
                self.camera.set_exposure_limit(cmd.max_us)
            elif isinstance(cmd, ctl.TakePhoto):
                photo = True
            elif isinstance(cmd, ctl.StartRecording):
                self._recording_path = self.media_dir / media_name("video", ".mp4")
                self.camera.start_recording(self._recording_path)
            elif isinstance(cmd, ctl.StopRecording):
                self.camera.stop_recording()
                self._recording_path = None
            elif isinstance(cmd, ctl.Shutdown):
                self.running = False
                self._shutdown_requested = True
        return photo

    def _inputs(self) -> list:
        cmds = []
        for name, long in self.io.button_events():
            cmds += self.controller.press(name, long)
        for action in self.hub.pop_actions():
            try:
                if "press" in action:
                    cmds += self.controller.press(action["press"], action.get("long", False))
                elif "ir_level" in action:
                    cmds += self.controller.set_ir_level(action["ir_level"])
            except ValueError as e:  # 网页传来不认识的按键
                self.controller.state.message = str(e)
        return cmds

    # -------------------------------------------------------------- 一帧

    def step(self) -> None:
        photo = self.execute(self._inputs())
        frame, meta = self.camera.capture()
        exposure = float(meta.get("ExposureTime", 33_333))
        gain = float(meta.get("AnalogueGain", 1.0))
        photo |= self.execute(self.controller.on_frame(mean_luma(frame), exposure, gain))

        state = self.controller.state
        out = self.pipeline.process(frame, state)
        self.display.show(out)
        if photo:
            self.last_photo = self.media_dir / media_name("photo", ".jpg")
            self.last_photo.write_bytes(encode_jpeg(out, 92))
        if self.hub.has_viewers:  # 没人看就不压缩 JPEG，省电
            if self.stream_size and out.shape[1::-1] != self.stream_size:
                out = digital_zoom(out, 1, self.stream_size)
            self.hub.submit(out)

        self.stats.tick(self.clock())
        self.hub.state = state.to_dict() | {"fps": round(self.stats.fps, 1),
                                            "exposure_ms": round(exposure / 1000, 1), "gain": round(gain, 1)}

    def run(self, max_frames: int | None = None) -> None:
        try:
            while self.running and (max_frames is None or self.stats.frames < max_frames):
                self.step()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
        if self._shutdown_requested:
            self.io.shutdown()

    def close(self) -> None:
        self.io.set_ir(0.0)  # 关机前一定关掉红外灯
        self.camera.close()
        self.display.close()
        self.io.close()
        self.hub.close()
