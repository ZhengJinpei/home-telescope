import json
import time
import urllib.request

import numpy as np
import pytest

from nightscope import controller as ctl
from nightscope import design as d
from nightscope import imaging as im
from nightscope.__main__ import main
from nightscope.app import App
from nightscope.hardware import (FramebufferDisplay, NullDisplay, SimulatedCamera, SimulatedIO,
                                 encode_jpeg, letterbox, to_rgb565)
from nightscope.server import Hub, start_server

LENS = d.Lens(50, 1.4)


# ------------------------------------------------------------------ 设计计算


def test_field_of_view_and_magnification():
    h, v, diag = d.field_of_view_deg(d.IMX462, LENS)
    assert h == pytest.approx(6.4, abs=0.05)  # 5.57 mm 宽的传感器配 50 mm 镜头
    assert v < h < diag
    assert d.MICRO_OLED.apparent_fov_deg == pytest.approx(29.9, abs=0.2)
    assert d.magnification(d.IMX462, LENS, d.MICRO_OLED) == pytest.approx(4.7, abs=0.1)
    assert d.magnification(d.IMX462, LENS, d.MICRO_OLED, 2) == pytest.approx(9.4, abs=0.1)
    assert d.lossless_zoom(d.IMX462, d.MICRO_OLED) == pytest.approx(1.875)


def test_pixels_and_resolution_range_agree():
    px = d.pixels_on_target(d.IMX462, LENS, 1.0, 200)
    assert px == pytest.approx(86, abs=1)  # 200 m 外 1 m 高的鹿占 86 个像素
    assert d.resolution_range_m(d.IMX462, LENS, 1.0, px) == pytest.approx(200)


def test_ir_range_scaling():
    ref = d.DEFAULT_IR_REF
    assert d.ir_range_m(ref.power_w, ref.beam_deg, ref.f_number) == pytest.approx(ref.range_m)
    # 功率翻 4 倍射程翻倍；光束角减半（立体角约 1/4）射程也约翻倍；光圈数翻倍射程减半
    assert d.ir_range_m(4 * ref.power_w, ref.beam_deg, ref.f_number) == pytest.approx(2 * ref.range_m)
    assert d.ir_range_m(ref.power_w, ref.beam_deg / 2, ref.f_number) == pytest.approx(2 * ref.range_m, rel=0.01)
    assert d.ir_range_m(ref.power_w, ref.beam_deg, 2 * ref.f_number) == pytest.approx(ref.range_m / 2)
    # 推荐光束刚好盖住对角线
    assert d.recommended_beam_deg(d.IMX462, LENS) == pytest.approx(1.2 * d.field_of_view_deg(d.IMX462, LENS)[2])


def test_battery_hours_drop_with_ir():
    hours = [d.battery_hours(level) for level in range(len(d.IR_LEVELS))]
    assert hours == sorted(hours, reverse=True)
    assert 4 < hours[0] < 8 and 3 < hours[-1] < hours[0]
    assert d.battery_hours(0, phone=False) > hours[0]


def test_cli_design(capsys):
    main(["design"])
    out = capsys.readouterr().out
    assert "放大 4.7 倍" in out
    assert "红外灯" in out and "续航" in out


# ------------------------------------------------------------------ 控制逻辑


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_startup_is_day_with_ircut_engaged():
    cmds = ctl.Controller().startup()
    assert ctl.SetIRCut(True) in cmds and ctl.SetIR(0.0) in cmds
    assert ctl.SetZoom(1) in cmds


def test_auto_day_night_with_hysteresis():
    clock = FakeClock()
    c = ctl.Controller(clock=clock)
    dark = dict(mean_luma=0.05, exposure_us=33_333, gain=64.0)  # 场景亮度约 0.02
    bright = dict(mean_luma=0.4, exposure_us=5_000, gain=1.0)  # 约 80
    assert c.on_frame(**dark) == []  # 刚变暗不马上切
    clock.t = 2.0
    assert c.on_frame(**dark) == []
    clock.t = 3.1
    cmds = c.on_frame(**dark)
    assert c.state.mode == ctl.NIGHT
    assert cmds == [ctl.SetIRCut(False), ctl.SetIR(0.0)]
    # 中途亮了一下（车灯）又暗了，不切回去
    clock.t = 4.0
    c.on_frame(**bright)
    clock.t = 6.0
    c.on_frame(**dark)
    clock.t = 10.0
    c.on_frame(**bright)
    assert c.state.mode == ctl.NIGHT
    clock.t = 15.5
    cmds = c.on_frame(**bright)
    assert c.state.mode == ctl.DAY and ctl.SetIRCut(True) in cmds


def test_no_auto_day_while_ir_is_on():
    clock = FakeClock()
    c = ctl.Controller(clock=clock, state=ctl.State(mode=ctl.NIGHT, ir_level=3))
    for t in range(0, 30):
        clock.t = t
        c.on_frame(0.5, 1_000, 1.0)  # 很亮，但那是红外灯照的
    assert c.state.mode == ctl.NIGHT


def test_manual_mode_disables_auto():
    clock = FakeClock()
    c = ctl.Controller(clock=clock)
    c.press("mode")
    assert c.state.mode == ctl.NIGHT and not c.state.auto_mode
    for t in range(0, 20):
        clock.t = t
        c.on_frame(0.4, 5_000, 1.0)
    assert c.state.mode == ctl.NIGHT
    c.press("mode")  # 回到白天
    c.press("mode", long=True)  # 白天长按：恢复自动
    assert c.state.auto_mode


def test_ir_button_cycles_and_is_ignored_in_day():
    c = ctl.Controller()
    assert c.press("ir") == []
    c.press("mode")
    levels = [c.press("ir")[0].power for _ in range(4)]
    assert levels == [0.25, 0.5, 1.0, 0.0]
    c.press("ir")
    assert c.press("ir", long=True) == [ctl.SetIR(0.0)]
    assert c.set_ir_level(9) == [ctl.SetIR(1.0)]


def test_zoom_shortens_exposure():
    c = ctl.Controller()
    cmds = c.press("zoom")
    assert ctl.SetZoom(2) in cmds
    limits = [cmd.max_us for cmd in cmds if isinstance(cmd, ctl.SetExposureLimit)]
    assert limits == [ctl.MAX_EXPOSURE_US[2]] and limits[0] < ctl.MAX_EXPOSURE_US[1]
    c.press("zoom")
    assert c.state.zoom == 4
    c.press("zoom", long=True)
    assert c.state.zoom == 1


def test_palette_cycles_in_night_with_long_press():
    c = ctl.Controller(state=ctl.State(mode=ctl.NIGHT))
    seen = [c.state.palette]
    for _ in range(3):
        c.press("mode", long=True)
        seen.append(c.state.palette)
    assert seen[0] == seen[-1] and set(seen) == set(ctl.PALETTES)


def test_shutter_and_power():
    c = ctl.Controller()
    assert c.press("shutter") == [ctl.TakePhoto()]
    assert c.press("shutter", long=True) == [ctl.StartRecording()]
    assert c.press("power") == []  # 碰一下不关机
    cmds = c.press("power", long=True)
    assert cmds == [ctl.StopRecording(), ctl.SetIR(0.0), ctl.Shutdown()]
    with pytest.raises(ValueError):
        c.press("nope")


# ------------------------------------------------------------------ 图像处理


def test_gray_and_mean_luma():
    frame = np.zeros((8, 8, 3), np.uint8)
    frame[..., 1] = 200
    assert im.gray(frame)[0, 0] == 100  # (0 + 2×200 + 0) / 4
    assert im.mean_luma(np.full((16, 16, 3), 255, np.uint8), step=4) == pytest.approx(1.0)


def test_digital_zoom_takes_center():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[45:55, 95:105] = 255  # 中间一个白块
    out = im.digital_zoom(frame, 4)
    assert out.shape == frame.shape
    assert out[50, 100].tolist() == [255, 255, 255]
    assert out[5, 5].tolist() == [0, 0, 0]
    assert im.digital_zoom(frame, 1, (100, 50)).shape == (50, 100, 3)


def test_denoiser_reduces_noise_without_ghosting():
    rng = np.random.default_rng(0)
    clean = np.tile(np.linspace(40, 120, 256, dtype=np.float32), (128, 1))
    den = im.TemporalDenoiser(shift=2)
    for _ in range(30):
        noisy = np.clip(clean + rng.normal(0, 30, clean.shape), 0, 255).astype(np.uint8)
        out = den.process(noisy)
    assert np.std(out - clean) < 0.5 * np.std(noisy - clean)
    # 移动的亮块：新位置马上是亮的，旧位置马上变回背景
    den.reset()
    bg = np.full((128, 256), 50, np.uint8)
    for k in range(6):
        f = bg.copy()
        f[40:80, 16 + 32 * k:56 + 32 * k] = 200
        out = den.process(f)
    assert out[60, 16 + 32 * 5 + 20] == 200
    assert out[60, 16 + 32 * 4 + 4] == 50


def test_stretch_lut():
    img = np.concatenate([np.full(500, 20), np.full(500, 60)]).astype(np.uint8).reshape(10, 100)
    lut = im.stretch_lut(img, step=1, lift=False)
    assert lut[20] == 0.0 and lut[60] == 1.0
    flat = im.stretch_lut(np.full((10, 10), 5, np.uint8), step=1)
    assert flat[5] == 0.0 and flat[13] == 1.0  # 全是同一个值：最少拉开 8 级，不放大噪点


def test_night_palettes():
    rng = np.random.default_rng(0)
    frame = (rng.random((64, 96, 3)) * 80).astype(np.uint8)
    green = im.night_palette(frame, "green")
    assert green.shape == frame.shape and green.dtype == np.uint8
    assert green[..., 1].mean() > 2 * green[..., 0].mean()  # 绿色为主
    white = im.night_palette(frame, "white")
    assert (white[..., 0] == white[..., 1]).all()
    color = im.night_palette(frame, "color", im.TemporalDenoiser())
    assert color.max() > frame.max()  # 拉了对比度


def test_pipeline_modes():
    frame = np.full((72, 128, 3), 100, np.uint8)
    p = im.Pipeline()
    day = p.process(frame, ctl.State())
    assert day[36, 64 - 6 - 1].tolist() == [255, 60, 60]  # 十字线
    assert frame[36, 64 - 6 - 1].tolist() == [100, 100, 100]  # 不改原画面
    assert p.process(frame, ctl.State(reticle=False)) is frame
    night = p.process(frame, ctl.State(mode=ctl.NIGHT, reticle=False))
    assert night[0, 0, 1] >= night[0, 0, 0]


# ------------------------------------------------------------------ 屏幕


def test_letterbox_and_rgb565():
    frame = np.full((576, 1024, 3), 255, np.uint8)
    out = letterbox(frame, 1024, 768)
    assert out.shape == (768, 1024, 3)
    assert out[0, 0].tolist() == [0, 0, 0] and out[384, 512].tolist() == [255, 255, 255]
    small = letterbox(frame, 512, 512)
    assert small[256, 256].tolist() == [255, 255, 255] and small[0, 0].tolist() == [0, 0, 0]
    px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
    assert np.frombuffer(to_rgb565(px), "<u2").tolist() == [0xF800, 0x07E0, 0x001F]


@pytest.mark.parametrize("bpp", [16, 32])
def test_framebuffer_display(tmp_path, bpp):
    sysfs = tmp_path / "fb0"
    sysfs.mkdir()
    (sysfs / "virtual_size").write_text("64,48\n")
    (sysfs / "bits_per_pixel").write_text(f"{bpp}\n")
    dev = tmp_path / "fb"
    dev.write_bytes(b"\0" * (64 * 48 * bpp // 8))
    fb = FramebufferDisplay(str(dev), str(sysfs))
    frame = np.zeros((36, 64, 3), np.uint8)
    frame[..., 0] = 255
    fb.show(frame)
    fb.close()
    data = dev.read_bytes()
    assert len(data) == 64 * 48 * bpp // 8
    mid = (24 * 64 + 32) * bpp // 8
    if bpp == 16:
        assert data[mid:mid + 2] == (0xF800).to_bytes(2, "little")
    else:
        assert data[mid:mid + 4] == bytes([0, 0, 255, 255])  # B G R X
    assert data[:3] == b"\0\0\0"  # 上面的黑边（32 位时第 4 个字节是不透明度）


# ------------------------------------------------------------------ 模拟相机和整机


def test_simulated_camera_ir_needs_ircut_open():
    cam = SimulatedCamera((160, 90), ambient=0.002)
    _, dark = cam.capture()
    cam.ir_power = 1.0
    _, blocked = cam.capture()
    assert blocked["AnalogueGain"] == pytest.approx(dark["AnalogueGain"], rel=0.01)  # 滤镜挡住了红外
    cam.ircut_engaged = False
    _, lit = cam.capture()
    assert lit["AnalogueGain"] < dark["AnalogueGain"]
    cam.ambient = 1.0
    frame, day = cam.capture()
    assert day["AnalogueGain"] == 1.0 and day["ExposureTime"] < 33_333
    assert frame.shape == (90, 160, 3)


def _app(tmp_path, ambient=1.0, clock=None):
    cam = SimulatedCamera((160, 90), ambient=ambient)
    io = SimulatedIO(cam)
    hub = Hub(tmp_path / "media", encode=lambda f: encode_jpeg(f, 60))
    app = App(cam, io, NullDisplay(), hub, tmp_path / "media", clock=clock or time.monotonic)
    return app, cam, io, hub


def test_app_photo_record_and_shutdown(tmp_path):
    app, cam, io, hub = _app(tmp_path)
    io.press("shutter")
    app.step()
    assert app.last_photo is not None and app.last_photo.read_bytes()[:2] == b"\xff\xd8"
    io.press("shutter", long=True)
    app.step()
    assert cam.recording_path is not None
    app.step()
    io.press("power", long=True)
    app.run(max_frames=100)
    assert not app.running and io.shut_down
    assert cam.recording_path is None  # 关机前停止录像
    assert io.ir_power == 0.0
    names = sorted(p.suffix for p in (tmp_path / "media").iterdir())
    assert names == [".jpg", ".mjpeg"]


def test_app_switches_to_night_and_applies_ir(tmp_path):
    clock = FakeClock()
    app, cam, io, hub = _app(tmp_path, ambient=0.002, clock=clock)
    for k in range(5):
        clock.t = k
        app.step()
    assert app.controller.state.mode == ctl.NIGHT
    assert not io.ircut_engaged and not cam.ircut_engaged
    hub.push_action({"ir_level": 3})
    hub.push_action({"press": "zoom"})
    hub.push_action({"press": "bogus"})  # 网页传错了也不能崩
    app.step()
    assert cam.ir_power == 1.0 and cam.zoom == 2
    assert hub.state["zoom"] == 2 and "bogus" in hub.state["message"]


def _get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


def _post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_web_control_and_stream(tmp_path):
    app, cam, io, hub = _app(tmp_path)
    server = start_server(hub, "127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        app.step()
        status, kind, body = _get(base + "/")
        assert status == 200 and "夜视望远镜" in body.decode()
        assert json.loads(_get(base + "/api/state")[2])["mode"] == "day"
        assert _post(base + "/api/press", {"button": "shutter"}) == {"ok": True}
        app.step()
        media = json.loads(_get(base + "/media")[2])
        assert len(media) == 1 and media[0]["name"].endswith(".jpg")
        status, kind, body = _get(base + "/media/" + media[0]["name"])
        assert kind == "image/jpeg" and body[:2] == b"\xff\xd8"
        with pytest.raises(urllib.error.HTTPError):
            _get(base + "/media/..%2Fsecret")

        # 画面推流：开一个连接，主循环推几帧，读到 JPEG
        stream = urllib.request.urlopen(base + "/stream.mjpg", timeout=5)
        deadline = time.time() + 3
        while not hub.has_viewers and time.time() < deadline:
            time.sleep(0.01)
        for _ in range(3):
            app.step()
        chunk = stream.read(2000)
        stream.close()
        assert b"--nightscope-frame" in chunk and b"\xff\xd8" in chunk
    finally:
        hub.close()
        server.shutdown()


def test_cli_run_simulated(tmp_path, capsys):
    main(["run", "--simulate", "--port", "0", "--frames", "3", "--media", str(tmp_path), "--size", "160x90"])
    assert "夜视望远镜已启动" in capsys.readouterr().out
