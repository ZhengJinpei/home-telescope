"""命令行：

    python3 -m nightscope design                 算放大倍数、视场、红外灯射程、续航
    python3 -m nightscope run --simulate         在电脑上用模拟相机跑，手机/浏览器打开 http://localhost:8080
    python3 -m nightscope run                    在树莓派上跑真机
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from . import design as d


def cmd_design(args) -> None:
    sensor = d.SENSORS[args.sensor]
    lens = d.Lens(args.focal, args.fnum)
    viewer = d.Viewer(d.MICRO_OLED.name, d.MICRO_OLED.width_px, d.MICRO_OLED.height_px,
                      d.MICRO_OLED.width_mm, args.eyepiece)
    h, v, diag = d.field_of_view_deg(sensor, lens)
    lossless = d.lossless_zoom(sensor, viewer)

    print(f"传感器：{sensor.name}，{sensor.width_px}×{sensor.height_px}，像素 {sensor.pixel_um:g} µm")
    print(f"镜头：焦距 {lens.focal_mm:g} mm，F{lens.f_number:g}（通光口径 {lens.aperture_mm:.0f} mm）")
    print(f"目镜屏幕：{viewer.name}，配 {viewer.eyepiece_focal_mm:g} mm 目镜，看起来宽 {viewer.apparent_fov_deg:.0f}°")

    print(f"\n视场：{h:.1f}° × {v:.1f}°（100 m 外能看到 {2 * 100 * math.tan(math.radians(h / 2)):.0f} m 宽）")
    for z in (1, 2, 4):
        note = "" if z <= lossless + 0.01 else "（超过传感器像素，会变糊一点）"
        print(f"  {z}× 数码变倍：放大 {d.magnification(sensor, lens, viewer, z):.1f} 倍{note}")
    print(f"  {lossless:.1f}× 以内的数码变倍不损失清晰度（传感器像素比屏幕多）")

    print("\n分辨率够用的距离（镜头和传感器决定，按约翰逊准则）：")
    for target, size in d.TARGETS.items():
        cells = "，".join(f"{k} {d.resolution_range_m(sensor, lens, size, px):.0f} m"
                         for k, px in d.JOHNSON_PX.items())
        print(f"  {target}：{cells}")

    beam = d.recommended_beam_deg(sensor, lens)
    rng = d.ir_range_m(args.ir_power, args.ir_beam, lens.f_number)
    print(f"\n红外灯：{args.ir_power:g} W，光束 {args.ir_beam:g}°（建议 {beam:.0f}° 左右，刚好盖住画面）")
    print(f"  晴朗的夜里大约能看清 {rng:.0f} m（估计值，实测后可校准）")
    print("  天黑以后真正限制看多远的是红外灯，不是分辨率")

    print("\n续航（两节 3000 mAh 18650 并联）：")
    for level, name in enumerate(["关", "弱", "中", "强"]):
        print(f"  红外{name}：约 {d.battery_hours(level):.1f} 小时")


def cmd_run(args) -> None:
    from .app import App
    from .hardware import (FramebufferDisplay, GpioIO, NullDisplay, PiCamera, SimulatedCamera,
                           SimulatedIO, encode_jpeg)
    from .server import Hub, start_server

    w, h = (int(v) for v in args.size.lower().split("x"))
    media = Path(args.media).expanduser()
    if args.simulate:
        camera = SimulatedCamera((w, h), ambient=args.ambient)
        io = SimulatedIO(camera)
    else:
        camera = PiCamera((w, h))
        io = GpioIO()
    display = NullDisplay() if args.simulate or args.no_display else FramebufferDisplay()
    hub = Hub(media, encode=lambda frame: encode_jpeg(frame, args.quality))
    server = start_server(hub, args.host, args.port)
    print(f"夜视望远镜已启动，手机或浏览器打开 http://{'localhost' if args.simulate else '10.42.0.1'}:{server.server_address[1]}")
    App(camera, io, display, hub, media).run(args.frames)
    server.shutdown()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nightscope", description="家用数码夜视望远镜")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("design", help="算放大倍数、视场、红外灯射程、续航")
    sp.add_argument("--sensor", choices=list(d.SENSORS), default="imx462", help="传感器，默认 imx462")
    sp.add_argument("--focal", type=float, default=50, help="镜头焦距 mm，默认 50")
    sp.add_argument("--fnum", type=float, default=1.4, help="镜头光圈数，默认 1.4")
    sp.add_argument("--eyepiece", type=float, default=d.MICRO_OLED.eyepiece_focal_mm, help="目镜焦距 mm，默认 15")
    sp.add_argument("--ir-power", type=float, default=3, help="红外灯电功率 W，默认 3")
    sp.add_argument("--ir-beam", type=float, default=10, help="红外光束角度，默认 10°")
    sp.set_defaults(func=cmd_design)

    sp = sub.add_parser("run", help="运行夜视仪")
    sp.add_argument("--simulate", action="store_true", help="没有硬件，用模拟相机在电脑上跑")
    sp.add_argument("--ambient", type=float, default=0.002,
                    help="模拟时的环境光：1 是白天，0.002 左右是有月光的夜里（默认）")
    sp.add_argument("--size", default="1024x576", help="相机输出画面大小，默认 1024x576（和目镜屏幕一样宽）")
    sp.add_argument("--host", default="0.0.0.0")
    sp.add_argument("--port", type=int, default=8080)
    sp.add_argument("--media", default="~/nightscope", help="照片和录像存在哪里")
    sp.add_argument("--quality", type=int, default=70, help="推到手机的 JPEG 画质 1–95，默认 70")
    sp.add_argument("--no-display", action="store_true", help="不输出到目镜屏幕（只用手机看）")
    sp.add_argument("--frames", type=int, help="跑这么多帧就退出（测试用）")
    sp.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyError, ValueError) as e:
        print(f"出错了：{e.args[0] if e.args else e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
