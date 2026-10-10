"""命令行：python -m telescope <命令>"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from . import astro, optics
from .catalog import ALL_TARGETS, find
from .mount import AltAzMount, SimulatedDriver, UnsafeTarget
from .sky import CITIES, Observer, locate, observer_for_city, plan_tonight

STARS = "★★★"


def _observer(args) -> Observer:
    if args.lat is not None and args.lon is not None:
        return Observer(args.lat, args.lon, args.tz)
    obs = observer_for_city(args.city)
    return Observer(obs.lat, obs.lon, args.tz)


def _when(args, obs: Observer) -> datetime:
    if args.time:
        return datetime.strptime(args.time, "%Y-%m-%d %H:%M").replace(tzinfo=obs.tz)
    return datetime.now(obs.tz)


def _scope(args) -> optics.Telescope | None:
    if args.aperture and args.focal:
        return optics.Telescope(args.aperture, args.focal)
    return None


def _hm(t: datetime | None) -> str:
    return t.strftime("%H:%M") if t else "--:--"


def cmd_optics(args) -> None:
    scope = optics.Telescope(args.aperture, args.focal)
    print(f"望远镜：口径 {scope.aperture_mm:g} mm，焦距 {scope.focal_length_mm:g} mm，焦比 f/{scope.focal_ratio:.1f}")
    print(f"  集光能力：人眼的 {scope.light_gathering_vs_eye:.0f} 倍")
    print(f"  极限星等：约 {scope.limiting_magnitude:.1f} 等（城市里通常再减 1–2 等）")
    print(f"  分辨率：{scope.dawes_limit_arcsec:.2f} 角秒（能分开这么近的两颗星）")
    print(f"  有效倍率：{scope.min_useful_magnification:.0f}–{scope.max_useful_magnification:.0f} 倍")

    for fl in args.eyepiece:
        ep = optics.Eyepiece(fl, args.afov, args.barlow)
        v = optics.view(scope, ep)
        barlow = f" + {args.barlow:g}× 巴洛" if args.barlow != 1 else ""
        print(f"\n目镜 {fl:g} mm{barlow}：{v.magnification:.0f} 倍，视场 {v.true_fov_deg:.2f}°，出瞳 {v.exit_pupil_mm:.1f} mm")
        print("  " + ("能装下整个月亮" if v.moon_fits else "装不下整个月亮，只能看月面局部"))
        for w in v.warnings:
            print(f"  ⚠ {w}")

    print("\n推荐目镜焦距：")
    for label, fl in optics.suggest_eyepieces(scope).items():
        print(f"  {label}：约 {fl:g} mm")

    if args.camera:
        cam = {"hq": optics.PI_HQ_CAMERA, "v3": optics.PI_CAMERA_3}[args.camera]
        cv = optics.camera_view(scope, cam)
        print(f"\n接 {cam.name} 直焦拍摄：")
        print(f"  {cv.arcsec_per_pixel:.2f} 角秒/像素，画面 {cv.fov_width_deg:.2f}° × {cv.fov_height_deg:.2f}°")
        print("  " + ("能拍下整个月亮" if cv.moon_fits else "拍不下整个月亮"))
        print(f"  {cv.sampling}")


def cmd_tonight(args) -> None:
    obs = _observer(args)
    night = date.fromisoformat(args.date) if args.date else datetime.now(obs.tz).date()
    plan = plan_tonight(night, obs, _scope(args))
    print(f"{night:%Y-%m-%d} 晚上（北纬 {obs.lat:.1f}°，东经 {obs.lon:.1f}°）")
    print(f"  天黑：{_hm(plan.dusk)}  完全黑：{_hm(plan.dark_start)} – {_hm(plan.dark_end)}  天亮：{_hm(plan.dawn)}")
    print(f"  月相：{plan.moon_phase}（亮面 {plan.moon_illumination:.0%}）")
    if not plan.items:
        print("\n今晚没有合适的目标。")
        return
    print()
    shown = plan.items if args.top <= 0 else plan.items[: args.top]
    for i, item in enumerate(shown, 1):
        t = item.target
        mag = f"{item.magnitude:+.1f} 等" if item.magnitude is not None else ""
        print(
            f"{i:2d}. {t.name}（{t.kind}，{mag}）难度 {STARS[:item.difficulty]:<3}  "
            f"最佳 {_hm(item.best_time)}，{astro.compass(item.best_az)}方高 {item.best_alt:.0f}°"
        )
        for n in item.notes:
            print(f"      · {n}")
    if len(shown) < len(plan.items):
        print(f"\n还有 {len(plan.items) - len(shown)} 个目标，用 --top 0 查看全部")


def cmd_where(args) -> None:
    obs = _observer(args)
    when = _when(args, obs)
    t = find(args.target)
    s = locate(t, when, obs)
    print(f"{t.name}（{t.kind}） {when:%Y-%m-%d %H:%M}")
    print(f"  赤经 {astro.hms(s.ra / 15)}  赤纬 {astro.dms(s.dec)}")
    print(f"  方位 {s.az:.1f}°（{astro.compass(s.az)}）  高度 {s.alt:.1f}°")
    if s.alt < 0:
        print("  现在在地平线下面，看不到。")
    if s.magnitude is not None:
        print(f"  亮度 {s.magnitude:+.1f} 等")
    print(f"\n{t.story}")


def cmd_goto(args) -> None:
    obs = _observer(args)
    when = _when(args, obs)
    mount = AltAzMount(SimulatedDriver(), obs)
    if args.align:
        mount.align(args.align, when)
        print(f"已用 {find(args.align).name} 校准：当前指向 高度 {mount.alt:.1f}° 方位 {mount.az:.1f}°")
    try:
        plan = mount.goto(args.target, when)
    except UnsafeTarget as e:
        print(f"⛔ {e}")
        sys.exit(1)
    print(f"转向 {plan.target.name}：高度 {plan.alt:.2f}°，方位 {plan.az:.2f}°（{astro.compass(plan.az)}）")
    print(f"  上下轴 {plan.alt_steps:+d} 步，左右轴 {plan.az_steps:+d} 步（模拟）")
    for w in plan.warnings:
        print(f"  ⚠ {w}")
    print(f"\n{plan.target.story}")


def cmd_list(args) -> None:
    kinds: dict[str, list[str]] = {}
    for t in ALL_TARGETS:
        kinds.setdefault(t.kind, []).append(f"{t.name}({t.key})")
    for kind, names in kinds.items():
        print(f"{kind}：{'、'.join(names)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="telescope", description="家庭智能望远镜")
    sub = p.add_subparsers(dest="command", required=True)

    def add_place(sp):
        sp.add_argument("--city", default="北京", help=f"城市，可选：{'、'.join(CITIES)}")
        sp.add_argument("--lat", type=float, help="纬度（北纬为正），和 --lon 一起用时忽略 --city")
        sp.add_argument("--lon", type=float, help="经度（东经为正）")
        sp.add_argument("--tz", type=float, default=8.0, help="时区，默认 8（北京时间）")

    def add_scope(sp, required=False):
        sp.add_argument("--aperture", type=float, required=required, help="口径 mm")
        sp.add_argument("--focal", type=float, required=required, help="焦距 mm")

    sp = sub.add_parser("optics", help="计算倍率、视场、极限星等")
    add_scope(sp, required=True)
    sp.add_argument("--eyepiece", type=float, nargs="*", default=[], help="目镜焦距 mm，可写多个")
    sp.add_argument("--afov", type=float, default=52.0, help="目镜视场角，默认 52°")
    sp.add_argument("--barlow", type=float, default=1.0, help="巴洛镜倍数")
    sp.add_argument("--camera", choices=["hq", "v3"], help="树莓派相机：hq 或 v3")
    sp.set_defaults(func=cmd_optics)

    sp = sub.add_parser("tonight", help="今晚看什么")
    add_place(sp)
    add_scope(sp)
    sp.add_argument("--date", help="日期，例如 2026-10-10，默认今天")
    sp.add_argument("--top", type=int, default=12, help="只显示最值得看的前几个，0 表示全部")
    sp.set_defaults(func=cmd_tonight)

    for name, func, help_ in (("where", cmd_where, "目标现在在哪里"), ("goto", cmd_goto, "模拟自动寻星")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("target", help="目标，例如 土星、M42、织女星")
        add_place(sp)
        sp.add_argument("--time", help="当地时间，例如 '2026-10-10 21:00'，默认现在")
        if name == "goto":
            sp.add_argument("--align", help="先用这颗星校准，例如 织女星")
        sp.set_defaults(func=func)

    sp = sub.add_parser("list", help="列出全部目标")
    sp.set_defaults(func=cmd_list)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (KeyError, ValueError) as e:
        print(f"出错了：{e.args[0] if e.args else e}")
        sys.exit(2)
