"""命令行：python -m telescope <命令>"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime

from pathlib import Path

from . import astro, build, mirror, optics
from .events import Event, find_events, highlights
from .catalog import ALL_TARGETS, find
from .mount import AltAzMount, SimulatedDriver, UnsafeTarget
from .sky import CITIES, Observer, locate, observer_for_city, plan_tonight

STARS = "★★★"
WEEKDAYS = "一二三四五六日"
TAGS = {1: "", 2: "【推荐】", 3: "【不要错过】"}


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


def _day(t: datetime) -> str:
    return f"{t:%m-%d} 周{WEEKDAYS[t.weekday()]}"


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
        cam = optics.CAMERAS[args.camera]
        cv = optics.camera_view(scope, cam)
        print(f"\n接 {cam.name} 直焦拍摄：")
        print(f"  {cv.arcsec_per_pixel:.2f} 角秒/像素，画面 {cv.fov_width_deg:.2f}° × {cv.fov_height_deg:.2f}°")
        print("  " + ("能拍下整个月亮" if cv.moon_fits else "拍不下整个月亮"))
        print(f"  {cv.sampling}")


def cmd_build(args) -> None:
    spec = build.NewtonianSpec(
        aperture_mm=args.aperture, focal_length_mm=args.focal, mirror_thickness_mm=args.thickness,
        tube_od_mm=args.tube_od, tube_wall_mm=args.tube_wall, tube_kg_per_m=args.tube_kg,
        focuser_height_mm=args.focuser_height, focuser_travel_mm=args.focuser_travel,
        cell_depth_mm=build.cell_depth(args.plywood), camera=optics.CAMERAS[args.camera],
    )
    n = build.design_newtonian(spec)
    dob = build.design_dobsonian(n, args.plywood)
    scope = n.telescope
    print(f"牛顿反射镜 {spec.aperture_mm:g} mm f/{scope.focal_ratio:.1f}（焦距 {spec.focal_length_mm:g} mm）")
    print(f"  看得见 {scope.limiting_magnitude:.1f} 等星，分辨率 {scope.dawes_limit_arcsec:.2f} 角秒，"
          f"最高有效倍率约 {scope.max_useful_magnification:.0f} 倍")

    print("\n镜筒（位置都从镜筒尾端量起）")
    print(f"  镜筒：外径 {spec.tube_od_mm:g} mm，切成 {n.tube_length:.0f} mm 长")
    print(f"  主镜镜面：距尾端 {n.mirror_face_pos:.0f} mm（主镜座占 {spec.cell_depth_mm:g} mm，镜厚 {spec.mirror_thickness_mm:g} mm）")
    print(f"  调焦座开孔中心：距尾端 {n.focuser_pos:.0f} mm")
    cam = spec.camera
    print(f"  副镜：短轴 {n.secondary_minor_axis:g} mm，挡光 {n.obstruction:.0%}，中心对准开孔中心，"
          f"再往主镜方向、背离调焦座方向各偏 {n.secondary_offset:.1f} mm")
    print(f"  焦点：调焦座收到底时在抽筒顶端外 {n.focal_plane_above_drawtube:.1f} mm（按{cam.name}设计）")
    print(f"    相机对焦时抽筒伸出约 {n.camera_racked_out:.0f} mm（调焦行程 {spec.focuser_travel_mm:g} mm）")
    ext = f"，要先插一节 {n.extension_tube_mm:g} mm 的 2 寸延长筒" if n.extension_tube_mm else ""
    print(f"    目镜对焦时抽筒伸出约 {n.eyepiece_racked_out[0]:.0f}–{n.eyepiece_racked_out[1]:.0f} mm{ext}")
    print(f"  照明：中心 {n.fully_illuminated_mm:.0f} mm 范围全亮，相机画面四角亮度 {n.corner_illumination:.0%}"
          + ("（拍平场就能修正）" if n.corner_illumination < 0.95 else ""))
    if n.corner_coma_arcsec > 3:
        print(f"  彗差：画面四角的星点被拉长约 {n.corner_coma_arcsec:.0f} 角秒（中间 {n.coma_free_mm:.1f} mm 范围没有彗差）。"
              "拍月亮没问题，拍星云星团建议加彗差改正镜")
    print(f"  重量：主镜约 {n.mirror_kg:.1f} kg，镜筒总重约 {n.total_kg:.1f} kg（目视）")
    print(f"  重心：目视时距尾端 {n.balance_pos:.0f} mm，换上相机后 {n.balance_camera_pos:.0f} mm")
    for w in n.warnings:
        print(f"  ⚠ {w}")

    print(f"\n多布森底座（{args.plywood:g} mm 多层板）")
    print(f"  侧轴承圆盘直径 {2 * dob.bearing_radius:.0f} mm，转轴离摇臂箱底板 {dob.pivot_height:.0f} mm")
    print(f"  地板上的三块铁氟龙垫片离中心 {dob.az_pad_radius:.0f} mm（图纸上有定位孔）")
    shift = abs(n.balance_camera_pos - n.balance_pos)
    print(f"  托架中心对准镜筒重心（目视时距尾端 {n.balance_pos:.0f} mm）；换上相机后把镜筒在托架里挪 {shift:.0f} mm")
    print(f"  目镜高度：指向天顶时离地约 {dob.eyepiece_height_zenith / 10:.0f} cm，"
          f"指向地平线时约 {dob.eyepiece_height_horizon / 10:.0f} cm")
    if dob.eyepiece_height_zenith > 1100:
        print("  小朋友看天顶附近的目标需要一个踏脚凳")
    print(f"  以后装电机（GT2 同步带 + 20 齿轮 + 16 细分）：高度轴 {dob.alt_steps_per_rev} 步/圈"
          f"（{1296000 / dob.alt_steps_per_rev:.0f} 角秒/步），方位轴 {dob.az_steps_per_rev} 步/圈"
          f"（{1296000 / dob.az_steps_per_rev:.0f} 角秒/步）")

    parts = build.cut_parts(n, dob)
    print("\n切割清单：")
    for p in parts:
        print(f"  {p.name} × {p.count}：{p.width:.0f} × {p.height:.0f} mm  {p.note}")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"telescope_{spec.aperture_mm:g}mm"
        build.write_dxf(parts, out / f"{stem}.dxf")
        build.write_svg(parts, out / f"{stem}.svg", args.plywood)
        print(f"\n图纸已保存：{out / stem}.dxf（发给 CNC 切割店）和 .svg（用浏览器打开预览）")
    else:
        print("\n加上 --out 文件夹 可以生成切割图纸")


def cmd_mirror(args) -> None:
    p = mirror.PrimarySpec(args.aperture, args.focal, args.thickness)
    n = build.design_newtonian(build.NewtonianSpec(
        aperture_mm=args.aperture, focal_length_mm=args.focal, mirror_thickness_mm=args.thickness))
    sec = mirror.SecondarySpec(n.secondary_minor_axis)
    lam = mirror.TEST_WAVELENGTH_NM

    print(f"主镜（{p.diameter_mm:g} mm f/{p.focal_ratio:g} 抛物面反射镜）")
    print(f"  直径 {p.diameter_mm:g} mm（-0/-0.2），有效口径 ≥ {p.clear_aperture_mm:g} mm")
    print(f"  厚度 {p.thickness_mm:g} mm（径厚比约 1:{p.diameter_mm / p.thickness_mm:.0f}），重约 {p.mass_kg:.1f} kg")
    print(f"  焦距 {p.focal_length_mm:g} mm ±1%，曲率半径 R = {p.radius_of_curvature_mm:g} mm，请附实测焦距")
    print(f"  面形：抛物面（圆锥系数 K = -1），镜面中心凹深 {p.sagitta_mm:.3f} mm")
    print(f"    从最接近的球面修成抛物面，只需改动约 {p.parabolizing_nm:.0f} nm")
    print(f"  精度（{lam:g} nm）：波面 PV ≤ λ/8（合格线 λ/6），RMS ≤ λ/25，斯特列尔比 ≥ 0.9")
    print(f"    即面形 PV ≤ λ/16。请附干涉检测报告；没有干涉仪可以附傅科检验读数")
    print("  材料：硼硅玻璃或微晶玻璃（热胀冷缩小）；K9 也能用，出门后要多等一会儿")
    print("  背面细磨平整，侧面细磨，两面边缘倒角 0.5–1 mm × 45°")
    print("  镀膜：增强铝膜（可见光平均反射率 ≥ 94%）+ 二氧化硅保护层")
    print("  中心做一个小标记（直径 5 mm 以内的圆点或圆环），校准光轴用")

    pv, rms, strehl = mirror.sphere_error(p.diameter_mm, p.focal_length_mm)
    if strehl < 0.7:
        verdict = "，星点明显发糊，必须修抛物面"
    elif strehl < 0.8:
        verdict = "，差一点到衍射极限（0.8），建议修抛物面"
    else:
        verdict = "，已经达到衍射极限，球面就够用"
    print(f"\n如果只能做球面：这块镜子做成球面，波面误差 PV {pv:.2f}λ，斯特列尔比 {strehl:.2f}{verdict}")
    f_min = mirror.min_focal_ratio_for_sphere(p.diameter_mm)
    print(f"  {p.diameter_mm:g} mm 的球面镜要到 f/{f_min:g}（焦距 {f_min * p.diameter_mm:.0f} mm）以上才达到衍射极限")

    print(f"\n傅科检验读数（在球心检验，光源随刀口移动，等面积分 4 个带区）")
    for z in mirror.foucault_zones(p.diameter_mm, p.focal_length_mm):
        print(f"  {z.inner_mm:5.1f}–{z.outer_mm:5.1f} mm 带区：刀口比中心后退 {z.reading_mm:.3f} mm")

    print(f"\n副镜（椭圆平面镜，45° 放置）")
    print(f"  短轴 {sec.minor_axis_mm:g} mm，长轴 {sec.major_axis_mm:.1f} mm，厚度 ≥ {sec.thickness_mm:g} mm")
    print(f"  平面度：面形 PV ≤ λ/10（{lam:g} nm）")
    print("  镀膜和主镜一样；边缘倒角，背面细磨")


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

    coming = highlights(find_events(night, 7, obs))
    if coming:
        print("\n近 7 天值得关注：")
        for e in coming:
            print(f"  {_day(e.when)}  {TAGS[e.importance]}{e.title}")
        print("  用 events 命令查看详情")


def _print_event(e: Event) -> None:
    hidden = "" if e.visible or "看不到" in e.title else "（本地看不到）"
    print(f"{_day(e.when)} {e.when:%H:%M}  {TAGS[e.importance]}{e.title}{hidden}")
    if e.view:
        print(f"      👀 {e.view}")
    for line in e.detail.splitlines():
        print(f"      {line}")


def cmd_events(args) -> None:
    obs = _observer(args)
    start = date.fromisoformat(args.date) if args.date else datetime.now(obs.tz).date()
    events = find_events(start, args.days, obs)
    shown = events if args.all else [e for e in events if e.visible]
    print(f"{start:%Y-%m-%d} 起 {args.days} 天的天象（北纬 {obs.lat:.1f}°，东经 {obs.lon:.1f}°）\n")
    if not shown:
        print("这段时间本地没有能看到的天象。")
    for e in shown:
        _print_event(e)
    if len(shown) < len(events):
        print(f"\n另有 {len(events) - len(shown)} 个天象本地看不到，用 --all 查看")


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
    sp.add_argument("--camera", choices=list(optics.CAMERAS), help="相机：hq（树莓派 HQ）、v3（树莓派相机 3）、80d（佳能 80D）")
    sp.set_defaults(func=cmd_optics)

    sp = sub.add_parser("build", help="自己做望远镜：算出镜筒和底座尺寸，生成切割图纸")
    sp.add_argument("--aperture", type=float, default=150, help="主镜口径 mm，默认 150")
    sp.add_argument("--focal", type=float, default=1200, help="主镜焦距 mm，默认 1200")
    sp.add_argument("--thickness", type=float, default=25, help="主镜厚度 mm，默认 25")
    sp.add_argument("--tube-od", type=float, default=200, help="镜筒外径 mm，默认 200（PVC 排水管）")
    sp.add_argument("--tube-wall", type=float, default=3.9, help="镜筒壁厚 mm，默认 3.9")
    sp.add_argument("--tube-kg", type=float, default=3.5, help="镜筒每米重量 kg，默认 3.5（200 mm PVC 管）")
    sp.add_argument("--focuser-height", type=float, default=55, help="调焦座收到底时的高度 mm")
    sp.add_argument("--focuser-travel", type=float, default=35, help="调焦行程 mm")
    sp.add_argument("--plywood", type=float, default=18, help="多层板厚度 mm，默认 18")
    sp.add_argument("--camera", choices=["hq", "80d"], default="hq",
                    help="按哪台相机设计焦点和副镜：hq（树莓派 HQ，默认）或 80d（佳能 80D 单反）")
    sp.add_argument("--out", help="把切割图纸（DXF + SVG）保存到这个文件夹")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("mirror", help="主镜和副镜的外协加工参数")
    sp.add_argument("--aperture", type=float, default=150, help="主镜口径 mm，默认 150")
    sp.add_argument("--focal", type=float, default=1200, help="主镜焦距 mm，默认 1200")
    sp.add_argument("--thickness", type=float, default=25, help="主镜厚度 mm，默认 25")
    sp.set_defaults(func=cmd_mirror)

    sp = sub.add_parser("tonight", help="今晚看什么")
    add_place(sp)
    add_scope(sp)
    sp.add_argument("--date", help="日期，例如 2026-10-10，默认今天")
    sp.add_argument("--top", type=int, default=12, help="只显示最值得看的前几个，0 表示全部")
    sp.set_defaults(func=cmd_tonight)

    sp = sub.add_parser("events", help="天象日历：月相、合月、冲日、流星雨、日月食")
    add_place(sp)
    sp.add_argument("--date", help="从哪天开始，例如 2026-10-10，默认今天")
    sp.add_argument("--days", type=int, default=30, help="往后看多少天，默认 30")
    sp.add_argument("--all", action="store_true", help="也列出本地看不到的天象")
    sp.set_defaults(func=cmd_events)

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
