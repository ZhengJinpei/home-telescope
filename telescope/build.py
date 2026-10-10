"""自己动手做牛顿反射望远镜：从主镜参数算出镜筒、副镜、底座的全部尺寸，并生成切割图纸。

坐标约定：沿镜筒的位置都从镜筒尾端（主镜那一头）量起，单位 mm。
底座是多布森式（Dobsonian）：镜筒放在托架上，托架两侧的圆盘是上下转的轴承，
摇臂箱在地板上左右转。木板全部按生成的 DXF 图纸交给 CNC 切割店加工。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .optics import PI_HQ_CAMERA, Camera, Telescope

GLASS_DENSITY_KG_MM3 = 2.5e-6  # 普通光学玻璃 / 硼硅玻璃约 2.2–2.5 g/cm³
# 淘宝上常见的副镜短轴尺寸（mm）
SECONDARY_SIZES = (22, 25, 30, 35, 40, 46, 50, 58, 63, 70)
# 常见的 2 寸延长筒长度（mm）
EXTENSION_TUBES = (25, 35, 50, 75)

# 焦点相对调焦座抽筒顶端的位置（mm，往外为正）
EYEPIECE_FOCUS = (-10.0, 5.0)  # 大多数 1.25 寸目镜的焦面在镜身台阶附近
EYEPIECE_KG = 0.3
CAMERA_MARGIN_MM = 5.0  # 相机对焦时抽筒至少伸出这么多，别顶到底
MIN_FULL_FIELD_MM = 10.0  # 目视：中心这么大的范围要 100% 亮
MIN_CORNER_LIGHT = 0.8  # 拍照：四角至少 80% 亮，剩下的暗角拍平场就能修正
WAVELENGTH_MM = 0.00055


@dataclass(frozen=True)
class NewtonianSpec:
    aperture_mm: float = 150.0
    focal_length_mm: float = 1200.0
    mirror_thickness_mm: float = 25.0
    tube_od_mm: float = 200.0  # 镜筒外径，默认 200 mm PVC 排水管
    tube_wall_mm: float = 3.9
    tube_kg_per_m: float = 3.5  # 200 mm PVC 排水管（壁厚 3.9）每米约 3.5 kg
    focuser_height_mm: float = 55.0  # 调焦座收到底时，从镜筒外表面到抽筒顶端
    focuser_travel_mm: float = 35.0
    cell_depth_mm: float = 50.0  # 镜筒尾端到主镜背面；自己做的主镜座 = 2 × 板厚 + 14
    camera: Camera = PI_HQ_CAMERA  # 按这台相机设计焦点位置和副镜大小
    # 指定 100% 照明的视场直径；不指定时自动选：中心至少 10 mm 全亮，相机四角至少 80% 亮
    illuminated_field_mm: float | None = None
    front_kg: float = 1.1  # 调焦座、副镜和支架、寻星镜、目镜（不含相机）
    cell_kg: float = 0.6

    def __post_init__(self):
        if self.focal_length_mm < 3 * self.aperture_mm:
            raise ValueError("焦比小于 f/3 的主镜不适合自己做第一台")
        if self.tube_od_mm - 2 * self.tube_wall_mm < self.aperture_mm + 20:
            raise ValueError("镜筒内径至少要比主镜大 20 mm，否则会挡光")
        if self.camera.back_focus_mm is None:
            raise ValueError(f"{self.camera.name}的镜头拆不下来，不能直焦拍摄")


@dataclass
class NewtonianDesign:
    spec: NewtonianSpec
    focal_plane_above_drawtube: float  # 调焦座收到底时，焦点在抽筒顶端外多少 mm
    extension_tube_mm: float  # 目镜要加的延长筒，0 表示不用
    eyepiece_racked_out: tuple[float, float]  # 目镜对焦时抽筒伸出的范围（已算上延长筒）
    camera_racked_out: float
    secondary_to_focus: float  # 副镜中心到焦点的光程
    secondary_minor_axis: float
    secondary_min_needed: float
    secondary_offset: float
    fully_illuminated_mm: float
    corner_illumination: float  # 相机画面四角的亮度（中心为 1）
    obstruction: float  # 按直径算
    coma_free_mm: float  # 彗差小于艾里斑的视场直径
    corner_coma_arcsec: float  # 相机画面四角星点被彗差拉长多少角秒
    mirror_face_pos: float  # 主镜镜面到镜筒尾端
    focuser_pos: float  # 调焦座开孔中心到镜筒尾端
    tube_length: float
    mirror_kg: float
    total_kg: float  # 目视时
    balance_pos: float  # 目视时重心到镜筒尾端
    balance_camera_pos: float  # 换上相机后的重心
    warnings: list[str] = field(default_factory=list)

    @property
    def telescope(self) -> Telescope:
        return Telescope(self.spec.aperture_mm, self.spec.focal_length_mm)

    @property
    def balance_range(self) -> tuple[float, float]:
        return min(self.balance_pos, self.balance_camera_pos), max(self.balance_pos, self.balance_camera_pos)


def _overlap_fraction(r_cone: float, r_sec: float, offset: float) -> float:
    """离轴光锥（半径 r_cone，偏心 offset）落在副镜（半径 r_sec）上的比例：两圆相交面积 / 光锥面积。"""
    if offset >= r_cone + r_sec:
        return 0.0
    if offset <= abs(r_sec - r_cone):
        return 1.0 if r_sec >= r_cone else (r_sec / r_cone) ** 2
    a = r_cone ** 2 * math.acos((offset ** 2 + r_cone ** 2 - r_sec ** 2) / (2 * offset * r_cone))
    b = r_sec ** 2 * math.acos((offset ** 2 + r_sec ** 2 - r_cone ** 2) / (2 * offset * r_sec))
    c = 0.5 * math.sqrt((-offset + r_cone + r_sec) * (offset + r_cone - r_sec)
                        * (offset - r_cone + r_sec) * (offset + r_cone + r_sec))
    return (a + b - c) / (math.pi * r_cone ** 2)


def design_newtonian(spec: NewtonianSpec = NewtonianSpec()) -> NewtonianDesign:
    warnings = []
    d, fl, travel = spec.aperture_mm, spec.focal_length_mm, spec.focuser_travel_mm
    cam = spec.camera
    f_ratio = fl / d

    # 焦点放在哪：相机要够得着（抽筒不能缩进负数），目镜不能超出调焦行程
    lo = max(cam.back_focus_mm + CAMERA_MARGIN_MM, EYEPIECE_FOCUS[1])
    hi = travel + EYEPIECE_FOCUS[0]
    extension = 0.0
    if lo <= hi:
        focal_out = (lo + hi) / 2
    else:
        # 单反这类后截距长的相机：按相机设计，目镜加一节延长筒
        focal_out = lo
        need = (focal_out - EYEPIECE_FOCUS[0] - travel, focal_out - EYEPIECE_FOCUS[1])
        fits = [e for e in EXTENSION_TUBES if need[0] <= e <= need[1]]
        extension = fits[0] if fits else min(EXTENSION_TUBES, key=lambda e: abs(e - sum(need) / 2))
        if not fits:
            warnings.append(f"找不到合适的延长筒（需要 {need[0]:.0f}–{need[1]:.0f} mm），换行程更长的调焦座")
    eye_out = (focal_out - extension - EYEPIECE_FOCUS[1], focal_out - extension - EYEPIECE_FOCUS[0])
    cam_out = focal_out - cam.back_focus_mm

    sec_to_focus = spec.tube_od_mm / 2 + spec.focuser_height_mm + focal_out
    corner = cam.diagonal_mm / 2

    def fully_lit(minor: float) -> float:
        return (minor * fl - d * sec_to_focus) / (fl - sec_to_focus)

    def corner_lit(minor: float) -> float:
        return _overlap_fraction(d / 2 * sec_to_focus / fl, minor / 2, corner * (1 - sec_to_focus / fl))

    # 经典公式：短轴 = 全亮视场 + (口径 - 全亮视场) × 副镜到焦点距离 / 焦距
    field_mm = spec.illuminated_field_mm or MIN_FULL_FIELD_MM
    needed = field_mm + (d - field_mm) * sec_to_focus / fl
    ok = [m for m in SECONDARY_SIZES if m >= needed
          and (spec.illuminated_field_mm is not None or corner_lit(m) >= MIN_CORNER_LIGHT)]
    if not ok:
        raise ValueError(f"副镜要 {needed:.0f} mm 以上，太大了，换更长焦距的主镜或更矮的调焦座")
    minor = ok[0]
    fully, corner_light = fully_lit(minor), corner_lit(minor)
    obstruction = minor / d
    if obstruction > 0.3:
        warnings.append(f"副镜挡光 {obstruction:.0%}（按直径），看行星的反差会变差")

    # 彗差：像高 h 处星点拉长 3h / (16 F²)；小于艾里斑 2.44 λ F 的范围算“没有彗差”
    coma_free = 2 * 2.44 * WAVELENGTH_MM * 16 * f_ratio ** 3 / 3
    corner_coma = 3 * corner / (16 * f_ratio ** 2) / fl * 206265

    mirror_face = spec.cell_depth_mm + spec.mirror_thickness_mm
    focuser_pos = mirror_face + fl - sec_to_focus
    tube_len = focuser_pos + max(100.0, 0.75 * spec.tube_od_mm)  # 前面多留一段挡杂光、防结露

    mirror_kg = math.pi * (d / 2) ** 2 * spec.mirror_thickness_mm * GLASS_DENSITY_KG_MM3
    tube_kg = spec.tube_kg_per_m * tube_len / 1000
    rear = [
        (mirror_kg, spec.cell_depth_mm + spec.mirror_thickness_mm / 2),
        (spec.cell_kg, spec.cell_depth_mm / 2),
        (tube_kg, tube_len / 2),
    ]

    def balance(front_kg: float) -> tuple[float, float]:
        parts = rear + [(front_kg, focuser_pos)]
        total = sum(m for m, _ in parts)
        return total, sum(m * z for m, z in parts) / total

    total, bal = balance(spec.front_kg)
    _, bal_cam = balance(spec.front_kg - EYEPIECE_KG + cam.mass_kg)

    return NewtonianDesign(
        spec, focal_out, extension, eye_out, cam_out, sec_to_focus, minor, needed,
        minor / (4 * f_ratio), fully, corner_light, obstruction, coma_free, corner_coma,
        mirror_face, focuser_pos, tube_len, mirror_kg, total, bal, bal_cam, warnings,
    )


# ---------------------------------------------------------------- 多布森底座


@dataclass
class DobsonianDesign:
    plywood_mm: float
    cradle_length: float  # 托架沿镜筒方向的长度（= 两块托架板外侧之间）
    cradle_width: float
    cradle_height: float  # 托架板高度（镜筒中心以下）
    bearing_radius: float  # 侧轴承圆盘半径
    pivot_height: float  # 上下转轴离摇臂箱底板上表面的高度
    rocker_inner_width: float  # 两块摇臂箱侧板内侧之间的距离
    rocker_side_width: float  # 摇臂箱侧板宽（沿镜筒方向）
    front_board_height: float
    floor_width: float
    ground_diameter: float
    az_pad_radius: float  # 地板上三块铁氟龙垫片离中心的距离（摇臂箱转到哪个角度都要压得到）
    eyepiece_height_zenith: float  # 指向天顶时目镜离地高度
    eyepiece_height_horizon: float
    alt_steps_per_rev: int  # 电动化以后每圈步数，给 mount.AltAzMount 用（高度轴绕在驱动盘上）
    az_steps_per_rev: int


PAD_MM = 3.0  # 聚四氟乙烯（铁氟龙）垫片厚度
# 高度轴的同步带不能绕在侧轴承圆盘边上（那里贴着垫片滑动），另外切一块小一圈的驱动盘
DRIVE_INSET_MM = 25.0
FEET_MM = 25.0  # 地板下三个脚垫的高度


def belt_steps_per_rev(radius_mm: float, pulley_teeth: int = 20, microsteps: int = 16) -> int:
    """GT2 同步带（齿距 2 mm）绕在半径 radius_mm 的圆盘上，电机 200 步/圈。"""
    ratio = 2 * math.pi * radius_mm / (pulley_teeth * 2.0)
    return round(200 * microsteps * ratio)


def design_dobsonian(newt: NewtonianDesign, plywood_mm: float = 18.0) -> DobsonianDesign:
    t = plywood_mm
    od = newt.spec.tube_od_mm
    cradle_len = od
    cradle_h = od / 2 + 30
    cradle_w = od + 50  # 半圆缺口两边各留 25 mm 木头
    # 托架板转起来不能超出圆盘，否则会刮到摇臂箱
    rb = 5 * math.ceil((math.hypot(cradle_len / 2, cradle_h) + 10) / 5)
    notch = rb + PAD_MM
    # 圆盘装在托架板外侧，摇臂箱侧板和圆盘在同一平面（各留 2 mm 余量），
    # 圆盘边缘正好压在侧板缺口的垫片上；托架板转动时始终在缺口圆以内，碰不到侧板
    inner = cradle_w + 2 * 2
    side_w = 2 * notch + 120
    # 镜筒尾端转到最低时也要离底板 25 mm 以上
    # 装相机后托架会往前挪，按两种重心里离尾端最远的算
    rear_reach = math.hypot(newt.balance_range[1], od / 2)
    pivot = 5 * math.ceil(max(rear_reach + 25, notch + 80) / 5)
    front_h = pivot - rb - 20
    floor_w = inner + 2 * t
    ground = 10 * math.ceil(max(floor_w, side_w) / 10)
    az_pad = 5 * math.floor((min(floor_w, side_w) / 2 - 20) / 5)
    base = FEET_MM + t + PAD_MM + t
    zenith = base + pivot + (newt.focuser_pos - newt.balance_pos)
    horizon = base + pivot + od / 2 + newt.spec.focuser_height_mm
    return DobsonianDesign(
        t, cradle_len, cradle_w, cradle_h, rb, pivot, inner, side_w, front_h, floor_w, ground, az_pad,
        zenith, horizon, belt_steps_per_rev(rb - DRIVE_INSET_MM), belt_steps_per_rev(ground / 2),
    )


# ---------------------------------------------------------------- 切割图纸


@dataclass
class Part:
    name: str
    count: int
    width: float
    height: float
    cuts: list[tuple] = field(default_factory=list)  # ("line", x1, y1, x2, y2) / ("arc", cx, cy, r, a0, a1) / ("circle", cx, cy, r)
    drills: list[tuple] = field(default_factory=list)  # 只打定位小孔，不切穿
    note: str = ""


def _rect_with_top_notch(w: float, h: float, r: float) -> list[tuple]:
    """矩形板，上边正中挖一个半径 r 的半圆。"""
    cx = w / 2
    return [
        ("line", 0, 0, w, 0),
        ("line", w, 0, w, h),
        ("line", w, h, cx + r, h),
        ("arc", cx, h, r, 180, 360),
        ("line", cx - r, h, 0, h),
        ("line", 0, h, 0, 0),
    ]


def _rect(w: float, h: float) -> list[tuple]:
    return [("line", 0, 0, w, 0), ("line", w, 0, w, h), ("line", w, h, 0, h), ("line", 0, h, 0, 0)]


CELL_SPRING_GAP_MM = 14.0  # 主镜座两块板之间弹簧的空隙


def cell_depth(plywood_mm: float) -> float:
    """自己做的主镜座：后板 + 弹簧空隙 + 托板。"""
    return 2 * plywood_mm + CELL_SPRING_GAP_MM


def _polar(r: float, angles: tuple[float, ...], cx: float, cy: float, hole_r: float) -> list[tuple]:
    return [("circle", cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)), hole_r)
            for a in angles]


def cell_parts(newt: NewtonianDesign) -> list[Part]:
    """主镜座两块圆板。三颗 M6 校准螺丝在 BOLTS 方向，托镜的三块软垫在 PADS 方向。"""
    s = newt.spec
    back_r = (s.tube_od_mm - 2 * s.tube_wall_mm) / 2 - 0.5
    front_r = s.aperture_mm / 2 - 5
    bolt_r = 0.8 * front_r
    bolts, pads, vents = (90.0, 210.0, 330.0), (30.0, 150.0, 270.0), (30.0, 150.0, 270.0)
    return [
        Part("主镜座后板", 1, 2 * back_r, 2 * back_r,
             [("circle", back_r, back_r, back_r)]
             + _polar(bolt_r, bolts, back_r, back_r, 3.5)
             + _polar(0.62 * back_r, vents, back_r, back_r, 15.0),
             note="正好塞进镜筒尾端，从管壁侧面拧 4–6 颗木螺丝固定。三个小孔穿 M6 校准螺丝，大孔通风"),
        Part("主镜托板", 1, 2 * front_r, 2 * front_r,
             [("circle", front_r, front_r, front_r), ("circle", front_r, front_r, 20.0)]
             + _polar(bolt_r, bolts, front_r, front_r, 5.0),
             _polar(0.7 * s.aperture_mm / 2, pads, front_r, front_r, 1.5),
             note="三个孔里敲进 M6 预埋螺母；三个定位小孔处贴软木垫托住主镜；中间大孔通风"),
    ]


def cut_parts(newt: NewtonianDesign, dob: DobsonianDesign) -> list[Part]:
    od = newt.spec.tube_od_mm
    rb, g = dob.bearing_radius, dob.ground_diameter / 2
    return cell_parts(newt) + [
        Part("侧轴承圆盘", 2, 2 * rb, 2 * rb, [("circle", rb, rb, rb)], [("circle", rb, rb, 1.5)],
             "中心小孔是转轴位置。装在托架板外侧，边缘贴一圈防火板，在摇臂箱的铁氟龙垫片上滑动"),
        Part("高度驱动盘", 1, 2 * (rb - DRIVE_INSET_MM), 2 * (rb - DRIVE_INSET_MM),
             [("circle", rb - DRIVE_INSET_MM, rb - DRIVE_INSET_MM, rb - DRIVE_INSET_MM)],
             [("circle", rb - DRIVE_INSET_MM, rb - DRIVE_INSET_MM, 1.5)],
             "第 3 阶段电动化用，先一起切好。同心地粘在一块侧轴承圆盘外侧，同步带绕在它边上"),
        Part("托架板", 2, dob.cradle_width, dob.cradle_height,
             _rect_with_top_notch(dob.cradle_width, dob.cradle_height, od / 2 + 1),
             note="夹在两个圆盘之间，半圆缺口托住镜筒，缺口上沿和圆盘中心孔齐平"),
        Part("摇臂箱侧板", 2, dob.rocker_side_width, dob.pivot_height,
             _rect_with_top_notch(dob.rocker_side_width, dob.pivot_height, rb + PAD_MM),
             note="半圆缺口左右 45° 处各贴一块铁氟龙垫片"),
        Part("摇臂箱前板", 1, dob.rocker_inner_width, dob.front_board_height,
             _rect(dob.rocker_inner_width, dob.front_board_height),
             note="装在镜筒口那一侧；镜筒尾端那一侧不装板，留给镜筒转到天顶"),
        Part("摇臂箱底板", 1, dob.floor_width, dob.rocker_side_width,
             _rect(dob.floor_width, dob.rocker_side_width) + [("circle", dob.floor_width / 2, dob.rocker_side_width / 2, 6.5)],
             note="中心孔穿 M12 螺栓当方位转轴，底面贴防火板"),
        Part("地板", 1, 2 * g, 2 * g, [("circle", g, g, g), ("circle", g, g, 6.5)],
             _polar(dob.az_pad_radius, (90.0, 210.0, 330.0), g, g, 1.5),
             note="中心孔穿 M12 螺栓；三个定位小孔处贴铁氟龙垫片，下面装三个脚"),
    ]


def _layout(parts: list[Part], gap: float = 20.0) -> list[tuple[Part, float, float]]:
    """把每块板按数量排成一行，返回 (板, x 偏移, y 偏移)。"""
    placed, x = [], 0.0
    for p in parts:
        for _ in range(p.count):
            placed.append((p, x, 0.0))
            x += p.width + gap
    return placed


def write_dxf(parts: list[Part], path: Path) -> None:
    """AutoCAD R12 格式，单位 mm。CUT 图层是切割线，DRILL 是定位孔，LABEL 是文字（不切）。

    中文 CAD 软件按 GBK 读 R12 文件，所以用 GBK 编码保存。
    """
    out = ["0", "SECTION", "2", "HEADER", "9", "$ACADVER", "1", "AC1009",
           "9", "$DWGCODEPAGE", "3", "ANSI_936", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]

    def ent(kind: str, layer: str, *pairs) -> None:
        out.extend(["0", kind, "8", layer])
        for code, value in pairs:
            out.extend([str(code), f"{value:.3f}" if isinstance(value, float) else str(value)])

    def shape(prims, layer, ox, oy):
        for prim in prims:
            if prim[0] == "line":
                _, x1, y1, x2, y2 = prim
                ent("LINE", layer, (10, ox + x1), (20, oy + y1), (11, ox + x2), (21, oy + y2))
            elif prim[0] == "arc":
                _, cx, cy, r, a0, a1 = prim
                ent("ARC", layer, (10, ox + cx), (20, oy + cy), (40, float(r)), (50, float(a0)), (51, float(a1)))
            else:
                _, cx, cy, r = prim
                ent("CIRCLE", layer, (10, ox + cx), (20, oy + cy), (40, float(r)))

    for p, ox, oy in _layout(parts):
        shape(p.cuts, "CUT", ox, oy)
        shape(p.drills, "DRILL", ox, oy)
        ent("TEXT", "LABEL", (10, ox), (20, oy - 30.0), (40, 15.0), (1, f"{p.name} {p.width:.0f}x{p.height:.0f}"))
    out += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(out) + "\n", encoding="gbk")


def write_svg(parts: list[Part], path: Path, plywood_mm: float) -> None:
    """给人看的预览图，尺寸单位 mm。"""
    placed = _layout(parts)
    width = max(ox + p.width for p, ox, _ in placed) + 20
    height = max(p.height for p, _, _ in placed) + 100

    def y(v: float) -> float:
        return height - 50 - v  # SVG 的 y 轴朝下

    body = []
    for p, ox, oy in placed:
        for prims, color in ((p.cuts, "#c0392b"), (p.drills, "#2471a3")):
            for prim in prims:
                if prim[0] == "line":
                    _, x1, y1, x2, y2 = prim
                    body.append(f'<line x1="{ox + x1:.1f}" y1="{y(oy + y1):.1f}" x2="{ox + x2:.1f}" '
                                f'y2="{y(oy + y2):.1f}" stroke="{color}"/>')
                elif prim[0] == "arc":
                    _, cx, cy, r, a0, a1 = prim
                    sx, sy = ox + cx + r * math.cos(math.radians(a0)), oy + cy + r * math.sin(math.radians(a0))
                    ex, ey = ox + cx + r * math.cos(math.radians(a1)), oy + cy + r * math.sin(math.radians(a1))
                    large = 1 if (a1 - a0) % 360 > 180 else 0
                    body.append(f'<path d="M {sx:.1f} {y(sy):.1f} A {r:.1f} {r:.1f} 0 {large} 0 {ex:.1f} {y(ey):.1f}" '
                                f'stroke="{color}" fill="none"/>')
                else:
                    _, cx, cy, r = prim
                    body.append(f'<circle cx="{ox + cx:.1f}" cy="{y(oy + cy):.1f}" r="{r:.1f}" stroke="{color}" fill="none"/>')
        body.append(f'<text x="{ox:.1f}" y="{y(oy) + 28:.1f}" font-size="16">{p.name} {p.width:.0f}×{p.height:.0f}</text>')
    title = f"主镜座和多布森底座切割图（单位 mm，板厚 {plywood_mm:g} mm，红线切割，蓝点为定位孔）"
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}mm" height="{height:.0f}mm" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">\n'
        f'<rect width="100%" height="100%" fill="white"/>\n'
        f'<text x="10" y="24" font-size="18">{title}</text>\n'
        f'<g stroke-width="1.5">\n' + "\n".join(body) + "\n</g>\n</svg>\n",
        encoding="utf-8",
    )
